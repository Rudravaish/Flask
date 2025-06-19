"""
Lightweight Model without PyTorch Dependencies
Fallback implementation using OpenCV and NumPy for skin lesion analysis
"""
import cv2
import numpy as np
from PIL import Image, ImageStat, ImageFilter, ImageEnhance
import logging
import os

# Configure logging
logger = logging.getLogger(__name__)

def analyze_color_asymmetry(image):
    """Analyze color asymmetry - melanomas often have uneven color distribution"""
    try:
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Split into quadrants
        h, w = image.shape[:2]
        mid_h, mid_w = h // 2, w // 2
        
        quadrants = [
            hsv[:mid_h, :mid_w],
            hsv[:mid_h, mid_w:],
            hsv[mid_h:, :mid_w],
            hsv[mid_h:, mid_w:]
        ]
        
        # Calculate color variance in each quadrant
        color_variances = []
        for quad in quadrants:
            if quad.size > 0:
                std = np.std(quad.reshape(-1, 3), axis=0)
                color_variances.append(np.mean(std))
        
        if not color_variances:
            return 0.2
            
        # High variance difference indicates asymmetry
        variance_diff = max(color_variances) - min(color_variances)
        asymmetry_score = min(variance_diff / 50.0, 1.0)
        
        return asymmetry_score
        
    except Exception as e:
        logger.warning(f"Color asymmetry analysis failed: {e}")
        return 0.2

def analyze_border_irregularity(image):
    """Analyze border irregularity - melanomas often have irregular borders"""
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Use adaptive thresholding
        binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 0.3
        
        # Get the largest contour (assumed to be the lesion)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Calculate irregularity metrics
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        
        if perimeter == 0:
            return 0.3
        
        # Circularity: 4π*area/perimeter²
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        # Irregularity score (1 - circularity, normalized)
        irregularity = max(0, 1 - circularity)
        border_score = min(irregularity * 2, 1.0)
        
        return border_score
        
    except Exception as e:
        logger.warning(f"Border irregularity analysis failed: {e}")
        return 0.3

def analyze_color_variation(image):
    """Analyze color variation - melanomas often have multiple colors"""
    try:
        # Convert to different color spaces for comprehensive analysis
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        # Calculate color diversity metrics
        rgb_std = np.std(image.reshape(-1, 3), axis=0)
        hsv_std = np.std(hsv.reshape(-1, 3), axis=0)
        lab_std = np.std(lab.reshape(-1, 3), axis=0)
        
        # Combine color variation metrics
        rgb_variation = np.mean(rgb_std) / 255.0
        hsv_variation = hsv_std[0] / 179.0  # Hue variation
        lab_variation = np.mean(lab_std) / 255.0
        
        # Weighted combination
        color_score = (rgb_variation * 0.4 + hsv_variation * 0.4 + lab_variation * 0.2)
        
        return min(color_score, 1.0)
        
    except Exception as e:
        logger.warning(f"Color variation analysis failed: {e}")
        return 0.4

def analyze_diameter_size(image):
    """Analyze relative size - larger lesions are more concerning"""
    try:
        h, w = image.shape[:2]
        total_pixels = h * w
        
        # Use the image dimensions as a proxy for lesion size
        # Assume the lesion takes up a significant portion of the image
        relative_size = min(total_pixels / 50000.0, 1.0)  # Normalize to 0-1
        
        # Convert to diameter-like metric
        diameter_score = min(relative_size * 0.8, 1.0)
        
        return diameter_score
        
    except Exception as e:
        logger.warning(f"Diameter analysis failed: {e}")
        return 0.4

def medical_risk_assessment(asymmetry, border, color, diameter):
    """
    Comprehensive medical risk assessment using ABCD criteria
    """
    try:
        # Weight the ABCD criteria based on medical literature
        weights = {
            'asymmetry': 0.25,
            'border': 0.25,
            'color': 0.25,
            'diameter': 0.25
        }
        
        # Calculate weighted score
        weighted_score = (
            asymmetry * weights['asymmetry'] +
            border * weights['border'] +
            color * weights['color'] +
            diameter * weights['diameter']
        )
        
        # Count concerning features (score > 0.6)
        concerning_features = sum(1 for score in [asymmetry, border, color, diameter] if score > 0.6)
        
        return weighted_score, concerning_features
        
    except Exception as e:
        logger.warning(f"Risk assessment failed: {e}")
        return 0.3, 0

def predict_lesion_lightweight(image_path, skin_type='III', body_part='other', has_evolved=False, evolution_weeks=0):
    """
    Lightweight prediction without PyTorch dependencies
    """
    try:
        logger.info(f"Processing image: {image_path}")
        
        # Load and preprocess image
        image = cv2.imread(image_path)
        if image is None:
            return "Error in Analysis", 0, None
            
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize for consistent analysis
        image_rgb = cv2.resize(image_rgb, (224, 224))
        
        # Perform ABCD analysis
        logger.info("Analyzing asymmetry...")
        asymmetry_score = analyze_color_asymmetry(image_rgb)
        
        logger.info("Analyzing border irregularity...")
        border_score = analyze_border_irregularity(image_rgb)
        
        logger.info("Analyzing color variation...")
        color_score = analyze_color_variation(image_rgb)
        
        logger.info("Analyzing size characteristics...")
        diameter_score = analyze_diameter_size(image_rgb)
        
        # Evolution analysis
        evolution_score = 0.0
        if has_evolved:
            # Evolution is highly concerning
            evolution_multiplier = min(evolution_weeks / 4.0, 1.0)  # Normalize weeks
            evolution_score = 0.8 * evolution_multiplier
            logger.info(f"Evolution detected: {evolution_weeks} weeks, score: {evolution_score:.2f}")
        
        # Medical risk assessment
        weighted_score, concerning_features = medical_risk_assessment(
            asymmetry_score, border_score, color_score, diameter_score
        )
        
        # Add evolution factor
        if evolution_score > 0:
            weighted_score = min(weighted_score + evolution_score * 0.3, 1.0)
            concerning_features += 1
        
        # Anatomical risk factor
        high_risk_locations = ['face', 'scalp', 'neck', 'trunk']
        location_multiplier = 1.2 if body_part in high_risk_locations else 1.0
        weighted_score *= location_multiplier
        
        # Skin type considerations
        skin_type_multipliers = {
            'I': 1.3, 'II': 1.2, 'III': 1.0, 'IV': 0.9, 'V': 0.8, 'VI': 0.8
        }
        skin_multiplier = skin_type_multipliers.get(skin_type, 1.0)
        weighted_score *= skin_multiplier
        
        # Ensure score stays within bounds
        weighted_score = min(weighted_score, 1.0)
        
        # Determine final prediction
        if weighted_score > 0.7 or concerning_features >= 3:
            final_prediction = "Highly Suspicious"
            final_confidence = min(weighted_score * 100, 95)
        elif weighted_score > 0.5 or concerning_features >= 2:
            final_prediction = "Suspicious"
            final_confidence = weighted_score * 100
        elif weighted_score > 0.35:
            final_prediction = "Moderately Concerning"
            final_confidence = weighted_score * 90
        else:
            final_prediction = "Benign"
            final_confidence = max(20, 100 - weighted_score * 100)
        
        # Create analysis summary
        analysis_summary = {
            'asymmetry': asymmetry_score,
            'border': border_score,
            'color': color_score,
            'diameter': diameter_score,
            'evolution': evolution_score,
            'detected_skin_tone': f'Type {skin_type}',
            'analysis_type': 'lightweight_analysis'
        }
        
        logger.info(f"Lightweight Analysis - Skin Type: {skin_type}")
        logger.info(f"Features - A:{asymmetry_score:.2f} B:{border_score:.2f} C:{color_score:.2f} D:{diameter_score:.2f}")
        logger.info(f"Final Assessment: {final_prediction}, Confidence: {final_confidence:.2f}%")
        
        return final_prediction, round(final_confidence, 2), analysis_summary
        
    except Exception as e:
        logger.error(f"Lightweight prediction failed: {e}")
        return "Error in Analysis", 0, None

# Main prediction function
def predict_lesion(image_path, skin_type='III', body_part='other', has_evolved=False, evolution_weeks=0, manual_length=None, manual_width=None):
    """Main prediction function using lightweight implementation"""
    return predict_lesion_lightweight(image_path, skin_type, body_part, has_evolved, evolution_weeks)