"""
Basic Skin Lesion Analysis using only Python built-in libraries
No external dependencies - uses PIL and basic image processing
"""
from PIL import Image, ImageStat, ImageFilter, ImageEnhance
import logging
import os
import math

# Configure logging
logger = logging.getLogger(__name__)

def analyze_basic_features(image_path):
    """Analyze basic image features using PIL only"""
    try:
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Resize for consistent analysis
        image = image.resize((224, 224))
        
        # Get image statistics
        stat = ImageStat.Stat(image)
        
        # Basic color analysis
        rgb_means = stat.mean
        rgb_stddevs = stat.stddev
        
        # Asymmetry analysis using color distribution
        width, height = image.size
        
        # Split image into quadrants
        left_half = image.crop((0, 0, width//2, height))
        right_half = image.crop((width//2, 0, width, height))
        top_half = image.crop((0, 0, width, height//2))
        bottom_half = image.crop((0, height//2, width, height))
        
        # Calculate color differences between halves
        left_stat = ImageStat.Stat(left_half)
        right_stat = ImageStat.Stat(right_half)
        top_stat = ImageStat.Stat(top_half)
        bottom_stat = ImageStat.Stat(bottom_half)
        
        # Asymmetry score based on color difference between halves
        horizontal_diff = sum(abs(l - r) for l, r in zip(left_stat.mean, right_stat.mean))
        vertical_diff = sum(abs(t - b) for t, b in zip(top_stat.mean, bottom_stat.mean))
        asymmetry_score = min((horizontal_diff + vertical_diff) / 510.0, 1.0)
        
        # Color variation analysis
        color_variation = sum(rgb_stddevs) / (255.0 * 3)
        
        # Border analysis using edge enhancement
        enhanced = image.filter(ImageFilter.FIND_EDGES)
        edge_stat = ImageStat.Stat(enhanced)
        border_score = min(sum(edge_stat.mean) / (255.0 * 3), 1.0)
        
        # Size analysis (relative to image dimensions)
        total_pixels = width * height
        size_score = min(total_pixels / 50176.0, 1.0)  # Normalize to 224x224
        
        return {
            'asymmetry': asymmetry_score,
            'border': border_score,
            'color': color_variation,
            'diameter': size_score,
            'evolution': 0.0  # Cannot determine from single image
        }
        
    except Exception as e:
        logger.error(f"Basic feature analysis failed: {e}")
        return {
            'asymmetry': 0.3,
            'border': 0.3,
            'color': 0.3,
            'diameter': 0.3,
            'evolution': 0.0
        }

def calculate_risk_factors(age, uv_exposure, family_history, skin_type, body_part, evolution_weeks):
    """Calculate additional risk factors"""
    risk_score = 0.0
    
    # Age factor
    if age > 50:
        risk_score += 0.3
    elif age > 30:
        risk_score += 0.2
    elif age > 20:
        risk_score += 0.1
    
    # UV exposure (0-10 scale)
    risk_score += min(uv_exposure / 10.0 * 0.3, 0.3)
    
    # Family history
    if family_history:
        risk_score += 0.3
    
    # Skin type (Fitzpatrick I-VI)
    skin_type_risks = {
        'I': 0.4, 'II': 0.3, 'III': 0.2, 'IV': 0.1, 'V': 0.05, 'VI': 0.05
    }
    risk_score += skin_type_risks.get(skin_type, 0.2)
    
    # Body location
    high_risk_locations = ['face', 'scalp', 'neck', 'trunk', 'back']
    if body_part in high_risk_locations:
        risk_score += 0.2
    
    # Evolution/changes
    if evolution_weeks > 0:
        risk_score += min(evolution_weeks / 12.0 * 0.4, 0.4)
    
    return min(risk_score, 1.0)

def predict_lesion(image_path, skin_type='III', body_part='other', has_evolved=False, evolution_weeks=0, manual_length=None, manual_width=None):
    """
    Basic prediction using only PIL and built-in Python libraries
    """
    try:
        logger.info(f"Processing image: {image_path}")
        
        # Analyze image features
        features = analyze_basic_features(image_path)
        
        # Calculate base ABCDE score
        base_score = (
            features['asymmetry'] * 0.25 +
            features['border'] * 0.25 +
            features['color'] * 0.25 +
            features['diameter'] * 0.25
        )
        
        # Add evolution factor
        evolution_score = 0.0
        if has_evolved and evolution_weeks > 0:
            evolution_score = min(evolution_weeks / 8.0, 1.0) * 0.8
            logger.info(f"Evolution detected: {evolution_weeks} weeks, score: {evolution_score:.2f}")
        
        # Combine scores
        combined_score = min(base_score + evolution_score * 0.3, 1.0)
        
        # Apply skin type adjustment
        skin_multipliers = {
            'I': 1.3, 'II': 1.2, 'III': 1.0, 'IV': 0.9, 'V': 0.8, 'VI': 0.8
        }
        skin_multiplier = skin_multipliers.get(skin_type, 1.0)
        final_score = min(combined_score * skin_multiplier, 1.0)
        
        # Apply location risk
        high_risk_locations = ['face', 'scalp', 'neck', 'trunk', 'back']
        if body_part in high_risk_locations:
            final_score = min(final_score * 1.2, 1.0)
        
        # Determine prediction and confidence
        if final_score > 0.7:
            prediction = "Highly Suspicious"
            confidence = min(final_score * 100, 95)
        elif final_score > 0.55:
            prediction = "Suspicious"
            confidence = final_score * 100
        elif final_score > 0.4:
            prediction = "Moderately Concerning"
            confidence = final_score * 90
        else:
            prediction = "Benign"
            confidence = max(25, 100 - final_score * 100)
        
        # Create analysis summary
        analysis_summary = {
            'asymmetry': features['asymmetry'],
            'border': features['border'],
            'color': features['color'],
            'diameter': features['diameter'],
            'evolution': evolution_score,
            'detected_skin_tone': f'Type {skin_type}',
            'analysis_type': 'basic_analysis'
        }
        
        logger.info(f"Basic Analysis - Skin Type: {skin_type}")
        logger.info(f"Features - A:{features['asymmetry']:.2f} B:{features['border']:.2f} C:{features['color']:.2f} D:{features['diameter']:.2f}")
        logger.info(f"Final Assessment: {prediction}, Confidence: {confidence:.2f}%")
        
        return prediction, round(confidence, 2), analysis_summary
        
    except Exception as e:
        logger.error(f"Basic prediction failed: {e}")
        return "Error in Analysis", 0, None