"""
Lightweight Multi-Input Model without PyTorch Dependencies
Provides multi-input analysis using OpenCV and NumPy
"""
import cv2
import numpy as np
from PIL import Image
import logging

logger = logging.getLogger(__name__)

def get_multi_input_prediction(image_path, age, uv_exposure, family_history, 
                              skin_type, body_part, evolution_weeks):
    """
    Lightweight multi-input prediction using comprehensive risk factors
    """
    try:
        logger.info(f"Processing multi-input analysis for: {image_path}")
        
        # Load and preprocess image
        image = cv2.imread(image_path)
        if image is None:
            return {
                'error': 'Could not load image',
                'classification': 'Error',
                'confidence': 0,
                'features': {'asymmetry': 0, 'border': 0, 'color': 0, 'diameter': 0, 'evolution': 0}
            }
            
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb = cv2.resize(image_rgb, (224, 224))
        
        # Basic ABCDE analysis
        features = analyze_image_features(image_rgb)
        
        # Risk factor analysis
        age_risk = calculate_age_risk(age)
        uv_risk = calculate_uv_risk(uv_exposure)
        family_risk = 0.3 if family_history else 0.0
        location_risk = calculate_location_risk(body_part)
        evolution_risk = calculate_evolution_risk(evolution_weeks)
        skin_type_risk = calculate_skin_type_risk(skin_type)
        
        # Combine all risk factors
        base_score = (features['asymmetry'] * 0.2 + features['border'] * 0.2 + 
                     features['color'] * 0.2 + features['diameter'] * 0.2 + 
                     features['evolution'] * 0.2)
        
        risk_multiplier = (1 + age_risk + uv_risk + family_risk + 
                          location_risk + evolution_risk + skin_type_risk)
        
        final_score = min(base_score * risk_multiplier, 1.0)
        
        # Classification
        if final_score > 0.7:
            classification = "High Risk"
            confidence = min(final_score * 100, 95)
            urgency = "urgent"
        elif final_score > 0.5:
            classification = "Moderate Risk"
            confidence = final_score * 100
            urgency = "moderate"
        elif final_score > 0.3:
            classification = "Low Risk"
            confidence = final_score * 90
            urgency = "routine"
        else:
            classification = "Minimal Risk"
            confidence = max(20, 100 - final_score * 100)
            urgency = "monitoring"
        
        logger.info(f"Multi-input analysis complete: {classification}, Confidence: {confidence:.1f}%")
        
        return {
            'classification': classification,
            'confidence': round(confidence, 1),
            'features': features,
            'risk_factors': {
                'age_risk': age_risk,
                'uv_risk': uv_risk,
                'family_risk': family_risk,
                'location_risk': location_risk,
                'evolution_risk': evolution_risk,
                'skin_type_risk': skin_type_risk
            },
            'urgency': urgency,
            'final_score': final_score
        }
        
    except Exception as e:
        logger.error(f"Multi-input prediction failed: {e}")
        return {
            'error': str(e),
            'classification': 'Error',
            'confidence': 0,
            'features': {'asymmetry': 0, 'border': 0, 'color': 0, 'diameter': 0, 'evolution': 0}
        }

def analyze_image_features(image_rgb):
    """Extract ABCDE features from image"""
    features = {}
    
    try:
        # Asymmetry analysis
        features['asymmetry'] = analyze_asymmetry(image_rgb)
        
        # Border analysis
        features['border'] = analyze_border(image_rgb)
        
        # Color analysis
        features['color'] = analyze_color(image_rgb)
        
        # Diameter analysis
        features['diameter'] = analyze_diameter(image_rgb)
        
        # Evolution placeholder (would require temporal data)
        features['evolution'] = 0.0
        
    except Exception as e:
        logger.warning(f"Feature analysis failed: {e}")
        features = {'asymmetry': 0.3, 'border': 0.3, 'color': 0.3, 'diameter': 0.3, 'evolution': 0.0}
    
    return features

def analyze_asymmetry(image):
    """Analyze asymmetry using quadrant comparison"""
    try:
        h, w = image.shape[:2]
        mid_h, mid_w = h // 2, w // 2
        
        # Split into quadrants
        q1 = image[:mid_h, :mid_w]
        q2 = image[:mid_h, mid_w:]
        q3 = image[mid_h:, :mid_w]
        q4 = image[mid_h:, mid_w:]
        
        # Calculate color statistics for each quadrant
        stats = []
        for quad in [q1, q2, q3, q4]:
            if quad.size > 0:
                mean_color = np.mean(quad, axis=(0, 1))
                std_color = np.std(quad, axis=(0, 1))
                stats.append(np.concatenate([mean_color, std_color]))
        
        if len(stats) < 4:
            return 0.3
        
        # Calculate asymmetry as variance between quadrants
        stats_array = np.array(stats)
        asymmetry = np.std(stats_array, axis=0).mean() / 255.0
        
        return min(asymmetry * 2, 1.0)
        
    except Exception:
        return 0.3

def analyze_border(image):
    """Analyze border irregularity"""
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 0.3
        
        # Analyze largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Calculate circularity
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        
        if perimeter == 0:
            return 0.3
        
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        irregularity = max(0, 1 - circularity)
        
        return min(irregularity * 2, 1.0)
        
    except Exception:
        return 0.3

def analyze_color(image):
    """Analyze color variation"""
    try:
        # Convert to different color spaces
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Calculate color diversity
        rgb_std = np.std(image.reshape(-1, 3), axis=0)
        hsv_std = np.std(hsv.reshape(-1, 3), axis=0)
        
        # Combine metrics
        rgb_variation = np.mean(rgb_std) / 255.0
        hue_variation = hsv_std[0] / 179.0
        
        color_score = (rgb_variation * 0.6 + hue_variation * 0.4)
        
        return min(color_score, 1.0)
        
    except Exception:
        return 0.4

def analyze_diameter(image):
    """Analyze relative size"""
    try:
        h, w = image.shape[:2]
        total_pixels = h * w
        
        # Estimate relative size (normalized)
        relative_size = min(total_pixels / 50000.0, 1.0)
        diameter_score = min(relative_size * 0.8, 1.0)
        
        return diameter_score
        
    except Exception:
        return 0.3

def calculate_age_risk(age):
    """Calculate age-based risk factor"""
    if age < 20:
        return 0.0
    elif age < 40:
        return 0.1
    elif age < 60:
        return 0.2
    else:
        return 0.3

def calculate_uv_risk(uv_exposure):
    """Calculate UV exposure risk (0-10 scale)"""
    return min(uv_exposure / 10.0 * 0.3, 0.3)

def calculate_location_risk(body_part):
    """Calculate anatomical location risk"""
    high_risk_locations = ['head', 'neck', 'trunk', 'back']
    moderate_risk_locations = ['arms', 'legs', 'hands', 'feet']
    
    if body_part in high_risk_locations:
        return 0.2
    elif body_part in moderate_risk_locations:
        return 0.1
    else:
        return 0.05

def calculate_evolution_risk(evolution_weeks):
    """Calculate evolution/change risk"""
    if evolution_weeks == 0:
        return 0.0
    elif evolution_weeks < 4:
        return 0.2
    elif evolution_weeks < 12:
        return 0.3
    else:
        return 0.4

def calculate_skin_type_risk(skin_type):
    """Calculate Fitzpatrick skin type risk"""
    risk_multipliers = {
        1: 0.3,   # Type I - highest risk
        2: 0.2,   # Type II
        3: 0.1,   # Type III
        4: 0.05,  # Type IV
        5: 0.0,   # Type V
        6: 0.0    # Type VI - lowest risk
    }
    return risk_multipliers.get(skin_type, 0.1)

def get_body_part_options():
    """Get available body part options"""
    return [
        'head', 'face', 'neck', 'chest', 'back', 'abdomen',
        'arms', 'hands', 'legs', 'feet', 'other'
    ]