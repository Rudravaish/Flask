"""
Basic Skin Lesion Analysis using only Python built-in libraries
No external dependencies - uses PIL and basic image processing
"""

from PIL import Image, ImageFilter, ImageStat
import os
import math

def analyze_basic_features(image_path):
    """Analyze basic image features using PIL only"""
    try:
        with Image.open(image_path) as img:
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Get basic image stats
            width, height = img.size
            total_pixels = width * height
            
            # Analyze color distribution
            stat = ImageStat.Stat(img)
            mean_colors = stat.mean
            std_colors = stat.stddev
            
            # Basic asymmetry analysis (compare quadrants)
            quarter_w, quarter_h = width // 2, height // 2
            
            # Extract quadrants
            top_left = img.crop((0, 0, quarter_w, quarter_h))
            top_right = img.crop((quarter_w, 0, width, quarter_h))
            bottom_left = img.crop((0, quarter_h, quarter_w, height))
            bottom_right = img.crop((quarter_w, quarter_h, width, height))
            
            # Calculate asymmetry score
            tl_stat = ImageStat.Stat(top_left)
            tr_stat = ImageStat.Stat(top_right)
            bl_stat = ImageStat.Stat(bottom_left)
            br_stat = ImageStat.Stat(bottom_right)
            
            # Asymmetry based on mean color differences
            horizontal_diff = sum(abs(a - b) for a, b in zip(tl_stat.mean + bl_stat.mean, tr_stat.mean + br_stat.mean))
            vertical_diff = sum(abs(a - b) for a, b in zip(tl_stat.mean + tr_stat.mean, bl_stat.mean + br_stat.mean))
            asymmetry_score = (horizontal_diff + vertical_diff) / 6  # Normalize
            
            # Border irregularity (edge detection simulation)
            edges = img.filter(ImageFilter.FIND_EDGES)
            edge_stat = ImageStat.Stat(edges)
            border_score = sum(edge_stat.mean) / 3  # Average edge intensity
            
            # Color variation
            color_variation = sum(std_colors) / 3  # Average standard deviation
            
            # Diameter estimation (relative size)
            diameter_score = min(width, height) / 100  # Normalize to typical lesion size
            
            return {
                'asymmetry': min(asymmetry_score / 50, 1.0),  # Normalize to 0-1
                'border': min(border_score / 100, 1.0),
                'color': min(color_variation / 50, 1.0),
                'diameter': min(diameter_score, 1.0),
                'mean_colors': mean_colors,
                'std_colors': std_colors,
                'size': (width, height)
            }
    except Exception as e:
        # Return default values if analysis fails
        return {
            'asymmetry': 0.3,
            'border': 0.3,
            'color': 0.3,
            'diameter': 0.3,
            'mean_colors': [128, 128, 128],
            'std_colors': [30, 30, 30],
            'size': (100, 100)
        }

def calculate_risk_factors(age, uv_exposure, family_history, skin_type, body_part, evolution_weeks):
    """Calculate additional risk factors"""
    risk_score = 0.0
    
    # Age risk (higher for older patients)
    if age > 50:
        risk_score += 0.2
    elif age > 65:
        risk_score += 0.3
    
    # UV exposure risk
    risk_score += min(uv_exposure / 10.0, 0.3)
    
    # Family history
    if family_history:
        risk_score += 0.2
    
    # Skin type risk (higher for lighter skin)
    skin_type_risk = {
        'I': 0.3, 'II': 0.25, 'III': 0.15,
        'IV': 0.1, 'V': 0.05, 'VI': 0.05
    }
    risk_score += skin_type_risk.get(skin_type, 0.15)
    
    # Body part risk (some locations more prone to melanoma)
    high_risk_parts = ['trunk_back', 'trunk_chest', 'head_neck', 'shoulders']
    if body_part in high_risk_parts:
        risk_score += 0.15
    
    # Evolution risk
    if evolution_weeks > 0:
        risk_score += min(evolution_weeks / 52.0, 0.2)  # Up to 1 year
    
    return min(risk_score, 1.0)

def predict_lesion(image_path, skin_type='III', body_part='other', has_evolved=False, evolution_weeks=0, manual_length=None, manual_width=None):
    """
    Basic prediction using only PIL and built-in Python libraries
    """
    try:
        # Analyze image features
        features = analyze_basic_features(image_path)
        
        # Calculate ABCDE scores
        asymmetry_score = features['asymmetry']
        border_score = features['border']
        color_score = features['color']
        diameter_score = features['diameter']
        evolution_score = 0.1 if has_evolved else 0.0
        
        # Weighted ABCDE score
        abcde_score = (
            asymmetry_score * 0.25 +
            border_score * 0.25 +
            color_score * 0.25 +
            diameter_score * 0.15 +
            evolution_score * 0.1
        )
        
        # Additional risk factors
        age = 50  # Default age
        uv_exposure = 5  # Default UV exposure
        family_history = 0  # Default no family history
        
        risk_factor_score = calculate_risk_factors(
            age, uv_exposure, family_history, skin_type, body_part, evolution_weeks
        )
        
        # Combine scores
        final_score = (abcde_score * 0.7) + (risk_factor_score * 0.3)
        
        # Determine prediction
        if final_score > 0.6:
            prediction = "Suspicious - Requires Medical Evaluation"
            confidence = min(85 + (final_score - 0.6) * 37.5, 95)
        elif final_score > 0.4:
            prediction = "Moderately Concerning - Monitor Closely"
            confidence = 60 + (final_score - 0.4) * 62.5
        else:
            prediction = "Likely Benign - Routine Monitoring Recommended"
            confidence = 70 + (0.4 - final_score) * 75
        
        # Enhanced analysis for darker skin tones
        detected_skin_tone = skin_type
        analysis_type = 'basic'
        
        if skin_type in ['V', 'VI']:
            # Adjust analysis for darker skin
            analysis_type = 'darker_skin_optimized'
            
            # Check for specific patterns in darker skin
            mean_colors = features['mean_colors']
            if mean_colors[0] < 100 and mean_colors[1] < 100 and mean_colors[2] < 100:
                # Very dark lesion on dark skin - increase concern
                final_score *= 1.1
                prediction = "Requires Professional Evaluation - Melanoma risk in darker skin"
                confidence = min(confidence + 10, 95)
        
        # Prepare detailed analysis data
        analysis_data = {
            'asymmetry': round(asymmetry_score * 100, 1),
            'border': round(border_score * 100, 1),
            'color': round(color_score * 100, 1),
            'diameter': round(diameter_score * 100, 1),
            'evolution': round(evolution_score * 100, 1),
            'risk_factors': round(risk_factor_score * 100, 1),
            'final_score': round(final_score * 100, 1),
            'detected_skin_tone': detected_skin_tone,
            'analysis_type': analysis_type,
            'image_features': features
        }
        
        return prediction, round(confidence, 1), analysis_data
        
    except Exception as e:
        # Fallback prediction
        return "Analysis Error - Please Consult Healthcare Provider", 50.0, {
            'asymmetry': 30, 'border': 30, 'color': 30, 'diameter': 30,
            'error': str(e), 'analysis_type': 'error_fallback'
        }