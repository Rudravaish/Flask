from torchvision import models, transforms
from PIL import Image, ImageStat, ImageFilter, ImageEnhance
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import logging
import os

# Configure logging
logger = logging.getLogger(__name__)

# Initialize model
model = None
transform = None

# Try to import advanced skin analysis
try:
    from enhanced_skin_analysis import get_advanced_skin_analysis
    advanced_analysis_available = True
except ImportError:
    logger.warning("Enhanced skin analysis not available")
    advanced_analysis_available = False

# Try to import ISIC feature enhancement
try:
    from isic_feature_extractor import get_isic_enhanced_analysis
    isic_enhancement_available = True
    logger.info("ISIC feature enhancement module loaded successfully")
except ImportError:
    isic_enhancement_available = False
    logger.warning("ISIC feature enhancement not available")

# Try to import enhanced ISIC model
try:
    from isic_model import get_enhanced_prediction
    enhanced_model_available = True
    logger.info("Enhanced ISIC model module loaded successfully")
except ImportError:
    enhanced_model_available = False
    logger.warning("Enhanced ISIC model not available, using fallback analysis")

# Try to import trained model as backup
try:
    from trained_model import get_trained_prediction, is_trained_model_available
    trained_model_available = True
    logger.info("Trained ISIC model module loaded successfully")
except ImportError:
    trained_model_available = False

def initialize_model():
    """Initialize the model and transforms."""
    global model, transform
    
    try:
        # Load a pretrained MobileNet v2 model
        logger.info("Loading MobileNet v2 model...")
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        model.eval()
        
        # Set model to use less memory
        if torch.cuda.is_available():
            model = model.cuda()
            torch.cuda.empty_cache()
        
        # Define image preprocessing transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        logger.info("Model loaded successfully")
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        # Continue without CNN model - use medical analysis only
        model = None
        transform = None
        logger.warning("Continuing with medical analysis only")

# Initialize the model when the module is imported
initialize_model()

def analyze_color_asymmetry(image):
    """Analyze color asymmetry - melanomas often have uneven color distribution"""
    try:
        # Convert to different color spaces for analysis
        hsv = image.convert('HSV')
        
        # Split image into quadrants and analyze color variance
        width, height = image.size
        if width < 4 or height < 4:
            return 0.5  # Default for very small images
            
        quadrants = [
            image.crop((0, 0, width//2, height//2)),
            image.crop((width//2, 0, width, height//2)),
            image.crop((0, height//2, width//2, height)),
            image.crop((width//2, height//2, width, height))
        ]
        
        color_variances = []
        for quad in quadrants:
            try:
                stat = ImageStat.Stat(quad)
                # Calculate variance in RGB channels
                if len(stat.stddev) >= 3:
                    variance = sum(stat.stddev[:3]) / 3
                    color_variances.append(variance)
            except Exception:
                color_variances.append(0.0)
        
        if not color_variances or all(v == 0 for v in color_variances):
            return 0.5
            
        # High variance between quadrants suggests asymmetry
        mean_variance = np.mean(color_variances)
        if mean_variance == 0:
            return 0.5
            
        asymmetry_score = np.std(color_variances) / (mean_variance + 1e-6)
        return min(float(asymmetry_score) / 50.0, 1.0)  # Normalize to 0-1
    except Exception as e:
        logger.warning(f"Error in asymmetry analysis: {e}")
        return 0.5

def analyze_border_irregularity(image):
    """Analyze border irregularity - melanomas often have irregular borders"""
    try:
        # Convert to grayscale and apply edge detection
        gray = image.convert('L')
        edges = gray.filter(ImageFilter.FIND_EDGES)
        
        # Convert to numpy for analysis
        edge_array = np.array(edges)
        
        if edge_array.size == 0:
            return 0.5
        
        # Find contours and analyze border roughness
        edge_pixels = np.where(edge_array > 50)
        if len(edge_pixels[0]) == 0:
            return 0.5
        
        # Calculate border irregularity based on edge pixel distribution
        try:
            edge_variance = np.var(edge_pixels[0]) + np.var(edge_pixels[1])
            border_score = min(float(edge_variance) / 10000.0, 1.0)
            return max(0.0, min(border_score, 1.0))
        except (ValueError, ZeroDivisionError):
            return 0.5
    except Exception as e:
        logger.warning(f"Error in border analysis: {e}")
        return 0.5

def analyze_color_variation(image):
    """Analyze color variation - melanomas often have multiple colors"""
    try:
        # Enhance contrast to better detect color variations
        enhancer = ImageEnhance.Contrast(image)
        enhanced = enhancer.enhance(1.5)
        
        # Convert to HSV for better color analysis
        hsv = enhanced.convert('HSV')
        hsv_channels = hsv.split()
        
        if len(hsv_channels) < 2:
            return 0.5
            
        hue_stat = ImageStat.Stat(hsv_channels[0])  # Hue channel
        sat_stat = ImageStat.Stat(hsv_channels[1])  # Saturation channel
        
        # High standard deviation in hue and saturation indicates color variation
        if len(hue_stat.stddev) == 0 or len(sat_stat.stddev) == 0:
            return 0.5
            
        hue_variation = hue_stat.stddev[0] / 255.0 if hue_stat.stddev[0] > 0 else 0.0
        sat_variation = sat_stat.stddev[0] / 255.0 if sat_stat.stddev[0] > 0 else 0.0
        
        color_score = (hue_variation + sat_variation) / 2.0
        return max(0.0, min(color_score * 2.0, 1.0))
    except Exception as e:
        logger.warning(f"Error in color variation analysis: {e}")
        return 0.5

def analyze_diameter_size(image):
    """Analyze relative size - larger lesions are more concerning"""
    width, height = image.size
    total_pixels = width * height
    
    # Estimate lesion size relative to image
    # Assume lesion takes up significant portion of image
    size_score = min(total_pixels / 100000.0, 1.0)  # Normalize
    return size_score

def medical_risk_assessment(asymmetry, border, color, diameter):
    """
    Comprehensive medical risk assessment using ABCD criteria:
    A - Asymmetry
    B - Border irregularity  
    C - Color variation
    D - Diameter/size
    """
    
    # Weight factors based on medical literature
    weights = {
        'asymmetry': 0.25,
        'border': 0.30,
        'color': 0.30,
        'diameter': 0.15
    }
    
    # Calculate weighted risk score
    risk_score = (
        asymmetry * weights['asymmetry'] +
        border * weights['border'] +
        color * weights['color'] +
        diameter * weights['diameter']
    )
    
    # Apply medical thresholds
    if risk_score > 0.7:
        return "Highly Suspicious", min(risk_score * 100, 95)
    elif risk_score > 0.5:
        return "Suspicious", risk_score * 100
    elif risk_score > 0.3:
        return "Moderately Concerning", risk_score * 100
    else:
        return "Benign", max(risk_score * 100, 15)

def enhance_cnn_prediction(cnn_output, medical_scores):
    """Combine CNN features with medical analysis"""
    asymmetry, border, color, diameter = medical_scores
    
    # Get top CNN predictions
    probabilities = F.softmax(cnn_output, dim=0)
    top_prob = torch.max(probabilities).item()
    
    # Look for skin-related ImageNet classes that might indicate suspicious features
    suspicious_classes = [
        # Classes that might correlate with lesion characteristics
        440, 441, 442,  # Various texture classes
        972, 973, 974,  # Medical/biological classes
        555, 556, 557,  # Pattern classes
    ]
    
    cnn_suspicion = 0
    for class_idx in suspicious_classes:
        if class_idx < len(probabilities):
            cnn_suspicion += probabilities[class_idx].item()
    
    # Combine medical analysis with CNN features
    medical_risk = (asymmetry + border + color + diameter) / 4.0
    combined_score = 0.7 * medical_risk + 0.3 * cnn_suspicion
    
    return combined_score

def analyze_diameter_size_manual(length_mm, width_mm):
    """Analyze diameter using manual measurements provided by user"""
    try:
        # Use the largest dimension as the effective diameter
        max_diameter = max(length_mm, width_mm)
        
        # Score based on melanoma risk thresholds
        # 6mm is the traditional ABCDE threshold, but modern guidelines consider smaller lesions
        if max_diameter >= 10:
            # Very large lesions are highly concerning
            diameter_score = 0.9
        elif max_diameter >= 6:
            # Classic ABCDE threshold - concerning
            diameter_score = 0.7
        elif max_diameter >= 4:
            # Moderate size - some concern
            diameter_score = 0.5
        elif max_diameter >= 2:
            # Small but notable
            diameter_score = 0.3
        else:
            # Very small lesions
            diameter_score = 0.1
        
        logger.info(f"Manual diameter analysis: {max_diameter}mm max diameter, score: {diameter_score:.2f}")
        return diameter_score
        
    except Exception as e:
        logger.warning(f"Manual diameter analysis failed: {e}")
        return 0.3  # Conservative fallback

def analyze_evolution(has_evolved, evolution_weeks):
    """Analyze Evolution (E in ABCDE) - changes over time are highly concerning"""
    try:
        if not has_evolved:
            return 0.1  # No reported changes = low concern
        
        # Evolution is highly concerning - melanomas change rapidly
        if evolution_weeks <= 4:
            # Recent changes (within 4 weeks) are very concerning
            evolution_score = 0.9
        elif evolution_weeks <= 12:
            # Changes within 3 months are concerning
            evolution_score = 0.7
        elif evolution_weeks <= 26:
            # Changes within 6 months are moderately concerning
            evolution_score = 0.5
        else:
            # Older changes still concerning but less urgent
            evolution_score = 0.4
        
        logger.info(f"Evolution analysis: has_evolved={has_evolved}, weeks={evolution_weeks}, score={evolution_score:.2f}")
        return evolution_score
        
    except Exception as e:
        logger.warning(f"Evolution analysis failed: {e}")
        return 0.3  # Conservative fallback

def calculate_anatomical_risk_factor(body_part):
    """Calculate risk multiplier based on anatomical location - melanoma incidence varies by body site"""
    # Risk factors based on melanoma epidemiology and sun exposure patterns
    risk_factors = {
        # High-risk sun-exposed areas
        "trunk_back": 1.4,         # Back - highest melanoma rates in men
        "trunk_chest": 1.3,        # Chest/abdomen - high UV exposure
        "head_neck": 1.35,         # Head/neck/face - chronic sun exposure
        "shoulders": 1.3,          # Shoulders - high intermittent sun exposure
        
        # Moderate-risk areas
        "arms_upper": 1.1,         # Upper arms - moderate sun exposure
        "arms_lower": 1.0,         # Lower arms/hands - moderate risk
        "legs_upper": 1.2,         # Upper legs/thighs - higher in women
        "legs_lower": 0.9,         # Lower legs/feet - lower overall risk
        
        # Special consideration areas
        "palms_soles": 1.5,        # Acral lentiginous melanoma - different pathogenesis
        "genital_area": 1.2,       # Mucosal melanoma - aggressive type
        "other": 1.0               # Default baseline risk
    }
    
    return risk_factors.get(body_part, 1.0)

def predict_lesion(image_path, skin_type='III', body_part='other', has_evolved=False, evolution_weeks=0, manual_length=None, manual_width=None):
    """
    Predict whether a skin lesion is benign or suspicious with enhanced medical parameters.
    
    Args:
        image_path (str): Path to the image file
        skin_type (str): Fitzpatrick skin type (I-VI)
        body_part (str): Anatomical location of the lesion
        has_evolved (bool): Whether the lesion has changed over time
        evolution_weeks (int): Number of weeks since changes were noticed
        manual_length (float): Manual measurement length in mm (optional)
        manual_width (float): Manual measurement width in mm (optional)
        
    Returns:
        tuple: (prediction, confidence_percentage, analysis_data)
    """
    image = None
    try:
        if not os.path.exists(image_path):
            raise ValueError(f"Image file not found: {image_path}")
        
        # Check file size to prevent memory issues
        file_size = os.path.getsize(image_path)
        if file_size > 50 * 1024 * 1024:  # 50MB limit
            raise ValueError("Image file too large")
        
        logger.info(f"Processing image: {image_path}")
        image = Image.open(image_path)
        
        # Validate image format and convert to RGB
        if image.format not in ['JPEG', 'PNG', 'BMP', 'WEBP', 'GIF']:
            raise ValueError(f"Unsupported image format: {image.format}")
            
        # Check image dimensions to prevent memory issues
        width, height = image.size
        if width * height > 50000000:  # ~50MP limit
            raise ValueError("Image resolution too high")
            
        image = image.convert('RGB')
        
        # Perform comprehensive medical analysis using enhanced ABCDE criteria
        logger.info("Analyzing asymmetry...")
        asymmetry_score = analyze_color_asymmetry(image)
        
        logger.info("Analyzing border irregularity...")
        border_score = analyze_border_irregularity(image)
        
        logger.info("Analyzing color variation...")
        color_score = analyze_color_variation(image)
        
        logger.info("Analyzing size characteristics...")
        # Use manual measurements if provided, otherwise analyze image
        if manual_length is not None and manual_width is not None:
            diameter_score = analyze_diameter_size_manual(manual_length, manual_width)
            logger.info(f"Using manual measurements: {manual_length}mm x {manual_width}mm")
        else:
            diameter_score = analyze_diameter_size(image)
        
        logger.info("Analyzing evolution (changes over time)...")
        evolution_score = analyze_evolution(has_evolved, evolution_weeks)
        
        logger.info("Calculating anatomical location risk factor...")
        anatomical_risk_factor = calculate_anatomical_risk_factor(body_part)
        
        # Try advanced skin tone analysis first (best for darker skin tones)
        if advanced_analysis_available:
            try:
                logger.info("Using advanced skin tone-aware analysis...")
                advanced_results = get_advanced_skin_analysis(image_path)
                
                if advanced_results and 'analysis_type' in advanced_results:
                    # Use advanced analysis weights based on detected skin tone
                    if advanced_results.get('detected_skin_tone') in ['V', 'VI']:
                        # Use darker skin weights
                        weights = {'asymmetry': 0.20, 'border': 0.25, 'color': 0.35, 'diameter': 0.15, 'evolution': 0.05}
                        logger.info(f"Applied darker skin tone analysis for Type {advanced_results.get('detected_skin_tone')}")
                    else:
                        # Use standard weights
                        weights = {'asymmetry': 0.25, 'border': 0.30, 'color': 0.30, 'diameter': 0.15, 'evolution': 0.00}
                    
                    # Calculate weighted score
                    weighted_score = 0
                    for criterion, weight in weights.items():
                        if criterion in advanced_results:
                            weighted_score += advanced_results[criterion] * weight
                    
                    # Enhanced classification with improved thresholds for all skin tones
                    concerning_features = sum(1 for criterion in ['asymmetry', 'border', 'color', 'diameter']
                                            if advanced_results.get(criterion, 0) > 0.6)
                    
                    # Adjust thresholds based on skin tone and feature combinations
                    if advanced_results.get('detected_skin_tone') in ['V', 'VI']:
                        # More conservative thresholds for darker skin due to analysis complexity
                        if weighted_score > 0.65 or concerning_features >= 3:
                            final_prediction = "Suspicious - Recommend Dermatologist"
                            final_confidence = min(weighted_score * 100, 92)
                        elif weighted_score > 0.45 or concerning_features >= 2:
                            final_prediction = "Moderately Concerning"
                            final_confidence = weighted_score * 100
                        elif weighted_score > 0.25:
                            final_prediction = "Monitor Closely"
                            final_confidence = weighted_score * 85
                        else:
                            final_prediction = "Likely Benign"
                            final_confidence = max(25, 100 - weighted_score * 100)
                    else:
                        # Standard thresholds for lighter skin tones
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
                    
                    logger.info(f"Advanced Analysis - Skin Type: {advanced_results.get('detected_skin_tone', 'Unknown')}")
                    logger.info(f"Features - A:{advanced_results.get('asymmetry', 0):.2f} B:{advanced_results.get('border', 0):.2f} C:{advanced_results.get('color', 0):.2f} D:{advanced_results.get('diameter', 0):.2f}")
                    logger.info(f"Final Assessment: {final_prediction}, Confidence: {final_confidence:.2f}%")
                    
                    # Apply ISIC enhancement if available
                    if isic_enhancement_available:
                        try:
                            enhanced_features = get_isic_enhanced_analysis(advanced_results)
                            if enhanced_features.get('isic_enhanced'):
                                logger.info("Applied ISIC-based feature enhancement")
                                # Recalculate weighted score with enhanced features
                                enhanced_asymmetry = enhanced_features.get('asymmetry', asymmetry_score)
                                enhanced_border = enhanced_features.get('border', border_score)
                                enhanced_weighted_score = (enhanced_asymmetry * 0.25 + enhanced_border * 0.25 + 
                                                         color_score * 0.20 + diameter_score * 0.15 + evolution_score * 0.15)
                                
                                # Update final prediction with enhanced scoring
                                if enhanced_weighted_score > 0.7 or (enhanced_asymmetry > 0.7 and enhanced_border > 0.7):
                                    final_prediction = "Highly Suspicious"
                                    final_confidence = min(enhanced_weighted_score * 100, 95)
                                elif enhanced_weighted_score > 0.5:
                                    final_prediction = "Suspicious"
                                    final_confidence = enhanced_weighted_score * 100
                                elif enhanced_weighted_score > 0.35:
                                    final_prediction = "Moderately Concerning"
                                    final_confidence = enhanced_weighted_score * 90
                                else:
                                    final_prediction = "Benign"
                                    final_confidence = max(20, 100 - enhanced_weighted_score * 100)
                                
                                advanced_results = enhanced_features
                                logger.info(f"ISIC-Enhanced Assessment: {final_prediction}, Confidence: {final_confidence:.2f}%")
                        except Exception as e:
                            logger.warning(f"ISIC enhancement failed: {e}")
                    
                    # Return results with additional metadata for chatbot
                    return final_prediction, round(final_confidence, 2), advanced_results
                    
            except Exception as e:
                logger.warning(f"Advanced skin analysis failed, trying fallback: {e}")
        
        # Try to use enhanced ISIC model as backup
        if enhanced_model_available:
            try:
                logger.info("Using enhanced ISIC-based prediction...")
                enhanced_prediction, enhanced_confidence = get_enhanced_prediction(image_path)
                if enhanced_prediction != "Error in Analysis":
                    logger.info(f"Enhanced ISIC Model: {enhanced_prediction}, Confidence: {enhanced_confidence:.2f}%")
                    logger.info(f"Medical Analysis - A:{asymmetry_score:.2f} B:{border_score:.2f} C:{color_score:.2f} D:{diameter_score:.2f}")
                    return enhanced_prediction, enhanced_confidence, None
            except Exception as e:
                logger.warning(f"Enhanced model failed, trying backup: {e}")
        
        # Try trained model as backup
        if trained_model_available:
            try:
                logger.info("Using trained ISIC model for prediction...")
                trained_prediction, trained_confidence = get_trained_prediction(image_path)
                if trained_prediction is not None:
                    logger.info(f"ISIC Model: {trained_prediction}, Confidence: {trained_confidence:.2f}%")
                    
                    # Combine with medical analysis for enhanced accuracy
                    medical_prediction, medical_confidence = medical_risk_assessment(
                        asymmetry_score, border_score, color_score, diameter_score
                    )
                    
                    # Weight the trained model more heavily but consider medical analysis
                    final_confidence = (trained_confidence * 0.7 + medical_confidence * 0.3)
                    
                    logger.info(f"Medical Analysis - A:{asymmetry_score:.2f} B:{border_score:.2f} C:{color_score:.2f} D:{diameter_score:.2f}")
                    logger.info(f"Combined Assessment: {trained_prediction}, Confidence: {final_confidence:.2f}%")
                    
                    return trained_prediction, round(final_confidence, 2)
            except Exception as e:
                logger.warning(f"Trained model failed, falling back to medical analysis: {e}")
        
        # Enhanced ABCDE medical analysis including Evolution and anatomical location
        abcde_base_score = (asymmetry_score + border_score + color_score + diameter_score + evolution_score) / 5.0
        
        # Apply anatomical location risk factor
        final_risk_score = abcde_base_score * anatomical_risk_factor
        
        # Enhanced medical risk assessment with improved weighting including Evolution
        enhanced_asymmetry = min(asymmetry_score * 1.15, 1.0)  # Boost asymmetry importance
        enhanced_border = border_score
        enhanced_color = min(color_score * 1.1, 1.0)  # Boost color variation
        enhanced_diameter = diameter_score
        enhanced_evolution = evolution_score
        
        # Count concerning features for feature-based classification (including Evolution)
        concerning_features = sum(1 for score in [enhanced_asymmetry, enhanced_border, enhanced_color, enhanced_diameter, enhanced_evolution] if score > 0.6)
        
        # Weighted risk score with clinical relevance including Evolution and anatomical location
        weighted_risk_score = (enhanced_asymmetry * 0.25 + enhanced_border * 0.25 + enhanced_color * 0.2 + enhanced_diameter * 0.15 + enhanced_evolution * 0.15) * anatomical_risk_factor
        
        # Use the weighted risk score that includes anatomical location
        combined_score = weighted_risk_score
        
        # Enhanced classification logic based on features and weighted risk score
        if concerning_features >= 3 or weighted_risk_score > 0.75:
            final_prediction = "Highly Suspicious"
            final_confidence = min(80 + (weighted_risk_score * 15), 95)
        elif concerning_features >= 2 or weighted_risk_score > 0.55:
            final_prediction = "Suspicious"
            final_confidence = 65 + (weighted_risk_score * 25)
        elif concerning_features >= 1 or weighted_risk_score > 0.35:
            final_prediction = "Moderately Concerning"
            final_confidence = 50 + (weighted_risk_score * 30)
        else:
            final_prediction = "Benign"
            final_confidence = max(25, 80 - (weighted_risk_score * 55))
        
        # Create enhanced analysis data for UI display including Evolution and anatomical location
        enhanced_analysis = {
            'features': {
                'asymmetry': enhanced_asymmetry,
                'border': enhanced_border,
                'color': enhanced_color,
                'diameter': enhanced_diameter,
                'evolution': enhanced_evolution
            },
            'anatomical_risk_factor': anatomical_risk_factor,
            'body_part': body_part,
            'detected_skin_tone': skin_type,
            'analysis_type': 'enhanced_abcde_anatomical'
        }
        
        # Add optional data if provided
        if manual_length is not None and manual_width is not None:
            enhanced_analysis['manual_measurements'] = {
                'length': manual_length,
                'width': manual_width
            }
        
        if has_evolved:
            enhanced_analysis['evolution_info'] = {
                'has_evolved': has_evolved,
                'weeks': evolution_weeks
            }
        
        logger.info(f"Enhanced ABCDE Analysis - Features: {concerning_features}/5 concerning")
        logger.info(f"ABCDE scores - A:{enhanced_asymmetry:.2f} B:{enhanced_border:.2f} C:{enhanced_color:.2f} D:{enhanced_diameter:.2f} E:{enhanced_evolution:.2f}")
        logger.info(f"Anatomical Risk Factor: {anatomical_risk_factor:.2f}x for {body_part}")
        logger.info(f"Final Assessment: {final_prediction}, Confidence: {final_confidence:.2f}%")
        
        return final_prediction, round(final_confidence, 2), enhanced_analysis
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise
    finally:
        # Clean up image memory
        if image is not None:
            try:
                image.close()
            except Exception:
                pass
        
        # Clear torch cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
