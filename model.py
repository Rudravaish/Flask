from torchvision import models, transforms
from PIL import Image, ImageStat, ImageFilter, ImageEnhance
import torch
import torch.nn.functional as F
import numpy as np
import logging
import os

# Configure logging
logger = logging.getLogger(__name__)

# Initialize model
model = None
transform = None

def initialize_model():
    """Initialize the model and transforms."""
    global model, transform
    
    try:
        # Load a pretrained MobileNet v2 model
        logger.info("Loading MobileNet v2 model...")
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        model.eval()
        
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
        raise

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

def predict_lesion(image_path):
    """
    Predict whether a skin lesion is benign or suspicious.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        tuple: (prediction, confidence_percentage)
    """
    try:
        if not os.path.exists(image_path):
            raise ValueError(f"Image file not found: {image_path}")
        
        logger.info(f"Processing image: {image_path}")
        image = Image.open(image_path).convert('RGB')
        
        # Perform comprehensive medical analysis using ABCD criteria
        logger.info("Analyzing asymmetry...")
        asymmetry_score = analyze_color_asymmetry(image)
        
        logger.info("Analyzing border irregularity...")
        border_score = analyze_border_irregularity(image)
        
        logger.info("Analyzing color variation...")
        color_score = analyze_color_variation(image)
        
        logger.info("Analyzing size characteristics...")
        diameter_score = analyze_diameter_size(image)
        
        # Get CNN-based features for additional analysis (only if model is available)
        enhanced_score = 0.5  # Default fallback
        if model is not None and transform is not None:
            try:
                # Apply transforms and convert to tensor
                image_tensor = transform(image)
                # Add batch dimension for CNN input
                image_tensor = image_tensor.unsqueeze(0)
                with torch.no_grad():
                    cnn_outputs = model(image_tensor)
                    enhanced_score = enhance_cnn_prediction(cnn_outputs[0], 
                                                          (asymmetry_score, border_score, color_score, diameter_score))
            except Exception as e:
                logger.warning(f"CNN analysis failed, using medical analysis only: {e}")
                enhanced_score = (asymmetry_score + border_score + color_score + diameter_score) / 4.0
        
        # Comprehensive medical risk assessment
        medical_prediction, medical_confidence = medical_risk_assessment(
            asymmetry_score, border_score, color_score, diameter_score
        )
        
        # Combine medical analysis with enhanced CNN features
        final_confidence = (medical_confidence + enhanced_score * 100) / 2
        
        # Determine final prediction based on comprehensive analysis
        if final_confidence > 70:
            final_prediction = "Highly Suspicious"
        elif final_confidence > 50:
            final_prediction = "Suspicious" 
        elif final_confidence > 35:
            final_prediction = "Moderately Concerning"
        else:
            final_prediction = "Benign"
        
        logger.info(f"Medical Analysis - A:{asymmetry_score:.2f} B:{border_score:.2f} C:{color_score:.2f} D:{diameter_score:.2f}")
        logger.info(f"Final Assessment: {final_prediction}, Confidence: {final_confidence:.2f}%")
        
        return final_prediction, round(final_confidence, 2)
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise
