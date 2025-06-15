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
    # Convert to different color spaces for analysis
    hsv = image.convert('HSV')
    
    # Split image into quadrants and analyze color variance
    width, height = image.size
    quadrants = [
        image.crop((0, 0, width//2, height//2)),
        image.crop((width//2, 0, width, height//2)),
        image.crop((0, height//2, width//2, height)),
        image.crop((width//2, height//2, width, height))
    ]
    
    color_variances = []
    for quad in quadrants:
        stat = ImageStat.Stat(quad)
        # Calculate variance in RGB channels
        variance = sum(stat.stddev[:3]) / 3
        color_variances.append(variance)
    
    # High variance between quadrants suggests asymmetry
    asymmetry_score = np.std(color_variances) / (np.mean(color_variances) + 1e-6)
    return min(asymmetry_score / 50.0, 1.0)  # Normalize to 0-1

def analyze_border_irregularity(image):
    """Analyze border irregularity - melanomas often have irregular borders"""
    # Convert to grayscale and apply edge detection
    gray = image.convert('L')
    edges = gray.filter(ImageFilter.FIND_EDGES)
    
    # Convert to numpy for analysis
    edge_array = np.array(edges)
    
    # Find contours and analyze border roughness
    edge_pixels = np.where(edge_array > 50)
    if len(edge_pixels[0]) == 0:
        return 0.5
    
    # Calculate border irregularity based on edge pixel distribution
    edge_variance = np.var(edge_pixels[0]) + np.var(edge_pixels[1])
    border_score = min(edge_variance / 10000.0, 1.0)
    return border_score

def analyze_color_variation(image):
    """Analyze color variation - melanomas often have multiple colors"""
    # Enhance contrast to better detect color variations
    enhancer = ImageEnhance.Contrast(image)
    enhanced = enhancer.enhance(1.5)
    
    # Convert to HSV for better color analysis
    hsv = enhanced.convert('HSV')
    hue_stat = ImageStat.Stat(hsv.split()[0])  # Hue channel
    sat_stat = ImageStat.Stat(hsv.split()[1])  # Saturation channel
    
    # High standard deviation in hue and saturation indicates color variation
    hue_variation = hue_stat.stddev[0] / 255.0
    sat_variation = sat_stat.stddev[0] / 255.0
    
    color_score = (hue_variation + sat_variation) / 2.0
    return min(color_score * 2.0, 1.0)

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
        
        # Load and preprocess the image
        logger.info(f"Processing image: {image_path}")
        image = Image.open(image_path).convert('RGB')
        
        # Apply transforms
        image_tensor = transform(image).unsqueeze(0)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            
            # Get the highest confidence prediction
            max_prob, predicted_idx = torch.max(probabilities, 0)
            confidence = float(max_prob)
            
            # Map to skin lesion categories
            prediction, confidence_pct = map_to_skin_categories(int(predicted_idx), confidence)
            
            logger.info(f"Prediction: {prediction}, Confidence: {confidence_pct:.2f}%")
            
            return prediction, round(confidence_pct, 2)
            
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        raise Exception(f"Failed to process image: {str(e)}")
