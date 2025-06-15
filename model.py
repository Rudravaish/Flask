from torchvision import models, transforms
from PIL import Image
import torch
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

# For this demo, we'll map some ImageNet classes to skin lesion categories
# In production, you would use a model specifically trained on skin lesion data
def map_to_skin_categories(prediction_idx, confidence):
    """
    Map ImageNet predictions to skin lesion categories.
    This is a simplified mapping for demonstration purposes.
    In production, use a model trained specifically on skin lesion data.
    """
    # This is a mock mapping - in reality you'd use a skin lesion-specific model
    # We'll use a simple heuristic based on the prediction confidence
    if confidence > 0.7:
        return "Suspicious", confidence * 100
    else:
        return "Benign", confidence * 100

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
