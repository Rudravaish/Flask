"""
Trained ISIC Skin Cancer Classification Model
This module handles loading and inference with a properly trained model
"""

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import logging
import pickle
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class SkinCancerClassifier(nn.Module):
    """Enhanced skin cancer classifier trained on ISIC data"""
    
    def __init__(self, num_classes=9, pretrained=False):
        super(SkinCancerClassifier, self).__init__()
        
        # Use EfficientNet as backbone
        self.backbone = models.efficientnet_b3(weights=None)
        
        # Replace classifier with custom head
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        
        # Custom classifier with dropout and batch norm
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        features = self.backbone(x)
        output = self.classifier(features)
        return output

class ISICSkinCancerModel:
    """Wrapper for trained ISIC skin cancer model"""
    
    def __init__(self):
        self.model = None
        self.label_encoder = None
        self.transform = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.load_model()
    
    def load_model(self):
        """Load the trained model and label encoder"""
        try:
            model_path = 'best_skin_cancer_model.pth'
            encoder_path = 'label_encoder.pkl'
            
            if os.path.exists(model_path) and os.path.exists(encoder_path):
                # Load label encoder
                with open(encoder_path, 'rb') as f:
                    self.label_encoder = pickle.load(f)
                
                # Create model
                num_classes = len(self.label_encoder.classes_)
                self.model = SkinCancerClassifier(num_classes=num_classes)
                
                # Load trained weights
                checkpoint = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.to(self.device)
                self.model.eval()
                
                # Define transform
                self.transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                
                logger.info(f"Loaded trained ISIC model with {num_classes} classes")
                logger.info(f"Classes: {list(self.label_encoder.classes_)}")
                
            else:
                logger.warning("Trained model not found, using fallback analysis")
                self.model = None
                
        except Exception as e:
            logger.error(f"Error loading trained model: {e}")
            self.model = None
    
    def predict(self, image):
        """Make prediction using trained model"""
        if self.model is None:
            return None, 0.0
        
        try:
            # Preprocess image
            if isinstance(image, str):
                image = Image.open(image).convert('RGB')
            
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted_class = torch.max(probabilities, 1)
                
                # Convert to class name
                class_name = self.label_encoder.inverse_transform([predicted_class.cpu().item()])[0]
                confidence_score = confidence.cpu().item() * 100
                
                # Map to our classification system
                prediction = self.map_to_classification(class_name, confidence_score)
                
                return prediction, confidence_score
                
        except Exception as e:
            logger.error(f"Error during model prediction: {e}")
            return None, 0.0
    
    def map_to_classification(self, isic_class, confidence):
        """Map ISIC classes to our binary classification system"""
        # ISIC classes mapping
        malignant_classes = ['MEL', 'BCC', 'SCC']  # Melanoma, Basal Cell Carcinoma, Squamous Cell Carcinoma
        benign_classes = ['NV', 'BKL', 'AK', 'DF', 'VASC']  # Nevus, Benign Keratosis, etc.
        
        if isic_class in malignant_classes:
            if confidence > 80:
                return "Highly Suspicious"
            elif confidence > 60:
                return "Suspicious"
            else:
                return "Moderately Concerning"
        elif isic_class in benign_classes:
            if confidence > 80:
                return "Benign"
            else:
                return "Moderately Concerning"
        else:
            return "Moderately Concerning"

# Global instance
trained_model = ISICSkinCancerModel()

def get_trained_prediction(image_path):
    """Get prediction from trained model"""
    return trained_model.predict(image_path)

def is_trained_model_available():
    """Check if trained model is available"""
    return trained_model.model is not None