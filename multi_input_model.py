"""
Advanced Multi-Input Skin Lesion Analysis Model
EfficientNetB0 backbone with patient metadata integration
Outputs ABCDE features, risk assessment, and dermatologist appointment urgency
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import models
import cv2
import numpy as np
from PIL import Image
import logging
import base64
import io
from typing import Dict, Tuple, Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultiInputSkinLesionModel(nn.Module):
    """
    Multi-input model combining EfficientNetB0 image features with patient metadata
    """
    
    def __init__(self, num_body_parts=20, num_skin_types=6):
        super(MultiInputSkinLesionModel, self).__init__()
        
        # EfficientNetB0 backbone for image processing
        self.efficientnet = models.efficientnet_b0(pretrained=True)
        # Remove the final classifier
        self.efficientnet.classifier = nn.Identity()
        
        # Get feature dimension from EfficientNet
        self.image_feature_dim = 1280  # EfficientNetB0 output dimension
        
        # Embedding layers for categorical variables
        self.skin_type_embedding = nn.Embedding(num_skin_types, 8)
        self.body_part_embedding = nn.Embedding(num_body_parts, 16)
        
        # Metadata processing layers
        # Numeric inputs: age, UV exposure, family history (binary), evolution weeks
        self.metadata_dim = 4 + 8 + 16  # numeric + embeddings
        
        # Combined feature processing
        self.combined_dim = self.image_feature_dim + self.metadata_dim
        
        # Shared feature layers
        self.shared_layers = nn.Sequential(
            nn.Linear(self.combined_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Task-specific heads
        
        # Risk level classification (Low=0, Medium=1, High=2)
        self.risk_classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 3)
        )
        
        # Days to dermatologist (regression)
        self.days_regressor = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )
        
        # ABCDE feature extractors
        self.asymmetry_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.border_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.color_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.diameter_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        self.evolution_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, image, age, uv_exposure, family_history, skin_type, body_part, evolution_weeks):
        """
        Forward pass through the multi-input model
        
        Args:
            image: Tensor of shape (batch_size, 3, 224, 224)
            age: Tensor of shape (batch_size, 1)
            uv_exposure: Tensor of shape (batch_size, 1) 
            family_history: Tensor of shape (batch_size, 1) - binary
            skin_type: Tensor of shape (batch_size,) - categorical 0-5
            body_part: Tensor of shape (batch_size,) - categorical 0-19
            evolution_weeks: Tensor of shape (batch_size, 1)
        """
        
        # Extract image features using EfficientNet
        image_features = self.efficientnet(image)  # (batch_size, 1280)
        
        # Process categorical embeddings
        skin_type_emb = self.skin_type_embedding(skin_type)  # (batch_size, 8)
        body_part_emb = self.body_part_embedding(body_part)  # (batch_size, 16)
        
        # Combine numeric metadata
        numeric_metadata = torch.cat([
            age.unsqueeze(1) if age.dim() == 1 else age,
            uv_exposure.unsqueeze(1) if uv_exposure.dim() == 1 else uv_exposure,
            family_history.unsqueeze(1) if family_history.dim() == 1 else family_history,
            evolution_weeks.unsqueeze(1) if evolution_weeks.dim() == 1 else evolution_weeks
        ], dim=1)  # (batch_size, 4)
        
        # Combine all metadata
        metadata_features = torch.cat([
            numeric_metadata,
            skin_type_emb,
            body_part_emb
        ], dim=1)  # (batch_size, 28)
        
        # Combine image and metadata features
        combined_features = torch.cat([image_features, metadata_features], dim=1)
        
        # Process through shared layers
        shared_output = self.shared_layers(combined_features)
        
        # Generate predictions for each task
        risk_logits = self.risk_classifier(shared_output)
        days_to_derm = self.days_regressor(shared_output)
        
        # ABCDE features
        asymmetry = self.asymmetry_head(shared_output)
        border = self.border_head(shared_output)
        color = self.color_head(shared_output)
        diameter = self.diameter_head(shared_output)
        evolution = self.evolution_head(shared_output)
        
        return {
            'risk_logits': risk_logits,
            'risk_probs': F.softmax(risk_logits, dim=1),
            'days_to_dermatologist': days_to_derm,
            'abcde': {
                'asymmetry': asymmetry,
                'border': border,
                'color': color,
                'diameter': diameter,
                'evolution': evolution
            }
        }

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping for visual explanation
    """
    
    def __init__(self, model, target_layer_name='efficientnet.features'):
        self.model = model
        self.target_layer_name = target_layer_name
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks"""
        
        def forward_hook(module, input, output):
            self.activations = output
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        # Get the target layer
        target_layer = dict(self.model.named_modules())[self.target_layer_name]
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)
    
    def generate_cam(self, image, target_class=None):
        """
        Generate Grad-CAM heatmap
        
        Args:
            image: Input image tensor
            target_class: Target class for CAM (if None, uses predicted class)
        
        Returns:
            numpy array: Heatmap of shape (224, 224)
        """
        
        # Forward pass
        self.model.eval()
        output = self.model(image, 
                          torch.tensor([50.0]), # dummy metadata for CAM
                          torch.tensor([3.0]),
                          torch.tensor([0.0]),
                          torch.tensor([2]),
                          torch.tensor([10]),
                          torch.tensor([0.0]))
        
        # Get risk prediction for CAM
        risk_logits = output['risk_logits']
        
        if target_class is None:
            target_class = torch.argmax(risk_logits, dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        risk_logits[0, target_class].backward(retain_graph=True)
        
        # Generate CAM
        gradients = self.gradients[0]  # (C, H, W)
        activations = self.activations[0]  # (C, H, W)
        
        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=(1, 2))  # (C,)
        
        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # Apply ReLU
        cam = F.relu(cam)
        
        # Normalize
        cam = cam - cam.min()
        cam = cam / cam.max()
        
        # Resize to input image size
        cam = F.interpolate(cam.unsqueeze(0).unsqueeze(0), 
                          size=(224, 224), mode='bilinear', align_corners=False)
        
        return cam.squeeze().detach().numpy()

class SkinLesionPredictor:
    """
    Main predictor class that handles the complete pipeline
    """
    
    def __init__(self, model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = MultiInputSkinLesionModel()
        
        if model_path:
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                logger.info(f"Loaded trained model from {model_path}")
            except:
                logger.warning(f"Could not load model from {model_path}, using untrained model")
        
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize Grad-CAM
        self.grad_cam = GradCAM(self.model)
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Risk level mapping
        self.risk_levels = ['Low', 'Medium', 'High']
        
        # Body part mapping (0-19)
        self.body_parts = [
            'head', 'neck', 'chest', 'back', 'abdomen', 'arm_upper_left', 
            'arm_upper_right', 'arm_lower_left', 'arm_lower_right', 'hand_left',
            'hand_right', 'leg_upper_left', 'leg_upper_right', 'leg_lower_left',
            'leg_lower_right', 'foot_left', 'foot_right', 'genital', 'palm', 'sole'
        ]
    
    def preprocess_image(self, image_path):
        """Preprocess image for model input"""
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0)
            return image_tensor.to(self.device), image
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            raise
    
    def predict(self, image_path, age, uv_exposure, family_history, 
                skin_type, body_part, evolution_weeks):
        """
        Complete prediction pipeline
        
        Args:
            image_path: Path to lesion image
            age: Patient age (float)
            uv_exposure: UV exposure level 0-10 (float)
            family_history: Family history of skin cancer (0/1)
            skin_type: Fitzpatrick skin type 1-6 (int)
            body_part: Body part location 0-19 (int)
            evolution_weeks: Weeks since lesion changes (float)
        
        Returns:
            Dict containing all predictions and visualizations
        """
        
        try:
            # Preprocess inputs
            image_tensor, original_image = self.preprocess_image(image_path)
            
            # Convert inputs to tensors
            age_tensor = torch.tensor([float(age)], device=self.device)
            uv_tensor = torch.tensor([float(uv_exposure)], device=self.device)
            family_tensor = torch.tensor([float(family_history)], device=self.device)
            skin_tensor = torch.tensor([int(skin_type) - 1], device=self.device)  # Convert to 0-5
            body_tensor = torch.tensor([int(body_part)], device=self.device)
            evolution_tensor = torch.tensor([float(evolution_weeks)], device=self.device)
            
            # Model prediction
            with torch.no_grad():
                outputs = self.model(image_tensor, age_tensor, uv_tensor, 
                                   family_tensor, skin_tensor, body_tensor, evolution_tensor)
            
            # Process predictions
            risk_probs = outputs['risk_probs'].cpu().numpy()[0]
            risk_level = self.risk_levels[np.argmax(risk_probs)]
            days_to_derm = max(1, int(outputs['days_to_dermatologist'].cpu().item()))
            
            # ABCDE scores
            abcde_scores = {
                'asymmetry': float(outputs['abcde']['asymmetry'].cpu().item()),
                'border': float(outputs['abcde']['border'].cpu().item()),
                'color': float(outputs['abcde']['color'].cpu().item()),
                'diameter': max(0, float(outputs['abcde']['diameter'].cpu().item())),
                'evolution': float(outputs['abcde']['evolution'].cpu().item())
            }
            
            # Generate Grad-CAM
            grad_cam_heatmap = self.grad_cam.generate_cam(image_tensor)
            grad_cam_base64 = self._create_grad_cam_overlay(original_image, grad_cam_heatmap)
            
            results = {
                'risk_level': risk_level,
                'risk_probabilities': {
                    'Low': float(risk_probs[0]),
                    'Medium': float(risk_probs[1]),
                    'High': float(risk_probs[2])
                },
                'days_to_dermatologist': days_to_derm,
                'abcde_scores': abcde_scores,
                'grad_cam_base64': grad_cam_base64,
                'metadata_used': {
                    'age': age,
                    'uv_exposure': uv_exposure,
                    'family_history': bool(family_history),
                    'skin_type': skin_type,
                    'body_part': self.body_parts[body_part] if body_part < len(self.body_parts) else 'unknown',
                    'evolution_weeks': evolution_weeks
                }
            }
            
            logger.info(f"Prediction completed: Risk={risk_level}, Days={days_to_derm}")
            return results
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise
    
    def _create_grad_cam_overlay(self, original_image, heatmap):
        """Create Grad-CAM overlay and convert to base64"""
        try:
            # Resize original image to match heatmap
            img_resized = original_image.resize((224, 224))
            img_array = np.array(img_resized)
            
            # Create heatmap overlay
            heatmap_colored = cv2.applyColorMap(
                (heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET
            )
            heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
            
            # Blend with original image
            overlay = cv2.addWeighted(img_array, 0.6, heatmap_colored, 0.4, 0)
            
            # Convert to base64
            overlay_img = Image.fromarray(overlay)
            buffer = io.BytesIO()
            overlay_img.save(buffer, format='PNG')
            img_base64 = base64.b64encode(buffer.getvalue()).decode()
            
            return img_base64
            
        except Exception as e:
            logger.error(f"Grad-CAM overlay creation error: {e}")
            return None

# Global predictor instance
predictor = None

def initialize_predictor(model_path=None):
    """Initialize the global predictor"""
    global predictor
    predictor = SkinLesionPredictor(model_path)
    return predictor

def get_multi_input_prediction(image_path, age, uv_exposure, family_history, 
                              skin_type, body_part, evolution_weeks):
    """
    Main prediction function for the multi-input model
    """
    global predictor
    
    if predictor is None:
        predictor = initialize_predictor()
    
    return predictor.predict(image_path, age, uv_exposure, family_history,
                           skin_type, body_part, evolution_weeks)

def get_body_part_options():
    """Get available body part options"""
    return [
        'head', 'neck', 'chest', 'back', 'abdomen', 'arm_upper_left', 
        'arm_upper_right', 'arm_lower_left', 'arm_lower_right', 'hand_left',
        'hand_right', 'leg_upper_left', 'leg_upper_right', 'leg_lower_left',
        'leg_lower_right', 'foot_left', 'foot_right', 'genital', 'palm', 'sole'
    ]

if __name__ == "__main__":
    # Test the model
    predictor = SkinLesionPredictor()
    print("Multi-input skin lesion model initialized successfully")
    print(f"Body part options: {get_body_part_options()}")