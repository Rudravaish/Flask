"""
Enhanced ISIC Model Training with Segmentation Data
Uses ISIC 2016 segmentation masks to improve lesion boundary detection and feature extraction
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b0
import cv2
import numpy as np
import os
from PIL import Image
import json
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ISICSegmentationDataset(Dataset):
    """Dataset for ISIC segmentation data"""
    
    def __init__(self, image_paths, mask_paths, transform=None, mask_transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.mask_transform = mask_transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image = Image.open(self.image_paths[idx]).convert('RGB')
        
        # Load segmentation mask
        mask = Image.open(self.mask_paths[idx]).convert('L')
        
        if self.transform:
            image = self.transform(image)
        
        if self.mask_transform:
            mask = self.mask_transform(mask)
        
        # Extract features from mask for lesion analysis
        mask_np = np.array(mask)
        features = self._extract_lesion_features(mask_np)
        
        return {
            'image': image,
            'mask': mask,
            'features': torch.tensor(features, dtype=torch.float32),
            'path': self.image_paths[idx]
        }
    
    def _extract_lesion_features(self, mask):
        """Extract ABCDE-related features from segmentation mask"""
        # Normalize mask to 0-255 range
        if mask.max() <= 1:
            mask = (mask * 255).astype(np.uint8)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return [0.0] * 8  # Return zero features if no lesion found
        
        # Get largest contour (main lesion)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Feature 1: Asymmetry (based on contour moments)
        moments = cv2.moments(largest_contour)
        if moments['m00'] > 0:
            cx = int(moments['m10'] / moments['m00'])
            cy = int(moments['m01'] / moments['m00'])
            
            # Split mask into quadrants and compare areas
            h, w = mask.shape
            q1 = mask[:cy, :cx].sum()
            q2 = mask[:cy, cx:].sum()
            q3 = mask[cy:, :cx].sum()
            q4 = mask[cy:, cx:].sum()
            
            total_area = mask.sum()
            if total_area > 0:
                asymmetry = abs(q1 + q4 - q2 - q3) / total_area
            else:
                asymmetry = 0.0
        else:
            asymmetry = 0.0
        
        # Feature 2: Border irregularity (perimeter to area ratio)
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        if area > 0:
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            border_irregularity = 1.0 - circularity
        else:
            border_irregularity = 0.0
        
        # Feature 3: Compactness
        if perimeter > 0:
            compactness = area / (perimeter * perimeter)
        else:
            compactness = 0.0
        
        # Feature 4: Solidity (area / convex hull area)
        hull = cv2.convexHull(largest_contour)
        hull_area = cv2.contourArea(hull)
        if hull_area > 0:
            solidity = area / hull_area
        else:
            solidity = 0.0
        
        # Feature 5: Extent (area / bounding box area)
        x, y, w, h = cv2.boundingRect(largest_contour)
        bbox_area = w * h
        if bbox_area > 0:
            extent = area / bbox_area
        else:
            extent = 0.0
        
        # Feature 6: Equivalent diameter
        if area > 0:
            equiv_diameter = np.sqrt(4 * area / np.pi)
        else:
            equiv_diameter = 0.0
        
        # Feature 7: Major axis length (ellipse fitting)
        if len(largest_contour) >= 5:
            ellipse = cv2.fitEllipse(largest_contour)
            major_axis = max(ellipse[1])
        else:
            major_axis = 0.0
        
        # Feature 8: Minor axis length
        if len(largest_contour) >= 5:
            minor_axis = min(ellipse[1])
        else:
            minor_axis = 0.0
        
        return [asymmetry, border_irregularity, compactness, solidity, 
                extent, equiv_diameter, major_axis, minor_axis]

class EnhancedSkinLesionModel(nn.Module):
    """Enhanced model using EfficientNet backbone with segmentation features"""
    
    def __init__(self, num_classes=2, num_features=8):
        super(EnhancedSkinLesionModel, self).__init__()
        
        # EfficientNet backbone for image features
        self.backbone = efficientnet_b0(pretrained=True)
        self.backbone.classifier = nn.Identity()
        
        # Feature fusion layers
        self.image_features = 1280  # EfficientNet-B0 output
        self.fusion_layer = nn.Sequential(
            nn.Linear(self.image_features + num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        
        # ABCDE feature prediction heads
        self.asymmetry_head = nn.Linear(256, 1)
        self.border_head = nn.Linear(256, 1)
        self.color_head = nn.Linear(256, 1)
        self.diameter_head = nn.Linear(256, 1)
        self.evolution_head = nn.Linear(256, 1)
        
        # Final classification head
        self.classifier = nn.Linear(256, num_classes)
        
        # Segmentation head (optional)
        self.segmentation_head = nn.Sequential(
            nn.ConvTranspose2d(1280, 512, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, 4, 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, image, features=None):
        # Extract image features
        img_features = self.backbone(image)
        
        if features is not None:
            # Combine image and extracted features
            combined_features = torch.cat([img_features, features], dim=1)
        else:
            combined_features = img_features
        
        # Fusion layer
        fused = self.fusion_layer(combined_features)
        
        # ABCDE predictions
        asymmetry = torch.sigmoid(self.asymmetry_head(fused))
        border = torch.sigmoid(self.border_head(fused))
        color = torch.sigmoid(self.color_head(fused))
        diameter = torch.sigmoid(self.diameter_head(fused))
        evolution = torch.sigmoid(self.evolution_head(fused))
        
        # Final classification
        classification = self.classifier(fused)
        
        return {
            'classification': classification,
            'asymmetry': asymmetry,
            'border': border,
            'color': color,
            'diameter': diameter,
            'evolution': evolution
        }

def prepare_training_data():
    """Prepare ISIC data for training"""
    logger.info("Preparing ISIC training data...")
    
    # Get all segmentation masks
    mask_dir = 'isic_data/ISBI2016_ISIC_Part1_Training_GroundTruth'
    mask_files = [f for f in os.listdir(mask_dir) if f.endswith('.png')]
    
    # Create dummy image paths (in real scenario, you'd have the actual images)
    image_paths = []
    mask_paths = []
    
    for mask_file in mask_files:
        mask_path = os.path.join(mask_dir, mask_file)
        # For training purposes, we'll create synthetic labels based on lesion characteristics
        mask_paths.append(mask_path)
        # In real implementation, you'd have corresponding image files
        image_paths.append(mask_path)  # Using mask as placeholder
    
    logger.info(f"Found {len(mask_files)} training samples")
    return image_paths, mask_paths

def create_synthetic_labels(mask_paths):
    """Create training labels based on lesion characteristics from masks"""
    logger.info("Creating synthetic labels from segmentation data...")
    
    labels = []
    for mask_path in mask_paths:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Extract features to determine label
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            
            # Simple heuristic: larger, more irregular lesions are more suspicious
            if area > 5000 and perimeter > 500:
                label = 1  # Suspicious
            else:
                label = 0  # Benign
        else:
            label = 0  # No lesion = benign
        
        labels.append(label)
    
    return labels

def train_enhanced_model():
    """Train the enhanced ISIC model"""
    logger.info("Starting enhanced ISIC model training...")
    
    # Prepare data
    image_paths, mask_paths = prepare_training_data()
    labels = create_synthetic_labels(mask_paths)
    
    # Split data
    train_images, val_images, train_masks, val_masks, train_labels, val_labels = train_test_split(
        image_paths, mask_paths, labels, test_size=0.2, random_state=42
    )
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    mask_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    # Create datasets
    train_dataset = ISICSegmentationDataset(train_images, train_masks, transform, mask_transform)
    val_dataset = ISICSegmentationDataset(val_images, val_masks, transform, mask_transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EnhancedSkinLesionModel().to(device)
    
    # Loss functions and optimizer
    classification_criterion = nn.CrossEntropyLoss()
    feature_criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    # Training loop
    num_epochs = 25
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch in train_loader:
            images = batch['image'].to(device)
            features = batch['features'].to(device)
            labels_batch = torch.tensor([train_labels[i] for i in range(len(batch['path']))]).to(device)
            
            optimizer.zero_grad()
            
            outputs = model(images, features)
            
            # Classification loss
            class_loss = classification_criterion(outputs['classification'], labels_batch)
            
            # Feature regression loss (using extracted features as targets)
            feature_loss = (
                feature_criterion(outputs['asymmetry'].squeeze(), features[:, 0]) +
                feature_criterion(outputs['border'].squeeze(), features[:, 1]) +
                feature_criterion(outputs['diameter'].squeeze(), features[:, 5])
            ) / 3.0
            
            total_loss = class_loss + 0.1 * feature_loss
            total_loss.backward()
            optimizer.step()
            
            train_loss += total_loss.item()
            _, predicted = torch.max(outputs['classification'].data, 1)
            train_total += labels_batch.size(0)
            train_correct += (predicted == labels_batch).sum().item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                features = batch['features'].to(device)
                labels_batch = torch.tensor([val_labels[i] for i in range(len(batch['path']))]).to(device)
                
                outputs = model(images, features)
                
                class_loss = classification_criterion(outputs['classification'], labels_batch)
                val_loss += class_loss.item()
                
                _, predicted = torch.max(outputs['classification'].data, 1)
                val_total += labels_batch.size(0)
                val_correct += (predicted == labels_batch).sum().item()
        
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        
        logger.info(f'Epoch [{epoch+1}/{num_epochs}]')
        logger.info(f'Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%')
        logger.info(f'Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'enhanced_isic_model.pth')
            logger.info(f'New best model saved with validation accuracy: {val_acc:.2f}%')
        
        scheduler.step()
    
    logger.info(f'Training completed. Best validation accuracy: {best_val_acc:.2f}%')
    return model

def integrate_with_existing_model():
    """Integrate enhanced model with existing prediction pipeline"""
    logger.info("Integrating enhanced model with existing pipeline...")
    
    # Load the trained model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EnhancedSkinLesionModel().to(device)
    
    if os.path.exists('enhanced_isic_model.pth'):
        model.load_state_dict(torch.load('enhanced_isic_model.pth', map_location=device))
        model.eval()
        logger.info("Enhanced model loaded successfully")
        return model
    else:
        logger.warning("Enhanced model not found, training new model...")
        return train_enhanced_model()

if __name__ == "__main__":
    # Train and save the enhanced model
    enhanced_model = train_enhanced_model()
    logger.info("Enhanced ISIC model training completed!")