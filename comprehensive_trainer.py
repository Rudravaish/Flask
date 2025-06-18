
"""
Comprehensive Training Pipeline for Multi-Input Skin Lesion Model
Incorporates patient metadata, anatomical location, and temporal evolution data
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from PIL import Image
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from multi_input_model import MultiInputSkinLesionModel
import json
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ComprehensiveDataset(Dataset):
    """Dataset that includes images and comprehensive patient metadata"""
    
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load image
        try:
            image = Image.open(row['image_path']).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            logger.warning(f"Error loading image {row['image_path']}: {e}")
            # Return black image as fallback
            image = torch.zeros((3, 224, 224))
        
        # Extract metadata
        metadata = {
            'age': float(row.get('age', 50.0)),
            'uv_exposure': float(row.get('uv_exposure', 5.0)),
            'family_history': float(row.get('family_history', 0.0)),
            'skin_type': int(row.get('skin_type', 2)),
            'body_part': int(row.get('body_part', 0)),
            'evolution_weeks': float(row.get('evolution_weeks', 0.0)),
            'lesion_diameter': float(row.get('lesion_diameter', 5.0))
        }
        
        # Target variables
        targets = {
            'risk_level': int(row.get('risk_level', 0)),  # 0=Low, 1=Medium, 2=High
            'days_to_derm': float(row.get('days_to_derm', 30.0)),
            'asymmetry': float(row.get('asymmetry_score', 0.3)),
            'border': float(row.get('border_score', 0.3)),
            'color': float(row.get('color_score', 0.3)),
            'diameter': float(row.get('diameter_score', 0.3)),
            'evolution': float(row.get('evolution_score', 0.1))
        }
        
        return image, metadata, targets

def create_synthetic_dataset(n_samples=1000):
    """Create synthetic dataset for training demonstration"""
    np.random.seed(42)
    
    data = []
    for i in range(n_samples):
        # Generate synthetic patient data
        age = np.random.normal(55, 15)  # Age distribution
        age = max(18, min(90, age))
        
        uv_exposure = np.random.exponential(3)  # UV exposure (0-10 scale)
        uv_exposure = min(10, uv_exposure)
        
        family_history = np.random.choice([0, 1], p=[0.85, 0.15])  # 15% have family history
        skin_type = np.random.choice(range(6), p=[0.1, 0.2, 0.3, 0.25, 0.1, 0.05])
        body_part = np.random.choice(range(20))
        
        evolution_weeks = np.random.exponential(8) if np.random.random() < 0.3 else 0
        lesion_diameter = np.random.lognormal(1.5, 0.5)  # Log-normal distribution for size
        
        # Risk calculation based on factors
        risk_factors = []
        if age > 65: risk_factors.append(1)
        if uv_exposure > 7: risk_factors.append(1)
        if family_history: risk_factors.append(2)
        if evolution_weeks > 4: risk_factors.append(2)
        if lesion_diameter > 6: risk_factors.append(1)
        if skin_type < 2: risk_factors.append(1)  # Fair skin
        
        risk_score = sum(risk_factors)
        if risk_score >= 4:
            risk_level = 2  # High
            days_to_derm = np.random.randint(1, 7)
        elif risk_score >= 2:
            risk_level = 1  # Medium
            days_to_derm = np.random.randint(7, 30)
        else:
            risk_level = 0  # Low
            days_to_derm = np.random.randint(30, 90)
        
        # ABCDE scores (synthetic based on risk)
        base_score = 0.2 + (risk_level * 0.2)
        asymmetry = max(0, min(1, np.random.normal(base_score, 0.15)))
        border = max(0, min(1, np.random.normal(base_score, 0.15)))
        color = max(0, min(1, np.random.normal(base_score, 0.15)))
        diameter_score = max(0, min(1, lesion_diameter / 10.0))
        evolution_score = max(0, min(1, evolution_weeks / 20.0))
        
        data.append({
            'image_path': f'synthetic_image_{i}.jpg',  # Placeholder
            'age': age,
            'uv_exposure': uv_exposure,
            'family_history': family_history,
            'skin_type': skin_type,
            'body_part': body_part,
            'evolution_weeks': evolution_weeks,
            'lesion_diameter': lesion_diameter,
            'risk_level': risk_level,
            'days_to_derm': days_to_derm,
            'asymmetry_score': asymmetry,
            'border_score': border,
            'color_score': color,
            'diameter_score': diameter_score,
            'evolution_score': evolution_score
        })
    
    return pd.DataFrame(data)

def train_comprehensive_model():
    """Train the comprehensive multi-input model"""
    
    # Create synthetic dataset (replace with real ISIC data loading)
    logger.info("Creating synthetic dataset...")
    df = create_synthetic_dataset(1000)
    
    # Split data
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, 
                                       stratify=df['risk_level'])
    
    # Create datasets (note: images would need to exist for real training)
    # For now, we'll create dummy datasets
    from torchvision import transforms
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Initialize model
    model = MultiInputSkinLesionModel()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Define loss functions for multi-task learning
    risk_criterion = nn.CrossEntropyLoss()
    regression_criterion = nn.MSELoss()
    abcde_criterion = nn.MSELoss()
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    
    # Training configuration
    num_epochs = 50
    best_val_loss = float('inf')
    
    logger.info("Starting training...")
    
    for epoch in range(num_epochs):
        model.train()
        
        # Simulate training batch (replace with real DataLoader)
        batch_size = 32
        for batch_idx in range(len(train_df) // batch_size):
            # Create dummy batch
            images = torch.randn(batch_size, 3, 224, 224).to(device)
            
            # Sample metadata from training data
            batch_data = train_df.sample(batch_size)
            
            ages = torch.tensor(batch_data['age'].values, dtype=torch.float32).to(device)
            uv_exposure = torch.tensor(batch_data['uv_exposure'].values, dtype=torch.float32).to(device)
            family_history = torch.tensor(batch_data['family_history'].values, dtype=torch.float32).to(device)
            skin_types = torch.tensor(batch_data['skin_type'].values, dtype=torch.long).to(device)
            body_parts = torch.tensor(batch_data['body_part'].values, dtype=torch.long).to(device)
            evolution_weeks = torch.tensor(batch_data['evolution_weeks'].values, dtype=torch.float32).to(device)
            
            # Targets
            risk_labels = torch.tensor(batch_data['risk_level'].values, dtype=torch.long).to(device)
            days_targets = torch.tensor(batch_data['days_to_derm'].values, dtype=torch.float32).to(device)
            asymmetry_targets = torch.tensor(batch_data['asymmetry_score'].values, dtype=torch.float32).to(device)
            border_targets = torch.tensor(batch_data['border_score'].values, dtype=torch.float32).to(device)
            color_targets = torch.tensor(batch_data['color_score'].values, dtype=torch.float32).to(device)
            diameter_targets = torch.tensor(batch_data['diameter_score'].values, dtype=torch.float32).to(device)
            evolution_targets = torch.tensor(batch_data['evolution_score'].values, dtype=torch.float32).to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images, ages, uv_exposure, family_history, 
                          skin_types, body_parts, evolution_weeks)
            
            # Calculate losses
            risk_loss = risk_criterion(outputs['risk_logits'], risk_labels)
            days_loss = regression_criterion(outputs['days_to_dermatologist'].squeeze(), days_targets)
            asymmetry_loss = abcde_criterion(outputs['abcde']['asymmetry'].squeeze(), asymmetry_targets)
            border_loss = abcde_criterion(outputs['abcde']['border'].squeeze(), border_targets)
            color_loss = abcde_criterion(outputs['abcde']['color'].squeeze(), color_targets)
            diameter_loss = abcde_criterion(outputs['abcde']['diameter'].squeeze(), diameter_targets)
            evolution_loss = abcde_criterion(outputs['abcde']['evolution'].squeeze(), evolution_targets)
            
            # Combined loss with weights
            total_loss = (risk_loss * 2.0 + 
                         days_loss * 0.1 +
                         asymmetry_loss * 0.5 +
                         border_loss * 0.5 +
                         color_loss * 0.5 +
                         diameter_loss * 0.3 +
                         evolution_loss * 0.4)
            
            total_loss.backward()
            optimizer.step()
            
            if batch_idx % 10 == 0:
                logger.info(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, '
                          f'Total Loss: {total_loss.item():.4f}, '
                          f'Risk Loss: {risk_loss.item():.4f}')
        
        scheduler.step()
        
        # Save model checkpoint
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_df_sample': train_df.head().to_dict()
            }, f'comprehensive_model_epoch_{epoch}.pth')
            logger.info(f'Saved checkpoint at epoch {epoch}')
    
    # Save final model
    torch.save(model.state_dict(), 'comprehensive_skin_model_final.pth')
    logger.info("Training completed and model saved!")
    
    return model

if __name__ == "__main__":
    train_comprehensive_model()
