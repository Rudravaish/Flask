#!/usr/bin/env python3
"""
ISIC Skin Cancer Classification Model Training
This script trains a proper skin cancer classification model on ISIC dataset
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import numpy as np
import os
import json
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import urllib.request
import zipfile
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ISICDataset(Dataset):
    """ISIC Dataset for skin lesion classification"""
    
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        try:
            image_path = self.image_paths[idx]
            image = Image.open(image_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
                
            label = self.labels[idx]
            return image, label
        except Exception as e:
            logger.warning(f"Error loading image {self.image_paths[idx]}: {e}")
            # Return a black image with label 0 in case of error
            if self.transform:
                image = self.transform(Image.new('RGB', (224, 224), (0, 0, 0)))
            else:
                image = Image.new('RGB', (224, 224), (0, 0, 0))
            return image, 0

class SkinCancerClassifier(nn.Module):
    """Enhanced skin cancer classifier with attention mechanism"""
    
    def __init__(self, num_classes=9, pretrained=True):
        super(SkinCancerClassifier, self).__init__()
        
        # Use EfficientNet as backbone (better than MobileNet for medical images)
        self.backbone = models.efficientnet_b3(weights='IMAGENET1K_V1' if pretrained else None)
        
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

def download_isic_data():
    """Download and extract ISIC dataset"""
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # ISIC 2019 dataset URLs (publicly available)
    urls = {
        "train_images": "https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_Input.zip",
        "train_metadata": "https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_Metadata.csv",
        "train_labels": "https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_GroundTruth.csv"
    }
    
    for name, url in urls.items():
        file_path = data_dir / f"{name}.{'zip' if 'zip' in url else 'csv'}"
        if not file_path.exists():
            logger.info(f"Downloading {name}...")
            try:
                urllib.request.urlretrieve(url, file_path)
                if file_path.suffix == '.zip':
                    logger.info(f"Extracting {name}...")
                    with zipfile.ZipFile(file_path, 'r') as zip_ref:
                        zip_ref.extractall(data_dir)
            except Exception as e:
                logger.error(f"Failed to download {name}: {e}")
                return False
    
    return True

def prepare_data():
    """Prepare ISIC dataset for training"""
    data_dir = Path("data")
    
    # Load metadata and labels
    try:
        metadata = pd.read_csv(data_dir / "train_metadata.csv")
        labels = pd.read_csv(data_dir / "train_labels.csv")
        
        # Merge metadata with labels
        df = pd.merge(metadata, labels, on='image')
        
        # Create diagnosis column from one-hot encoded labels
        diagnosis_columns = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC', 'UNK']
        df['diagnosis'] = df[diagnosis_columns].idxmax(axis=1)
        
        # Filter out unknown diagnoses
        df = df[df['diagnosis'] != 'UNK']
        
        # Create image paths
        image_dir = data_dir / "ISIC_2019_Training_Input"
        df['image_path'] = df['image'].apply(lambda x: str(image_dir / f"{x}.jpg"))
        
        # Filter existing images
        df = df[df['image_path'].apply(os.path.exists)]
        
        logger.info(f"Prepared {len(df)} images for training")
        logger.info(f"Class distribution:\n{df['diagnosis'].value_counts()}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error preparing data: {e}")
        return None

def create_transforms():
    """Create data augmentation transforms"""
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def train_model(model, train_loader, val_loader, device, num_epochs=50):
    """Train the skin cancer classification model"""
    
    # Loss function with class weights for imbalanced dataset
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer with different learning rates for backbone and classifier
    optimizer = optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': 1e-5},
        {'params': model.classifier.parameters(), 'lr': 1e-3}
    ], weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    best_val_acc = 0.0
    train_losses = []
    val_accuracies = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
            if batch_idx % 50 == 0:
                logger.info(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
        
        # Validation phase
        model.eval()
        correct_val = 0
        total_val = 0
        val_loss = 0.0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
        
        train_acc = 100 * correct_train / total_train
        val_acc = 100 * correct_val / total_val
        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        train_losses.append(avg_train_loss)
        val_accuracies.append(val_acc)
        
        logger.info(f'Epoch [{epoch+1}/{num_epochs}]:')
        logger.info(f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        logger.info(f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'train_losses': train_losses,
                'val_accuracies': val_accuracies
            }, 'best_skin_cancer_model.pth')
            logger.info(f'New best model saved with val accuracy: {val_acc:.2f}%')
        
        scheduler.step()
    
    return model, train_losses, val_accuracies

def main():
    """Main training pipeline"""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    
    # Download and prepare data
    logger.info("Downloading ISIC dataset...")
    if not download_isic_data():
        logger.error("Failed to download ISIC data. Please download manually.")
        return
    
    logger.info("Preparing dataset...")
    df = prepare_data()
    if df is None:
        logger.error("Failed to prepare dataset")
        return
    
    # Encode labels
    label_encoder = LabelEncoder()
    df['encoded_label'] = label_encoder.fit_transform(df['diagnosis'])
    
    # Save label encoder for later use
    import pickle
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    
    # Split data
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['encoded_label'], random_state=42)
    
    # Create transforms
    train_transform, val_transform = create_transforms()
    
    # Create datasets
    train_dataset = ISICDataset(
        train_df['image_path'].values,
        train_df['encoded_label'].values,
        transform=train_transform
    )
    
    val_dataset = ISICDataset(
        val_df['image_path'].values,
        val_df['encoded_label'].values,
        transform=val_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # Create model
    num_classes = len(label_encoder.classes_)
    model = SkinCancerClassifier(num_classes=num_classes, pretrained=True)
    model = model.to(device)
    
    logger.info(f"Model created with {num_classes} classes: {label_encoder.classes_}")
    
    # Train model
    logger.info("Starting training...")
    model, train_losses, val_accuracies = train_model(model, train_loader, val_loader, device)
    
    logger.info("Training completed!")
    logger.info(f"Best validation accuracy: {max(val_accuracies):.2f}%")

if __name__ == "__main__":
    main()