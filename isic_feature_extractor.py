"""
ISIC Feature Extractor - Extract enhanced features from segmentation masks
"""
import cv2
import numpy as np
import os
from PIL import Image
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ISICFeatureExtractor:
    """Extract enhanced lesion features from ISIC segmentation masks"""
    
    def __init__(self):
        self.feature_cache = {}
        self.load_isic_features()
    
    def load_isic_features(self):
        """Load pre-computed features from ISIC dataset"""
        try:
            mask_dir = 'isic_data/ISBI2016_ISIC_Part1_Training_GroundTruth'
            if os.path.exists(mask_dir):
                self.extract_all_features(mask_dir)
                logger.info(f"Loaded features from {len(self.feature_cache)} ISIC samples")
        except Exception as e:
            logger.warning(f"Could not load ISIC features: {e}")
    
    def extract_all_features(self, mask_dir):
        """Extract features from all available masks"""
        mask_files = [f for f in os.listdir(mask_dir) if f.endswith('.png')]
        
        for mask_file in mask_files[:50]:  # Process first 50 for efficiency
            mask_path = os.path.join(mask_dir, mask_file)
            try:
                features = self.extract_lesion_features_from_mask(mask_path)
                self.feature_cache[mask_file] = features
            except Exception as e:
                logger.warning(f"Error processing {mask_file}: {e}")
    
    def extract_lesion_features_from_mask(self, mask_path):
        """Extract comprehensive lesion features from segmentation mask"""
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if mask is None:
            return self.get_default_features()
        
        # Threshold mask to ensure binary
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return self.get_default_features()
        
        # Get largest contour (main lesion)
        largest_contour = max(contours, key=cv2.contourArea)
        
        return self.compute_advanced_features(largest_contour, mask)
    
    def compute_advanced_features(self, contour, mask):
        """Compute advanced ABCDE features from contour and mask"""
        features = {}
        
        # Basic measurements
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        # 1. Asymmetry Analysis
        features['asymmetry'] = self.compute_asymmetry(contour, mask)
        
        # 2. Border Irregularity
        features['border_irregularity'] = self.compute_border_irregularity(area, perimeter)
        
        # 3. Compactness and Shape Features
        features['compactness'] = self.compute_compactness(area, perimeter)
        features['solidity'] = self.compute_solidity(contour)
        features['extent'] = self.compute_extent(contour)
        
        # 4. Size Features
        features['area_normalized'] = min(area / 10000.0, 1.0)  # Normalize to 0-1
        features['equivalent_diameter'] = np.sqrt(4 * area / np.pi) if area > 0 else 0
        
        # 5. Ellipse Features
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            features['major_axis'] = max(ellipse[1])
            features['minor_axis'] = min(ellipse[1])
            features['eccentricity'] = self.compute_eccentricity(ellipse[1])
        else:
            features['major_axis'] = 0
            features['minor_axis'] = 0
            features['eccentricity'] = 0
        
        # 6. Convexity Defects
        features['convexity_defects'] = self.compute_convexity_defects(contour)
        
        return features
    
    def compute_asymmetry(self, contour, mask):
        """Compute asymmetry score using multiple axes"""
        moments = cv2.moments(contour)
        if moments['m00'] == 0:
            return 0.0
        
        # Centroid
        cx = int(moments['m10'] / moments['m00'])
        cy = int(moments['m01'] / moments['m00'])
        
        h, w = mask.shape
        
        # Check bounds
        if cx >= w or cy >= h or cx < 0 or cy < 0:
            return 0.0
        
        # Vertical asymmetry
        left_area = mask[:, :cx].sum() if cx > 0 else 0
        right_area = mask[:, cx:].sum() if cx < w else 0
        
        # Horizontal asymmetry
        top_area = mask[:cy, :].sum() if cy > 0 else 0
        bottom_area = mask[cy:, :].sum() if cy < h else 0
        
        total_area = mask.sum()
        if total_area == 0:
            return 0.0
        
        vertical_asym = abs(left_area - right_area) / total_area
        horizontal_asym = abs(top_area - bottom_area) / total_area
        
        return (vertical_asym + horizontal_asym) / 2.0
    
    def compute_border_irregularity(self, area, perimeter):
        """Compute border irregularity using circularity"""
        if perimeter == 0:
            return 0.0
        
        # Circularity (1.0 = perfect circle, lower = more irregular)
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        # Convert to irregularity score (higher = more irregular)
        irregularity = 1.0 - min(circularity, 1.0)
        return irregularity
    
    def compute_compactness(self, area, perimeter):
        """Compute compactness measure"""
        if perimeter == 0:
            return 0.0
        return area / (perimeter * perimeter)
    
    def compute_solidity(self, contour):
        """Compute solidity (area / convex hull area)"""
        area = cv2.contourArea(contour)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        
        if hull_area == 0:
            return 0.0
        
        return area / hull_area
    
    def compute_extent(self, contour):
        """Compute extent (area / bounding box area)"""
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        bbox_area = w * h
        
        if bbox_area == 0:
            return 0.0
        
        return area / bbox_area
    
    def compute_eccentricity(self, axes):
        """Compute eccentricity of fitted ellipse"""
        major_axis, minor_axis = max(axes), min(axes)
        
        if major_axis == 0:
            return 0.0
        
        # Eccentricity formula
        eccentricity = np.sqrt(1 - (minor_axis / major_axis) ** 2)
        return eccentricity
    
    def compute_convexity_defects(self, contour):
        """Compute convexity defects as a measure of border irregularity"""
        hull = cv2.convexHull(contour, returnPoints=False)
        
        if len(hull) < 3 or len(contour) < 4:
            return 0.0
        
        try:
            defects = cv2.convexityDefects(contour, hull)
            if defects is None:
                return 0.0
            
            # Sum of defect depths normalized by contour length
            total_defect = 0
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                total_defect += d
            
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                return 0.0
            
            return total_defect / (perimeter * 256)  # Normalize
            
        except:
            return 0.0
    
    def get_default_features(self):
        """Return default features when extraction fails"""
        return {
            'asymmetry': 0.0,
            'border_irregularity': 0.0,
            'compactness': 0.0,
            'solidity': 1.0,
            'extent': 1.0,
            'area_normalized': 0.0,
            'equivalent_diameter': 0.0,
            'major_axis': 0.0,
            'minor_axis': 0.0,
            'eccentricity': 0.0,
            'convexity_defects': 0.0
        }
    
    def get_enhanced_features_for_analysis(self, image_analysis_result):
        """Enhance existing analysis with ISIC-derived features"""
        # Use statistical patterns from ISIC data to improve feature accuracy
        if not self.feature_cache:
            return image_analysis_result
        
        # Calculate statistical benchmarks from ISIC data
        asymmetry_values = [f['asymmetry'] for f in self.feature_cache.values()]
        border_values = [f['border_irregularity'] for f in self.feature_cache.values()]
        
        if asymmetry_values and border_values:
            asymmetry_mean = np.mean(asymmetry_values)
            asymmetry_std = np.std(asymmetry_values)
            border_mean = np.mean(border_values)
            border_std = np.std(border_values)
            
            # Adjust current analysis based on ISIC statistical patterns
            current_asymmetry = image_analysis_result.get('asymmetry', 0)
            current_border = image_analysis_result.get('border', 0)
            
            # Z-score normalization based on ISIC data
            if asymmetry_std > 0:
                asymmetry_zscore = (current_asymmetry - asymmetry_mean) / asymmetry_std
                enhanced_asymmetry = min(max(0.5 + asymmetry_zscore * 0.2, 0), 1)
            else:
                enhanced_asymmetry = current_asymmetry
            
            if border_std > 0:
                border_zscore = (current_border - border_mean) / border_std
                enhanced_border = min(max(0.5 + border_zscore * 0.2, 0), 1)
            else:
                enhanced_border = current_border
            
            # Update analysis with enhanced features
            image_analysis_result['asymmetry'] = enhanced_asymmetry
            image_analysis_result['border'] = enhanced_border
            image_analysis_result['isic_enhanced'] = True
        
        return image_analysis_result

# Global instance
isic_extractor = ISICFeatureExtractor()

def get_isic_enhanced_analysis(analysis_result):
    """Get ISIC-enhanced analysis for better accuracy"""
    return isic_extractor.get_enhanced_features_for_analysis(analysis_result)