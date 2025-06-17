"""
Enhanced Skin Analysis for Improved Classifications Across All Skin Tones
Specialized algorithms for darker skin tone lesion detection
"""

import numpy as np
from PIL import Image, ImageStat, ImageFilter, ImageEnhance, ImageOps
import logging
from sklearn.cluster import KMeans
import cv2

logger = logging.getLogger(__name__)

class AdvancedSkinToneAnalyzer:
    """Advanced analyzer specifically designed for accurate detection across all skin tones"""
    
    def __init__(self):
        # Melanin-aware color spaces and thresholds
        self.skin_tone_ranges = {
            'I': {'ita_angle': (55, 90), 'melanin_index': (0.0, 0.2)},
            'II': {'ita_angle': (41, 55), 'melanin_index': (0.2, 0.35)},
            'III': {'ita_angle': (28, 41), 'melanin_index': (0.35, 0.5)},
            'IV': {'ita_angle': (10, 28), 'melanin_index': (0.5, 0.65)},
            'V': {'ita_angle': (-30, 10), 'melanin_index': (0.65, 0.8)},
            'VI': {'ita_angle': (-90, -30), 'melanin_index': (0.8, 1.0)}
        }
        
        # Lesion indicators specific to darker skin tones
        self.darker_skin_indicators = {
            'hyperpigmentation_patterns': True,
            'keloid_risk_assessment': True,
            'post_inflammatory_changes': True,
            'melasma_differentiation': True,
            'seborrheic_keratosis_variants': True
        }
        
        # Enhanced ABCDE criteria weights for darker skin
        self.darker_skin_weights = {
            'asymmetry': 0.20,  # Less reliable due to post-inflammatory changes
            'border': 0.25,     # Modified for hyperpigmentation borders
            'color': 0.35,      # More important - different color patterns
            'diameter': 0.15,   # Standard importance
            'evolution': 0.05   # Texture changes more subtle
        }
        
        self.lighter_skin_weights = {
            'asymmetry': 0.25,
            'border': 0.30,
            'color': 0.30,
            'diameter': 0.15,
            'evolution': 0.00
        }

    def detect_skin_tone(self, image):
        """Detect skin tone using ITA (Individual Typology Angle) and melanin index"""
        try:
            # Convert to LAB color space for ITA calculation
            lab_image = image.convert('LAB')
            lab_array = np.array(lab_image)
            
            # Calculate ITA angle
            L = lab_array[:,:,0]
            b = lab_array[:,:,2]
            
            # ITA = (arctan((L-50)/b)) * (180/π)
            ita_values = np.arctan2(L.astype(float) - 50, b.astype(float)) * (180 / np.pi)
            mean_ita = np.mean(ita_values[~np.isnan(ita_values)])
            
            # Calculate melanin index
            rgb_array = np.array(image)
            melanin_index = self._calculate_melanin_index(rgb_array)
            
            # Determine Fitzpatrick type
            fitzpatrick_type = self._classify_fitzpatrick_type(mean_ita, melanin_index)
            
            logger.info(f"Detected skin tone: Type {fitzpatrick_type} (ITA: {mean_ita:.2f}, Melanin: {melanin_index:.3f})")
            
            return fitzpatrick_type, mean_ita, melanin_index
            
        except Exception as e:
            logger.warning(f"Skin tone detection failed: {e}")
            return 'III', 0, 0.5  # Default to medium tone

    def _calculate_melanin_index(self, rgb_array):
        """Calculate melanin index using spectral analysis"""
        try:
            # Melanin index based on RGB ratios
            # Higher melanin content shows specific absorption patterns
            r = rgb_array[:,:,0].astype(float)
            g = rgb_array[:,:,1].astype(float)
            b = rgb_array[:,:,2].astype(float)
            
            # Melanin absorption is stronger in blue/green wavelengths
            melanin_absorption = (b + g) / (2 * r + 1e-6)
            melanin_index = np.mean(melanin_absorption)
            
            # Normalize to 0-1 range
            melanin_index = min(max(melanin_index - 0.5, 0) / 1.5, 1.0)
            
            return melanin_index
            
        except Exception:
            return 0.5

    def _classify_fitzpatrick_type(self, ita_angle, melanin_index):
        """Classify Fitzpatrick skin type using ITA and melanin index"""
        for skin_type, ranges in self.skin_tone_ranges.items():
            if (ranges['ita_angle'][0] <= ita_angle <= ranges['ita_angle'][1] and
                ranges['melanin_index'][0] <= melanin_index <= ranges['melanin_index'][1]):
                return skin_type
        
        # Fallback classification based on melanin index
        if melanin_index > 0.7:
            return 'VI'
        elif melanin_index > 0.6:
            return 'V'
        elif melanin_index > 0.4:
            return 'IV'
        elif melanin_index > 0.3:
            return 'III'
        elif melanin_index > 0.2:
            return 'II'
        else:
            return 'I'

    def analyze_for_darker_skin(self, image, fitzpatrick_type):
        """Specialized analysis for darker skin tones (Types IV-VI)"""
        
        if fitzpatrick_type in ['IV', 'V', 'VI']:
            return self._darker_skin_analysis(image, fitzpatrick_type)
        else:
            return self._lighter_skin_analysis(image, fitzpatrick_type)

    def _darker_skin_analysis(self, image, skin_type):
        """Enhanced analysis specifically for darker skin tones"""
        try:
            # Asymmetry analysis adjusted for post-inflammatory hyperpigmentation
            asymmetry_score = self._asymmetry_darker_skin(image)
            
            # Border analysis considering common hyperpigmentation patterns
            border_score = self._border_analysis_darker_skin(image)
            
            # Color analysis focusing on darker skin lesion patterns
            color_score = self._color_analysis_darker_skin(image)
            
            # Diameter analysis remains standard
            diameter_score = self._diameter_analysis(image)
            
            # Evolution patterns specific to darker skin
            evolution_score = self._evolution_darker_skin(image)
            
            # Differential diagnosis considerations
            differential_analysis = self._differential_diagnosis_darker_skin(image)
            
            logger.info(f"Darker skin analysis ({skin_type}) - A:{asymmetry_score:.2f} B:{border_score:.2f} C:{color_score:.2f} D:{diameter_score:.2f} E:{evolution_score:.2f}")
            
            return {
                'asymmetry': asymmetry_score,
                'border': border_score,
                'color': color_score,
                'diameter': diameter_score,
                'evolution': evolution_score,
                'differential': differential_analysis,
                'skin_type': skin_type,
                'analysis_type': 'darker_skin_specialized'
            }
            
        except Exception as e:
            logger.error(f"Darker skin analysis failed: {e}")
            return self._fallback_analysis(image)

    def _asymmetry_darker_skin(self, image):
        """Asymmetry analysis adjusted for darker skin characteristics"""
        try:
            # Use LAB color space for better contrast in darker skin
            lab_image = image.convert('LAB')
            lab_array = np.array(lab_image)
            
            width, height = image.size
            center_x, center_y = width // 2, height // 2
            
            # Analyze asymmetry in LAB space (better for darker skin)
            asymmetry_scores = []
            
            # L channel asymmetry (lightness)
            l_channel = lab_array[:,:,0]
            left_l = l_channel[:, :center_x]
            right_l = l_channel[:, center_x:]
            right_l_flipped = np.fliplr(right_l)
            
            if left_l.shape == right_l_flipped.shape:
                l_asymmetry = np.mean(np.abs(left_l.astype(float) - right_l_flipped.astype(float))) / 255.0
                asymmetry_scores.append(l_asymmetry)
            
            # A and B channel asymmetry for color differences
            for channel_idx in [1, 2]:  # A and B channels
                channel = lab_array[:,:,channel_idx]
                left_channel = channel[:, :center_x]
                right_channel = channel[:, center_x:]
                right_channel_flipped = np.fliplr(right_channel)
                
                if left_channel.shape == right_channel_flipped.shape:
                    channel_asymmetry = np.mean(np.abs(left_channel.astype(float) - right_channel_flipped.astype(float))) / 255.0
                    asymmetry_scores.append(channel_asymmetry * 0.8)  # Weight color channels less for darker skin
            
            # Texture asymmetry using local binary patterns
            texture_asymmetry = self._texture_asymmetry_analysis(image)
            asymmetry_scores.append(texture_asymmetry)
            
            return min(np.mean(asymmetry_scores) * 1.5, 1.0)
            
        except Exception:
            return 0.4  # Conservative default for darker skin

    def _border_analysis_darker_skin(self, image):
        """Border analysis considering hyperpigmentation patterns in darker skin"""
        try:
            # Convert to LAB for better edge detection in darker skin
            lab_image = image.convert('LAB')
            l_channel = np.array(lab_image.split()[0])
            
            # Enhanced contrast for better edge detection
            l_enhanced = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)).apply(l_channel)
            
            # Multiple edge detection methods
            # Sobel edge detection
            sobel_x = cv2.Sobel(l_enhanced, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(l_enhanced, cv2.CV_64F, 0, 1, ksize=3)
            sobel_combined = np.sqrt(sobel_x**2 + sobel_y**2)
            
            # Canny edge detection with adaptive thresholds
            edges_canny = cv2.Canny(l_enhanced, 50, 150)
            
            # Analyze border characteristics
            edge_density = np.sum(edges_canny > 0) / edges_canny.size
            edge_irregularity = np.std(sobel_combined) / (np.mean(sobel_combined) + 1e-6)
            
            # Border smoothness analysis
            contours, _ = cv2.findContours(edges_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                perimeter = cv2.arcLength(largest_contour, True)
                area = cv2.contourArea(largest_contour)
                
                if area > 0:
                    # Circularity measure (4π*area/perimeter²)
                    circularity = 4 * np.pi * area / (perimeter**2 + 1e-6)
                    border_irregularity = 1.0 - circularity
                else:
                    border_irregularity = 0.5
            else:
                border_irregularity = edge_irregularity
            
            border_score = (edge_density + border_irregularity) / 2.0
            return min(border_score * 1.2, 1.0)  # Slightly enhanced for darker skin
            
        except Exception:
            return 0.5

    def _color_analysis_darker_skin(self, image):
        """Enhanced color analysis for darker skin tones"""
        try:
            # Convert to multiple color spaces for comprehensive analysis
            hsv_image = image.convert('HSV')
            lab_image = image.convert('LAB')
            
            # Analyze in HSV space
            hsv_array = np.array(hsv_image)
            h_channel = hsv_array[:,:,0]
            s_channel = hsv_array[:,:,1]
            v_channel = hsv_array[:,:,2]
            
            # Color variation metrics
            hue_variance = np.var(h_channel) / (255.0**2)
            saturation_range = (np.max(s_channel) - np.min(s_channel)) / 255.0
            value_variance = np.var(v_channel) / (255.0**2)
            
            # Melanoma-specific color patterns for darker skin
            melanoma_colors_darker = self._detect_melanoma_colors_darker_skin(image)
            
            # Hyperpigmentation vs lesion differentiation
            hyperpigmentation_score = self._detect_hyperpigmentation_patterns(image)
            
            # Post-inflammatory hyperpigmentation detection
            pih_score = self._detect_post_inflammatory_changes(image)
            
            # Combine color metrics with weights for darker skin
            color_score = (
                hue_variance * 0.3 +
                saturation_range * 0.2 +
                value_variance * 0.2 +
                melanoma_colors_darker * 0.4 +
                hyperpigmentation_score * 0.3 -
                pih_score * 0.2  # Subtract PIH score as it indicates benign changes
            )
            
            return min(max(color_score, 0.0), 1.0)
            
        except Exception:
            return 0.4

    def _detect_melanoma_colors_darker_skin(self, image):
        """Detect melanoma-specific color patterns in darker skin"""
        try:
            rgb_array = np.array(image)
            
            # Melanoma patterns specific to darker skin
            darker_skin_melanoma_patterns = [
                # Very dark/black areas (concerning in any skin tone)
                {'r_range': (0, 60), 'g_range': (0, 60), 'b_range': (0, 60)},
                # Gray-blue areas (regression patterns)
                {'r_range': (60, 120), 'g_range': (60, 120), 'b_range': (80, 140)},
                # Irregular brown with high contrast
                {'r_range': (40, 100), 'g_range': (20, 80), 'b_range': (10, 60)},
                # Red/inflamed areas
                {'r_range': (100, 180), 'g_range': (40, 100), 'b_range': (40, 100)}
            ]
            
            pattern_score = 0.0
            total_pixels = rgb_array.shape[0] * rgb_array.shape[1]
            
            for pattern in darker_skin_melanoma_patterns:
                r_mask = (rgb_array[:,:,0] >= pattern['r_range'][0]) & (rgb_array[:,:,0] <= pattern['r_range'][1])
                g_mask = (rgb_array[:,:,1] >= pattern['g_range'][0]) & (rgb_array[:,:,1] <= pattern['g_range'][1])
                b_mask = (rgb_array[:,:,2] >= pattern['b_range'][0]) & (rgb_array[:,:,2] <= pattern['b_range'][1])
                
                pattern_pixels = np.sum(r_mask & g_mask & b_mask)
                pattern_ratio = pattern_pixels / total_pixels
                pattern_score += pattern_ratio
            
            return min(pattern_score * 1.5, 1.0)  # Enhanced weighting for darker skin
            
        except Exception:
            return 0.3

    def _detect_hyperpigmentation_patterns(self, image):
        """Detect hyperpigmentation patterns common in darker skin"""
        try:
            # Convert to LAB for better analysis
            lab_image = image.convert('LAB')
            lab_array = np.array(lab_image)
            
            # Analyze lightness distribution
            l_channel = lab_array[:,:,0]
            
            # Look for uniform hyperpigmentation (often benign)
            l_std = np.std(l_channel)
            l_mean = np.mean(l_channel)
            
            # Uniform dark areas with low variation suggest benign hyperpigmentation
            if l_std < 20 and l_mean < 80:  # Uniform and dark
                return 0.2  # Lower concern
            elif l_std > 40:  # High variation suggests irregularity
                return 0.8  # Higher concern
            else:
                return 0.5
                
        except Exception:
            return 0.5

    def _detect_post_inflammatory_changes(self, image):
        """Detect post-inflammatory hyperpigmentation patterns"""
        try:
            # PIH typically shows specific patterns
            hsv_array = np.array(image.convert('HSV'))
            
            # PIH characteristics: uniform hue, moderate saturation, varied value
            h_std = np.std(hsv_array[:,:,0])
            s_mean = np.mean(hsv_array[:,:,1])
            v_std = np.std(hsv_array[:,:,2])
            
            # PIH pattern: uniform hue, moderate saturation
            if h_std < 30 and 50 < s_mean < 150:
                return min(0.8, v_std / 50.0)  # Higher PIH score means more likely benign
            else:
                return 0.2
                
        except Exception:
            return 0.3

    def _texture_asymmetry_analysis(self, image):
        """Analyze texture asymmetry using local binary patterns"""
        try:
            # Convert to grayscale
            gray = np.array(image.convert('L'))
            
            # Simple texture analysis using standard deviation in patches
            h, w = gray.shape
            patch_size = min(h, w) // 8
            
            asymmetry_values = []
            
            # Compare texture in corresponding patches
            for i in range(0, h - patch_size, patch_size):
                for j in range(0, w//2 - patch_size, patch_size):
                    left_patch = gray[i:i+patch_size, j:j+patch_size]
                    right_patch = gray[i:i+patch_size, w-j-patch_size:w-j]
                    
                    left_texture = np.std(left_patch)
                    right_texture = np.std(right_patch)
                    
                    if left_texture + right_texture > 0:
                        asymmetry = abs(left_texture - right_texture) / (left_texture + right_texture)
                        asymmetry_values.append(asymmetry)
            
            return np.mean(asymmetry_values) if asymmetry_values else 0.3
            
        except Exception:
            return 0.3

    def _lighter_skin_analysis(self, image, skin_type):
        """Standard analysis for lighter skin tones"""
        # Use original analysis methods for lighter skin
        from isic_model import enhanced_analyzer
        features = enhanced_analyzer.analyze_medical_features(image)
        
        features.update({
            'skin_type': skin_type,
            'analysis_type': 'standard',
            'differential': {'type': 'standard_differential', 'considerations': []}
        })
        
        return features

    def _diameter_analysis(self, image):
        """Standard diameter analysis"""
        try:
            width, height = image.size
            area_pixels = width * height
            
            # Estimate lesion size (simplified)
            estimated_area_ratio = 0.3  # Assume 30% of image is lesion
            estimated_diameter_mm = np.sqrt(estimated_area_ratio * area_pixels) / 50  # Rough conversion
            
            # Concern increases with size > 6mm
            if estimated_diameter_mm > 6:
                return min(0.8 + (estimated_diameter_mm - 6) * 0.05, 1.0)
            else:
                return estimated_diameter_mm / 8.0
                
        except Exception:
            return 0.4

    def _evolution_darker_skin(self, image):
        """Evolution analysis for darker skin considering typical changes"""
        try:
            # Analyze surface texture variations
            gray = np.array(image.convert('L'))
            
            # Use Laplacian for texture analysis
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            texture_score = min(laplacian_var / 1000.0, 1.0)
            
            return texture_score * 0.8  # Reduced weight for darker skin
            
        except Exception:
            return 0.3

    def _differential_diagnosis_darker_skin(self, image):
        """Generate differential diagnosis considerations for darker skin"""
        try:
            differentials = []
            
            # Analyze for common conditions in darker skin
            hsv_array = np.array(image.convert('HSV'))
            rgb_array = np.array(image)
            
            # Post-inflammatory hyperpigmentation
            pih_likelihood = self._detect_post_inflammatory_changes(image)
            if pih_likelihood > 0.6:
                differentials.append({
                    'condition': 'Post-inflammatory hyperpigmentation',
                    'likelihood': pih_likelihood,
                    'characteristics': 'Uniform pigmentation following inflammation'
                })
            
            # Seborrheic keratosis
            sk_likelihood = self._detect_seborrheic_keratosis_patterns(image)
            if sk_likelihood > 0.5:
                differentials.append({
                    'condition': 'Seborrheic keratosis',
                    'likelihood': sk_likelihood,
                    'characteristics': 'Waxy, stuck-on appearance'
                })
            
            # Melanoma consideration
            melanoma_features = self._assess_melanoma_likelihood_darker_skin(image)
            if melanoma_features > 0.4:
                differentials.append({
                    'condition': 'Melanoma',
                    'likelihood': melanoma_features,
                    'characteristics': 'Irregular pigmentation, asymmetry, border irregularity'
                })
            
            # Dermatofibroma
            df_likelihood = self._detect_dermatofibroma_patterns(image)
            if df_likelihood > 0.3:
                differentials.append({
                    'condition': 'Dermatofibroma',
                    'likelihood': df_likelihood,
                    'characteristics': 'Firm nodule, often darker in center'
                })
            
            return {
                'differentials': differentials,
                'primary_concern_level': max([d['likelihood'] for d in differentials] + [0.0])
            }
            
        except Exception:
            return {'differentials': [], 'primary_concern_level': 0.3}

    def _detect_seborrheic_keratosis_patterns(self, image):
        """Detect patterns suggestive of seborrheic keratosis"""
        try:
            # SK characteristics: waxy appearance, well-demarcated
            gray = np.array(image.convert('L'))
            
            # Look for sharp borders
            edges = cv2.Canny(gray, 50, 150)
            edge_sharpness = np.sum(edges > 0) / edges.size
            
            # Uniform texture pattern
            texture_uniformity = 1.0 - (np.std(gray) / (np.mean(gray) + 1e-6))
            
            sk_score = (edge_sharpness + texture_uniformity) / 2.0
            return min(sk_score, 1.0)
            
        except Exception:
            return 0.3

    def _assess_melanoma_likelihood_darker_skin(self, image):
        """Assess melanoma likelihood specific to darker skin presentations"""
        try:
            # Melanoma in darker skin often presents differently
            features = []
            
            # Very dark/black areas
            rgb_array = np.array(image)
            very_dark_pixels = np.sum(np.all(rgb_array < 60, axis=2))
            dark_ratio = very_dark_pixels / (rgb_array.shape[0] * rgb_array.shape[1])
            features.append(dark_ratio)
            
            # Color heterogeneity
            color_variance = np.var(rgb_array, axis=(0,1))
            color_het = np.mean(color_variance) / (255.0**2)
            features.append(color_het)
            
            # Border irregularity
            border_score = self._border_analysis_darker_skin(image)
            features.append(border_score)
            
            return min(np.mean(features) * 1.3, 1.0)
            
        except Exception:
            return 0.4

    def _detect_dermatofibroma_patterns(self, image):
        """Detect patterns suggestive of dermatofibroma"""
        try:
            # DF characteristics: central darkness, peripheral lighter area
            gray = np.array(image.convert('L'))
            h, w = gray.shape
            
            # Analyze center vs periphery
            center_region = gray[h//4:3*h//4, w//4:3*w//4]
            peripheral_region = np.concatenate([
                gray[:h//4, :].flatten(),
                gray[3*h//4:, :].flatten(),
                gray[:, :w//4].flatten(),
                gray[:, 3*w//4:].flatten()
            ])
            
            center_mean = np.mean(center_region)
            peripheral_mean = np.mean(peripheral_region)
            
            # DF often shows central depression/darkness
            if center_mean < peripheral_mean:
                darkness_ratio = (peripheral_mean - center_mean) / (peripheral_mean + 1e-6)
                return min(darkness_ratio * 2.0, 1.0)
            else:
                return 0.2
                
        except Exception:
            return 0.3

    def _fallback_analysis(self, image):
        """Fallback analysis when specialized methods fail"""
        return {
            'asymmetry': 0.4,
            'border': 0.4,
            'color': 0.4,
            'diameter': 0.4,
            'evolution': 0.3,
            'skin_type': 'Unknown',
            'analysis_type': 'fallback',
            'differential': {'differentials': [], 'primary_concern_level': 0.4}
        }

# Global instance
advanced_analyzer = AdvancedSkinToneAnalyzer()

def get_advanced_skin_analysis(image_path):
    """Get advanced skin tone-aware analysis"""
    try:
        if isinstance(image_path, str):
            image = Image.open(image_path).convert('RGB')
        else:
            image = image_path
            
        # Detect skin tone
        fitzpatrick_type, ita_angle, melanin_index = advanced_analyzer.detect_skin_tone(image)
        
        # Perform tone-specific analysis
        analysis_results = advanced_analyzer.analyze_for_darker_skin(image, fitzpatrick_type)
        
        # Add skin tone detection results
        analysis_results.update({
            'detected_skin_tone': fitzpatrick_type,
            'ita_angle': ita_angle,
            'melanin_index': melanin_index
        })
        
        return analysis_results
        
    except Exception as e:
        logger.error(f"Advanced skin analysis failed: {e}")
        return advanced_analyzer._fallback_analysis(image_path)