"""
AI Medical Chatbot for Skin Lesion Analysis Explanations
Provides detailed medical justifications using ABCDE criteria and differential diagnoses
"""

import logging
from typing import Dict, List, Any
import json

logger = logging.getLogger(__name__)

class MedicalExplanationChatbot:
    """AI chatbot that explains medical analysis and provides differential diagnoses"""
    
    def __init__(self):
        self.abcde_criteria = {
            'A': {
                'name': 'Asymmetry',
                'description': 'One half of the lesion does not match the other half',
                'normal_range': (0.0, 0.3),
                'concerning_range': (0.6, 1.0),
                'clinical_significance': 'Asymmetry suggests uncontrolled, irregular growth patterns typical of malignant lesions'
            },
            'B': {
                'name': 'Border Irregularity',
                'description': 'Edges are ragged, notched, or blurred',
                'normal_range': (0.0, 0.4),
                'concerning_range': (0.7, 1.0),
                'clinical_significance': 'Irregular borders indicate invasive growth patterns and loss of growth control'
            },
            'C': {
                'name': 'Color Variation',
                'description': 'Multiple colors or uneven distribution of color',
                'normal_range': (0.0, 0.3),
                'concerning_range': (0.5, 1.0),
                'clinical_significance': 'Color variation reflects different cell populations and varying levels of melanin production'
            },
            'D': {
                'name': 'Diameter',
                'description': 'Size greater than 6mm (about the size of a pencil eraser)',
                'normal_range': (0.0, 0.4),
                'concerning_range': (0.8, 1.0),
                'clinical_significance': 'Larger lesions have higher malignant potential, though small melanomas can occur'
            },
            'E': {
                'name': 'Evolution',
                'description': 'Changes in size, shape, color, or texture over time',
                'normal_range': (0.0, 0.2),
                'concerning_range': (0.4, 1.0),
                'clinical_significance': 'Evolution indicates dynamic cellular activity and potential malignant transformation'
            }
        }
        
        self.differential_diagnoses = {
            'melanoma': {
                'description': 'Malignant melanoma - most serious form of skin cancer',
                'key_features': ['asymmetry', 'irregular borders', 'color variation', 'diameter >6mm', 'evolution'],
                'risk_factors': ['fair skin', 'sun exposure', 'family history', 'multiple moles', 'immunosuppression'],
                'urgency': 'URGENT - requires immediate dermatological evaluation',
                'prognosis': 'Excellent if caught early, poor if metastatic'
            },
            'basal_cell_carcinoma': {
                'description': 'Most common skin cancer, rarely metastasizes',
                'key_features': ['pearly appearance', 'rolled borders', 'central ulceration', 'slow growth'],
                'risk_factors': ['sun exposure', 'fair skin', 'older age'],
                'urgency': 'Semi-urgent - should be evaluated within weeks',
                'prognosis': 'Excellent with treatment, local invasion if untreated'
            },
            'seborrheic_keratosis': {
                'description': 'Benign, waxy, "stuck-on" appearing lesion',
                'key_features': ['waxy texture', 'well-demarcated borders', 'uniform color', 'stuck-on appearance'],
                'risk_factors': ['age', 'genetics', 'sun exposure'],
                'urgency': 'Routine monitoring - cosmetic concerns mainly',
                'prognosis': 'Benign, no malignant potential'
            },
            'dysplastic_nevus': {
                'description': 'Atypical mole with some irregular features',
                'key_features': ['asymmetry', 'irregular borders', 'color variation', 'diameter >5mm'],
                'risk_factors': ['genetics', 'sun exposure', 'fair skin'],
                'urgency': 'Monitor closely - increased melanoma risk',
                'prognosis': 'Benign but marker of increased melanoma risk'
            },
            'post_inflammatory_hyperpigmentation': {
                'description': 'Darkening of skin following inflammation, common in darker skin tones',
                'key_features': ['uniform pigmentation', 'well-defined borders', 'history of trauma/inflammation'],
                'risk_factors': ['darker skin types', 'acne', 'trauma', 'eczema'],
                'urgency': 'Routine - mainly cosmetic concern',
                'prognosis': 'Benign, may fade over time'
            }
        }
        
        self.skin_tone_considerations = {
            'I-II': {
                'description': 'Very fair to fair skin',
                'melanoma_presentation': 'Classic ABCDE criteria highly applicable',
                'common_conditions': ['melanoma', 'basal_cell_carcinoma', 'squamous_cell_carcinoma'],
                'special_considerations': 'High UV sensitivity, classic presentations'
            },
            'III-IV': {
                'description': 'Light to moderate brown skin',
                'melanoma_presentation': 'May present with less obvious color changes',
                'common_conditions': ['melanoma', 'seborrheic_keratosis', 'solar_lentigo'],
                'special_considerations': 'Intermediate risk, monitor for subtle changes'
            },
            'V-VI': {
                'description': 'Dark brown to black skin',
                'melanoma_presentation': 'Often acral locations, amelanotic variants, delayed diagnosis',
                'common_conditions': ['post_inflammatory_hyperpigmentation', 'dermatosis_papulosa_nigra', 'acral_melanoma'],
                'special_considerations': 'Higher risk of acral melanoma, PIH common, delayed presentation'
            }
        }

    def generate_medical_explanation(self, analysis_results: Dict[str, Any], skin_type: str = 'III') -> Dict[str, Any]:
        """Generate comprehensive medical explanation of the analysis"""
        try:
            explanation = {
                'abcde_analysis': self._explain_abcde_scores(analysis_results),
                'overall_assessment': self._generate_overall_assessment(analysis_results),
                'differential_diagnosis': self._generate_differential_diagnosis(analysis_results),
                'skin_tone_considerations': self._explain_skin_tone_factors(skin_type, analysis_results),
                'recommendations': self._generate_recommendations(analysis_results),
                'educational_content': self._provide_educational_content(analysis_results),
                'follow_up_guidance': self._provide_follow_up_guidance(analysis_results)
            }
            
            return explanation
            
        except Exception as e:
            logger.error(f"Medical explanation generation failed: {e}")
            return self._generate_fallback_explanation()

    def _explain_abcde_scores(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Explain each ABCDE criterion score in medical terms"""
        abcde_explanation = {}
        
        for criterion in ['asymmetry', 'border', 'color', 'diameter', 'evolution']:
            if criterion in analysis_results:
                score = analysis_results[criterion]
                criterion_key = criterion[0].upper()
                criterion_info = self.abcde_criteria[criterion_key]
                
                # Determine severity level
                if score <= criterion_info['normal_range'][1]:
                    severity = 'Normal'
                    concern_level = 'Low'
                elif score >= criterion_info['concerning_range'][0]:
                    severity = 'Concerning'
                    concern_level = 'High'
                else:
                    severity = 'Borderline'
                    concern_level = 'Moderate'
                
                abcde_explanation[criterion] = {
                    'score': score,
                    'severity': severity,
                    'concern_level': concern_level,
                    'explanation': self._generate_criterion_explanation(criterion, score, severity),
                    'clinical_significance': criterion_info['clinical_significance']
                }
        
        return abcde_explanation

    def _generate_criterion_explanation(self, criterion: str, score: float, severity: str) -> str:
        """Generate detailed explanation for each criterion"""
        explanations = {
            'asymmetry': {
                'Normal': f"Asymmetry score of {score:.2f} indicates the lesion is relatively symmetric. Benign lesions are typically symmetric when divided along any axis.",
                'Borderline': f"Asymmetry score of {score:.2f} shows some irregularity. While not highly concerning, this warrants monitoring for changes.",
                'Concerning': f"Asymmetry score of {score:.2f} indicates significant asymmetry. Malignant lesions often show marked asymmetry due to uncontrolled growth patterns."
            },
            'border': {
                'Normal': f"Border score of {score:.2f} indicates well-defined, regular borders. Benign lesions typically have smooth, even borders.",
                'Borderline': f"Border score of {score:.2f} shows some irregularity. The borders may be slightly uneven but not dramatically irregular.",
                'Concerning': f"Border score of {score:.2f} indicates irregular, poorly defined borders. This suggests invasive growth patterns typical of malignant lesions."
            },
            'color': {
                'Normal': f"Color score of {score:.2f} indicates uniform coloration. Benign lesions typically show consistent color throughout.",
                'Borderline': f"Color score of {score:.2f} shows some color variation. There may be slight differences in pigmentation within the lesion.",
                'Concerning': f"Color score of {score:.2f} indicates significant color variation. Multiple colors within a lesion suggest different cell populations and varying melanin production."
            },
            'diameter': {
                'Normal': f"Diameter score of {score:.2f} indicates a smaller lesion. While size alone doesn't determine malignancy, smaller lesions are statistically less likely to be melanoma.",
                'Borderline': f"Diameter score of {score:.2f} indicates moderate size. The lesion may be approaching the 6mm threshold that increases concern.",
                'Concerning': f"Diameter score of {score:.2f} indicates a larger lesion. Lesions larger than 6mm have higher statistical risk of malignancy."
            },
            'evolution': {
                'Normal': f"Evolution score of {score:.2f} suggests stable characteristics. Stable lesions are less concerning than rapidly changing ones.",
                'Borderline': f"Evolution score of {score:.2f} indicates some textural or surface changes. Monitor for continued evolution.",
                'Concerning': f"Evolution score of {score:.2f} suggests significant changes. Rapidly evolving lesions require immediate attention as this is a key warning sign."
            }
        }
        
        return explanations.get(criterion, {}).get(severity, f"Score of {score:.2f} for {criterion}")

    def _generate_overall_assessment(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall medical assessment"""
        
        # Count concerning features
        concerning_features = 0
        total_score = 0
        feature_count = 0
        
        for criterion in ['asymmetry', 'border', 'color', 'diameter']:
            if criterion in analysis_results:
                score = analysis_results[criterion]
                total_score += score
                feature_count += 1
                
                criterion_key = criterion[0].upper()
                if score >= self.abcde_criteria[criterion_key]['concerning_range'][0]:
                    concerning_features += 1
        
        average_score = total_score / feature_count if feature_count > 0 else 0.5
        
        # Determine risk level
        if concerning_features >= 3:
            risk_level = 'High'
            urgency = 'Urgent evaluation recommended'
        elif concerning_features >= 2:
            risk_level = 'Moderate-High'
            urgency = 'Prompt evaluation recommended'
        elif concerning_features >= 1:
            risk_level = 'Moderate'
            urgency = 'Routine evaluation recommended'
        else:
            risk_level = 'Low'
            urgency = 'Routine monitoring appropriate'
        
        return {
            'risk_level': risk_level,
            'concerning_features_count': concerning_features,
            'average_score': average_score,
            'urgency': urgency,
            'summary': self._generate_assessment_summary(concerning_features, risk_level)
        }

    def _generate_assessment_summary(self, concerning_features: int, risk_level: str) -> str:
        """Generate summary of assessment"""
        if concerning_features >= 3:
            return "Multiple concerning features present. This lesion demonstrates several characteristics associated with malignant transformation and requires urgent dermatological evaluation."
        elif concerning_features >= 2:
            return "Some concerning features identified. While not definitively malignant, the presence of multiple irregular features warrants prompt professional assessment."
        elif concerning_features >= 1:
            return "At least one concerning feature noted. While many benign lesions can show irregular features, monitoring and routine evaluation are recommended."
        else:
            return "Features appear within normal limits. This analysis suggests characteristics more consistent with benign lesions, though routine monitoring remains important."

    def _generate_differential_diagnosis(self, analysis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate differential diagnosis based on analysis"""
        differentials = []
        
        # Check if differential analysis was performed
        if 'differential' in analysis_results and analysis_results['differential'].get('differentials'):
            # Use advanced differential analysis if available
            for diff in analysis_results['differential']['differentials']:
                condition_key = diff['condition'].lower().replace(' ', '_').replace('-', '_')
                if condition_key in self.differential_diagnoses:
                    condition_info = self.differential_diagnoses[condition_key]
                    differentials.append({
                        'condition': diff['condition'],
                        'likelihood': diff['likelihood'],
                        'description': condition_info['description'],
                        'key_features': condition_info['key_features'],
                        'urgency': condition_info['urgency'],
                        'characteristics_found': diff.get('characteristics', '')
                    })
        else:
            # Generate basic differential based on ABCDE scores
            differentials = self._generate_basic_differential(analysis_results)
        
        # Sort by likelihood
        differentials.sort(key=lambda x: x['likelihood'], reverse=True)
        
        return differentials[:5]  # Return top 5 differentials

    def _generate_basic_differential(self, analysis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate basic differential diagnosis from ABCDE scores"""
        differentials = []
        
        # Calculate melanoma likelihood
        melanoma_features = 0
        for criterion in ['asymmetry', 'border', 'color', 'diameter']:
            if criterion in analysis_results:
                score = analysis_results[criterion]
                criterion_key = criterion[0].upper()
                if score >= self.abcde_criteria[criterion_key]['concerning_range'][0]:
                    melanoma_features += 1
        
        melanoma_likelihood = min(melanoma_features / 4.0 * 0.8 + 0.2, 1.0)
        
        if melanoma_likelihood > 0.4:
            differentials.append({
                'condition': 'Melanoma',
                'likelihood': melanoma_likelihood,
                'description': self.differential_diagnoses['melanoma']['description'],
                'key_features': self.differential_diagnoses['melanoma']['key_features'],
                'urgency': self.differential_diagnoses['melanoma']['urgency'],
                'characteristics_found': f"ABCDE analysis shows {melanoma_features} concerning features"
            })
        
        # Add other common differentials based on pattern
        if analysis_results.get('border', 0) > 0.6 and analysis_results.get('color', 0) < 0.4:
            differentials.append({
                'condition': 'Seborrheic Keratosis',
                'likelihood': 0.6,
                'description': self.differential_diagnoses['seborrheic_keratosis']['description'],
                'key_features': self.differential_diagnoses['seborrheic_keratosis']['key_features'],
                'urgency': self.differential_diagnoses['seborrheic_keratosis']['urgency'],
                'characteristics_found': 'Well-defined borders with uniform coloration'
            })
        
        return differentials

    def _explain_skin_tone_factors(self, skin_type: str, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Explain how skin tone affects analysis and interpretation"""
        
        # Determine skin tone category
        if skin_type in ['I', 'II']:
            tone_category = 'I-II'
        elif skin_type in ['III', 'IV']:
            tone_category = 'III-IV'
        else:
            tone_category = 'V-VI'
        
        tone_info = self.skin_tone_considerations[tone_category]
        
        explanations = {
            'skin_type_detected': skin_type,
            'category': tone_category,
            'description': tone_info['description'],
            'melanoma_presentation': tone_info['melanoma_presentation'],
            'common_conditions': tone_info['common_conditions'],
            'analysis_adjustments': self._explain_analysis_adjustments(skin_type, analysis_results),
            'special_considerations': tone_info['special_considerations']
        }
        
        return explanations

    def _explain_analysis_adjustments(self, skin_type: str, analysis_results: Dict[str, Any]) -> str:
        """Explain how analysis was adjusted for skin tone"""
        if skin_type in ['V', 'VI']:
            return ("Analysis adjusted for darker skin tone: Enhanced contrast techniques used for border detection, "
                   "specialized color analysis for hyperpigmentation patterns, and consideration of post-inflammatory "
                   "changes. Color variation criteria weighted more heavily as melanomas in darker skin often show "
                   "different pigmentation patterns.")
        elif skin_type in ['III', 'IV']:
            return ("Analysis used balanced approach for medium skin tone: Standard ABCDE criteria applied with "
                   "moderate adjustments for pigmentation differences. Monitoring for both typical and atypical "
                   "presentations recommended.")
        else:
            return ("Analysis optimized for lighter skin tone: Standard ABCDE criteria highly applicable. "
                   "Classic melanoma presentations expected with clear color and border distinctions.")

    def _generate_recommendations(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Generate specific medical recommendations"""
        recommendations = []
        
        # Get overall risk assessment
        concerning_features = sum(1 for criterion in ['asymmetry', 'border', 'color', 'diameter']
                                if analysis_results.get(criterion, 0) >= 0.6)
        
        if concerning_features >= 3:
            recommendations.extend([
                "URGENT: Schedule dermatological evaluation within 1-2 weeks",
                "Consider dermoscopy or dermatoscopic evaluation",
                "Document lesion with high-quality photographs",
                "Avoid sun exposure to the lesion area",
                "Do not attempt self-treatment or removal"
            ])
        elif concerning_features >= 2:
            recommendations.extend([
                "Schedule dermatological evaluation within 2-4 weeks",
                "Monitor for any changes in size, color, or texture",
                "Photograph lesion for comparison monitoring",
                "Use sun protection measures",
                "Consider total body skin examination"
            ])
        else:
            recommendations.extend([
                "Routine dermatological monitoring recommended",
                "Annual skin examinations appropriate",
                "Monthly self-examinations of all moles",
                "Use broad-spectrum sunscreen daily",
                "Document any changes and report to healthcare provider"
            ])
        
        return recommendations

    def _provide_educational_content(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Provide educational content about skin cancer and prevention"""
        return {
            'abcde_overview': ("The ABCDE criteria are the gold standard for melanoma screening. "
                              "Each letter represents a key feature that dermatologists look for when "
                              "evaluating suspicious lesions."),
            'early_detection_importance': ("Early detection of melanoma dramatically improves outcomes. "
                                         "When caught early, melanoma has a 5-year survival rate exceeding 95%. "
                                         "However, once metastasized, survival rates drop significantly."),
            'prevention_tips': [
                "Use broad-spectrum SPF 30+ sunscreen daily",
                "Seek shade during peak UV hours (10 AM - 4 PM)",
                "Wear protective clothing and wide-brimmed hats",
                "Avoid tanning beds and excessive sun exposure",
                "Perform monthly self-examinations",
                "Schedule annual dermatological screenings"
            ],
            'when_to_see_doctor': [
                "Any new mole or lesion appearing after age 30",
                "Changes in existing moles (size, color, texture)",
                "Bleeding, itching, or tenderness in a mole",
                "Asymmetry, irregular borders, or multiple colors",
                "Lesions larger than 6mm in diameter"
            ]
        }

    def _provide_follow_up_guidance(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Provide specific follow-up guidance"""
        
        # Determine follow-up timeline based on risk
        concerning_features = sum(1 for criterion in ['asymmetry', 'border', 'color', 'diameter']
                                if analysis_results.get(criterion, 0) >= 0.6)
        
        if concerning_features >= 3:
            timeline = "1-2 weeks"
            urgency = "Urgent"
        elif concerning_features >= 2:
            timeline = "2-4 weeks"
            urgency = "Prompt"
        else:
            timeline = "3-6 months"
            urgency = "Routine"
        
        return {
            'timeline': timeline,
            'urgency': urgency,
            'monitoring_instructions': [
                "Take monthly photographs from the same angle and lighting",
                "Measure lesion dimensions if safely accessible",
                "Note any changes in color, texture, or symptoms",
                "Keep a lesion diary with dates and observations"
            ],
            'red_flags': [
                "Rapid increase in size",
                "Development of bleeding or ulceration",
                "New symptoms (pain, itching, tenderness)",
                "Satellite lesions appearing nearby",
                "Changes in surface texture or elevation"
            ],
            'documentation_tips': [
                "Use consistent lighting and camera settings",
                "Include a ruler or coin for size reference",
                "Take both close-up and contextual photos",
                "Record date and any relevant symptoms"
            ]
        }

    def _generate_fallback_explanation(self) -> Dict[str, Any]:
        """Generate fallback explanation when analysis fails"""
        return {
            'message': 'Unable to generate detailed medical explanation',
            'general_advice': [
                'Consult with a dermatologist for proper evaluation',
                'Monitor the lesion for any changes',
                'Practice sun safety measures',
                'Perform regular self-examinations'
            ],
            'disclaimer': 'This analysis is not a substitute for professional medical evaluation'
        }

    def generate_conversational_response(self, question: str, analysis_results: Dict[str, Any]) -> str:
        """Generate conversational response to user questions"""
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['abcd', 'abcde', 'criteria', 'features']):
            return self._explain_abcde_conversational(analysis_results)
        elif any(word in question_lower for word in ['differential', 'diagnosis', 'what could', 'possibilities']):
            return self._explain_differential_conversational(analysis_results)
        elif any(word in question_lower for word in ['skin tone', 'skin type', 'darker skin', 'lighter skin']):
            return self._explain_skin_tone_conversational(analysis_results)
        elif any(word in question_lower for word in ['urgency', 'urgent', 'when', 'how soon']):
            return self._explain_urgency_conversational(analysis_results)
        elif any(word in question_lower for word in ['next steps', 'follow up', 'what should', 'recommendations']):
            return self._explain_next_steps_conversational(analysis_results)
        else:
            return self._general_explanation_conversational(analysis_results)

    def _explain_abcde_conversational(self, analysis_results: Dict[str, Any]) -> str:
        """Conversational explanation of ABCDE criteria"""
        explanation = "The ABCDE criteria help evaluate skin lesions systematically:\n\n"
        
        for criterion in ['asymmetry', 'border', 'color', 'diameter']:
            if criterion in analysis_results:
                score = analysis_results[criterion]
                criterion_name = self.abcde_criteria[criterion[0].upper()]['name']
                
                if score > 0.6:
                    explanation += f"• **{criterion_name}**: Score {score:.2f} - This shows concerning features that warrant attention.\n"
                else:
                    explanation += f"• **{criterion_name}**: Score {score:.2f} - This appears within normal limits.\n"
        
        explanation += "\nHigher scores indicate features that are more commonly seen in malignant lesions."
        return explanation

    def _explain_differential_conversational(self, analysis_results: Dict[str, Any]) -> str:
        """Conversational explanation of differential diagnosis"""
        if 'differential' in analysis_results and analysis_results['differential'].get('differentials'):
            explanation = "Based on the analysis, here are the most likely possibilities:\n\n"
            
            for diff in analysis_results['differential']['differentials'][:3]:
                likelihood_percent = diff['likelihood'] * 100
                explanation += f"• **{diff['condition']}** ({likelihood_percent:.0f}% likelihood)\n"
                explanation += f"  {diff.get('characteristics', 'Analysis suggests this possibility')}\n\n"
        else:
            explanation = "The analysis suggests this lesion should be evaluated by a dermatologist to determine the exact nature and rule out any concerning possibilities."
        
        return explanation

    def _explain_skin_tone_conversational(self, analysis_results: Dict[str, Any]) -> str:
        """Conversational explanation of skin tone considerations"""
        detected_type = analysis_results.get('detected_skin_tone', 'III')
        
        if detected_type in ['V', 'VI']:
            return ("Your analysis was optimized for darker skin tones. In darker skin, melanomas can present differently - "
                   "they're more common on palms, soles, and nail beds, and may be less pigmented than expected. "
                   "Post-inflammatory hyperpigmentation is also common and usually benign.")
        elif detected_type in ['I', 'II']:
            return ("Your analysis used standard criteria optimized for lighter skin tones. Fair skin shows higher UV "
                   "sensitivity and classic melanoma presentations with clear color and border changes are more typical.")
        else:
            return ("Your skin tone analysis used balanced criteria. Medium skin tones can show both typical and atypical "
                   "presentations, so monitoring for various types of changes is important.")

    def _explain_urgency_conversational(self, analysis_results: Dict[str, Any]) -> str:
        """Conversational explanation of urgency"""
        concerning_features = sum(1 for criterion in ['asymmetry', 'border', 'color', 'diameter']
                                if analysis_results.get(criterion, 0) >= 0.6)
        
        if concerning_features >= 3:
            return ("This analysis shows multiple concerning features. I recommend scheduling a dermatologist appointment "
                   "within 1-2 weeks. While this doesn't mean it's definitely cancer, the combination of features "
                   "warrants prompt professional evaluation.")
        elif concerning_features >= 2:
            return ("The analysis identified some concerning features. Schedule a dermatologist appointment within 2-4 weeks "
                   "for evaluation. In the meantime, monitor for any changes and protect the area from sun exposure.")
        else:
            return ("The features appear relatively stable, but routine monitoring is still important. Schedule your next "
                   "skin check within 3-6 months, and watch for any changes in the meantime.")

    def _explain_next_steps_conversational(self, analysis_results: Dict[str, Any]) -> str:
        """Conversational explanation of next steps"""
        return ("Here are my recommendations:\n\n"
               "1. **Professional evaluation** - See a dermatologist for definitive assessment\n"
               "2. **Documentation** - Take monthly photos to track any changes\n"
               "3. **Protection** - Use sunscreen and avoid further UV damage\n"
               "4. **Monitoring** - Watch for changes in size, color, or texture\n"
               "5. **Follow-up** - Don't ignore new symptoms like bleeding or itching\n\n"
               "Remember, this analysis is a screening tool - only a dermatologist can provide a definitive diagnosis.")

    def _general_explanation_conversational(self, analysis_results: Dict[str, Any]) -> str:
        """General conversational explanation"""
        return ("This analysis uses medical criteria to evaluate skin lesions systematically. The ABCDE criteria "
               "(Asymmetry, Border, Color, Diameter, Evolution) are the gold standard for melanoma screening. "
               "Higher scores indicate features more commonly seen in concerning lesions. However, only a dermatologist "
               "can provide a definitive diagnosis. The analysis is designed to help identify lesions that warrant "
               "professional evaluation.")

# Global chatbot instance
medical_chatbot = MedicalExplanationChatbot()

def get_medical_explanation(analysis_results: Dict[str, Any], skin_type: str = 'III') -> Dict[str, Any]:
    """Get comprehensive medical explanation"""
    return medical_chatbot.generate_medical_explanation(analysis_results, skin_type)

def chat_with_medical_bot(question: str, analysis_results: Dict[str, Any]) -> str:
    """Chat with medical explanation bot"""
    return medical_chatbot.generate_conversational_response(question, analysis_results)