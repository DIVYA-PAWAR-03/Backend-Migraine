"""
Enhanced Machine Learning Service for Migraine Prediction.

This service provides:
- Symptom-based migraine type classification
- Lifestyle-based migraine risk prediction
- Combined analysis for comprehensive results
- Detailed trigger detection and suggestions
"""

import joblib
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import logging
from datetime import datetime

from ..config import settings
from ..models.schemas import (
    HealthDataInput, 
    PredictionResponse, 
    RiskLevel,
    SymptomInput,
    SymptomClassificationResponse,
    ComprehensiveAnalysisResponse
)

logger = logging.getLogger(__name__)


class EnhancedMLService:
    """Enhanced ML service with symptom classification and risk prediction."""
    
    # Thresholds for trigger detection (based on medical research)
    THRESHOLDS = {
        "stress_high": 7,
        "stress_very_high": 9,
        "sleep_low": 6,
        "sleep_very_low": 4,
        "heart_rate_high": 90,
        "heart_rate_very_high": 100,
        "activity_low": 3,
        "activity_very_low": 2,
        "weather_pressure_low": 1000,
        "weather_pressure_high": 1025,
        "aqi_moderate": 50,
        "aqi_high": 100,
        "aqi_very_high": 150,
    }
    
    # Symptom descriptions for user-friendly output
    SYMPTOM_DESCRIPTIONS = {
        'Nausea': 'Feeling of sickness with urge to vomit',
        'Vomit': 'Actual vomiting episodes',
        'Phonophobia': 'Sensitivity to sound',
        'Photophobia': 'Sensitivity to light',
        'Visual': 'Visual disturbances (flashing lights, blind spots)',
        'Sensory': 'Sensory changes (tingling, numbness)',
        'Vertigo': 'Dizziness or spinning sensation',
        'Tinnitus': 'Ringing in the ears',
        'Dysphasia': 'Difficulty speaking',
        'Paresthesia': 'Abnormal skin sensations',
    }
    
    # Migraine type descriptions and recommendations
    MIGRAINE_TYPE_INFO = {
        'Migraine without aura': {
            'description': 'Common migraine without warning symptoms',
            'recommendations': [
                'Keep a headache diary to identify triggers',
                'Maintain regular sleep schedule',
                'Stay hydrated and avoid skipping meals',
                'Practice stress management techniques'
            ]
        },
        'Typical aura with migraine': {
            'description': 'Migraine preceded by visual or sensory warning signs',
            'recommendations': [
                'When aura starts, take medication immediately if prescribed',
                'Rest in a dark, quiet room during aura phase',
                'Avoid bright lights and loud sounds',
                'Consider preventive medications with your doctor'
            ]
        },
        'Basilar-type aura': {
            'description': 'Aura originating from brainstem, causing dizziness and vision changes',
            'recommendations': [
                'Consult a neurologist for specialized care',
                'Avoid activities that trigger vertigo',
                'Be cautious with certain medications',
                'Emergency awareness for severe symptoms'
            ]
        },
        'Sporadic hemiplegic migraine': {
            'description': 'Migraine with temporary weakness on one side of body',
            'recommendations': [
                'Seek immediate care if symptoms are new',
                'Have a stroke protocol awareness',
                'Regular neurological monitoring',
                'Avoid known triggers strictly'
            ]
        },
        'Familial hemiplegic migraine': {
            'description': 'Inherited form of hemiplegic migraine',
            'recommendations': [
                'Genetic counseling may be helpful',
                'Family members should be aware of symptoms',
                'Specialized treatment plan needed',
                'Regular follow-ups with neurologist'
            ]
        },
        'Typical aura without migraine': {
            'description': 'Visual or sensory aura without subsequent headache',
            'recommendations': [
                'Monitor for changes in pattern',
                'Rule out other neurological conditions',
                'Generally good prognosis',
                'Track frequency and duration'
            ]
        },
        'Other': {
            'description': 'Headache not classified as specific migraine type',
            'recommendations': [
                'Further evaluation may be needed',
                'Consider tension-type headache treatments',
                'Lifestyle modifications can help',
                'Consult if symptoms worsen'
            ]
        }
    }
    
    def __init__(self):
        """Initialize the enhanced ML service."""
        self.symptom_model = None
        self.symptom_scaler = None
        self.label_encoder = None
        self.symptom_features = []
        self.symptom_classes = []
        
        self.risk_model = None
        self.risk_scaler = None
        self.risk_features = []
        
        self.model_info = {}
        
        self._load_models()
    
    def _load_models(self) -> None:
        """Load all trained models from disk."""
        ml_path = Path(settings.MODEL_PATH).parent
        
        # Load symptom classifier
        symptom_path = ml_path / "symptom_classifier.pkl"
        if symptom_path.exists():
            try:
                symptom_data = joblib.load(symptom_path)
                self.symptom_model = symptom_data.get('model')
                self.symptom_scaler = symptom_data.get('scaler')
                self.label_encoder = symptom_data.get('label_encoder')
                self.symptom_features = symptom_data.get('features', [])
                self.symptom_classes = symptom_data.get('classes', [])
                logger.info(f"Loaded symptom classifier: {len(self.symptom_classes)} classes")
            except Exception as e:
                logger.error(f"Error loading symptom classifier: {e}")
        
        # Load risk predictor
        risk_path = Path(settings.MODEL_PATH)
        if risk_path.exists():
            try:
                risk_data = joblib.load(risk_path)
                if isinstance(risk_data, dict):
                    self.risk_model = risk_data.get('model')
                    self.risk_scaler = risk_data.get('scaler')
                    self.risk_features = risk_data.get('features', [
                        'stress_level', 'sleep_hours', 'heart_rate',
                        'activity_level', 'weather_pressure', 'aqi'
                    ])
                    self.model_info = risk_data.get('info', {})
                else:
                    self.risk_model = risk_data
                logger.info(f"Loaded risk predictor: {type(self.risk_model).__name__}")
            except Exception as e:
                logger.error(f"Error loading risk predictor: {e}")
        
        # Load scaler if separate
        scaler_path = Path(settings.SCALER_PATH)
        if scaler_path.exists() and self.risk_scaler is None:
            try:
                self.risk_scaler = joblib.load(scaler_path)
                logger.info("Loaded risk scaler")
            except Exception as e:
                logger.error(f"Error loading scaler: {e}")
    
    def is_symptom_model_ready(self) -> bool:
        """Check if symptom classifier is ready."""
        return self.symptom_model is not None
    
    def is_risk_model_ready(self) -> bool:
        """Check if risk predictor is ready."""
        return self.risk_model is not None
    
    def classify_symptoms(self, symptoms: 'SymptomInput') -> 'SymptomClassificationResponse':
        """
        Classify migraine type based on symptoms.
        
        Args:
            symptoms: SymptomInput with symptom data
            
        Returns:
            SymptomClassificationResponse with classification and recommendations
        """
        if not self.is_symptom_model_ready():
            # Fallback classification based on rules
            return self._rule_based_classification(symptoms)
        
        # Prepare features
        features = self._prepare_symptom_features(symptoms)
        
        # Scale features
        if self.symptom_scaler:
            features = self.symptom_scaler.transform(features)
        
        # Get prediction probabilities
        if hasattr(self.symptom_model, 'predict_proba'):
            probabilities = self.symptom_model.predict_proba(features)[0]
            predicted_class_idx = np.argmax(probabilities)
            confidence = float(probabilities[predicted_class_idx])
            
            # Get top 3 predictions
            top_indices = np.argsort(probabilities)[::-1][:3]
            top_predictions = [
                {
                    'type': self.symptom_classes[i],
                    'probability': float(probabilities[i])
                }
                for i in top_indices
            ]
        else:
            predicted_class_idx = self.symptom_model.predict(features)[0]
            confidence = 0.7
            top_predictions = [{'type': self.symptom_classes[predicted_class_idx], 'probability': confidence}]
        
        # Decode prediction
        predicted_type = self.symptom_classes[predicted_class_idx]
        
        # Get type info
        type_info = self.MIGRAINE_TYPE_INFO.get(predicted_type, self.MIGRAINE_TYPE_INFO['Other'])
        
        # Identify key symptoms
        key_symptoms = self._identify_key_symptoms(symptoms)
        
        return SymptomClassificationResponse(
            migraine_type=predicted_type,
            confidence=confidence,
            description=type_info['description'],
            recommendations=type_info['recommendations'],
            key_symptoms=key_symptoms,
            top_predictions=top_predictions,
            timestamp=datetime.utcnow()
        )
    
    def _prepare_symptom_features(self, symptoms: 'SymptomInput') -> np.ndarray:
        """Prepare feature array from symptom input."""
        feature_values = []
        
        # Match features expected by the model
        feature_mapping = {
            'Age': symptoms.age,
            'Duration': symptoms.duration,
            'Frequency': symptoms.frequency,
            'Location': symptoms.location,
            'Character': symptoms.character,
            'Intensity': symptoms.intensity,
            'Nausea': 1 if symptoms.nausea else 0,
            'Vomit': 1 if symptoms.vomit else 0,
            'Phonophobia': 1 if symptoms.phonophobia else 0,
            'Photophobia': 1 if symptoms.photophobia else 0,
            'Visual': symptoms.visual,
            'Sensory': symptoms.sensory,
            'Dysphasia': 1 if symptoms.dysphasia else 0,
            'Dysarthria': 1 if symptoms.dysarthria else 0,
            'Vertigo': 1 if symptoms.vertigo else 0,
            'Tinnitus': 1 if symptoms.tinnitus else 0,
            'Hypoacusis': 1 if symptoms.hypoacusis else 0,
            'Diplopia': 1 if symptoms.diplopia else 0,
            'Defect': 1 if symptoms.defect else 0,
            'Ataxia': 1 if symptoms.ataxia else 0,
            'Conscience': 1 if symptoms.conscience else 0,
            'Paresthesia': 1 if symptoms.paresthesia else 0,
            'DPF': 1 if symptoms.dpf else 0,
        }
        
        for feature in self.symptom_features:
            feature_values.append(feature_mapping.get(feature, 0))
        
        return np.array([feature_values])
    
    def _rule_based_classification(self, symptoms: 'SymptomInput') -> 'SymptomClassificationResponse':
        """Fallback rule-based classification when ML model unavailable."""
        
        # Check for aura symptoms
        has_visual = symptoms.visual > 0
        has_sensory = symptoms.sensory > 0
        has_aura = has_visual or has_sensory
        
        # Check for brainstem symptoms
        has_brainstem = symptoms.vertigo or symptoms.tinnitus or symptoms.dysarthria
        
        # Check for motor symptoms
        has_motor = symptoms.ataxia or symptoms.paresthesia
        
        # Classification logic
        if has_motor:
            if symptoms.dpf:
                predicted_type = "Familial hemiplegic migraine"
            else:
                predicted_type = "Sporadic hemiplegic migraine"
            confidence = 0.65
        elif has_brainstem and has_aura:
            predicted_type = "Basilar-type aura"
            confidence = 0.7
        elif has_aura and symptoms.intensity >= 2:
            predicted_type = "Typical aura with migraine"
            confidence = 0.75
        elif has_aura and symptoms.intensity < 2:
            predicted_type = "Typical aura without migraine"
            confidence = 0.6
        elif symptoms.nausea and symptoms.phonophobia and symptoms.photophobia:
            predicted_type = "Migraine without aura"
            confidence = 0.8
        else:
            predicted_type = "Other"
            confidence = 0.5
        
        type_info = self.MIGRAINE_TYPE_INFO.get(predicted_type, self.MIGRAINE_TYPE_INFO['Other'])
        key_symptoms = self._identify_key_symptoms(symptoms)
        
        return SymptomClassificationResponse(
            migraine_type=predicted_type,
            confidence=confidence,
            description=type_info['description'],
            recommendations=type_info['recommendations'],
            key_symptoms=key_symptoms,
            top_predictions=[{'type': predicted_type, 'probability': confidence}],
            timestamp=datetime.utcnow()
        )
    
    def _identify_key_symptoms(self, symptoms: 'SymptomInput') -> List[str]:
        """Identify and describe key symptoms present."""
        key_symptoms = []
        
        if symptoms.nausea:
            key_symptoms.append("Nausea present")
        if symptoms.vomit:
            key_symptoms.append("Vomiting episodes")
        if symptoms.phonophobia:
            key_symptoms.append("Sound sensitivity (phonophobia)")
        if symptoms.photophobia:
            key_symptoms.append("Light sensitivity (photophobia)")
        if symptoms.visual > 0:
            key_symptoms.append(f"Visual disturbances (severity: {symptoms.visual})")
        if symptoms.sensory > 0:
            key_symptoms.append(f"Sensory symptoms (severity: {symptoms.sensory})")
        if symptoms.vertigo:
            key_symptoms.append("Vertigo/dizziness")
        if symptoms.tinnitus:
            key_symptoms.append("Tinnitus (ringing in ears)")
        if symptoms.paresthesia:
            key_symptoms.append("Paresthesia (tingling/numbness)")
        
        # Add intensity description
        intensity_desc = {1: "Mild", 2: "Moderate", 3: "Severe"}
        if symptoms.intensity in intensity_desc:
            key_symptoms.insert(0, f"{intensity_desc[symptoms.intensity]} pain intensity")
        
        return key_symptoms
    
    def detect_triggers(self, data: HealthDataInput) -> List[Dict[str, str]]:
        """
        Detect and categorize potential migraine triggers.
        
        Returns list of triggers with severity and recommendations.
        """
        triggers = []
        
        # Stress triggers
        if data.stress_level >= self.THRESHOLDS["stress_very_high"]:
            triggers.append({
                'trigger': f"Very high stress level ({data.stress_level}/10)",
                'severity': 'high',
                'category': 'stress',
                'recommendation': 'Practice deep breathing, take a break, or try relaxation techniques'
            })
        elif data.stress_level >= self.THRESHOLDS["stress_high"]:
            triggers.append({
                'trigger': f"High stress level ({data.stress_level}/10)",
                'severity': 'medium',
                'category': 'stress',
                'recommendation': 'Consider taking short breaks and practicing mindfulness'
            })
        
        # Sleep triggers
        if data.sleep_hours <= self.THRESHOLDS["sleep_very_low"]:
            triggers.append({
                'trigger': f"Very poor sleep ({data.sleep_hours} hours)",
                'severity': 'high',
                'category': 'sleep',
                'recommendation': 'Prioritize rest, avoid screens before bed, consider a nap if possible'
            })
        elif data.sleep_hours <= self.THRESHOLDS["sleep_low"]:
            triggers.append({
                'trigger': f"Insufficient sleep ({data.sleep_hours} hours)",
                'severity': 'medium',
                'category': 'sleep',
                'recommendation': 'Try to get 7-8 hours tonight, maintain consistent sleep schedule'
            })
        
        # Heart rate triggers
        if data.heart_rate >= self.THRESHOLDS["heart_rate_very_high"]:
            triggers.append({
                'trigger': f"Elevated heart rate ({data.heart_rate} bpm)",
                'severity': 'high',
                'category': 'cardiovascular',
                'recommendation': 'Rest, practice calm breathing, avoid caffeine and stimulants'
            })
        elif data.heart_rate >= self.THRESHOLDS["heart_rate_high"]:
            triggers.append({
                'trigger': f"Elevated heart rate ({data.heart_rate} bpm)",
                'severity': 'medium',
                'category': 'cardiovascular',
                'recommendation': 'Monitor your heart rate, reduce physical exertion if needed'
            })
        
        # Activity triggers
        if data.activity_level <= self.THRESHOLDS["activity_very_low"]:
            triggers.append({
                'trigger': f"Very low physical activity ({data.activity_level}/10)",
                'severity': 'medium',
                'category': 'activity',
                'recommendation': 'Light stretching or short walk can help, avoid prolonged sitting'
            })
        elif data.activity_level <= self.THRESHOLDS["activity_low"]:
            triggers.append({
                'trigger': f"Low physical activity ({data.activity_level}/10)",
                'severity': 'low',
                'category': 'activity',
                'recommendation': 'Consider adding light exercise to your routine'
            })
        
        # Weather pressure triggers
        if data.weather_pressure <= self.THRESHOLDS["weather_pressure_low"]:
            triggers.append({
                'trigger': f"Low barometric pressure ({data.weather_pressure} hPa)",
                'severity': 'medium',
                'category': 'environmental',
                'recommendation': 'Stay hydrated, avoid altitude changes, rest if symptoms appear'
            })
        elif data.weather_pressure >= self.THRESHOLDS["weather_pressure_high"]:
            triggers.append({
                'trigger': f"High barometric pressure ({data.weather_pressure} hPa)",
                'severity': 'low',
                'category': 'environmental',
                'recommendation': 'Monitor for symptoms, maintain hydration'
            })
        
        # Air quality triggers
        if data.aqi >= self.THRESHOLDS["aqi_very_high"]:
            triggers.append({
                'trigger': f"Very poor air quality (AQI: {data.aqi})",
                'severity': 'high',
                'category': 'environmental',
                'recommendation': 'Stay indoors, use air purifier, avoid outdoor activities'
            })
        elif data.aqi >= self.THRESHOLDS["aqi_high"]:
            triggers.append({
                'trigger': f"Poor air quality (AQI: {data.aqi})",
                'severity': 'medium',
                'category': 'environmental',
                'recommendation': 'Limit outdoor exposure, keep windows closed'
            })
        elif data.aqi >= self.THRESHOLDS["aqi_moderate"]:
            triggers.append({
                'trigger': f"Moderate air quality (AQI: {data.aqi})",
                'severity': 'low',
                'category': 'environmental',
                'recommendation': 'Be aware of air quality if sensitive'
            })
        
        return triggers
    
    def _calculate_risk_level(self, probability: float, triggers: List[Dict]) -> RiskLevel:
        """Calculate risk level based on probability and triggers."""
        # Count high severity triggers
        high_triggers = sum(1 for t in triggers if t.get('severity') == 'high')
        medium_triggers = sum(1 for t in triggers if t.get('severity') == 'medium')
        low_triggers = sum(1 for t in triggers if t.get('severity') == 'low')
        
        # Calculate trigger-based score (0 to 1)
        trigger_score = (high_triggers * 0.15) + (medium_triggers * 0.08) + (low_triggers * 0.03)
        
        # Weighted combination of model probability and trigger score
        # Model probability has 60% weight, triggers have 40% weight
        adjusted_prob = (probability * 0.6) + (trigger_score * 0.4)
        adjusted_prob = min(1.0, max(0.0, adjusted_prob))
        
        # Clear thresholds for risk levels
        if adjusted_prob >= 0.60 or high_triggers >= 2:
            return RiskLevel.HIGH
        elif adjusted_prob >= 0.30 or high_triggers >= 1 or medium_triggers >= 2:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    def predict_risk(self, data: HealthDataInput) -> PredictionResponse:
        """
        Predict migraine risk based on lifestyle factors.
        
        Args:
            data: HealthDataInput with current health metrics
            
        Returns:
            PredictionResponse with comprehensive risk assessment
        """
        # Detect triggers first
        triggers_detailed = self.detect_triggers(data)
        trigger_descriptions = [t['trigger'] for t in triggers_detailed]
        
        # Calculate risk score based on input values directly
        # This ensures predictions vary properly based on inputs
        risk_score = self._calculate_input_based_risk(data)
        
        # Count triggers
        high_count = sum(1 for t in triggers_detailed if t.get('severity') == 'high')
        medium_count = sum(1 for t in triggers_detailed if t.get('severity') == 'medium')
        low_count = sum(1 for t in triggers_detailed if t.get('severity') == 'low')
        
        # Combine input-based score with trigger score
        trigger_score = (high_count * 0.12) + (medium_count * 0.06) + (low_count * 0.02)
        
        # Final probability combines both scores
        probability = (risk_score * 0.7) + (trigger_score * 0.3)
        probability = min(0.95, max(0.05, probability))
        
        # Determine risk level from probability
        if probability >= 0.55:
            risk_level = RiskLevel.HIGH
        elif probability >= 0.30:
            risk_level = RiskLevel.MEDIUM
        else:
            risk_level = RiskLevel.LOW
        
        # Calculate confidence based on how clear-cut the prediction is
        if probability < 0.20 or probability > 0.75:
            confidence = 0.85
        elif probability < 0.35 or probability > 0.55:
            confidence = 0.7
        else:
            confidence = 0.55
        
        return PredictionResponse(
            risk_level=risk_level,
            probability=round(probability, 3),
            confidence=round(confidence, 3),
            triggers=trigger_descriptions,
            prediction_window="24-48 hours"
        )
    
    def _calculate_input_based_risk(self, data: HealthDataInput) -> float:
        """
        Calculate risk score directly from input values.
        
        This provides more predictable and balanced predictions
        based on the actual input values.
        """
        score = 0.0
        
        # Stress contribution (1-10 scale) -> 0 to 0.25
        # Low stress (1-3) = low risk, High stress (8-10) = high risk
        stress_norm = (data.stress_level - 1) / 9.0  # 0 to 1
        score += stress_norm * 0.25
        
        # Sleep contribution (inverse) -> 0 to 0.25
        # Good sleep (7-9 hrs) = low risk, Poor sleep (<5 hrs) = high risk
        if data.sleep_hours >= 7:
            sleep_risk = 0.0
        elif data.sleep_hours >= 5:
            sleep_risk = (7 - data.sleep_hours) / 2.0  # 0 to 1
        else:
            sleep_risk = 1.0
        score += sleep_risk * 0.25
        
        # Heart rate contribution -> 0 to 0.15
        # Normal (60-80) = low risk, High (>95) = high risk
        if data.heart_rate <= 80:
            hr_risk = 0.0
        elif data.heart_rate <= 95:
            hr_risk = (data.heart_rate - 80) / 15.0  # 0 to 1
        else:
            hr_risk = 1.0
        score += hr_risk * 0.15
        
        # Activity contribution (inverse) -> 0 to 0.10
        # Active (7-10) = low risk, Sedentary (1-3) = medium risk
        activity_norm = (10 - data.activity_level) / 9.0  # 0 to 1
        score += activity_norm * 0.10
        
        # Weather pressure contribution -> 0 to 0.10
        # Normal (1010-1020) = low risk, Extreme = higher risk
        pressure_deviation = abs(data.weather_pressure - 1013) / 25.0
        pressure_risk = min(1.0, pressure_deviation)
        score += pressure_risk * 0.10
        
        # AQI contribution -> 0 to 0.15
        # Good (<50) = low risk, Unhealthy (>100) = high risk
        if data.aqi <= 50:
            aqi_risk = 0.0
        elif data.aqi <= 100:
            aqi_risk = (data.aqi - 50) / 50.0  # 0 to 1
        else:
            aqi_risk = 1.0
        score += aqi_risk * 0.15
        
        return min(1.0, max(0.0, score))

    def get_comprehensive_analysis(
        self, 
        health_data: HealthDataInput, 
        symptoms: Optional['SymptomInput'] = None
    ) -> 'ComprehensiveAnalysisResponse':
        """
        Get comprehensive analysis combining symptom classification and risk prediction.
        
        Args:
            health_data: Current lifestyle/health metrics
            symptoms: Optional symptom data for classification
            
        Returns:
            ComprehensiveAnalysisResponse with full analysis
        """
        # Get risk prediction
        risk_prediction = self.predict_risk(health_data)
        
        # Get symptom classification if symptoms provided
        symptom_result = None
        if symptoms:
            symptom_result = self.classify_symptoms(symptoms)
        
        # Get detailed triggers
        triggers_detailed = self.detect_triggers(health_data)
        
        # Generate combined recommendations
        recommendations = self._generate_recommendations(
            risk_prediction, symptom_result, triggers_detailed
        )
        
        return ComprehensiveAnalysisResponse(
            risk_prediction=risk_prediction,
            symptom_classification=symptom_result,
            detailed_triggers=triggers_detailed,
            recommendations=recommendations,
            analysis_timestamp=datetime.utcnow()
        )
    
    def _generate_recommendations(
        self, 
        risk: PredictionResponse, 
        symptoms: Optional['SymptomClassificationResponse'],
        triggers: List[Dict]
    ) -> List[str]:
        """Generate personalized recommendations based on all analyses."""
        recommendations = []
        
        # Add trigger-specific recommendations
        for trigger in triggers[:3]:  # Top 3 triggers
            if trigger.get('recommendation'):
                recommendations.append(trigger['recommendation'])
        
        # Add symptom-based recommendations
        if symptoms and symptoms.recommendations:
            recommendations.extend(symptoms.recommendations[:2])
        
        # Add risk-level based recommendations
        if risk.risk_level == RiskLevel.HIGH:
            recommendations.insert(0, "⚠️ High risk detected - consider taking preventive medication if prescribed")
            recommendations.append("Avoid known personal triggers today")
        elif risk.risk_level == RiskLevel.MEDIUM:
            recommendations.insert(0, "Moderate risk - be proactive with prevention measures")
        
        # Deduplicate while preserving order
        seen = set()
        unique_recommendations = []
        for rec in recommendations:
            if rec not in seen:
                seen.add(rec)
                unique_recommendations.append(rec)
        
        return unique_recommendations[:6]  # Return max 6 recommendations
    
    def get_model_info(self) -> Dict:
        """Get information about loaded models."""
        return {
            'symptom_classifier': {
                'loaded': self.is_symptom_model_ready(),
                'type': type(self.symptom_model).__name__ if self.symptom_model else None,
                'classes': self.symptom_classes
            },
            'risk_predictor': {
                'loaded': self.is_risk_model_ready(),
                'type': type(self.risk_model).__name__ if self.risk_model else None,
                'info': self.model_info
            }
        }


# Singleton instance
enhanced_ml_service = EnhancedMLService()
