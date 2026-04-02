"""
Machine Learning Service for Migraine Prediction.

This service handles:
- Loading trained ML model
- Making predictions
- Detecting triggers
- Calculating risk levels
"""

import joblib
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import logging

from ..config import settings
from ..models.schemas import HealthDataInput, PredictionResponse, RiskLevel

logger = logging.getLogger(__name__)


class MLService:
    """Service class for ML-based migraine prediction."""
    
    # Thresholds for trigger detection
    THRESHOLDS = {
        "stress_high": 7,
        "sleep_low": 6,
        "sleep_very_low": 4,
        "heart_rate_high": 90,
        "heart_rate_very_high": 100,
        "activity_low": 3,
        "weather_pressure_low": 1000,
        "weather_pressure_high": 1025,
        "aqi_moderate": 50,
        "aqi_high": 100,
    }
    
    def __init__(self):
        """Initialize ML service and load model."""
        self.model = None
        self.scaler = None
        self.model_info = {}
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the trained model and scaler from disk."""
        try:
            model_path = Path(settings.MODEL_PATH)
            scaler_path = Path(settings.SCALER_PATH)
            
            if model_path.exists():
                model_data = joblib.load(model_path)
                self.model = model_data.get("model")
                self.model_info = model_data.get("info", {})
                logger.info(f"Loaded model: {self.model_info.get('model_type', 'Unknown')}")
            else:
                logger.warning(f"Model file not found at {model_path}. Please train the model first.")
            
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
                logger.info("Loaded scaler successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
    
    def is_model_ready(self) -> bool:
        """Check if model is loaded and ready for predictions."""
        return self.model is not None
    
    def _prepare_features(self, data: HealthDataInput) -> np.ndarray:
        """
        Prepare feature array from input data.
        
        Args:
            data: Health data input from user
            
        Returns:
            numpy array of features
        """
        features = np.array([[
            data.stress_level,
            data.sleep_hours,
            data.heart_rate,
            data.activity_level,
            data.weather_pressure,
            data.aqi
        ]])
        
        # Scale features if scaler is available
        if self.scaler is not None:
            features = self.scaler.transform(features)
        
        return features
    
    def detect_triggers(self, data: HealthDataInput) -> List[str]:
        """
        Detect potential migraine triggers from input data.
        
        Args:
            data: Health data input
            
        Returns:
            List of detected trigger descriptions
        """
        triggers = []
        
        # Stress triggers
        if data.stress_level >= self.THRESHOLDS["stress_high"]:
            triggers.append(f"High stress level ({data.stress_level}/10)")
        
        # Sleep triggers
        if data.sleep_hours <= self.THRESHOLDS["sleep_very_low"]:
            triggers.append(f"Very low sleep ({data.sleep_hours} hours)")
        elif data.sleep_hours <= self.THRESHOLDS["sleep_low"]:
            triggers.append(f"Insufficient sleep ({data.sleep_hours} hours)")
        
        # Heart rate triggers
        if data.heart_rate >= self.THRESHOLDS["heart_rate_very_high"]:
            triggers.append(f"Very elevated heart rate ({data.heart_rate} bpm)")
        elif data.heart_rate >= self.THRESHOLDS["heart_rate_high"]:
            triggers.append(f"Elevated heart rate ({data.heart_rate} bpm)")
        
        # Activity triggers
        if data.activity_level <= self.THRESHOLDS["activity_low"]:
            triggers.append(f"Low physical activity ({data.activity_level}/10)")
        
        # Weather pressure triggers
        if data.weather_pressure <= self.THRESHOLDS["weather_pressure_low"]:
            triggers.append(f"Low barometric pressure ({data.weather_pressure} hPa)")
        elif data.weather_pressure >= self.THRESHOLDS["weather_pressure_high"]:
            triggers.append(f"High barometric pressure ({data.weather_pressure} hPa)")
        
        # Air quality triggers
        if data.aqi >= self.THRESHOLDS["aqi_high"]:
            triggers.append(f"Poor air quality (AQI: {data.aqi})")
        elif data.aqi >= self.THRESHOLDS["aqi_moderate"]:
            triggers.append(f"Moderate air quality concern (AQI: {data.aqi})")
        
        return triggers
    
    def _calculate_risk_level(self, probability: float, triggers: List[str]) -> RiskLevel:
        """
        Calculate risk level based on probability and trigger count.
        
        Args:
            probability: Model prediction probability
            triggers: List of detected triggers
            
        Returns:
            RiskLevel enum value
        """
        # Count triggers and calculate weight
        trigger_count = len(triggers)
        trigger_weight = trigger_count * 0.08  # Reduced weight per trigger
        
        # Weighted combination of probability and triggers
        adjusted_prob = (probability * 0.7) + (trigger_weight * 0.3)
        adjusted_prob = min(1.0, max(0.0, adjusted_prob))
        
        # Clear thresholds for balanced predictions
        if adjusted_prob >= 0.55 or trigger_count >= 4:
            return RiskLevel.HIGH
        elif adjusted_prob >= 0.25 or trigger_count >= 2:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    def predict(self, data: HealthDataInput) -> PredictionResponse:
        """
        Make migraine risk prediction.
        
        Args:
            data: Health data input from user
            
        Returns:
            PredictionResponse with risk level, probability, and triggers
        """
        # Detect triggers first
        triggers = self.detect_triggers(data)
        
        if not self.is_model_ready():
            # Fallback prediction based on triggers if model not available
            logger.warning("Model not available, using trigger-based prediction")
            trigger_score = len(triggers) / 6.0  # Normalize by max possible triggers
            probability = min(0.95, trigger_score + 0.2)
            risk_level = self._calculate_risk_level(probability, triggers)
            
            return PredictionResponse(
                risk_level=risk_level,
                probability=round(probability, 3),
                confidence=0.5,  # Lower confidence without ML model
                triggers=triggers,
                prediction_window="24-48 hours"
            )
        
        # Prepare features and predict
        features = self._prepare_features(data)
        
        # Get prediction probability
        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(features)[0]
            # Assuming binary classification: [no_migraine, migraine]
            probability = probabilities[1] if len(probabilities) > 1 else probabilities[0]
        else:
            # For models without predict_proba
            prediction = self.model.predict(features)[0]
            probability = float(prediction)
        
        # Calculate risk level
        risk_level = self._calculate_risk_level(probability, triggers)
        
        # Calculate confidence based on probability distance from 0.5
        confidence = abs(probability - 0.5) * 2
        
        return PredictionResponse(
            risk_level=risk_level,
            probability=round(float(probability), 3),
            confidence=round(float(confidence), 3),
            triggers=triggers,
            prediction_window="24-48 hours"
        )
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model."""
        if not self.model_info:
            return {
                "status": "not_loaded",
                "message": "Model not trained or loaded"
            }
        return self.model_info


# Singleton instance
ml_service = MLService()
