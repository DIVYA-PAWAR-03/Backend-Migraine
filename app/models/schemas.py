"""
Pydantic schemas for API request/response validation.

Includes schemas for:
- Health data input and prediction
- Symptom-based classification
- Comprehensive analysis
- AI suggestions
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class RiskLevel(str, Enum):
    """Migraine risk level classification."""
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"


class MigraineType(str, Enum):
    """Classification of migraine types."""
    MIGRAINE_WITHOUT_AURA = "Migraine without aura"
    TYPICAL_AURA_WITH_MIGRAINE = "Typical aura with migraine"
    BASILAR_TYPE_AURA = "Basilar-type aura"
    SPORADIC_HEMIPLEGIC = "Sporadic hemiplegic migraine"
    FAMILIAL_HEMIPLEGIC = "Familial hemiplegic migraine"
    TYPICAL_AURA_WITHOUT_MIGRAINE = "Typical aura without migraine"
    OTHER = "Other"


class HealthDataInput(BaseModel):
    """Input schema for health data logging and prediction."""
    user_id: Optional[str] = Field(default="default_user", description="User identifier")
    stress_level: int = Field(..., ge=1, le=10, description="Stress level (1-10)")
    sleep_hours: float = Field(..., ge=0, le=24, description="Hours of sleep")
    heart_rate: int = Field(..., ge=40, le=200, description="Heart rate in BPM")
    activity_level: int = Field(..., ge=1, le=10, description="Activity level (1-10)")
    weather_pressure: float = Field(..., ge=900, le=1100, description="Barometric pressure in hPa")
    aqi: int = Field(..., ge=0, le=500, description="Air Quality Index")
    had_migraine: Optional[bool] = Field(default=None, description="Whether user had migraine")
    notes: Optional[str] = Field(default=None, description="Additional notes")
    
    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "user123",
                "stress_level": 7,
                "sleep_hours": 5.5,
                "heart_rate": 85,
                "activity_level": 4,
                "weather_pressure": 1013,
                "aqi": 75,
                "had_migraine": False,
                "notes": "Feeling tired today"
            }
        }


class PredictionResponse(BaseModel):
    """Response schema for migraine prediction."""
    risk_level: RiskLevel
    probability: float = Field(..., ge=0, le=1, description="Prediction probability")
    confidence: float = Field(..., ge=0, le=1, description="Model confidence")
    triggers: List[str] = Field(default=[], description="Detected trigger factors")
    prediction_window: str = Field(default="24-48 hours", description="Prediction time window")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_schema_extra = {
            "example": {
                "risk_level": "High",
                "probability": 0.78,
                "confidence": 0.85,
                "triggers": ["High stress", "Low sleep", "High AQI"],
                "prediction_window": "24-48 hours",
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }


class HealthDataRecord(BaseModel):
    """Schema for stored health data record."""
    id: Optional[str] = None
    user_id: str
    stress_level: int
    sleep_hours: float
    heart_rate: int
    activity_level: int
    weather_pressure: float
    aqi: int
    had_migraine: Optional[bool] = None
    notes: Optional[str] = None
    prediction: Optional[dict] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class HistoryResponse(BaseModel):
    """Response schema for user history."""
    user_id: str
    total_records: int
    migraine_count: int
    average_risk: float
    records: List[HealthDataRecord]
    trigger_frequency: dict = Field(default={}, description="Frequency of each trigger")
    trends: dict = Field(default={}, description="Trend analysis data")


class AISuggestionRequest(BaseModel):
    """Request schema for AI suggestions."""
    user_id: Optional[str] = "default_user"
    triggers: List[str] = Field(default=[], description="Detected triggers")
    risk_level: RiskLevel
    stress_level: int = Field(..., ge=1, le=10)
    sleep_hours: float = Field(..., ge=0, le=24)
    heart_rate: int = Field(..., ge=40, le=200)
    activity_level: int = Field(..., ge=1, le=10)
    weather_pressure: float = Field(..., ge=900, le=1100)
    aqi: int = Field(..., ge=0, le=500)
    
    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "user123",
                "triggers": ["High stress", "Low sleep"],
                "risk_level": "High",
                "stress_level": 8,
                "sleep_hours": 4,
                "heart_rate": 90,
                "activity_level": 3,
                "weather_pressure": 1005,
                "aqi": 85
            }
        }


class AISuggestionResponse(BaseModel):
    """Response schema for AI-generated suggestions."""
    suggestions: List[str] = Field(..., description="List of actionable suggestions")
    summary: str = Field(..., description="Brief summary of the advice")
    urgency: str = Field(default="moderate", description="Urgency level of action needed")
    generated_at: datetime = Field(default_factory=datetime.utcnow)


class ModelInfoResponse(BaseModel):
    """Response schema for ML model information."""
    model_type: str
    accuracy: float
    f1_score: float
    features: List[str]
    trained_at: str
    sample_count: int


# ========================
# SYMPTOM CLASSIFICATION SCHEMAS
# ========================

class SymptomInput(BaseModel):
    """Input schema for symptom-based classification."""
    user_id: Optional[str] = Field(default="default_user", description="User identifier")
    
    # Demographics
    age: int = Field(..., ge=1, le=120, description="Patient age")
    
    # Headache characteristics
    duration: int = Field(..., ge=1, le=3, description="Duration: 1=<4hrs, 2=4-72hrs, 3=>72hrs")
    frequency: int = Field(..., ge=1, le=10, description="Frequency of attacks per month")
    location: int = Field(default=1, ge=1, le=3, description="Location: 1=Unilateral, 2=Bilateral, 3=Other")
    character: int = Field(default=1, ge=1, le=3, description="Character: 1=Pulsating, 2=Pressing, 3=Other")
    intensity: int = Field(..., ge=1, le=3, description="Intensity: 1=Mild, 2=Moderate, 3=Severe")
    
    # Associated symptoms
    nausea: bool = Field(default=False, description="Nausea present")
    vomit: bool = Field(default=False, description="Vomiting present")
    phonophobia: bool = Field(default=False, description="Sensitivity to sound")
    photophobia: bool = Field(default=False, description="Sensitivity to light")
    
    # Aura symptoms
    visual: int = Field(default=0, ge=0, le=4, description="Visual symptoms: 0=None to 4=Severe")
    sensory: int = Field(default=0, ge=0, le=2, description="Sensory symptoms: 0=None, 1=Present, 2=Severe")
    
    # Neurological symptoms
    dysphasia: bool = Field(default=False, description="Speech difficulty")
    dysarthria: bool = Field(default=False, description="Articulation difficulty")
    vertigo: bool = Field(default=False, description="Vertigo/dizziness")
    tinnitus: bool = Field(default=False, description="Ringing in ears")
    hypoacusis: bool = Field(default=False, description="Hearing decrease")
    diplopia: bool = Field(default=False, description="Double vision")
    defect: bool = Field(default=False, description="Visual field defect")
    ataxia: bool = Field(default=False, description="Coordination problems")
    conscience: bool = Field(default=False, description="Consciousness changes")
    paresthesia: bool = Field(default=False, description="Tingling/numbness")
    
    # Family history
    dpf: bool = Field(default=False, description="Direct family history of migraine")
    
    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "user123",
                "age": 35,
                "duration": 2,
                "frequency": 4,
                "location": 1,
                "character": 1,
                "intensity": 2,
                "nausea": True,
                "vomit": False,
                "phonophobia": True,
                "photophobia": True,
                "visual": 2,
                "sensory": 1,
                "dysphasia": False,
                "dysarthria": False,
                "vertigo": False,
                "tinnitus": False,
                "hypoacusis": False,
                "diplopia": False,
                "defect": False,
                "ataxia": False,
                "conscience": False,
                "paresthesia": False,
                "dpf": True
            }
        }


class SymptomClassificationResponse(BaseModel):
    """Response schema for symptom-based classification."""
    migraine_type: str = Field(..., description="Classified migraine type")
    confidence: float = Field(..., ge=0, le=1, description="Classification confidence")
    description: str = Field(..., description="Description of the migraine type")
    recommendations: List[str] = Field(default=[], description="Type-specific recommendations")
    key_symptoms: List[str] = Field(default=[], description="Key symptoms identified")
    top_predictions: List[Dict[str, Any]] = Field(default=[], description="Top predicted types with probabilities")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_schema_extra = {
            "example": {
                "migraine_type": "Typical aura with migraine",
                "confidence": 0.85,
                "description": "Migraine preceded by visual or sensory warning signs",
                "recommendations": [
                    "When aura starts, take medication immediately if prescribed",
                    "Rest in a dark, quiet room during aura phase"
                ],
                "key_symptoms": ["Moderate pain intensity", "Visual disturbances", "Light sensitivity"],
                "top_predictions": [
                    {"type": "Typical aura with migraine", "probability": 0.85},
                    {"type": "Migraine without aura", "probability": 0.10}
                ]
            }
        }


class TriggerDetail(BaseModel):
    """Detailed trigger information."""
    trigger: str
    severity: str = Field(description="low, medium, or high")
    category: str
    recommendation: str


class ComprehensiveAnalysisResponse(BaseModel):
    """Response schema for comprehensive analysis."""
    risk_prediction: PredictionResponse
    symptom_classification: Optional[SymptomClassificationResponse] = None
    detailed_triggers: List[TriggerDetail] = Field(default=[])
    recommendations: List[str] = Field(default=[])
    analysis_timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_schema_extra = {
            "example": {
                "risk_prediction": {
                    "risk_level": "High",
                    "probability": 0.78,
                    "confidence": 0.85,
                    "triggers": ["High stress", "Low sleep"]
                },
                "symptom_classification": {
                    "migraine_type": "Migraine without aura",
                    "confidence": 0.82
                },
                "recommendations": [
                    "Practice stress reduction techniques",
                    "Ensure adequate sleep tonight"
                ]
            }
        }


# ========================
# CHATBOT SCHEMAS
# ========================

class ChatMessage(BaseModel):
    """Schema for chat messages."""
    role: str = Field(..., description="'user' or 'assistant'")
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ChatRequest(BaseModel):
    """Request schema for chatbot."""
    user_id: Optional[str] = "default_user"
    message: str = Field(..., min_length=1, max_length=1000, description="User message")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Additional context")
    
    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "user123",
                "message": "What are common migraine triggers?",
                "context": {"recent_prediction": "High risk"}
            }
        }


class ChatResponse(BaseModel):
    """Response schema for chatbot."""
    response: str = Field(..., description="AI response")
    suggestions: List[str] = Field(default=[], description="Follow-up suggestions")
    related_topics: List[str] = Field(default=[], description="Related topics to explore")
    timestamp: datetime = Field(default_factory=datetime.utcnow)