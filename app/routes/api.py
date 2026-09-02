"""
API Routes for Migraine Trigger Tracker.

Endpoints:
- POST /predict - Get migraine risk prediction
- POST /log-data - Store user daily health data
- GET /history - Fetch user trends and history
- POST /ai-suggestion - Get AI-powered prevention advice
- GET /model-info - Get ML model information
- POST /classify-symptoms - Classify migraine type from symptoms
- POST /comprehensive-analysis - Get full analysis with symptoms and risk
- POST /chat - Interactive chatbot for migraine help
- POST /generate-daily-report - Generate daily PDF report
- POST /generate-weekly-report - Generate weekly PDF report
"""

from fastapi import APIRouter, HTTPException, Query, Depends, UploadFile, File
from fastapi.responses import Response
from typing import Optional, List, Literal
from pydantic import BaseModel, EmailStr, Field
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
import logging

from ..models.schemas import (
    HealthDataInput,
    PredictionResponse,
    HistoryResponse,
    AISuggestionRequest,
    AISuggestionResponse,
    ModelInfoResponse,
    SymptomInput,
    SymptomClassificationResponse,
    ComprehensiveAnalysisResponse,
    ChatRequest,
    ChatResponse,
    ReportAnalysisResponse
)
from ..services.ml_service import ml_service
from ..services.enhanced_ml_service import enhanced_ml_service
from ..services.groq_service import groq_service
from ..services.db_service import db_service
from ..services.auth_service import auth_service

report_service = None
report_service_error = None
try:
    from ..services.report_service import report_service
except Exception as exc:
    report_service_error = str(exc)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Migraine Tracker"])
security = HTTPBearer(auto_error=False)


class RegisterRequest(BaseModel):
    full_name: str = Field(..., min_length=2, max_length=100)
    email: EmailStr
    password: str = Field(..., min_length=6, max_length=128)
    age: int = Field(..., ge=1, le=120)
    gender: Literal["Male", "Female", "Other"]


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=6, max_length=128)


class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: dict


class PatientInfo(BaseModel):
    full_name: Optional[str] = "Patient"
    patient_id: Optional[str] = None
    email: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[str] = None


class DailyReportRequest(BaseModel):
    """Request model for daily report generation."""
    health_data: HealthDataInput
    prediction: dict
    ai_suggestions: Optional[List[str]] = None
    user_name: Optional[str] = "User"
    patient_info: Optional[PatientInfo] = None


class WeeklyReportRequest(BaseModel):
    """Request model for weekly report generation."""
    weekly_data: Optional[List[dict]] = None
    user_name: Optional[str] = "User"
    patient_info: Optional[PatientInfo] = None


def _patient_from_user(user: dict) -> dict:
    return {
        "full_name": user.get("full_name"),
        "patient_id": user.get("patient_id"),
        "email": user.get("email"),
        "age": user.get("age"),
        "gender": user.get("gender"),
    }


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    if not credentials or not credentials.credentials:
        raise HTTPException(status_code=401, detail="Authentication required")

    payload = auth_service.verify_access_token(credentials.credentials)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    user = await db_service.get_user_by_id(payload.get("user_id", ""))
    if not user:
        raise HTTPException(status_code=401, detail="User not found")

    return user


async def get_optional_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    if not credentials or not credentials.credentials:
        return {"_id": "guest_user", "full_name": "Guest User"}
    payload = auth_service.verify_access_token(credentials.credentials)
    if not payload:
        return {"_id": "guest_user", "full_name": "Guest User"}
    user = await db_service.get_user_by_id(payload.get("user_id", ""))
    return user or {"_id": "guest_user", "full_name": "Guest User"}



@router.post("/register", response_model=AuthResponse)
async def register_user(request: RegisterRequest):
    """Register a new user and return an access token."""
    try:
        user = await db_service.register_user(
            full_name=request.full_name,
            email=request.email,
            password=request.password,
            age=request.age,
            gender=request.gender,
        )
        if not user:
            raise HTTPException(status_code=400, detail="Email already exists")

        token = auth_service.create_access_token(user)
        return AuthResponse(
            access_token=token,
            user=db_service.serialize_user_profile(user),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Registration error: {e}")
        raise HTTPException(status_code=500, detail="Registration failed")


@router.post("/login", response_model=AuthResponse)
async def login_user(request: LoginRequest):
    """Authenticate user and return token."""
    try:
        user = await db_service.authenticate_user(request.email, request.password)
        if not user:
            raise HTTPException(status_code=401, detail="Invalid email or password")

        token = auth_service.create_access_token(user)
        return AuthResponse(
            access_token=token,
            user=db_service.serialize_user_profile(user),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(status_code=500, detail="Login failed")


@router.get("/me")
async def get_me(current_user: dict = Depends(get_current_user)):
    return {"user": db_service.serialize_user_profile(current_user)}


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "Migraine Trigger Tracker API",
        "ml_model_loaded": ml_service.is_model_ready(),
        "groq_available": groq_service.is_available(),
        "database_connected": db_service.is_connected()
    }


@router.post("/predict", response_model=PredictionResponse)
async def predict_migraine_risk(
    data: HealthDataInput,
    current_user: dict = Depends(get_current_user)
):
    """
    Predict migraine risk for the next 24-48 hours.
    
    This endpoint analyzes the provided health metrics and returns:
    - Risk level (Low/Medium/High)
    - Probability of migraine occurrence
    - Detected triggers
    - Confidence score
    
    Args:
        data: HealthDataInput with current health metrics
        
    Returns:
        PredictionResponse with risk assessment
    """
    try:
        data.user_id = str(current_user.get("_id"))
        logger.info(f"Prediction request for user: {data.user_id}")
        
        # Get prediction from enhanced ML service
        prediction = enhanced_ml_service.predict(data)
        
        logger.info(f"Prediction result: {prediction.risk_level.value} ({prediction.probability:.2%})")
        
        return prediction
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error making prediction: {str(e)}"
        )


@router.post("/log-data")
async def log_health_data(
    data: HealthDataInput,
    current_user: dict = Depends(get_current_user)
):
    """
    Store user daily health data with prediction.
    
    Logs the health data to the database along with
    the migraine risk prediction for future analysis.
    
    Args:
        data: HealthDataInput with daily health metrics
        
    Returns:
        Confirmation with record ID and prediction
    """
    try:
        data.user_id = str(current_user.get("_id"))
        logger.info(f"Logging health data for user: {data.user_id}")
        
        # Get prediction
        prediction = ml_service.predict(data)
        
        # Save to database
        record_id = await db_service.save_health_data(
            data=data,
            prediction={
                "risk_level": prediction.risk_level.value,
                "probability": prediction.probability,
                "triggers": prediction.triggers,
                "confidence": prediction.confidence
            }
        )
        
        return {
            "success": True,
            "message": "Health data logged successfully",
            "record_id": record_id,
            "prediction": {
                "risk_level": prediction.risk_level.value,
                "probability": prediction.probability,
                "triggers": prediction.triggers
            }
        }
        
    except Exception as e:
        logger.error(f"Error logging data: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error logging health data: {str(e)}"
        )


@router.get("/history", response_model=HistoryResponse)
async def get_user_history(
    days: int = Query(default=30, ge=1, le=365, description="Number of days to fetch"),
    limit: int = Query(default=100, ge=1, le=500, description="Maximum records to return"),
    current_user: dict = Depends(get_current_user)
):
    """
    Fetch user's health data history and trends.
    
    Returns historical records with analytics including:
    - Total records and migraine count
    - Average risk score
    - Trigger frequency analysis
    - Trend data for visualization
    
    Args:
        user_id: User identifier
        days: Number of days to look back
        limit: Maximum records to return
        
    Returns:
        HistoryResponse with records and analytics
    """
    try:
        user_id = str(current_user.get("_id"))
        logger.info(f"Fetching history for user: {user_id}, days: {days}")
        
        history = await db_service.get_user_history(
            user_id=user_id,
            days=days,
            limit=limit
        )
        
        return history
        
    except Exception as e:
        logger.error(f"Error fetching history: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching history: {str(e)}"
        )


@router.post("/ai-suggestion", response_model=AISuggestionResponse)
async def get_ai_suggestions(
    request: AISuggestionRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Get AI-powered personalized migraine prevention advice.
    
    Uses Groq LLM to generate contextual suggestions based on:
    - Detected triggers
    - Current risk level
    - Recent lifestyle data
    
    Args:
        request: AISuggestionRequest with health data and triggers
        
    Returns:
        AISuggestionResponse with actionable suggestions
    """
    try:
        request.user_id = str(current_user.get("_id"))
        logger.info(f"AI suggestion request for risk level: {request.risk_level.value}")
        
        # Get AI suggestions from Groq service
        suggestions = await groq_service.get_suggestions(request)
        
        logger.info(f"Generated {len(suggestions.suggestions)} suggestions")
        
        return suggestions
        
    except Exception as e:
        logger.error(f"Error generating AI suggestions: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error generating suggestions: {str(e)}"
        )


@router.get("/model-info")
async def get_model_info():
    """
    Get information about the loaded ML model.
    
    Returns:
        Model type, accuracy, features, and training info
    """
    try:
        model_info = ml_service.get_model_info()
        return model_info
        
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting model info: {str(e)}"
        )


@router.get("/statistics")
async def get_user_statistics(
    current_user: dict = Depends(get_current_user)
):
    """
    Get aggregated statistics for a user.
    
    Args:
        user_id: User identifier
        
    Returns:
        Aggregated statistics including averages and counts
    """
    try:
        user_id = str(current_user.get("_id"))
        stats = await db_service.get_statistics(user_id)
        return {
            "user_id": user_id,
            "statistics": stats
        }
        
    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting statistics: {str(e)}"
        )


@router.post("/update-migraine-status")
async def update_migraine_status(
    record_id: str = Query(..., description="Record ID to update"),
    had_migraine: bool = Query(..., description="Whether migraine occurred"),
    current_user: dict = Depends(get_current_user),
):
    """
    Update migraine status for a previous record.
    
    Allows users to confirm whether a migraine actually occurred
    after a prediction was made, improving future predictions.
    
    Args:
        record_id: ID of the record to update
        had_migraine: Whether migraine occurred
        
    Returns:
        Confirmation of update
    """
    try:
        user_id = str(current_user.get("_id"))
        success = await db_service.update_migraine_status(
            record_id=record_id,
            had_migraine=had_migraine,
            user_id=user_id,
        )
        
        if success:
            return {
                "success": True,
                "message": "Migraine status updated successfully"
            }
        else:
            raise HTTPException(
                status_code=404,
                detail="Record not found or update failed"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating migraine status: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error updating status: {str(e)}"
        )


@router.get("/prompt-template")
async def get_prompt_template():
    """
    Get the Groq LLM prompt template used for suggestions.
    
    Useful for documentation and understanding how
    AI suggestions are generated.
    
    Returns:
        The prompt template string
    """
    return {
        "prompt_template": groq_service.get_prompt_template(),
        "model": "llama-3.3-70b-versatile",
        "description": "This prompt is used to generate personalized migraine prevention advice"
    }


# ========================
# SYMPTOM CLASSIFICATION ENDPOINTS
# ========================

@router.post("/classify-symptoms", response_model=SymptomClassificationResponse)
async def classify_symptoms(
    symptoms: SymptomInput,
    current_user: dict = Depends(get_current_user)
):
    """
    Classify migraine type based on symptoms.
    
    Analyzes the provided symptoms and returns:
    - Predicted migraine type
    - Confidence score
    - Type description
    - Specific recommendations
    
    Args:
        symptoms: SymptomInput with symptom data
        
    Returns:
        SymptomClassificationResponse with classification
    """
    try:
        symptoms.user_id = str(current_user.get("_id"))
        logger.info(f"Symptom classification request for user: {symptoms.user_id}")
        
        result = enhanced_ml_service.classify_symptoms(symptoms)
        
        logger.info(f"Classification: {result.migraine_type} (confidence: {result.confidence:.2%})")
        
        return result
        
    except Exception as e:
        logger.error(f"Classification error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error classifying symptoms: {str(e)}"
        )


@router.post("/comprehensive-analysis", response_model=ComprehensiveAnalysisResponse)
async def get_comprehensive_analysis(
    health_data: HealthDataInput,
    symptoms: Optional[SymptomInput] = None,
    current_user: dict = Depends(get_current_user)
):
    """
    Get comprehensive migraine analysis.
    
    Combines lifestyle-based risk prediction with symptom
    classification for a complete analysis.
    
    Args:
        health_data: Current health/lifestyle metrics
        symptoms: Optional symptom data for classification
        
    Returns:
        ComprehensiveAnalysisResponse with full analysis
    """
    try:
        user_id = str(current_user.get("_id"))
        health_data.user_id = user_id
        if symptoms:
            symptoms.user_id = user_id
        logger.info(f"Comprehensive analysis for user: {health_data.user_id}")
        
        result = enhanced_ml_service.get_comprehensive_analysis(health_data, symptoms)
        
        logger.info(f"Analysis complete - Risk: {result.risk_prediction.risk_level.value}")
        
        return result
        
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error performing analysis: {str(e)}"
        )


@router.post("/enhanced-predict", response_model=PredictionResponse)
async def enhanced_predict(
    data: HealthDataInput,
    current_user: dict = Depends(get_current_user)
):
    """
    Enhanced migraine risk prediction using improved model.
    
    Uses the enhanced ML service with better trigger detection
    and more accurate predictions.
    
    Args:
        data: HealthDataInput with current health metrics
        
    Returns:
        PredictionResponse with enhanced risk assessment
    """
    try:
        data.user_id = str(current_user.get("_id"))
        logger.info(f"Enhanced prediction for user: {data.user_id}")
        
        prediction = enhanced_ml_service.predict_risk(data)
        
        logger.info(f"Prediction: {prediction.risk_level.value} ({prediction.probability:.2%})")
        
        return prediction
        
    except Exception as e:
        logger.error(f"Enhanced prediction error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error making prediction: {str(e)}"
        )


@router.get("/detailed-triggers")
async def get_detailed_triggers(
    stress_level: int = Query(..., ge=1, le=10),
    sleep_hours: float = Query(..., ge=0, le=24),
    heart_rate: int = Query(..., ge=40, le=200),
    activity_level: int = Query(..., ge=1, le=10),
    weather_pressure: float = Query(..., ge=900, le=1100),
    aqi: int = Query(..., ge=0, le=500)
):
    """
    Get detailed trigger analysis without full prediction.
    
    Quick endpoint for trigger detection with recommendations.
    
    Returns:
        List of detailed triggers with severity and recommendations
    """
    try:
        data = HealthDataInput(
            stress_level=stress_level,
            sleep_hours=sleep_hours,
            heart_rate=heart_rate,
            activity_level=activity_level,
            weather_pressure=weather_pressure,
            aqi=aqi
        )
        
        triggers = enhanced_ml_service.detect_triggers(data)
        
        return {
            "trigger_count": len(triggers),
            "triggers": triggers,
            "summary": _get_trigger_summary(triggers)
        }
        
    except Exception as e:
        logger.error(f"Trigger detection error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error detecting triggers: {str(e)}"
        )


def _get_trigger_summary(triggers: list) -> str:
    """Generate a human-readable trigger summary."""
    if not triggers:
        return "No significant triggers detected. Your current metrics look healthy!"
    
    high_count = sum(1 for t in triggers if t.get('severity') == 'high')
    medium_count = sum(1 for t in triggers if t.get('severity') == 'medium')
    
    if high_count >= 2:
        return f"⚠️ High risk: {high_count} severe triggers detected. Take immediate preventive action."
    elif high_count == 1:
        return f"Caution: 1 severe trigger detected along with {medium_count} moderate triggers."
    elif medium_count >= 2:
        return f"Moderate risk: {medium_count} triggers detected. Consider preventive measures."
    else:
        return "Low risk: Minor triggers detected. Monitor your symptoms."


# ========================
# CHATBOT ENDPOINTS
# ========================

@router.post("/chat", response_model=ChatResponse)
async def chat_with_assistant(
    request: ChatRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Interactive chatbot for migraine-related questions.
    
    Provides helpful information about:
    - Migraine triggers and prevention
    - Symptoms and when to seek help
    - Lifestyle recommendations
    - General health guidance
    
    Args:
        request: ChatRequest with user message
        
    Returns:
        ChatResponse with AI reply and suggestions
    """
    try:
        request.user_id = str(current_user.get("_id"))
        logger.info(f"Chat request from user: {request.user_id}")
        
        response = await groq_service.chat(request)
        
        return response
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing chat: {str(e)}"
        )


@router.get("/enhanced-model-info")
async def get_enhanced_model_info():
    """
    Get information about all loaded ML models.
    
    Returns:
        Info about symptom classifier and risk predictor
    """
    try:
        return enhanced_ml_service.get_model_info()
        
    except Exception as e:
        logger.error(f"Error getting enhanced model info: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting model info: {str(e)}"
        )


@router.get("/migraine-types")
async def get_migraine_types():
    """
    Get list of migraine types with descriptions.
    
    Useful for understanding different migraine classifications.
    
    Returns:
        Dictionary of migraine types with info
    """
    return {
        "types": enhanced_ml_service.MIGRAINE_TYPE_INFO
    }


# ========================
# PDF REPORT GENERATION ENDPOINTS
# ========================


@router.post("/generate-daily-report")
async def generate_daily_report(
    request: DailyReportRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Generate a daily PDF report with prediction and health data.
    
    Args:
        request: DailyReportRequest with health data, prediction, and suggestions
        
    Returns:
        PDF file as downloadable response
    """
    if report_service is None:
        raise HTTPException(status_code=503, detail=f"Report service unavailable: {report_service_error}")

    try:
        user_id = str(current_user.get("_id"))
        request.health_data.user_id = user_id
        patient_info = request.patient_info.model_dump() if request.patient_info else _patient_from_user(current_user)
        report_user_name = patient_info.get("full_name") or request.user_name or "User"
        logger.info(f"Generating daily report for user: {report_user_name}")
        
        # Convert health_data to dict
        health_dict = {
            'sleep_hours': request.health_data.sleep_hours,
            'stress_level': request.health_data.stress_level,
            'heart_rate': request.health_data.heart_rate,
            'activity_level': request.health_data.activity_level,
            'weather_pressure': request.health_data.weather_pressure,
            'aqi': request.health_data.aqi
        }
        
        # Generate PDF
        pdf_bytes = report_service.generate_daily_report(
            prediction_data=request.prediction,
            health_data=health_dict,
            ai_suggestions=request.ai_suggestions,
            user_name=report_user_name,
            patient_info=patient_info,
        )
        
        # Return as downloadable PDF
        from datetime import datetime
        filename = f"migraine_daily_report_{datetime.now().strftime('%Y%m%d')}.pdf"
        
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename={filename}"
            }
        )
        
    except Exception as e:
        logger.error(f"Error generating daily report: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error generating report: {str(e)}"
        )


@router.post("/generate-weekly-report")
async def generate_weekly_report(
    request: WeeklyReportRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Generate a weekly PDF report with trend analysis.
    
    Args:
        request: WeeklyReportRequest with weekly data
        
    Returns:
        PDF file as downloadable response
    """
    if report_service is None:
        raise HTTPException(status_code=503, detail=f"Report service unavailable: {report_service_error}")

    try:
        user_id = str(current_user.get("_id"))
        patient_info = request.patient_info.model_dump() if request.patient_info else _patient_from_user(current_user)
        report_user_name = patient_info.get("full_name") or request.user_name or "User"
        logger.info(f"Generating weekly report for user: {report_user_name}")

        weekly_history = await db_service.get_user_history(user_id=user_id, days=7, limit=200)
        full_history = await db_service.get_user_history(user_id=user_id, days=90, limit=500)

        weekly_data = []
        for record in reversed(weekly_history.records):
            prediction = record.prediction or {}
            weekly_data.append({
                "date": record.created_at.strftime("%Y-%m-%d"),
                "risk_level": prediction.get("risk_level", "N/A"),
                "probability": prediction.get("probability", 0),
                "triggers": prediction.get("triggers", []),
                "health_data": {
                    "stress_level": record.stress_level,
                    "sleep_hours": record.sleep_hours,
                    "heart_rate": record.heart_rate,
                    "activity_level": record.activity_level,
                    "weather_pressure": record.weather_pressure,
                    "aqi": record.aqi,
                },
            })

        if not weekly_data and request.weekly_data:
            weekly_data = request.weekly_data

        top_trigger = "None detected"
        if full_history.trigger_frequency:
            top_trigger = max(full_history.trigger_frequency, key=full_history.trigger_frequency.get)

        history_summary = {
            "total_records": full_history.total_records,
            "migraine_count": full_history.migraine_count,
            "average_risk": full_history.average_risk,
            "top_trigger": top_trigger,
        }
        
        # Generate PDF
        pdf_bytes = report_service.generate_weekly_report(
            weekly_data=weekly_data,
            user_name=report_user_name,
            patient_info=patient_info,
            previous_history_summary=history_summary,
        )
        
        # Return as downloadable PDF
        from datetime import datetime
        filename = f"migraine_weekly_report_{datetime.now().strftime('%Y%m%d')}.pdf"
        
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename={filename}"
            }
        )
        
    except Exception as e:
        logger.error(f"Error generating weekly report: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error generating report: {str(e)}"
        )


@router.post("/generate-quick-report")
async def generate_quick_report(
    health_data: HealthDataInput,
    current_user: dict = Depends(get_current_user)
):
    """
    Quick endpoint to generate daily report from just health data.
    
    Automatically performs prediction and generates PDF.
    
    Args:
        health_data: HealthDataInput with current metrics
        
    Returns:
        PDF file as downloadable response
    """
    if report_service is None:
        raise HTTPException(status_code=503, detail=f"Report service unavailable: {report_service_error}")

    try:
        health_data.user_id = str(current_user.get("_id"))
        patient_info = _patient_from_user(current_user)
        logger.info(f"Generating quick report for user: {health_data.user_id}")
        
        # Get prediction
        prediction = enhanced_ml_service.predict_risk(health_data)
        
        # Get AI suggestions
        from ..models.schemas import AISuggestionRequest, RiskLevel
        suggestion_request = AISuggestionRequest(
            risk_level=prediction.risk_level,
            triggers=prediction.triggers
        )
        ai_response = await groq_service.get_suggestions(suggestion_request)
        
        # Convert health_data to dict
        health_dict = {
            'sleep_hours': health_data.sleep_hours,
            'stress_level': health_data.stress_level,
            'heart_rate': health_data.heart_rate,
            'activity_level': health_data.activity_level,
            'weather_pressure': health_data.weather_pressure,
            'aqi': health_data.aqi
        }
        
        # Generate PDF
        pdf_bytes = report_service.generate_daily_report(
            prediction_data={
                'risk_level': prediction.risk_level.value,
                'probability': prediction.probability,
                'triggers': prediction.triggers,
                'confidence': prediction.confidence
            },
            health_data=health_dict,
            ai_suggestions=ai_response.suggestions,
            user_name=patient_info.get("full_name") or health_data.user_id,
            patient_info=patient_info,
        )
        
        # Return as downloadable PDF
        from datetime import datetime
        filename = f"migraine_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename={filename}"
            }
        )
        
    except Exception as e:
        logger.error(f"Error generating quick report: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error generating report: {str(e)}"
        )


# ========================
# REPORT UPLOAD & ANALYSIS ENDPOINT
# ========================

@router.post("/analyze-report", response_model=ReportAnalysisResponse)
async def analyze_uploaded_report(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_optional_current_user)
):
    """
    Upload and analyze a medical report (PDF, TXT, CSV, JSON, MD, or Image).
    Extracts report content and uses Groq AI to evaluate key clinical findings,
    migraine triggers, risk level, and recommendations.
    """
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")

    filename = file.filename or "uploaded_report"
    ext = filename.split(".")[-1].lower() if "." in filename else ""
    
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    extracted_text = ""
    file_type_label = ext.upper() if ext else "UNKNOWN"

    try:
        if ext == "pdf":
            file_type_label = "PDF Document"
            import io
            import pypdf
            pdf_reader = pypdf.PdfReader(io.BytesIO(contents))
            text_pages = []
            for i, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text_pages.append(page_text)
            extracted_text = "\n".join(text_pages).strip()
        else:
            file_type_label = f"{ext.upper()} Document" if ext else "Text Document"
            try:
                extracted_text = contents.decode("utf-8", errors="ignore").strip()
            except Exception:
                extracted_text = ""
    except Exception as e:
        logger.error(f"Error parsing uploaded file {filename}: {e}")
        raise HTTPException(status_code=400, detail=f"Could not extract text from file: {str(e)}")

    if not extracted_text or len(extracted_text) < 5:
        extracted_text = f"Report File: {filename}\nFile Size: {len(contents)} bytes\nNote: Visual report or medical document uploaded."

    logger.info(f"Analyzing report '{filename}' ({file_type_label}) for user: {current_user.get('_id')}")
    
    analysis_detail = await groq_service.analyze_medical_report(
        report_text=extracted_text,
        filename=filename
    )

    preview = extracted_text[:300] + "..." if len(extracted_text) > 300 else extracted_text

    return ReportAnalysisResponse(
        success=True,
        filename=filename,
        file_type=file_type_label,
        extracted_text_preview=preview,
        analysis=analysis_detail
    )

