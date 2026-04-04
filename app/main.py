"""
Migraine Trigger Tracker - FastAPI Application

Main application entry point with:
- CORS configuration
- Database connection lifecycle
- Route registration
- API documentation
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from .config import settings
from .services.db_service import db_service

router = None
startup_error = None

try:
    from .routes.api import router
except Exception as e:
    startup_error = str(e)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    
    Handles startup and shutdown events:
    - Database connection on startup
    - Database disconnection on shutdown
    """
    # Startup
    logger.info("Starting Migraine Trigger Tracker API...")
    
    # Connect to database
    connected = await db_service.connect()
    if connected:
        logger.info("Database connected successfully")
    else:
        logger.warning("Database connection failed - running without persistence")
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")
    await db_service.disconnect()
    logger.info("Goodbye!")


# Create FastAPI application
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="""
## 🧠 AI-Powered Migraine Trigger Tracker

An intelligent health application that predicts migraine risk using machine learning 
and provides personalized prevention advice using Groq LLM.

### Features

- **ML Prediction**: Predicts migraine risk for next 24-48 hours
- **Trigger Detection**: Identifies stress, sleep, heart rate, activity, weather, and AQI triggers
- **AI Suggestions**: Personalized prevention advice using Groq LLM
- **History Tracking**: Track and visualize health trends over time

### Endpoints

- `POST /api/v1/predict` - Get migraine risk prediction
- `POST /api/v1/log-data` - Log daily health data
- `GET /api/v1/history` - Get user history and trends
- `POST /api/v1/ai-suggestion` - Get AI-generated advice
- `GET /api/v1/model-info` - Get ML model information

### Tech Stack

- **Backend**: FastAPI + Python
- **ML**: Scikit-learn (Random Forest / Logistic Regression / Decision Tree)
- **Database**: MongoDB
- **LLM**: Groq (Llama-3.3-70b-versatile)
    """,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
if router is not None:
    app.include_router(router)
else:
    logger.error(f"API router failed to load: {startup_error}")


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running" if startup_error is None else "degraded",
        "docs": "/docs",
        "health": "/api/v1/health",
        "startup_error": startup_error
    }


@app.get("/api/v1/health")
async def app_health():
    """Global health endpoint that remains available even when router load fails."""
    return {
        "status": "healthy" if startup_error is None else "degraded",
        "service": settings.APP_NAME,
        "router_loaded": router is not None,
        "database_connected": db_service.is_connected(),
        "startup_error": startup_error,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG
    )
