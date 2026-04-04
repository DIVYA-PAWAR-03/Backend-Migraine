"""
Configuration settings for the Migraine Trigger Tracker application.
Uses pydantic-settings for environment variable management.
"""

import json
import os

from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Application Settings
    APP_NAME: str = "Migraine Trigger Tracker"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True
    
    # MongoDB Settings
    MONGODB_URL: str = "mongodb://localhost:27017"
    DATABASE_NAME: str = "migraine_tracker"
    
    # Groq LLM Settings
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "llama-3.3-70b-versatile"
    
    # ML Model Settings
    MODEL_PATH: str = "app/ml/model.pkl"
    SCALER_PATH: str = "app/ml/scaler.pkl"
    
    # CORS Settings
    CORS_ORIGINS: str = "http://localhost:3000,http://127.0.0.1:3000"

    @property
    def cors_origins_list(self) -> list[str]:
        raw = (self.CORS_ORIGINS or "").strip()
        if not raw:
            return []
        if raw.startswith("["):
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    return [str(item).strip() for item in parsed if str(item).strip()]
            except Exception:
                pass
        return [item.strip() for item in raw.split(",") if item.strip()]

    # Auth Settings
    AUTH_SECRET: str = "change-this-in-production"
    AUTH_TOKEN_EXPIRY_HOURS: int = 168
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


try:
    settings = get_settings()
except Exception:
    class FallbackSettings:
        APP_NAME = os.getenv("APP_NAME", "Migraine Trigger Tracker")
        APP_VERSION = os.getenv("APP_VERSION", "1.0.0")
        DEBUG = os.getenv("DEBUG", "false").lower() == "true"
        MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
        DATABASE_NAME = os.getenv("DATABASE_NAME", "migraine_tracker")
        GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
        GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        MODEL_PATH = os.getenv("MODEL_PATH", "app/ml/model.pkl")
        SCALER_PATH = os.getenv("SCALER_PATH", "app/ml/scaler.pkl")
        CORS_ORIGINS = os.getenv(
            "CORS_ORIGINS",
            "http://localhost:3000,http://127.0.0.1:3000",
        )
        AUTH_SECRET = os.getenv("AUTH_SECRET", "change-this-in-production")
        AUTH_TOKEN_EXPIRY_HOURS = int(os.getenv("AUTH_TOKEN_EXPIRY_HOURS", "168"))

        @property
        def cors_origins_list(self) -> list[str]:
            raw = (self.CORS_ORIGINS or "").strip()
            if not raw:
                return []
            if raw.startswith("["):
                try:
                    parsed = json.loads(raw)
                    if isinstance(parsed, list):
                        return [str(item).strip() for item in parsed if str(item).strip()]
                except Exception:
                    pass
            return [item.strip() for item in raw.split(",") if item.strip()]

    settings = FallbackSettings()
