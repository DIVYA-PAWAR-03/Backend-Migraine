"""
Configuration settings for the Migraine Trigger Tracker application.
Uses pydantic-settings for environment variable management.
"""

import json

from pydantic import field_validator
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
    CORS_ORIGINS: list[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000"
    ]

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def parse_cors_origins(cls, value):
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            raw = value.strip()
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
        return value

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


settings = get_settings()
