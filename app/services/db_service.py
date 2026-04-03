"""
Database Service for MongoDB operations.

Handles all database interactions for storing and retrieving
user health data, predictions, and historical trends.
"""

import logging
import secrets
import hashlib
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.errors import ConnectionFailure
from bson import ObjectId

from ..config import settings
from ..models.schemas import HealthDataInput, HealthDataRecord, HistoryResponse

logger = logging.getLogger(__name__)


class DatabaseService:
    """Service class for MongoDB database operations."""
    
    COLLECTION_NAME = "health_records"
    USERS_COLLECTION_NAME = "users"
    
    def __init__(self):
        """Initialize database service."""
        self.client = None
        self.db = None
        self.collection = None
        self.users_collection = None
    
    async def connect(self) -> bool:
        """
        Establish connection to MongoDB.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            self.client = AsyncIOMotorClient(settings.MONGODB_URL)
            # Test connection
            await self.client.admin.command('ping')
            
            self.db = self.client[settings.DATABASE_NAME]
            self.collection = self.db[self.COLLECTION_NAME]
            self.users_collection = self.db[self.USERS_COLLECTION_NAME]
            
            # Create indexes for better query performance
            await self._create_indexes()
            
            logger.info(f"Connected to MongoDB: {settings.DATABASE_NAME}")
            return True
            
        except ConnectionFailure as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            return False
        except Exception as e:
            logger.error(f"Database connection error: {e}")
            return False
    
    async def _create_indexes(self) -> None:
        """Create database indexes for optimized queries."""
        try:
            await self.collection.create_index("user_id")
            await self.collection.create_index("created_at")
            await self.collection.create_index([("user_id", 1), ("created_at", -1)])
            await self.users_collection.create_index("email", unique=True)
            await self.users_collection.create_index("patient_id", unique=True)
            logger.info("Database indexes created successfully")
        except Exception as e:
            logger.warning(f"Error creating indexes: {e}")
    
    async def disconnect(self) -> None:
        """Close database connection."""
        if self.client:
            self.client.close()
            logger.info("Database connection closed")
    
    def is_connected(self) -> bool:
        """Check if database is connected."""
        return self.client is not None and self.collection is not None

    @staticmethod
    def _hash_password(password: str, salt: str) -> str:
        return hashlib.sha256(f"{salt}:{password}".encode("utf-8")).hexdigest()

    async def register_user(
        self,
        full_name: str,
        email: str,
        password: str,
        age: Optional[int] = None,
        gender: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Register a new user account."""
        if not self.is_connected() or self.users_collection is None:
            return None

        email = email.strip().lower()
        existing = await self.users_collection.find_one({"email": email})
        if existing:
            return None

        salt = secrets.token_hex(16)
        patient_id = f"PT-{secrets.token_hex(4).upper()}"
        user = {
            "full_name": full_name.strip(),
            "email": email,
            "patient_id": patient_id,
            "age": age,
            "gender": gender,
            "password_salt": salt,
            "password_hash": self._hash_password(password, salt),
            "created_at": datetime.utcnow(),
        }
        result = await self.users_collection.insert_one(user)
        user["_id"] = result.inserted_id
        return user

    async def authenticate_user(self, email: str, password: str) -> Optional[Dict[str, Any]]:
        """Validate user credentials and return user document when valid."""
        if not self.is_connected() or self.users_collection is None:
            return None

        email = email.strip().lower()
        user = await self.users_collection.find_one({"email": email})
        if not user:
            return None

        expected = self._hash_password(password, user.get("password_salt", ""))
        if expected != user.get("password_hash"):
            return None
        return user

    async def get_user_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Fetch user by id string."""
        if not self.is_connected() or self.users_collection is None:
            return None

        try:
            return await self.users_collection.find_one({"_id": ObjectId(user_id)})
        except Exception:
            return None

    @staticmethod
    def serialize_user_profile(user: Dict[str, Any]) -> Dict[str, Any]:
        """Return safe user profile payload."""
        return {
            "id": str(user.get("_id")),
            "full_name": user.get("full_name", "Patient"),
            "email": user.get("email"),
            "patient_id": user.get("patient_id"),
            "age": user.get("age"),
            "gender": user.get("gender"),
            "created_at": user.get("created_at"),
        }
    
    async def save_health_data(
        self, 
        data: HealthDataInput, 
        prediction: Optional[Dict] = None
    ) -> str:
        """
        Save health data record to database.
        
        Args:
            data: Health data input from user
            prediction: Optional prediction results to store
            
        Returns:
            str: ID of the inserted record
        """
        if not self.is_connected():
            logger.warning("Database not connected. Data not saved.")
            return ""
        
        try:
            record = {
                "user_id": data.user_id or "default_user",
                "stress_level": data.stress_level,
                "sleep_hours": data.sleep_hours,
                "heart_rate": data.heart_rate,
                "activity_level": data.activity_level,
                "weather_pressure": data.weather_pressure,
                "aqi": data.aqi,
                "had_migraine": data.had_migraine,
                "notes": data.notes,
                "prediction": prediction,
                "created_at": datetime.utcnow()
            }
            
            result = await self.collection.insert_one(record)
            logger.info(f"Saved health record: {result.inserted_id}")
            return str(result.inserted_id)
            
        except Exception as e:
            logger.error(f"Error saving health data: {e}")
            return ""
    
    async def get_user_history(
        self, 
        user_id: str = "default_user",
        days: int = 30,
        limit: int = 100
    ) -> HistoryResponse:
        """
        Get user's health data history.
        
        Args:
            user_id: User identifier
            days: Number of days to look back
            limit: Maximum number of records to return
            
        Returns:
            HistoryResponse with records and analytics
        """
        if not self.is_connected():
            logger.warning("Database not connected.")
            return HistoryResponse(
                user_id=user_id,
                total_records=0,
                migraine_count=0,
                average_risk=0.0,
                records=[],
                trigger_frequency={},
                trends={}
            )
        
        try:
            # Calculate date range
            start_date = datetime.utcnow() - timedelta(days=days)
            
            # Query records
            cursor = self.collection.find({
                "user_id": user_id,
                "created_at": {"$gte": start_date}
            }).sort("created_at", -1).limit(limit)
            
            records = []
            migraine_count = 0
            total_risk = 0.0
            trigger_counts: Dict[str, int] = {}
            
            async for doc in cursor:
                record = HealthDataRecord(
                    id=str(doc["_id"]),
                    user_id=doc["user_id"],
                    stress_level=doc["stress_level"],
                    sleep_hours=doc["sleep_hours"],
                    heart_rate=doc["heart_rate"],
                    activity_level=doc["activity_level"],
                    weather_pressure=doc["weather_pressure"],
                    aqi=doc["aqi"],
                    had_migraine=doc.get("had_migraine"),
                    notes=doc.get("notes"),
                    prediction=doc.get("prediction"),
                    created_at=doc["created_at"]
                )
                records.append(record)
                
                # Count migraines
                if doc.get("had_migraine"):
                    migraine_count += 1
                
                # Accumulate risk scores
                if doc.get("prediction") and "probability" in doc["prediction"]:
                    total_risk += doc["prediction"]["probability"]
                
                # Count triggers
                if doc.get("prediction") and "triggers" in doc["prediction"]:
                    for trigger in doc["prediction"]["triggers"]:
                        # Extract trigger type (e.g., "High stress" -> "stress")
                        trigger_key = trigger.split("(")[0].strip().lower()
                        trigger_counts[trigger_key] = trigger_counts.get(trigger_key, 0) + 1
            
            # Calculate averages
            total_records = len(records)
            average_risk = (total_risk / total_records) if total_records > 0 else 0.0
            
            # Calculate trends
            trends = await self._calculate_trends(records)
            
            return HistoryResponse(
                user_id=user_id,
                total_records=total_records,
                migraine_count=migraine_count,
                average_risk=round(average_risk, 3),
                records=records,
                trigger_frequency=trigger_counts,
                trends=trends
            )
            
        except Exception as e:
            logger.error(f"Error fetching user history: {e}")
            return HistoryResponse(
                user_id=user_id,
                total_records=0,
                migraine_count=0,
                average_risk=0.0,
                records=[],
                trigger_frequency={},
                trends={}
            )
    
    async def _calculate_trends(self, records: List[HealthDataRecord]) -> Dict[str, Any]:
        """
        Calculate trend data from records.
        
        Args:
            records: List of health data records
            
        Returns:
            Dictionary with trend analysis
        """
        if not records:
            return {}
        
        # Group by date
        daily_data: Dict[str, Dict] = {}
        
        for record in records:
            date_key = record.created_at.strftime("%Y-%m-%d")
            if date_key not in daily_data:
                daily_data[date_key] = {
                    "stress": [],
                    "sleep": [],
                    "heart_rate": [],
                    "risk": []
                }
            
            daily_data[date_key]["stress"].append(record.stress_level)
            daily_data[date_key]["sleep"].append(record.sleep_hours)
            daily_data[date_key]["heart_rate"].append(record.heart_rate)
            
            if record.prediction and "probability" in record.prediction:
                daily_data[date_key]["risk"].append(record.prediction["probability"])
        
        # Calculate daily averages
        trends = {
            "dates": [],
            "avg_stress": [],
            "avg_sleep": [],
            "avg_heart_rate": [],
            "avg_risk": []
        }
        
        for date_key in sorted(daily_data.keys()):
            data = daily_data[date_key]
            trends["dates"].append(date_key)
            trends["avg_stress"].append(sum(data["stress"]) / len(data["stress"]) if data["stress"] else 0)
            trends["avg_sleep"].append(sum(data["sleep"]) / len(data["sleep"]) if data["sleep"] else 0)
            trends["avg_heart_rate"].append(sum(data["heart_rate"]) / len(data["heart_rate"]) if data["heart_rate"] else 0)
            trends["avg_risk"].append(sum(data["risk"]) / len(data["risk"]) if data["risk"] else 0)
        
        return trends
    
    async def update_migraine_status(
        self, 
        record_id: str, 
        had_migraine: bool,
        user_id: str,
    ) -> bool:
        """
        Update migraine status for a record.
        
        Args:
            record_id: Record ID to update
            had_migraine: Whether migraine occurred
            user_id: User identifier to enforce record ownership
            
        Returns:
            bool: True if update successful
        """
        if not self.is_connected():
            return False
        
        try:
            result = await self.collection.update_one(
                {"_id": ObjectId(record_id), "user_id": user_id},
                {"$set": {"had_migraine": had_migraine}}
            )
            return result.modified_count > 0
        except Exception as e:
            logger.error(f"Error updating migraine status: {e}")
            return False
    
    async def get_statistics(self, user_id: str = "default_user") -> Dict[str, Any]:
        """
        Get aggregated statistics for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            Dictionary with various statistics
        """
        if not self.is_connected():
            return {}
        
        try:
            pipeline = [
                {"$match": {"user_id": user_id}},
                {"$group": {
                    "_id": None,
                    "total_records": {"$sum": 1},
                    "avg_stress": {"$avg": "$stress_level"},
                    "avg_sleep": {"$avg": "$sleep_hours"},
                    "avg_heart_rate": {"$avg": "$heart_rate"},
                    "migraine_count": {
                        "$sum": {"$cond": [{"$eq": ["$had_migraine", True]}, 1, 0]}
                    }
                }}
            ]
            
            cursor = self.collection.aggregate(pipeline)
            result = await cursor.to_list(length=1)
            
            if result:
                stats = result[0]
                stats.pop("_id", None)
                return stats
            return {}
            
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {}


# Singleton instance
db_service = DatabaseService()
