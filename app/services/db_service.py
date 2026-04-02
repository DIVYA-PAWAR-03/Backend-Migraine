"""
Database Service for MongoDB operations.

Handles all database interactions for storing and retrieving
user health data, predictions, and historical trends.
"""

import logging
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
    
    def __init__(self):
        """Initialize database service."""
        self.client: Optional[AsyncIOMotorClient] = None
        self.db = None
        self.collection = None
    
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
        had_migraine: bool
    ) -> bool:
        """
        Update migraine status for a record.
        
        Args:
            record_id: Record ID to update
            had_migraine: Whether migraine occurred
            
        Returns:
            bool: True if update successful
        """
        if not self.is_connected():
            return False
        
        try:
            result = await self.collection.update_one(
                {"_id": ObjectId(record_id)},
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
