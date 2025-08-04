from pymongo import MongoClient
from bson import ObjectId
from datetime import datetime
from typing import Optional, List, Dict, Any
from config import Config

class DatabaseManager:
    """MongoDB database manager with proper error handling"""
    
    def __init__(self):
        self.client = None
        self.transcription_collection = None
        self.chat_logs_collection = None
        self._initialize_connection()
    
    def _initialize_connection(self):
        """Initialize MongoDB connection"""
        if not Config.MONGODB_URI:
            raise ValueError("MongoDB URI not configured")
        
        try:
            self.client = MongoClient(Config.MONGODB_URI)
            self.transcription_collection = self.client[Config.MONGODB_DB][Config.MONGODB_COLLECTION]
            self.chat_logs_collection = self.client[Config.MONGODB_DB]["chat_logs"]
            
            # Test connection
            self.client.admin.command('ping')
        except Exception as e:
            raise ConnectionError(f"Failed to connect to MongoDB: {str(e)}")
    
    def save_transcription(self, data: Dict[str, Any]) -> Optional[str]:
        """Save transcription data to MongoDB"""
        try:
            if self.transcription_collection is None:
                raise ValueError("Transcription collection not initialized")
            
            result = self.transcription_collection.insert_one(data)
            return str(result.inserted_id)
        except Exception as e:
            raise Exception(f"Failed to save transcription: {str(e)}")
    
    def save_chat_log(self, transcription_id: str, question: str, answer: str) -> Optional[str]:
        """Save chat log to MongoDB"""
        try:
            if self.chat_logs_collection is None:
                raise ValueError("Chat logs collection not initialized")
            
            chat_log = {
                'transcription_id': transcription_id,
                'question': question,
                'answer': answer,
                'created_at': datetime.utcnow()
            }
            result = self.chat_logs_collection.insert_one(chat_log)
            return str(result.inserted_id)
        except Exception as e:
            raise Exception(f"Failed to save chat log: {str(e)}")
    
    def get_transcription(self, mongo_id: str) -> Optional[Dict[str, Any]]:
        """Get transcription by ID"""
        try:
            if self.transcription_collection is None:
                raise ValueError("Transcription collection not initialized")
            
            doc = self.transcription_collection.find_one({'_id': ObjectId(mongo_id)})
            if doc:
                doc['_id'] = str(doc['_id'])
            return doc
        except Exception as e:
            raise Exception(f"Failed to get transcription: {str(e)}")
    
    def update_transcription(self, mongo_id: str, update_data: Dict[str, Any]) -> bool:
        """Update transcription document"""
        try:
            if self.transcription_collection is None:
                raise ValueError("Transcription collection not initialized")
            
            result = self.transcription_collection.update_one(
                {'_id': ObjectId(mongo_id)}, 
                {'$set': update_data}
            )
            return result.modified_count > 0
        except Exception as e:
            raise Exception(f"Failed to update transcription: {str(e)}")
    
    def get_chat_history(self, transcription_id: str) -> List[Dict[str, Any]]:
        """Get chat history for a transcription"""
        try:
            if self.chat_logs_collection is None:
                raise ValueError("Chat logs collection not initialized")
            
            chat_logs = list(self.chat_logs_collection.find(
                {'transcription_id': transcription_id},
                {'_id': 1, 'question': 1, 'answer': 1, 'created_at': 1}
            ).sort('created_at', 1))
            
            # Convert ObjectId to string for JSON serialization
            for log in chat_logs:
                log['_id'] = str(log['_id'])
                log['created_at'] = log['created_at'].isoformat()
            
            return chat_logs
        except Exception as e:
            raise Exception(f"Failed to get chat history: {str(e)}")
    
    def close_connection(self):
        """Close MongoDB connection"""
        if self.client is not None:
            self.client.close()

# Global database instance
db = DatabaseManager() 