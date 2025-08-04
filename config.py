import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Centralized configuration management"""
    
    # OpenAI Configuration
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
    OPENAI_MODEL = os.environ.get('OPENAI_MODEL', 'gpt-3.5-turbo')
    
    # Folder paths
    AUDIO_FOLDER = os.environ.get('AUDIO_FOLDER', 'audio_files')
    VIDEO_FOLDER = os.environ.get('VIDEO_FOLDER', 'video_files')
    TRANSCRIPTS_FOLDER = os.environ.get('TRANSCRIPTS_FOLDER', 'transcripts')
    
    # MongoDB Configuration
    MONGODB_URI = os.environ.get('MONGODB_URI')
    MONGODB_DB = os.environ.get('MONGODB_DB')
    MONGODB_COLLECTION = os.environ.get('MONGODB_COLLECTION')
    
    # OpenRouter AI Model Configuration
    AI_MODEL_API_KEY = os.environ.get('AI_MODEL_API_KEY')
    AI_MODEL_NAME = os.environ.get('AI_MODEL_NAME', 'mistralai/mistral-7b-instruct')
    
    # Helpy API Configuration
    HELPY_API_KEY = os.environ.get('HELPY_API_KEY')
    
    # Application Configuration
    MAX_TRANSCRIPTION_LENGTH = 100000
    MAX_SUMMARY_LENGTH = 50000
    CONVERSATION_HISTORY_LIMIT = 5
    
    # API Configuration
    OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
    HELPY_URL = "https://mlapi.run/9331793d-efda-4839-8f97-ff66f7eaf605/v1/chat/completions"
    
    @classmethod
    def validate_config(cls):
        """Validate required configuration"""
        required_vars = [
            'MONGODB_URI',
            'MONGODB_DB', 
            'MONGODB_COLLECTION',
            'AI_MODEL_API_KEY'
        ]
        
        missing_vars = [var for var in required_vars if not getattr(cls, var)]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
        
        return True 