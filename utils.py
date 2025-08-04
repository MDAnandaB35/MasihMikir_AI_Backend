import os
import re
import json
import requests
from typing import List, Dict, Any, Optional
from config import Config

def validate_mongo_id(mongo_id: str) -> bool:
    """Validate MongoDB ObjectId format"""
    if not mongo_id or not isinstance(mongo_id, str):
        return False
    return bool(re.match(r'^[0-9a-fA-F]{24}$', mongo_id))

def validate_youtube_url(url: str) -> bool:
    """Validate YouTube URL format"""
    if not url or not isinstance(url, str):
        return False
    
    patterns = [
        r'(?:https?://)?(?:www\.)?youtube\.com/watch\?v=[\w-]+',
        r'(?:https?://)?(?:www\.)?youtu\.be/[\w-]+',
        r'(?:https?://)?(?:www\.)?youtube\.com/embed/[\w-]+'
    ]
    
    return any(re.match(pattern, url) for pattern in patterns)

def extract_video_id(youtube_url: str) -> str:
    """Extract video ID from YouTube URL"""
    if not validate_youtube_url(youtube_url):
        raise ValueError('Invalid YouTube URL')
    
    try:
        if 'v=' in youtube_url:
            video_id = youtube_url.split('v=')[1].split('&')[0]
        elif 'youtu.be/' in youtube_url:
            video_id = youtube_url.split('youtu.be/')[1].split('?')[0]
        elif 'embed/' in youtube_url:
            video_id = youtube_url.split('embed/')[1].split('?')[0]
        else:
            raise ValueError('Could not extract video ID')
        
        if not video_id or len(video_id) != 11:
            raise ValueError('Invalid video ID format')
        
        return video_id
    except Exception as e:
        raise ValueError(f'Could not extract video ID: {str(e)}')

def validate_filename(filename: str) -> bool:
    """Validate filename format"""
    if not filename or not isinstance(filename, str):
        return False
    
    # Check for valid characters and no path traversal
    invalid_chars = ['<', '>', ':', '"', '|', '?', '*', '\\', '/']
    return not any(char in filename for char in invalid_chars)

def validate_language_code(language: str) -> bool:
    """Validate language code format"""
    if not language or not isinstance(language, str):
        return False
    
    # Basic ISO 639-1 language code validation
    return bool(re.match(r'^[a-z]{2}$', language.lower()))

def truncate_text(text: str, max_length: int) -> str:
    """Truncate text to specified length"""
    if not text or len(text) <= max_length:
        return text
    return text[:max_length]

def format_time(seconds: float) -> str:
    """Format seconds to MM:SS format"""
    minutes = int(seconds) // 60
    secs = int(seconds) % 60
    return f"{minutes:02d}:{secs:02d}"

def format_transcript(transcript_data: List[Dict[str, Any]]) -> str:
    """Format transcript data with timestamps"""
    if not transcript_data:
        return ""
    
    lines = []
    for snippet in transcript_data:
        # Extract values from dict or object
        start = snippet['start'] if isinstance(snippet, dict) else snippet.start
        duration = snippet['duration'] if isinstance(snippet, dict) else snippet.duration
        text = snippet['text'] if isinstance(snippet, dict) else snippet.text

        end = start + duration
        timestamp = f"[{format_time(start)} - {format_time(end)}]"
        lines.append(f"{timestamp} {text}")
    
    return "\n".join(lines)

def make_openrouter_request(prompt: str, model: str = None, temperature: float = 0.5, max_tokens: int = 1000) -> str:
    """Make request to OpenRouter API"""
    if not Config.AI_MODEL_API_KEY:
        raise ValueError("OpenRouter API key not configured")
    
    model = model or Config.AI_MODEL_NAME
    
    headers = {
        "Authorization": f"Bearer {Config.AI_MODEL_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    
    try:
        response = requests.post(
            Config.OPENROUTER_URL, 
            headers=headers, 
            data=json.dumps(payload),
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"].strip()
        else:
            raise Exception(f"OpenRouter API error {response.status_code}: {response.text}")
    
    except requests.exceptions.Timeout:
        raise Exception("OpenRouter API request timed out")
    except requests.exceptions.RequestException as e:
        raise Exception(f"OpenRouter API request failed: {str(e)}")

def build_conversation_context(conversation_history: List[Dict[str, str]], limit: int = None) -> str:
    """Build conversation context from history"""
    if not conversation_history:
        return ""
    
    limit = limit or Config.CONVERSATION_HISTORY_LIMIT
    recent_history = conversation_history[-limit:]
    
    context_lines = ["\n\nPrevious conversation:"]
    for i, qa in enumerate(recent_history, 1):
        context_lines.append(f"Q{i}: {qa['question']}")
        context_lines.append(f"A{i}: {qa['answer']}\n")
    
    return "\n".join(context_lines)

def sanitize_input(text: str) -> str:
    """Basic input sanitization"""
    if not text:
        return ""
    
    # Remove potentially dangerous characters
    dangerous_chars = ['<script>', '</script>', 'javascript:', 'data:', 'vbscript:']
    sanitized = text
    for char in dangerous_chars:
        sanitized = sanitized.replace(char.lower(), '')
        sanitized = sanitized.replace(char.upper(), '')
    
    return sanitized.strip()

def ensure_directory_exists(directory: str) -> None:
    """Ensure directory exists, create if it doesn't"""
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True) 