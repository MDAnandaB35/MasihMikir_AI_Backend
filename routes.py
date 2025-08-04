from flask import Blueprint, request, jsonify
from functools import wraps
import logging
from typing import Dict, Any

from config import Config
from database import db
from services import TranscriptionService, TranscriptionManager
from utils import validate_mongo_id, validate_youtube_url, validate_filename, validate_language_code

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create blueprint
api = Blueprint('api', __name__)

def handle_errors(f):
    """Decorator for standardized error handling"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except ValueError as e:
            logger.warning(f"Validation error: {str(e)}")
            return jsonify({'error': str(e)}), 400
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            return jsonify({'error': 'Internal server error'}), 500
    return decorated_function

def validate_json_data(required_fields: list):
    """Decorator to validate JSON request data"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not request.is_json:
                return jsonify({'error': 'Content-Type must be application/json'}), 400
            
            data = request.get_json()
            if not data:
                return jsonify({'error': 'No JSON data provided'}), 400
            
            missing_fields = [field for field in required_fields if field not in data or not data[field]]
            if missing_fields:
                return jsonify({'error': f'Missing required fields: {", ".join(missing_fields)}'}), 400
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator

@api.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        # Test database connection
        db.client.admin.command('ping')
        return jsonify({
            'status': 'healthy',
            'database': 'connected',
            'config_validated': Config.validate_config()
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500

@api.route('/summarize_transcription', methods=['POST'])
@handle_errors
@validate_json_data(['_id'])
def summarize_transcription():
    """Summarize a transcription from MongoDB by _id"""
    data = request.get_json()
    mongo_id = data['_id']
    
    if not validate_mongo_id(mongo_id):
        return jsonify({'error': 'Invalid MongoDB ID format'}), 400
    
    # Get transcription
    doc = db.get_transcription(mongo_id)
    if not doc:
        return jsonify({'error': f'No document found for _id: {mongo_id}'}), 404
    
    transcription = doc.get('transcription')
    if not transcription:
        return jsonify({'error': 'No transcription found in document'}), 400
    
    # Generate summary
    summary = TranscriptionManager.summarize_and_save(mongo_id)
    
    return jsonify({
        '_id': mongo_id,
        'transcription': transcription,
        'summary': summary
    })

@api.route('/youtube_subtitle_transcribe', methods=['POST'])
@handle_errors
@validate_json_data(['url'])
def youtube_subtitle_transcribe():
    """Transcribe YouTube video using subtitles"""
    data = request.get_json()
    youtube_url = data['url']
    
    if not validate_youtube_url(youtube_url):
        return jsonify({'error': 'Invalid YouTube URL format'}), 400
    
    # Transcribe video
    result = TranscriptionService.transcribe_youtube_video(youtube_url)
    mongo_id = result['mongo_id']
    
    # Generate summary
    summary = TranscriptionManager.summarize_and_save(mongo_id) if mongo_id else None
    
    return jsonify({
        '_id': mongo_id,
        'transcription': result['transcription'],
        'summary': summary
    })

@api.route('/whisper_file_transcribe', methods=['POST'])
@handle_errors
@validate_json_data(['filename'])
def whisper_file_transcribe():
    """Transcribe audio file using Whisper API"""
    data = request.get_json()
    filename = data['filename']
    language = data.get('language', 'en')
    
    if not validate_filename(filename):
        return jsonify({'error': 'Invalid filename format'}), 400
    
    if not validate_language_code(language):
        return jsonify({'error': 'Invalid language code'}), 400
    
    # Transcribe audio
    result = TranscriptionService.transcribe_audio_file(filename, language)
    mongo_id = result['mongo_id']
    
    # Generate summary
    summary = TranscriptionManager.summarize_and_save(mongo_id) if mongo_id else None
    
    return jsonify({
        '_id': mongo_id,
        'transcription': result['transcription'],
        'summary': summary
    })

@api.route('/video_to_audio_transcribe', methods=['POST'])
@handle_errors
@validate_json_data(['filename'])
def video_to_audio_transcribe():
    """Convert video to audio and transcribe"""
    data = request.get_json()
    filename = data['filename']
    language = data.get('language', 'en')
    
    if not validate_filename(filename):
        return jsonify({'error': 'Invalid filename format'}), 400
    
    if not validate_language_code(language):
        return jsonify({'error': 'Invalid language code'}), 400
    
    # Transcribe video
    result = TranscriptionService.transcribe_video_file(filename, language)
    mongo_id = result['mongo_id']
    
    # Generate summary
    summary = TranscriptionManager.summarize_and_save(mongo_id) if mongo_id else None
    
    return jsonify({
        '_id': mongo_id,
        'transcription': result['transcription'],
        'summary': summary
    })

@api.route('/generate_quiz', methods=['POST'])
@handle_errors
@validate_json_data(['_id', 'quiz_level'])
def generate_quiz():
    """Generate quiz from transcription"""
    data = request.get_json()
    mongo_id = data['_id']
    quiz_level = data['quiz_level']
    
    if not validate_mongo_id(mongo_id):
        return jsonify({'error': 'Invalid MongoDB ID format'}), 400
    
    if quiz_level not in ['easy', 'medium', 'hard']:
        return jsonify({'error': 'Quiz level must be easy, medium, or hard'}), 400
    
    # Get transcription
    doc = db.get_transcription(mongo_id)
    if not doc:
        return jsonify({'error': f'No document found for _id: {mongo_id}'}), 404
    
    transcription = doc.get('transcription')
    if not transcription:
        return jsonify({'error': 'No transcription found in document'}), 400
    
    # Generate quiz
    mcqs = TranscriptionManager.generate_quiz_and_save(mongo_id, quiz_level)
    
    return jsonify({
        '_id': mongo_id,
        'transcription': transcription,
        'mcqs': mcqs
    })

@api.route('/ask_question', methods=['POST'])
@handle_errors
@validate_json_data(['_id', 'question'])
def ask_question():
    """Ask a question about a transcription"""
    data = request.get_json()
    mongo_id = data['_id']
    question = data['question']
    
    if not validate_mongo_id(mongo_id):
        return jsonify({'error': 'Invalid MongoDB ID format'}), 400
    
    if not question or len(question.strip()) == 0:
        return jsonify({'error': 'Question cannot be empty'}), 400
    
    # Get transcription
    doc = db.get_transcription(mongo_id)
    if not doc:
        return jsonify({'error': f'No document found for _id: {mongo_id}'}), 404
    
    transcription = doc.get('transcription')
    if not transcription:
        return jsonify({'error': 'No transcription found in document'}), 400
    
    # Ask question and save
    result = TranscriptionManager.ask_question_and_save(mongo_id, question)
    
    return jsonify({
        '_id': mongo_id,
        'transcription': transcription,
        'question': question,
        'answer': result['answer'],
        'chat_log_id': result['chat_log_id'],
        'conversation_history_count': result['conversation_history_count']
    })

@api.route('/get_chat_history', methods=['POST'])
@handle_errors
@validate_json_data(['_id'])
def get_chat_history():
    """Get chat history for a transcription"""
    data = request.get_json()
    mongo_id = data['_id']
    
    if not validate_mongo_id(mongo_id):
        return jsonify({'error': 'Invalid MongoDB ID format'}), 400
    
    # Verify transcription exists
    doc = db.get_transcription(mongo_id)
    if not doc:
        return jsonify({'error': f'No document found for _id: {mongo_id}'}), 404
    
    # Get chat history
    chat_history = db.get_chat_history(mongo_id)
    
    return jsonify({
        '_id': mongo_id,
        'chat_history': chat_history
    })

@api.route('/get_transcription/<mongo_id>', methods=['GET'])
@handle_errors
def get_transcription(mongo_id):
    """Get transcription by ID"""
    if not validate_mongo_id(mongo_id):
        return jsonify({'error': 'Invalid MongoDB ID format'}), 400
    
    doc = db.get_transcription(mongo_id)
    if not doc:
        return jsonify({'error': f'No document found for _id: {mongo_id}'}), 404
    
    return jsonify(doc)

# Error handlers
@api.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@api.errorhandler(405)
def method_not_allowed(error):
    return jsonify({'error': 'Method not allowed'}), 405

@api.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({'error': 'Internal server error'}), 500 