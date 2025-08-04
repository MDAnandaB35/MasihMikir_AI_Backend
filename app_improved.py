from flask import Flask, jsonify
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import logging
import os
import atexit

from config import Config
from database import db
from routes import api

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_app():
    """Application factory pattern"""
    app = Flask(__name__)
    
    # Configure app
    app.config['JSON_SORT_KEYS'] = False
    app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True
    
    # Security headers
    @app.after_request
    def add_security_headers(response):
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers['X-XSS-Protection'] = '1; mode=block'
        response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
        return response
    
    # CORS configuration
    CORS(app, origins=['*'], methods=['GET', 'POST', 'PUT', 'DELETE'])
    
    # Rate limiting
    limiter = Limiter(
        app=app,
        key_func=get_remote_address,
        default_limits=["200 per day", "50 per hour"]
    )
    
    # Register blueprints
    app.register_blueprint(api, url_prefix='/api/v1')
    
    # Health check endpoint
    @app.route('/health', methods=['GET'])
    def health_check():
        """Root health check endpoint"""
        try:
            # Test database connection
            db.client.admin.command('ping')
            return jsonify({
                'status': 'healthy',
                'service': 'MasihMeeting AI Backend',
                'version': '2.0.0',
                'database': 'connected'
            })
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return jsonify({
                'status': 'unhealthy',
                'error': str(e)
            }), 500
    
    # Root endpoint
    @app.route('/', methods=['GET'])
    def root():
        """Root endpoint with API information"""
        return jsonify({
            'service': 'MasihMeeting AI Backend',
            'version': '2.0.0',
            'description': 'AI-powered meeting transcription and analysis service',
            'endpoints': {
                'health': '/health',
                'api_docs': '/api/v1/health',
                'transcription': {
                    'youtube': '/api/v1/youtube_subtitle_transcribe',
                    'audio': '/api/v1/whisper_file_transcribe',
                    'video': '/api/v1/video_to_audio_transcribe'
                },
                'analysis': {
                    'summarize': '/api/v1/summarize_transcription',
                    'quiz': '/api/v1/generate_quiz',
                    'question': '/api/v1/ask_question'
                },
                'data': {
                    'transcription': '/api/v1/get_transcription/<id>',
                    'chat_history': '/api/v1/get_chat_history'
                }
            }
        })
    
    # Error handlers
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({
            'error': 'Endpoint not found',
            'message': 'The requested endpoint does not exist'
        }), 404
    
    @app.errorhandler(405)
    def method_not_allowed(error):
        return jsonify({
            'error': 'Method not allowed',
            'message': 'The HTTP method is not supported for this endpoint'
        }), 405
    
    @app.errorhandler(500)
    def internal_error(error):
        logger.error(f"Internal server error: {error}")
        return jsonify({
            'error': 'Internal server error',
            'message': 'An unexpected error occurred'
        }), 500
    
    @app.errorhandler(Exception)
    def handle_exception(error):
        logger.error(f"Unhandled exception: {error}")
        return jsonify({
            'error': 'Internal server error',
            'message': 'An unexpected error occurred'
        }), 500
    
    return app

def cleanup():
    """Cleanup function to close database connections"""
    try:
        db.close_connection()
        logger.info("Database connection closed")
    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}")

def main():
    """Main application entry point"""
    try:
        # Validate configuration
        Config.validate_config()
        logger.info("Configuration validated successfully")
        
        # Create app
        app = create_app()
        
        # Register cleanup function
        atexit.register(cleanup)
        
        # Get port from environment or use default
        port = int(os.environ.get('PORT', 6969))
        host = os.environ.get('HOST', '0.0.0.0')
        debug = os.environ.get('DEBUG', 'False').lower() == 'true'
        
        logger.info(f"Starting server on {host}:{port}")
        logger.info(f"Debug mode: {debug}")
        
        # Run app
        app.run(
            host=host,
            port=port,
            debug=debug,
            threaded=True
        )
        
    except Exception as e:
        logger.error(f"Failed to start application: {str(e)}")
        raise

if __name__ == '__main__':
    main() 