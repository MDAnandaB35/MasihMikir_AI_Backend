# MasihMeeting AI Backend - Improved Version

A robust, production-ready AI-powered meeting transcription and analysis service built with Flask.

## 🚀 Features

- **YouTube Video Transcription**: Extract and transcribe YouTube videos using subtitles
- **Audio File Transcription**: Transcribe audio files using OpenAI Whisper API
- **Video to Audio Conversion**: Convert video files to audio and transcribe
- **AI-Powered Summarization**: Generate detailed meeting summaries
- **Quiz Generation**: Create multiple-choice questions from transcriptions
- **Interactive Q&A**: Ask questions about transcriptions with conversation memory
- **MongoDB Integration**: Persistent storage with proper indexing
- **Rate Limiting**: API protection against abuse
- **Security Headers**: Enhanced security with proper HTTP headers
- **Comprehensive Logging**: Detailed logging for monitoring and debugging
- **Input Validation**: Robust validation for all inputs
- **Error Handling**: Standardized error responses

## 📁 Project Structure

```
AI_Backend/
├── config.py              # Configuration management
├── database.py            # MongoDB operations
├── services.py            # Business logic
├── routes.py              # API endpoints
├── utils.py               # Utility functions
├── app_improved.py        # Main application
├── requirements_improved.txt
├── README_IMPROVED.md
└── .env                   # Environment variables
```

## 🛠️ Installation

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd AI_Backend
   ```

2. **Create virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements_improved.txt
   ```

4. **Set up environment variables**

   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. **Install ffmpeg** (for video processing)

   ```bash
   # Ubuntu/Debian
   sudo apt update && sudo apt install ffmpeg

   # macOS
   brew install ffmpeg

   # Windows
   # Download from https://ffmpeg.org/download.html
   ```

## ⚙️ Configuration

Create a `.env` file with the following variables:

```env
# MongoDB Configuration
MONGODB_URI=mongodb://localhost:27017
MONGODB_DB=masihmeeting
MONGODB_COLLECTION=transcriptions

# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-3.5-turbo

# OpenRouter AI Configuration
AI_MODEL_API_KEY=your_openrouter_api_key
AI_MODEL_NAME=mistralai/mistral-7b-instruct

# Helpy API Configuration (optional)
HELPY_API_KEY=your_helpy_api_key

# Folder Configuration
AUDIO_FOLDER=audio_files
VIDEO_FOLDER=video_files
TRANSCRIPTS_FOLDER=transcripts

# Application Configuration
PORT=6969
HOST=0.0.0.0
DEBUG=False
```

## 🚀 Running the Application

### Development Mode

```bash
python app_improved.py
```

### Production Mode

```bash
export FLASK_ENV=production
python app_improved.py
```

### Using Gunicorn (Recommended for Production)

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:6969 app_improved:create_app()
```

## 📚 API Endpoints

### Health Check

- `GET /health` - Application health status
- `GET /api/v1/health` - API health status

### Transcription

- `POST /api/v1/youtube_subtitle_transcribe` - Transcribe YouTube video
- `POST /api/v1/whisper_file_transcribe` - Transcribe audio file
- `POST /api/v1/video_to_audio_transcribe` - Convert video to audio and transcribe

### Analysis

- `POST /api/v1/summarize_transcription` - Generate summary
- `POST /api/v1/generate_quiz` - Generate quiz questions
- `POST /api/v1/ask_question` - Ask questions about transcription

### Data Retrieval

- `GET /api/v1/get_transcription/<id>` - Get transcription by ID
- `POST /api/v1/get_chat_history` - Get chat history

## 📝 API Usage Examples

### Transcribe YouTube Video

```bash
curl -X POST http://localhost:6969/api/v1/youtube_subtitle_transcribe \
  -H "Content-Type: application/json" \
  -d '{"url": "https://www.youtube.com/watch?v=VIDEO_ID"}'
```

### Transcribe Audio File

```bash
curl -X POST http://localhost:6969/api/v1/whisper_file_transcribe \
  -H "Content-Type: application/json" \
  -d '{"filename": "audio.mp3", "language": "en"}'
```

### Generate Summary

```bash
curl -X POST http://localhost:6969/api/v1/summarize_transcription \
  -H "Content-Type: application/json" \
  -d '{"_id": "transcription_id"}'
```

### Generate Quiz

```bash
curl -X POST http://localhost:6969/api/v1/generate_quiz \
  -H "Content-Type: application/json" \
  -d '{"_id": "transcription_id", "quiz_level": "medium"}'
```

### Ask Question

```bash
curl -X POST http://localhost:6969/api/v1/ask_question \
  -H "Content-Type: application/json" \
  -d '{"_id": "transcription_id", "question": "What was discussed about the budget?"}'
```

## 🔧 Improvements Made

### 1. **Code Organization**

- ✅ Split monolithic file into modules
- ✅ Separated concerns (config, database, services, routes)
- ✅ Added proper imports and dependencies

### 2. **Error Handling**

- ✅ Standardized error responses
- ✅ Comprehensive exception handling
- ✅ Input validation for all endpoints
- ✅ Proper HTTP status codes

### 3. **Security**

- ✅ Added CORS support
- ✅ Rate limiting to prevent abuse
- ✅ Security headers
- ✅ Input sanitization
- ✅ Path traversal protection

### 4. **Database Operations**

- ✅ Centralized database management
- ✅ Connection pooling
- ✅ Proper error handling
- ✅ Connection cleanup

### 5. **Configuration Management**

- ✅ Centralized configuration
- ✅ Environment variable validation
- ✅ Default values
- ✅ Configuration validation

### 6. **Logging and Monitoring**

- ✅ Comprehensive logging
- ✅ Health check endpoints
- ✅ Application metrics
- ✅ Error tracking

### 7. **Code Quality**

- ✅ Type hints
- ✅ Docstrings
- ✅ Consistent code style
- ✅ Modular design

### 8. **Performance**

- ✅ Async operations where appropriate
- ✅ Connection pooling
- ✅ Request timeouts
- ✅ Resource cleanup

## 🧪 Testing

```bash
# Install test dependencies
pip install pytest pytest-flask

# Run tests
pytest

# Run with coverage
pytest --cov=.
```

## 📊 Monitoring

The application includes:

- Health check endpoints
- Comprehensive logging
- Error tracking
- Performance metrics

## 🔒 Security Considerations

- Rate limiting prevents API abuse
- Input validation and sanitization
- Security headers protect against common attacks
- CORS configuration for cross-origin requests
- Environment variable management

## 🚀 Deployment

### Docker (Recommended)

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements_improved.txt .
RUN pip install -r requirements_improved.txt

COPY . .
EXPOSE 6969

CMD ["python", "app_improved.py"]
```

### Environment Variables for Production

```env
FLASK_ENV=production
DEBUG=False
MONGODB_URI=mongodb://production-db:27017
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🆘 Support

For support and questions:

- Create an issue in the repository
- Check the documentation
- Review the API examples

## 🔄 Migration from Original

To migrate from the original `app.py`:

1. **Backup your data**
2. **Update environment variables**
3. **Install new dependencies**
4. **Test the new endpoints**
5. **Update your frontend to use new API structure**

The new API maintains backward compatibility while adding new features and improvements.
