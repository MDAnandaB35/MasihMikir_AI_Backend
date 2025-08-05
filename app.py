# Importing libraries
from flask import Flask, request, jsonify
import os
import tempfile
import requests
import yt_dlp
import openai
import asyncio
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from pymongo import MongoClient
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()
from bson import ObjectId
import json
import subprocess
import re
import logging
import time

app = Flask(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Console handler
        logging.FileHandler('ai_backend.log')  # File handler
    ]
)
logger = logging.getLogger(__name__)

# Initializing environment variables
# OpenAI
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
OPENAI_MODEL = os.environ.get('OPENAI_MODEL')

# Folder paths (Optional)
AUDIO_FOLDER = os.environ.get('AUDIO_FOLDER', 'audio_files')  # Folder for audio files
VIDEO_FOLDER = os.environ.get('VIDEO_FOLDER', 'video_files') # Folder for video files
TRANSCRIPTS_FOLDER = os.environ.get('TRANSCRIPTS_FOLDER', 'transcripts')

# MongoDB
MONGODB_URI = os.environ.get('MONGODB_URI')
MONGODB_DB = os.environ.get('MONGODB_DB')
MONGODB_COLLECTION = os.environ.get('MONGODB_COLLECTION')

# OpenRouter AI Model
AI_MODEL_API_KEY = os.environ.get('AI_MODEL_API_KEY')
AI_MODEL_NAME = os.environ.get('AI_MODEL_NAME')

# Helpy API Key
HELPY_API_KEY = os.environ.get('HELPY_API_KEY')

# AI Model Selection
AI_MODEL_IN_USE = os.environ.get('AI_MODEL_IN_USE', 'openrouter').lower()



ytt_api = YouTubeTranscriptApi()

# MongoDB setup
mongo_client = None
mongo_collection = None
chat_logs_collection = None
if MONGODB_URI:
    mongo_client = MongoClient(MONGODB_URI)
    mongo_collection = mongo_client[MONGODB_DB][MONGODB_COLLECTION]
    chat_logs_collection = mongo_client[MONGODB_DB]["chat_logs"]

# Saving transcription to MongoDB Atlas database
def save_transcription_to_mongodb(data):
    """
    Save transcription data to MongoDB Atlas.
    Args:
        data: Data to insert

    Returns:
        Inserted ID if successful, otherwise None
    """
    if mongo_collection is not None:
        result = mongo_collection.insert_one(data)
        return str(result.inserted_id)
    return None

# Saving chat logs to MongoDB Atlas database
def save_chat_log_to_mongodb(transcription_id, question, answer):
    """
    Save chat log data to MongoDB Atlas chat_logs collection.
    Args:
        transcription_id (str): ID of the original transcription document
        question (str): User's question
        answer (str): AI's answer

    Returns:
        Inserted ID if successful, otherwise None
    """
    if chat_logs_collection is not None:
        chat_log = {
            'transcription_id': transcription_id,
            'question': question,
            'answer': answer,
            'created_at': datetime.utcnow()
        }
        result = chat_logs_collection.insert_one(chat_log)
        return str(result.inserted_id)
    return None

# Helper function to format transcript for LLM
# Without timestamp
# def format_transcript(transcript_data):
#     """
#     Given a list of transcript segments (from YouTubeTranscriptApi),
#     return a single string suitable for OpenAI input.
#     """
#     if len(transcript_data) == 0:
#         return ""
#     if isinstance(transcript_data[0], dict):
#         return " ".join([snippet['text'] for snippet in transcript_data])
#     else:
#         return " ".join([snippet.text for snippet in transcript_data])

# Including timestamp
def format_transcript(transcript_data):
    """
    Converts transcript data to a string with timestamps for each line.
    Works with both dict and object-based transcript formats.
    Example:
    [00:00 - 00:05] Hello world
    """
    lines = []
    for snippet in transcript_data:
        # Extract values from dict or object
        start = snippet['start'] if isinstance(snippet, dict) else snippet.start
        duration = snippet['duration'] if isinstance(snippet, dict) else snippet.duration
        text = snippet['text'] if isinstance(snippet, dict) else snippet.text

        end = start + duration

        # Format start and end timestamps as MM:SS
        def format_time(seconds):
            minutes = int(seconds) // 60
            secs = int(seconds) % 60
            return f"{minutes:02d}:{secs:02d}"

        timestamp = f"[{format_time(start)} - {format_time(end)}]"
        lines.append(f"{timestamp} {text}")
    return "\n".join(lines)


# Helper function to extract video ID from YouTube URL
def extract_video_id(youtube_url):
    """
    Extract video ID from YouTube URL.
    Args:
        youtube_url (str): YouTube URL

    Returns:
        str: Video ID

    Raises:
        ValueError: If URL is invalid or video ID cannot be extracted
    """
    try:
        if 'v=' in youtube_url:
            video_id = youtube_url.split('v=')[1].split('&')[0]
        elif 'youtu.be/' in youtube_url:
            video_id = youtube_url.split('youtu.be/')[1].split('?')[0]
        else:
            raise ValueError('Invalid YouTube URL')
        return video_id
    except Exception:
        raise ValueError('Could not extract video ID')

# Main function to transcribe YouTube video (fetches, formats, and saves to MongoDB)
def transcribe_youtube_video(youtube_url):
    """
    Transcribe YouTube video: fetch transcript, format, and save to MongoDB.
    Args:
        youtube_url (str): YouTube URL
    Returns:
        dict: Dictionary containing transcription and MongoDB ID
    Raises:
        Exception: If transcription fails
    """
    # Extract video ID
    video_id = extract_video_id(youtube_url)
    # Fetch transcript (try Indonesian, then English, then any)
    transcript_data = None
    for lang in ['en', 'id', None]:
        try:
            if lang:
                transcript_data = ytt_api.fetch(video_id, languages=[lang])
            else:
                transcript_data = ytt_api.fetch(video_id)
            break
        except (NoTranscriptFound, TranscriptsDisabled):
            continue
        except Exception as e:
            raise Exception(f'Failed to fetch transcript: {str(e)}')
    if not transcript_data:
        raise Exception('No transcript found for this video')
    # Format for OpenAI
    full_text = format_transcript(transcript_data)
    # Save to MongoDB
    doc = {
        'type': 'youtube',
        'video_id': video_id,
        'url': youtube_url,
        'transcription': full_text,
        'created_at': datetime.utcnow()
    }
    inserted_id = save_transcription_to_mongodb(doc)
    return {
        'mongo_id': inserted_id,
        'transcription': full_text,
    }

# Transcribing audio file using OpenAI Whisper API
# With timestamp
async def transcribe_audio_file_async(filename, language='en'):
    """
    Transcribe audio file using OpenAI Whisper API and return detailed result with timestamps.

    Returns:
        dict: Contains 'text', 'segments', and 'formatted'
    """
    if not OPENAI_API_KEY:
        raise Exception('OpenAI API key not set in OPENAI_API_KEY env variable')

    file_path = os.path.join(AUDIO_FOLDER, filename)
    if not os.path.isfile(file_path):
        raise Exception(f'File not found: {file_path}')

    try:
        with open(file_path, 'rb') as audio_file:
            audio_data = audio_file.read()

        client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY)

        transcript = await client.audio.transcriptions.create(
            model="whisper-1",
            file=(filename, audio_data),
            language=language,
            response_format="verbose_json",
            timestamp_granularities=["segment"],
        )

        # Helper functions
        def format_time(seconds):
            minutes = int(seconds) // 60
            secs = int(seconds) % 60
            return f"{minutes:02d}:{secs:02d}"

        def format_segments(segments):
            lines = []
            for seg in segments:
                start = format_time(seg.start)
                end = format_time(seg.end)
                lines.append(f"[{start} - {end}] {seg.text}")
            return "\n".join(lines)

        formatted_text = format_segments(transcript.segments)

        return {
            "text": transcript.text,
            "segments": [s.model_dump() for s in transcript.segments],
            "formatted": formatted_text
        }

    except Exception as e:
        raise Exception(f'Failed to transcribe audio: {str(e)}')


# Saving to MongoDB
def transcribe_audio_file(filename, language='en'):
    """
    Synchronous wrapper for audio transcription.
    
    Returns:
        str: Formatted transcription with timestamps
    """
    result = asyncio.run(transcribe_audio_file_async(filename, language))

    doc = {
        'type': 'audio',
        'filename': filename,
        'transcription': result['formatted'],
        'created_at': datetime.utcnow()
    }

    inserted_id = save_transcription_to_mongodb(doc)
    return {
            'mongo_id': inserted_id,
            'transcription': result['formatted'],
        }



# Unified summarize_text function with AI model selection
def summarize_text(text):
    """
    Summarize meeting transcript using the selected AI model based on AI_MODEL_IN_USE environment variable.
    """
    start_time = time.time()
    logger.info(f"Starting text summarization using AI model: {AI_MODEL_IN_USE.upper()}")
    
    prompt = (
        "You are a professional meeting summarizer. "
        "Read the following meeting transcript and produce a detailed, structured summary. "
        "Organize the summary into bullet points grouped by main topics. "
        "For each topic, include sub-bullets for key decisions, action items, and important discussions. "
        "Be as specific and complete as possible, including names or roles if mentioned. "
        "Make the summary clear and useful for someone who did not attend the meeting. "
        "(IMPORTANT!) If timestamps are present in the transcript, you MUST use them to group and label each section accordingly. Each main topic should include the relevant timestamp(s) where the discussion occurred."
        "(CRITICAL!) You MUST respond in the EXACT SAME LANGUAGE as the input text. Do not translate or change languages."
        "\n\nMeeting transcript:\n" + text
    )

    if AI_MODEL_IN_USE == 'openai':
        if not OPENAI_API_KEY:
            raise Exception('OpenAI API key not set in OPENAI_API_KEY env variable')
        
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant. You MUST always respond in the same language as the input text provided by the user. Do not translate or change languages."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=4096,
            temperature=0.5,
        )
        result = response.choices[0].message.content.strip()
    
    elif AI_MODEL_IN_USE == 'helpy':
        if not HELPY_API_KEY:
            raise Exception('Helpy API key not set in HELPY_API_KEY env variable')
        
        url = "https://mlapi.run/9331793d-efda-4839-8f97-ff66f7eaf605/v1/chat/completions"
        payload = {
            "model": "helpy-v-reasoning-c",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ],
        }
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "Authorization": f"Bearer {HELPY_API_KEY}"
        }
        response = requests.post(url, json=payload, headers=headers)
        if response.status_code == 200:
            result = response.json()
            result = result["choices"][0]["message"]["content"].strip()
        else:
            raise Exception(f"Error {response.status_code}: {response.text}")
    
    elif AI_MODEL_IN_USE == 'openrouter':
        if not AI_MODEL_API_KEY:
            raise Exception('OpenRouter API key not set in AI_MODEL_API_KEY env variable')
        
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {AI_MODEL_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": AI_MODEL_NAME,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant. You MUST always respond in the same language as the input text provided by the user. Do not translate or change languages."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.5
        }
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        if response.status_code == 200:
            result = response.json()
            result = result["choices"][0]["message"]["content"].strip()
        else:
            raise Exception(f"Error {response.status_code}: {response.text}")
    
    else:
        raise Exception(f'Unsupported AI model: {AI_MODEL_IN_USE}. Supported values: openai, helpy, openrouter')
    
    end_time = time.time()
    processing_time = end_time - start_time
    logger.info(f"Text summarization completed in {processing_time:.2f} seconds using {AI_MODEL_IN_USE.upper()}")
    
    return result




# Unified generate_mcqs function with AI model selection
def generate_mcqs(text_content, quiz_level):
    """
    Generate quiz from text content using the selected AI model based on AI_MODEL_IN_USE environment variable.
    """
    start_time = time.time()
    logger.info(f"Starting quiz generation using AI model: {AI_MODEL_IN_USE.upper()}")
    RESPONSE_JSON = {
      "mcqs" : [
        {
            "mcq": "multiple choice question1",
            "options": {
                "a": "choice here1",
                "b": "choice here2",
                "c": "choice here3",
                "d": "choice here4",
            },
            "correct": "correct choice option in the form of a, b, c or d",
        },
        {
            "mcq": "multiple choice question",
            "options": {
                "a": "choice here",
                "b": "choice here",
                "c": "choice here",
                "d": "choice here",
            },
            "correct": "correct choice option in the form of a, b, c or d",
        },
        {   
            "mcq": "multiple choice question",
            "options": {
                "a": "choice here",
                "b": "choice here",
                "c": "choice here",
                "d": "choice here",
            },
            "correct": "correct choice option in the form of a, b, c or d",
        }
      ]
    }

    PROMPT_TEMPLATE = f"""
    Text: {text_content}

    You are a specialized AI that creates high-quality, multiple-choice questions (MCQs) from a given text. 
    Given the above text, create a quiz of 5 multiple choice questions keeping difficulty level as {quiz_level}. 
    G1: Each question must test comprehension of the key concepts, facts, or statements in the input text, not just trivial details.
    G2: The correct answer must be unambiguously supported by the input text.
    G3: All distractors (incorrect options) must be plausible and relevant to the context of the question but clearly incorrect based on the input text.
    G4: The questions and options MUST be in the same language as the input text.
    G5: Do not repeat questions or test the exact same concept multiple times.
    (CRITICAL!) You MUST respond in the EXACT SAME LANGUAGE as the input text. Do not translate or change languages.
    Make sure to format your response like RESPONSE_JSON below and use it as a guide.
    Ensure to make an array of 5 MCQs referring the following response json.

    Here is the RESPONSE_JSON: 

    {json.dumps(RESPONSE_JSON, indent=2)}
    """

    def parse_mcqs_response(content):
        """Helper function to parse MCQs from response content"""
        try:
            # Try parsing directly
            mcqs = json.loads(content).get("mcqs", [])
        except Exception:
            # Try extracting JSON if extra explanation text is included
            match = re.search(r'\{[\s\S]*\}', content)
            if match:
                try:
                    mcqs = json.loads(match.group()).get("mcqs", [])
                except:
                    mcqs = []
            else:
                mcqs = []
        return mcqs

    if AI_MODEL_IN_USE == 'openai':
        if not OPENAI_API_KEY:
            raise Exception('OpenAI API key not set in OPENAI_API_KEY env variable')
        
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant. You MUST always respond in the same language as the input text provided by the user. Do not translate or change languages."},
                {"role": "user", "content": PROMPT_TEMPLATE}
            ],
            temperature=0.3,
            max_tokens=1000,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        content = response.choices[0].message.content
        result = parse_mcqs_response(content)
    
    elif AI_MODEL_IN_USE == 'helpy':
        if not HELPY_API_KEY:
            raise Exception('Helpy API key not set in HELPY_API_KEY env variable')
        
        url = "https://mlapi.run/9331793d-efda-4839-8f97-ff66f7eaf605/v1/chat/completions"
        payload = {
            "model": "helpy-v-reasoning-c",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": PROMPT_TEMPLATE
                        }
                    ]
                }
            ],
        }
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "Authorization": f"Bearer {HELPY_API_KEY}"
        }
        response = requests.post(url, json=payload, headers=headers)
        if response.status_code == 200:
            content = response.json()["choices"][0]["message"]["content"]
            result = parse_mcqs_response(content)
        else:
            raise Exception(f"Error {response.status_code}: {response.text}")
    
    elif AI_MODEL_IN_USE == 'openrouter':
        if not AI_MODEL_API_KEY:
            raise Exception('OpenRouter API key not set in AI_MODEL_API_KEY env variable')
        
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {AI_MODEL_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": AI_MODEL_NAME,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant. You MUST always respond in the same language as the input text provided by the user. Do not translate or change languages."},
                {"role": "user", "content": PROMPT_TEMPLATE}
            ],
            "temperature": 0.3,
            "max_tokens": 10000,
            "top_p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0
        }
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        if response.status_code == 200:
            content = response.json()["choices"][0]["message"]["content"]
            result = parse_mcqs_response(content)
        else:
            raise Exception(f"Error {response.status_code}: {response.text}")
    
    else:
        raise Exception(f'Unsupported AI model: {AI_MODEL_IN_USE}. Supported values: openai, helpy, openrouter')
    
    end_time = time.time()
    processing_time = end_time - start_time
    logger.info(f"Quiz generation completed in {processing_time:.2f} seconds using {AI_MODEL_IN_USE.upper()}")
    
    return result

# Unified ask_question_about_transcription function with AI model selection
def ask_question_about_transcription(transcription, question, conversation_history=None):
    """
    Ask a question about a transcription using the selected AI model based on AI_MODEL_IN_USE environment variable.
    
    Args:
        transcription (str): The transcription text to use as context
        question (str): The user's question
        conversation_history (list): List of previous Q&A pairs for context
        
    Returns:
        str: The AI's answer to the question
    """
    start_time = time.time()
    logger.info(f"Starting question answering using AI model: {AI_MODEL_IN_USE.upper()}")
    
    # Build conversation context
    conversation_context = ""
    if conversation_history and len(conversation_history) > 0:
        conversation_context = "\n\nPrevious conversation:\n"
        for i, qa in enumerate(conversation_history[-5:], 1):  # Keep last 5 Q&A pairs for context
            conversation_context += f"Q{i}: {qa['question']}\n"
            conversation_context += f"A{i}: {qa['answer']}\n\n"
    
    prompt = (
        f"You are a helpful assistant that answers questions based on a meeting transcript. "
        f"Please answer the following question based on the transcript provided below. "
        f"Only use information that is explicitly mentioned in the transcript. "
        f"If the answer cannot be found in the transcript, say so clearly. "
        f"Provide a clear, concise, and accurate answer.\n\n"
        f"(CRITICAL!) You MUST respond in the EXACT SAME LANGUAGE as the input text. Do not translate or change languages."
        f"Meeting Transcript:\n{transcription}\n"
        f"{conversation_context}"
        f"Current Question: {question}\n\n"
        f"Answer:"
    )

    if AI_MODEL_IN_USE == 'openai':
        if not OPENAI_API_KEY:
            raise Exception('OpenAI API key not set in OPENAI_API_KEY env variable')
        
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant. You MUST always respond in the same language as the input text provided by the user. Do not translate or change languages."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.3,
        )
        result = response.choices[0].message.content.strip()
    
    elif AI_MODEL_IN_USE == 'helpy':
        if not HELPY_API_KEY:
            raise Exception('Helpy API key not set in HELPY_API_KEY env variable')
        
        url = "https://mlapi.run/9331793d-efda-4839-8f97-ff66f7eaf605/v1/chat/completions"
        payload = {
            "model": "helpy-v-reasoning-c",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ],
        }
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "Authorization": f"Bearer {HELPY_API_KEY}"
        }
        response = requests.post(url, json=payload, headers=headers)
        if response.status_code == 200:
            result = response.json()
            result = result["choices"][0]["message"]["content"].strip()
        else:
            raise Exception(f"Error {response.status_code}: {response.text}")
    
    elif AI_MODEL_IN_USE == 'openrouter':
        if not AI_MODEL_API_KEY:
            raise Exception('OpenRouter API key not set in AI_MODEL_API_KEY env variable')
        
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {AI_MODEL_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": AI_MODEL_NAME,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant. You MUST always respond in the same language as the input text provided by the user. Do not translate or change languages."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 1000
        }
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        if response.status_code == 200:
            result = response.json()
            result = result["choices"][0]["message"]["content"].strip()
        else:
            raise Exception(f"Error {response.status_code}: {response.text}")
    
    else:
        raise Exception(f'Unsupported AI model: {AI_MODEL_IN_USE}. Supported values: openai, helpy, openrouter')
    
    end_time = time.time()
    processing_time = end_time - start_time
    logger.info(f"Question answering completed in {processing_time:.2f} seconds using {AI_MODEL_IN_USE.upper()}")
    
    return result


# Helper function to summarize and save to MongoDB
def summarize_and_save(mongo_id):
    """
    Summarize transcription from MongoDB by _id and save the summary back to the same document.
    """
    if not mongo_id or mongo_collection is None:
        return None
    doc = mongo_collection.find_one({'_id': ObjectId(mongo_id)})
    if not doc:
        return None
    transcription = doc.get('transcription')
    if not transcription:
        return None
    # Truncate if too long for the model
    max_chars = 100000
    if len(transcription) > max_chars:
        transcription = transcription[:max_chars]
    summary = summarize_text(transcription)
    mongo_collection.update_one({'_id': ObjectId(mongo_id)}, {'$set': {'summary': summary}})
    return summary

# App Routes

# Summarizing transcription
@app.route('/summarize_transcription', methods=['POST'])
def summarize_transcription():
    """Summarize a transcription from MongoDB by _id and save the summary back to the same document."""
    start_time = time.time()
    logger.info("API: /summarize_transcription endpoint called")
    
    try:
        data = request.get_json()
        mongo_id = data.get('_id')
        if not mongo_id:
            logger.error("API: /summarize_transcription - No _id provided")
            return jsonify({'error': 'No _id provided'}), 400
        if mongo_collection is None:
            logger.error("API: /summarize_transcription - MongoDB not configured")
            return jsonify({'error': 'MongoDB not configured'}), 500
        doc = mongo_collection.find_one({'_id': ObjectId(mongo_id)})
        if not doc:
            logger.error(f"API: /summarize_transcription - No document found for _id: {mongo_id}")
            return jsonify({'error': f'No document found for _id: {mongo_id}'}), 404
        transcription = doc.get('transcription')
        if not transcription:
            logger.error("API: /summarize_transcription - No transcription found in document")
            return jsonify({'error': 'No transcription found in document'}), 400
        max_chars = 50000
        if len(transcription) > max_chars:
            transcription = transcription[:max_chars]
        summary = summarize_text(transcription)
        mongo_collection.update_one({'_id': ObjectId(mongo_id)}, {'$set': {'summary': summary}})
        
        end_time = time.time()
        processing_time = end_time - start_time
        logger.info(f"API: /summarize_transcription completed in {processing_time:.2f} seconds")
        
        return jsonify({
            '_id': mongo_id, 
            'transcription': transcription,
            'summary': summary
        })
    except Exception as e:
        logger.error(f"API: /summarize_transcription error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/youtube_subtitle_transcribe', methods=['POST'])
def youtube_subtitle_transcribe():
    """Route to transcribe YouTube video using subtitles."""
    try:
        data = request.get_json()
        youtube_url = data.get('url')
        if not youtube_url:
            return jsonify({'error': 'No URL provided'}), 400
        result = transcribe_youtube_video(youtube_url)
        mongo_id = result.get('mongo_id')
        summary = summarize_and_save(mongo_id) if mongo_id else None
        return jsonify({'_id': mongo_id, 'transcription': result.get('transcription'), 'summary': summary})
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/whisper_file_transcribe', methods=['POST'])
def whisper_file_transcribe():
    """Route to transcribe audio file using Whisper API."""
    try:
        data = request.get_json()
        filename = data.get('filename')
        language = data.get('language', 'id')
        if not filename:
            return jsonify({'error': 'No filename provided'}), 400
        result = transcribe_audio_file(filename, language)
        mongo_id = result.get('mongo_id')
        summary = summarize_and_save(mongo_id) if mongo_id else None
        return jsonify({
            '_id': mongo_id, 
            'transcription': result.get('transcription'), 
            'summary': summary
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/video_to_audio_transcribe', methods=['POST'])
def video_to_audio_transcribe():
    """
    Convert an mp4 video to mp3 audio, transcribe using Whisper, and save transcription to MongoDB.
    Expects JSON: {"filename": "video.mp4", "language": "en"}
    Returns: {"_id": ..., "transcription": ..., "summary": ...}
    """
    try:
        data = request.get_json()
        filename = data.get('filename')
        language = data.get('language', 'en')
        if not filename:
            return jsonify({'error': 'No filename provided'}), 400
        video_path = os.path.join(VIDEO_FOLDER, filename)
        if not os.path.isfile(video_path):
            return jsonify({'error': f'File not found: {video_path}'}), 404
        base_name = os.path.splitext(filename)[0]
        mp3_filename = f"{base_name}.mp3"
        mp3_path = os.path.join(AUDIO_FOLDER, mp3_filename)
        ffmpeg_cmd = [
            'ffmpeg', '-y', '-i', video_path, '-vn', '-acodec', 'libmp3lame', mp3_path
        ]
        result = subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            return jsonify({'error': f'ffmpeg failed: {result.stderr.decode()}'}), 500
        transcription_result = transcribe_audio_file(mp3_filename, language)
        mongo_id = transcription_result.get('mongo_id')
        summary = summarize_and_save(mongo_id) if mongo_id else None
        return jsonify({
            '_id': mongo_id, 
            'transcription': transcription_result.get('transcription'), 
            'summary': summary
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/generate_quiz', methods=['POST'])
def generate_quiz():
    """Generate quiz from transcription in MongoDB by _id and quiz_level, and save to the document."""
    start_time = time.time()
    logger.info("API: /generate_quiz endpoint called")
    
    try:
        data = request.get_json()
        mongo_id = data.get('_id')
        quiz_level = data.get('quiz_level')
        if not mongo_id or not quiz_level:
            logger.error("API: /generate_quiz - Both _id and quiz_level are required")
            return jsonify({'error': 'Both _id and quiz_level are required'}), 400
        if mongo_collection is None:
            logger.error("API: /generate_quiz - MongoDB not configured")
            return jsonify({'error': 'MongoDB not configured'}), 500
        doc = mongo_collection.find_one({'_id': ObjectId(mongo_id)})
        if not doc:
            logger.error(f"API: /generate_quiz - No document found for _id: {mongo_id}")
            return jsonify({'error': f'No document found for _id: {mongo_id}'}), 404
        transcription = doc.get('transcription')
        if not transcription:
            logger.error("API: /generate_quiz - No transcription found in document")
            return jsonify({'error': 'No transcription found in document'}), 400
        mcqs = generate_mcqs(transcription, quiz_level)
        mongo_collection.update_one({'_id': ObjectId(mongo_id)}, {'$set': {'mcqs': mcqs}})
        
        end_time = time.time()
        processing_time = end_time - start_time
        logger.info(f"API: /generate_quiz completed in {processing_time:.2f} seconds")
        
        return jsonify({
            '_id': mongo_id, 
            'transcription': transcription,
            'mcqs': mcqs
        })
    except Exception as e:
        logger.error(f"API: /generate_quiz error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/ask_question', methods=['POST'])
def ask_question():
    """Ask a question about a transcription stored in MongoDB by _id with conversation memory."""
    start_time = time.time()
    logger.info("API: /ask_question endpoint called")
    
    try:
        data = request.get_json()
        mongo_id = data.get('_id')
        question = data.get('question')
        
        if not mongo_id or not question:
            logger.error("API: /ask_question - Both _id and question are required")
            return jsonify({'error': 'Both _id and question are required'}), 400
        
        if mongo_collection is None:
            logger.error("API: /ask_question - MongoDB not configured")
            return jsonify({'error': 'MongoDB not configured'}), 500
        
        if chat_logs_collection is None:
            logger.error("API: /ask_question - Chat logs collection not configured")
            return jsonify({'error': 'Chat logs collection not configured'}), 500
        
        doc = mongo_collection.find_one({'_id': ObjectId(mongo_id)})
        if not doc:
            logger.error(f"API: /ask_question - No document found for _id: {mongo_id}")
            return jsonify({'error': f'No document found for _id: {mongo_id}'}), 404
        
        transcription = doc.get('transcription')
        if not transcription:
            logger.error("API: /ask_question - No transcription found in document")
            return jsonify({'error': 'No transcription found in document'}), 400
        
        # Get conversation history for this transcription
        conversation_history = []
        if chat_logs_collection is not None:
            chat_logs = list(chat_logs_collection.find(
                {'transcription_id': mongo_id},
                {'question': 1, 'answer': 1}
            ).sort('created_at', 1))  # Sort by creation time, oldest first
            
            conversation_history = [
                {'question': log['question'], 'answer': log['answer']} 
                for log in chat_logs
            ]
        
        # Truncate transcription if too long for the model
        max_chars = 100000
        if len(transcription) > max_chars:
            transcription = transcription[:max_chars]
        
        # Pass conversation history to the function
        answer = ask_question_about_transcription(transcription, question, conversation_history)
        
        # Save the chat log to the chat_logs collection
        chat_log_id = save_chat_log_to_mongodb(mongo_id, question, answer)
        
        end_time = time.time()
        processing_time = end_time - start_time
        logger.info(f"API: /ask_question completed in {processing_time:.2f} seconds")
        
        return jsonify({
            '_id': mongo_id,
            'transcription': transcription,
            'question': question,
            'answer': answer,
            'chat_log_id': chat_log_id,
            'conversation_history_count': len(conversation_history)
        })
    except Exception as e:
        logger.error(f"API: /ask_question error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/get_chat_history', methods=['POST'])
def get_chat_history():
    """Get chat history for a specific transcription by _id."""
    try:
        data = request.get_json()
        mongo_id = data.get('_id')
        
        if not mongo_id:
            return jsonify({'error': 'No _id provided'}), 400
        
        if chat_logs_collection is None:
            return jsonify({'error': 'Chat logs collection not configured'}), 500
        
        # Find all chat logs for this transcription
        chat_logs = list(chat_logs_collection.find(
            {'transcription_id': mongo_id},
            {'_id': 1, 'question': 1, 'answer': 1, 'created_at': 1}
        ).sort('created_at', 1))  # Sort by creation time, oldest first
        
        # Convert ObjectId to string for JSON serialization
        for log in chat_logs:
            log['_id'] = str(log['_id'])
            log['created_at'] = log['created_at'].isoformat()
        
        return jsonify({
            '_id': mongo_id,
            'chat_history': chat_logs
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    logger.info(f"Starting AI Backend with AI model: {AI_MODEL_IN_USE.upper()}")
    logger.info(f"Server will run on http://0.0.0.0:6969")
    app.run(debug=True, host='0.0.0.0', port=6969)