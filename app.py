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

app = Flask(__name__)

# Initializing environment variables
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
DEEPSEEK_API_KEY = os.environ.get('DEEPSEEK_API_KEY')
AUDIO_FOLDER = os.environ.get('AUDIO_FOLDER', 'audio_files')  # Folder for audio files
VIDEO_FOLDER = os.environ.get('VIDEO_FOLDER', 'video_files') # Folder for video files
TRANSCRIPTS_FOLDER = os.environ.get('TRANSCRIPTS_FOLDER', 'transcripts')
MONGODB_URI = os.environ.get('MONGODB_URI')
MONGODB_DB = os.environ.get('MONGODB_DB')
MONGODB_COLLECTION = os.environ.get('MONGODB_COLLECTION')

ytt_api = YouTubeTranscriptApi()

# MongoDB setup
mongo_client = None
mongo_collection = None
if MONGODB_URI:
    mongo_client = MongoClient(MONGODB_URI)
    mongo_collection = mongo_client[MONGODB_DB][MONGODB_COLLECTION]

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
        'transcription': full_text,
        'mongo_id': inserted_id
    }

# Transcribing audio file using OpenAI Whisper API
# Without timestamp
# async def transcribe_audio_file_async(filename, language='en'):
#     """
#     Transcribe audio file using OpenAI Whisper API.
    
#     Args:
#         filename (str): Name of the audio file
#         language (str): Language code for transcription
        
#     Returns:
#         str: Transcription text
        
#     Raises:
#         Exception: If transcription fails
#     """
#     if not OPENAI_API_KEY:
#         raise Exception('OpenAI API key not set in OPENAI_API_KEY env variable')
    
#     file_path = os.path.join(AUDIO_FOLDER, filename)
#     if not os.path.isfile(file_path):
#         raise Exception(f'File not found: {file_path}')
    
#     try:
#         with open(file_path, 'rb') as audio_file:
#             audio_data = audio_file.read()
        
#         client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY)
#         transcript = await client.audio.transcriptions.create(
#             model="whisper-1",
#             file=(filename, audio_data),
#             language=language
#         )
#         return transcript.text
#     except Exception as e:
#         raise Exception(f'Failed to transcribe audio: {str(e)}')

# # Saving to MongoDB
# def transcribe_audio_file(filename, language='en'):
#     """
#     Synchronous wrapper for audio transcription.
    
#     Args:
#         filename (str): Name of the audio file
#         language (str): Language code for transcription
        
#     Returns:
#         str: Transcription text
#     """
#     transcription = asyncio.run(transcribe_audio_file_async(filename, language))
#     # Save to MongoDB
#     doc = {
#         'type': 'audio',
#         'filename': filename,
#         'language': language,
#         'transcription': transcription,
#         'created_at': datetime.utcnow()
#     }
#     inserted_id = save_transcription_to_mongodb(doc)
#     return transcription

# Transcribing audio file using OpenAI Whisper API
# With timestamp
async def transcribe_audio_file_async(filename, language='en'):
    """
    Transcribe audio file using OpenAI Whisper API and return detailed result with timestamps.
    
    Returns:
        dict: Contains 'text', 'segments', and optionally 'formatted'
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
            timestamp_granularities=["segment"]
        )

        # Format segments into readable string with timestamps
        def format_time(seconds):
            minutes = int(seconds) // 60
            secs = int(seconds) % 60
            return f"{minutes:02d}:{secs:02d}"

        def format_segments(segments):
            lines = []
            for seg in segments:
                start = format_time(seg["start"])
                end = format_time(seg["end"])
                lines.append(f"[{start} - {end}] {seg['text']}")
            return "\n".join(lines)

        formatted_text = format_segments(transcript.segments)

        return {
            "text": transcript.text,
            "segments": transcript.segments,
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
    return result['formatted']

# Uncomment if want to use OpenAI instead (paid method)
# Summarizing text using OpenAI
# def summarize_text(text, api_key=OPENAI_API_KEY, model='gpt-3.5-turbo', max_tokens=1024):
#     client = openai.OpenAI(api_key=api_key)
#     prompt = (
#         "You are a professional meeting summarizer. "
#         "Read the following meeting transcript and produce a detailed, structured summary. "
#         "Organize the summary into bullet points grouped by main topics. "
#         "For each topic, include sub-bullets for key decisions, action items, and important discussions. "
#         "Be as specific and complete as possible, including names or roles if mentioned. "
#         "Make the summary clear and useful for someone who did not attend the meeting. "
#         "\n\nMeeting transcript:\n" + text
#     )
#     response = client.chat.completions.create(
#         model=model,
#         messages=[
#             {"role": "system", "content": "You are a helpful assistant."},
#             {"role": "user", "content": prompt}
#         ],
#         max_tokens=max_tokens,
#         temperature=0.5,
#     )
#     return response.choices[0].message.content.strip()

# Use this for free ai model from OpenRouter (but slow)
# Summarizing text using DeepSeek
def summarize_text(text, deepseek_api_key=DEEPSEEK_API_KEY, model='deepseek/deepseek-r1-0528:free', max_tokens=1024):
    prompt = (
        "You are a professional meeting summarizer. "
        "Read the following meeting transcript and produce a detailed, structured summary. "
        "Organize the summary into bullet points grouped by main topics. "
        "For each topic, include sub-bullets for key decisions, action items, and important discussions. "
        "Be as specific and complete as possible, including names or roles if mentioned. "
        "Make the summary clear and useful for someone who did not attend the meeting. "
        "If a timestamp is present, make sure to utilize the timestamp to categorize certain topics into certain timestamps per section. "
        "\n\nMeeting transcript:\n" + text
    )

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {deepseek_api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.5
    }

    response = requests.post(url, headers=headers, data=json.dumps(payload))

    if response.status_code == 200:
        result = response.json()
        return result["choices"][0]["message"]["content"].strip()
    else:
        raise Exception(f"Error {response.status_code}: {response.text}")


# Generating quiz with OpenAI
# Generating Quiz
# def generate_mcqs(text_content, quiz_level, api_key=OPENAI_API_KEY, model="gpt-3.5-turbo"): 
#     RESPONSE_JSON = {
#       "mcqs" : [
#         {
#             "mcq": "multiple choice question1",
#             "options": {
#                 "a": "choice here1",
#                 "b": "choice here2",
#                 "c": "choice here3",
#                 "d": "choice here4",
#             },
#             "correct": "correct choice option in the form of a, b, c or d",
#         },
#         {
#             "mcq": "multiple choice question",
#             "options": {
#                 "a": "choice here",
#                 "b": "choice here",
#                 "c": "choice here",
#                 "d": "choice here",
#             },
#             "correct": "correct choice option in the form of a, b, c or d",
#         },
#         {
#             "mcq": "multiple choice question",
#             "options": {
#                 "a": "choice here",
#                 "b": "choice here",
#                 "c": "choice here",
#                 "d": "choice here",
#             },
#             "correct": "correct choice option in the form of a, b, c or d",
#         }
#       ]
#     }

#     PROMPT_TEMPLATE="""
#     Text: {text_content}
#     You are an expert in generating MCQ type quiz on the basis of provided content. 
#     Given the above text, create a quiz of 3 multiple choice questions keeping difficulty level as {quiz_level}. 
#     Make sure the questions are not repeated and check all the questions to be conforming the text as well.
#     Make sure to format your response like RESPONSE_JSON below and use it as a guide.
#     Ensure to make an array of 3 MCQs referring the following response json.
#     Here is the RESPONSE_JSON: 

#     {RESPONSE_JSON}

#     """

#     formatted_template = PROMPT_TEMPLATE.format(text_content=text_content, quiz_level=quiz_level, RESPONSE_JSON=RESPONSE_JSON)

#     client = openai.OpenAI(api_key=api_key)
#     response = client.chat.completions.create(
#         model=model,
#         messages=[
#             {
#                 "role": "user",
#                 "content" : formatted_template
#             }
#         ],
#         temperature=0.3,
#         max_tokens=1000,
#         top_p=1,
#         frequency_penalty=0,
#         presence_penalty=0
#     )

#     extracted_response = response.choices[0].message.content
#     try:
#         mcqs = json.loads(extracted_response).get("mcqs", [])
#     except Exception:
#         # Try to extract JSON from the response if not directly parsable
#         import re
#         match = re.search(r'\{[\s\S]*\}', extracted_response)
#         if match:
#             mcqs = json.loads(match.group()).get("mcqs", [])
#         else:
#             mcqs = []
#     return mcqs

# Generating quiz with OpenRouter Deepseek
def generate_mcqs(text_content, quiz_level, deepseek_api_key=DEEPSEEK_API_KEY, model="deepseek/deepseek-r1-0528:free"):
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

    You are an expert in generating MCQ type quiz on the basis of provided content. 
    Given the above text, create a quiz of 3 multiple choice questions keeping difficulty level as {quiz_level}. 
    Make sure the questions are not repeated and check all the questions to be conforming the text as well.

    Make sure to format your response like RESPONSE_JSON below and use it as a guide.
    Ensure to make an array of 3 MCQs referring the following response json.

    Here is the RESPONSE_JSON: 

    {json.dumps(RESPONSE_JSON, indent=2)}
    """

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {deepseek_api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": PROMPT_TEMPLATE}
        ],
        "temperature": 0.3,
        "max_tokens": 1000,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0
    }

    response = requests.post(url, headers=headers, data=json.dumps(payload))

    if response.status_code != 200:
        raise Exception(f"Request failed: {response.status_code} {response.text}")

    try:
        content = response.json()["choices"][0]["message"]["content"]
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

# App Routes

# Summarizing transcription
@app.route('/summarize_transcription', methods=['POST'])
def summarize_transcription():
    """Summarize a transcription from MongoDB by _id and save the summary back to the same document."""
    try:
        data = request.get_json()
        mongo_id = data.get('_id')
        if not mongo_id:
            return jsonify({'error': 'No _id provided'}), 400
        if mongo_collection is None:
            return jsonify({'error': 'MongoDB not configured'}), 500
        doc = mongo_collection.find_one({'_id': ObjectId(mongo_id)})
        if not doc:
            return jsonify({'error': f'No document found for _id: {mongo_id}'}), 404
        transcription = doc.get('transcription')
        if not transcription:
            return jsonify({'error': 'No transcription found in document'}), 400
        # Truncate if too long for the model
        max_chars = 50000
        if len(transcription) > max_chars:
            transcription = transcription[:max_chars]
        summary = summarize_text(transcription)
        # Save the summary back to MongoDB under the same _id
        mongo_collection.update_one({'_id': ObjectId(mongo_id)}, {'$set': {'summary': summary}})
        return jsonify({'summary': summary})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Transcribing YouTube video using subtitles
@app.route('/youtube_subtitle_transcribe', methods=['POST'])
def youtube_subtitle_transcribe():
    """Route to transcribe YouTube video using subtitles."""
    try:
        data = request.get_json()
        youtube_url = data.get('url')
        
        if not youtube_url:
            return jsonify({'error': 'No URL provided'}), 400
        
        result = transcribe_youtube_video(youtube_url)
        return jsonify(result)
        
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Transcribing audio file using Whisper API
@app.route('/whisper_file_transcribe', methods=['POST'])
def whisper_file_transcribe():
    """Route to transcribe audio file using Whisper API."""
    try:
        data = request.get_json()
        filename = data.get('filename')
        language = data.get('language', 'id')
        
        if not filename:
            return jsonify({'error': 'No filename provided'}), 400
        
        transcription = transcribe_audio_file(filename, language)
        return jsonify({'transcription': transcription})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Transcribing video to audio using Whisper API
@app.route('/video_to_audio_transcribe', methods=['POST'])
def video_to_audio_transcribe():
    """
    Convert an mp4 video to mp3 audio, transcribe using Whisper, and save transcription to MongoDB.
    Expects JSON: {"filename": "video.mp4", "language": "en"}
    Returns: {"transcription": ..., "mongo_id": ...}
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
        # Convert mp4 to mp3
        base_name = os.path.splitext(filename)[0]
        mp3_filename = f"{base_name}.mp3"
        mp3_path = os.path.join(AUDIO_FOLDER, mp3_filename)
        # Use ffmpeg to convert
        ffmpeg_cmd = [
            'ffmpeg', '-y', '-i', video_path, '-vn', '-acodec', 'libmp3lame', mp3_path
        ]
        result = subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            return jsonify({'error': f'ffmpeg failed: {result.stderr.decode()}'}), 500
        # Transcribe mp3 using Whisper
        transcription = transcribe_audio_file(mp3_filename, language)

        return jsonify({'transcription': transcription})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/generate_quiz', methods=['POST'])
def generate_quiz():
    """Generate quiz from transcription in MongoDB by _id and quiz_level, and save to the document."""
    try:
        data = request.get_json()
        mongo_id = data.get('_id')
        quiz_level = data.get('quiz_level')
        if not mongo_id or not quiz_level:
            return jsonify({'error': 'Both _id and quiz_level are required'}), 400
        if mongo_collection is None:
            return jsonify({'error': 'MongoDB not configured'}), 500
        doc = mongo_collection.find_one({'_id': ObjectId(mongo_id)})
        if not doc:
            return jsonify({'error': f'No document found for _id: {mongo_id}'}), 404
        transcription = doc.get('transcription')
        if not transcription:
            return jsonify({'error': 'No transcription found in document'}), 400
        mcqs = generate_mcqs(transcription, quiz_level)
        # Save questions back to MongoDB
        mongo_collection.update_one({'_id': ObjectId(mongo_id)}, {'$set': {'mcqs': mcqs}})
        return jsonify({'mcqs': mcqs})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True) 