import os
import asyncio
import subprocess
import json
import re
from typing import Dict, Any, List, Optional
from datetime import datetime

import openai
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound

from config import Config
from database import db
from utils import (
    extract_video_id, validate_filename, validate_language_code,
    truncate_text, format_transcript, make_openrouter_request,
    build_conversation_context, sanitize_input, ensure_directory_exists
)

class TranscriptionService:
    """Service for handling transcription operations"""
    
    @staticmethod
    def transcribe_youtube_video(youtube_url: str) -> Dict[str, Any]:
        """Transcribe YouTube video using subtitles"""
        # Validate URL
        if not youtube_url:
            raise ValueError("YouTube URL is required")
        
        youtube_url = sanitize_input(youtube_url)
        
        # Extract video ID
        video_id = extract_video_id(youtube_url)
        
        # Fetch transcript (try Indonesian, then English, then any)
        transcript_data = None
        ytt_api = YouTubeTranscriptApi()
        
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
        
        # Format transcript
        full_text = format_transcript(transcript_data)
        
        # Save to database
        doc = {
            'type': 'youtube',
            'video_id': video_id,
            'url': youtube_url,
            'transcription': full_text,
            'created_at': datetime.utcnow()
        }
        
        inserted_id = db.save_transcription(doc)
        
        return {
            'mongo_id': inserted_id,
            'transcription': full_text,
        }
    
    @staticmethod
    async def transcribe_audio_file_async(filename: str, language: str = 'en') -> Dict[str, Any]:
        """Transcribe audio file using OpenAI Whisper API"""
        if not Config.OPENAI_API_KEY:
            raise Exception('OpenAI API key not configured')
        
        if not validate_filename(filename):
            raise ValueError('Invalid filename')
        
        if not validate_language_code(language):
            raise ValueError('Invalid language code')
        
        # Ensure audio folder exists
        ensure_directory_exists(Config.AUDIO_FOLDER)
        
        file_path = os.path.join(Config.AUDIO_FOLDER, filename)
        if not os.path.isfile(file_path):
            raise Exception(f'File not found: {file_path}')
        
        try:
            with open(file_path, 'rb') as audio_file:
                audio_data = audio_file.read()
            
            client = openai.AsyncOpenAI(api_key=Config.OPENAI_API_KEY)
            
            transcript = await client.audio.transcriptions.create(
                model="whisper-1",
                file=(filename, audio_data),
                language=language,
                response_format="verbose_json",
                timestamp_granularities=["segment"],
            )
            
            # Format segments with timestamps
            lines = []
            for seg in transcript.segments:
                # Handle both dict and object formats
                if isinstance(seg, dict):
                    start_time = seg['start']
                    end_time = seg['end']
                    text = seg['text']
                else:
                    start_time = seg.start
                    end_time = seg.end
                    text = seg.text
                
                start = f"{int(start_time) // 60:02d}:{int(start_time) % 60:02d}"
                end = f"{int(end_time) // 60:02d}:{int(end_time) % 60:02d}"
                lines.append(f"[{start} - {end}] {text}")
            
            formatted_text = "\n".join(lines)
            
            return {
                "text": transcript.text,
                "segments": transcript.segments,
                "formatted": formatted_text
            }
        
        except Exception as e:
            raise Exception(f'Failed to transcribe audio: {str(e)}')
    
    @staticmethod
    def transcribe_audio_file(filename: str, language: str = 'en') -> Dict[str, Any]:
        """Synchronous wrapper for audio transcription"""
        result = asyncio.run(TranscriptionService.transcribe_audio_file_async(filename, language))
        
        doc = {
            'type': 'audio',
            'filename': filename,
            'language': language,
            'transcription': result['formatted'],
            'created_at': datetime.utcnow()
        }
        
        inserted_id = db.save_transcription(doc)
        
        return {
            'mongo_id': inserted_id,
            'transcription': result['formatted'],
        }
    
    @staticmethod
    def transcribe_video_file(filename: str, language: str = 'en') -> Dict[str, Any]:
        """Convert video to audio and transcribe"""
        if not validate_filename(filename):
            raise ValueError('Invalid filename')
        
        if not validate_language_code(language):
            raise ValueError('Invalid language code')
        
        # Ensure directories exist
        ensure_directory_exists(Config.VIDEO_FOLDER)
        ensure_directory_exists(Config.AUDIO_FOLDER)
        
        video_path = os.path.join(Config.VIDEO_FOLDER, filename)
        if not os.path.isfile(video_path):
            raise Exception(f'Video file not found: {video_path}')
        
        # Convert video to audio
        base_name = os.path.splitext(filename)[0]
        mp3_filename = f"{base_name}.mp3"
        mp3_path = os.path.join(Config.AUDIO_FOLDER, mp3_filename)
        
        ffmpeg_cmd = [
            'ffmpeg', '-y', '-i', video_path, '-vn', '-acodec', 'libmp3lame', mp3_path
        ]
        
        result = subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            raise Exception(f'ffmpeg failed: {result.stderr.decode()}')
        
        # Transcribe the audio
        return TranscriptionService.transcribe_audio_file(mp3_filename, language)

class AIService:
    """Service for AI operations (summarization, quiz generation, Q&A)"""
    
    @staticmethod
    def summarize_text(text: str) -> str:
        """Summarize text using OpenRouter AI"""
        if not text:
            raise ValueError("Text is required for summarization")
        
        # Truncate if too long
        text = truncate_text(text, Config.MAX_SUMMARY_LENGTH)
        
        prompt = (
            "You are a professional meeting summarizer. "
            "Read the following meeting transcript and produce a detailed, structured summary. "
            "Organize the summary into bullet points grouped by main topics. "
            "For each topic, include sub-bullets for key decisions, action items, and important discussions. "
            "Be as specific and complete as possible, including names or roles if mentioned. "
            "Make the summary clear and useful for someone who did not attend the meeting. "
            "(IMPORTANT!) If timestamps are present in the transcript, you MUST use them to group and label each section accordingly. Each main topic should include the relevant timestamp(s) where the discussion occurred."
            "(IMPORTANT!) Provide summary in the original language of the original text. "
            "\n\nMeeting transcript:\n" + text
        )
        
        return make_openrouter_request(prompt, temperature=0.5, max_tokens=2000)
    
    @staticmethod
    def generate_quiz(text: str, quiz_level: str) -> List[Dict[str, Any]]:
        """Generate quiz questions from text"""
        if not text:
            raise ValueError("Text is required for quiz generation")
        
        if not quiz_level or quiz_level not in ['easy', 'medium', 'hard']:
            raise ValueError("Quiz level must be 'easy', 'medium', or 'hard'")
        
        # Truncate if too long
        text = truncate_text(text, Config.MAX_TRANSCRIPTION_LENGTH)
        
        response_template = {
            "mcqs": [
                {
                    "mcq": "multiple choice question1",
                    "options": {
                        "a": "choice here1",
                        "b": "choice here2",
                        "c": "choice here3",
                        "d": "choice here4",
                    },
                    "correct": "correct choice option in the form of a, b, c or d",
                }
            ]
        }
        
        prompt = f"""
        Text: {text}

        You are a specialized AI that creates high-quality, multiple-choice questions (MCQs) from a given text. 
        Given the above text, create a quiz of 5 multiple choice questions keeping difficulty level as {quiz_level}. 
        G1: Each question must test comprehension of the key concepts, facts, or statements in the input text, not just trivial details.
        G2: The correct answer must be unambiguously supported by the input text.
        G3: All distractors (incorrect options) must be plausible and relevant to the context of the question but clearly incorrect based on the input text.
        G4: The questions and options MUST be in the same language as the input text.
        G5: Do not repeat questions or test the exact same concept multiple times.
        Make sure to format your response like RESPONSE_JSON below and use it as a guide.
        Ensure to make an array of 5 MCQs referring the following response json.

        Provide quiz in the original language of the original text.

        Here is the RESPONSE_JSON: 

        {json.dumps(response_template, indent=2)}
        """
        
        try:
            content = make_openrouter_request(prompt, temperature=0.3, max_tokens=10000)
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
    
    @staticmethod
    def ask_question(text: str, question: str, conversation_history: List[Dict[str, str]] = None) -> str:
        """Ask a question about the text with conversation memory"""
        if not text:
            raise ValueError("Text is required")
        
        if not question:
            raise ValueError("Question is required")
        
        # Sanitize inputs
        text = sanitize_input(text)
        question = sanitize_input(question)
        
        # Truncate text if too long
        text = truncate_text(text, Config.MAX_TRANSCRIPTION_LENGTH)
        
        # Build conversation context
        conversation_context = build_conversation_context(conversation_history or [])
        
        prompt = (
            f"You are a helpful assistant that answers questions based on a meeting transcript. "
            f"Please answer the following question based on the transcript provided below. "
            f"Only use information that is explicitly mentioned in the transcript. "
            f"If the answer cannot be found in the transcript, say so clearly. "
            f"Provide a clear, concise, and accurate answer.\n\n"
            f"Provide the answer in the original language of the original text. "
            f"Meeting Transcript:\n{text}\n"
            f"{conversation_context}"
            f"Current Question: {question}\n\n"
            f"Answer:"
        )
        
        return make_openrouter_request(prompt, temperature=0.3, max_tokens=1000)

class TranscriptionManager:
    """Manager for transcription operations with database integration"""
    
    @staticmethod
    def summarize_and_save(mongo_id: str) -> Optional[str]:
        """Summarize transcription and save to database"""
        if not mongo_id:
            return None
        
        doc = db.get_transcription(mongo_id)
        if not doc:
            return None
        
        transcription = doc.get('transcription')
        if not transcription:
            return None
        
        # Generate summary
        summary = AIService.summarize_text(transcription)
        
        # Save to database
        db.update_transcription(mongo_id, {'summary': summary})
        
        return summary
    
    @staticmethod
    def generate_quiz_and_save(mongo_id: str, quiz_level: str) -> List[Dict[str, Any]]:
        """Generate quiz and save to database"""
        if not mongo_id:
            raise ValueError("MongoDB ID is required")
        
        doc = db.get_transcription(mongo_id)
        if not doc:
            raise ValueError("Transcription not found")
        
        transcription = doc.get('transcription')
        if not transcription:
            raise ValueError("No transcription content found")
        
        # Generate quiz
        mcqs = AIService.generate_quiz(transcription, quiz_level)
        
        # Save to database
        db.update_transcription(mongo_id, {'mcqs': mcqs})
        
        return mcqs
    
    @staticmethod
    def ask_question_and_save(mongo_id: str, question: str) -> Dict[str, Any]:
        """Ask question and save to database"""
        if not mongo_id:
            raise ValueError("MongoDB ID is required")
        
        if not question:
            raise ValueError("Question is required")
        
        doc = db.get_transcription(mongo_id)
        if not doc:
            raise ValueError("Transcription not found")
        
        transcription = doc.get('transcription')
        if not transcription:
            raise ValueError("No transcription content found")
        
        # Get conversation history
        chat_history = db.get_chat_history(mongo_id)
        conversation_history = [
            {'question': log['question'], 'answer': log['answer']} 
            for log in chat_history
        ]
        
        # Ask question
        answer = AIService.ask_question(transcription, question, conversation_history)
        
        # Save to database
        chat_log_id = db.save_chat_log(mongo_id, question, answer)
        
        return {
            'answer': answer,
            'chat_log_id': chat_log_id,
            'conversation_history_count': len(conversation_history)
        } 