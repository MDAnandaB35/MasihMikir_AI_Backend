# MasihMeeting AI Backend

This is the backend service for MasihMeeting, providing APIs for transcribing, summarizing, and generating quizzes from YouTube videos, audio, and video files using AI models.

## Features

- **Transcribe YouTube Videos:** Extracts subtitles/transcripts from YouTube videos and saves them to MongoDB.
- **Transcribe Audio Files:** Uses OpenAI Whisper API to transcribe uploaded audio files.
- **Transcribe Video Files:** Converts video files to audio, then transcribes using Whisper.
- **Summarize Transcriptions:** Summarizes meeting transcripts using Helpy, OpenRouter or OpenAI.
- **Generate Quizzes:** Generates multiple-choice questions (MCQs) from transcriptions using Helpy, OpenRouter or OpenAI.
- **Asking Questions:** Provide answers to questions about the meeting.

## Folder Structure

```
.env
app.py
requirements.txt
MasihMeeting.postman_collection.json
audio_files/
video_files/
```

## Setup

1. **Clone the repository** and navigate to the backend directory.

2. **Install dependencies:**

   Create and activate a virtual environment to avoid polluting your base Python installation:

   On Windows:

   ```sh
   python -m venv .venv
   .venv\Scripts\activate
   ```

   On macOS/Linux:

   ```sh
   python -m venv .venv
   source .venv/bin/activate
   ```

   Then install the requirements:

   ```sh
   pip install -r requirements.txt
   ```

3. **Environment Variables:**
   Create a `.env` file with the following variables:

   ```
   OPENAI_API_KEY=your_openai_api_key
   OPENAI_MODEL=your_openai_model_name
   AI_MODEL_API_KEY=your_openrouter_api_key
   AI_MODEL_NAME=your_openrouter_model_name
   HELPY_API_KEY=your_helpy_api_key
   MONGODB_URI=your_mongodb_uri
   MONGODB_DB=your_db_name
   MONGODB_COLLECTION=your_collection_name
   AUDIO_FOLDER=audio_files
   VIDEO_FOLDER=video_files
   TRANSCRIPTS_FOLDER=transcripts
   ```

4. **Run the server:**
   ```sh
   python app.py
   ```

## API Endpoints

All endpoints return a consistent JSON structure with the MongoDB document ID and relevant data.

### Response Format

**For Transcription Endpoints** (YouTube, Audio, Video, Summary):

```json
{
  "_id": "mongodb_object_id",
  "transcription": "formatted_transcription_with_timestamps",
  "summary": "detailed_meeting_summary"
}
```

**For Quiz Generation Endpoint**:

```json
{
  "_id": "mongodb_object_id",
  "transcription": "formatted_transcription_with_timestamps",
  "mcqs": [
    {
      "mcq": "question_text",
      "options": {
        "a": "option_a",
        "b": "option_b",
        "c": "option_c",
        "d": "option_d"
      },
      "correct": "correct_option_letter"
    }
  ]
}
```

**For Question Answering Endpoint**:

```json
{
  "_id": "mongodb_object_id",
  "transcription": "formatted_transcription_with_timestamps",
  "question": "user_question",
  "answer": "ai_generated_answer",
  "chat_log_id": "chat_log_document_id"
}
```

**For Chat History Endpoint**:

```json
{
  "_id": "mongodb_object_id",
  "chat_history": [
    {
      "_id": "chat_log_id",
      "question": "user_question",
      "answer": "ai_generated_answer",
      "created_at": "2024-01-01T12:00:00.000Z"
    }
  ]
}
```

### Endpoint Details

| Name                  | Method | Endpoint                       | Body                                                                       | Returns                                                                                              |
| --------------------- | ------ | ------------------------------ | -------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| YouTube Transcription | `POST` | `/youtube_subtitle_transcribe` | `{ "url": "<YouTube URL>" }`                                               | `{ "_id": "...", "transcription": "...", "summary": "..." }`                                         |
| Audio Transcription   | `POST` | `/whisper_file_transcribe`     | `{ "filename": "<audio file>", "language": "<lang>" }`                     | `{ "_id": "...", "transcription": "...", "summary": "..." }`                                         |
| Video Transcription   | `POST` | `/video_to_audio_transcribe`   | `{ "filename": "<video file>", "language": "<lang>" }`                     | `{ "_id": "...", "transcription": "...", "summary": "..." }`                                         |
| Request Summary       | `POST` | `/summarize_transcription`     | `{ "_id": "<MongoDB document ID>" }`                                       | `{ "_id": "...", "transcription": "...", "summary": "..." }`                                         |
| Generate Quiz         | `POST` | `/generate_quiz`               | `{ "_id": "<MongoDB document ID>", "quiz_level": "<easy, medium, hard>" }` | `{ "_id": "...", "transcription": "...", "mcqs": [...] }`                                            |
| Ask Question          | `POST` | `/ask_question`                | `{ "_id": "<MongoDB document ID>", "question": "<user question>" }`        | `{ "_id": "...", "transcription": "...", "question": "...", "answer": "...", "chat_log_id": "..." }` |
| Get Chat History      | `POST` | `/get_chat_history`            | `{ "_id": "<MongoDB document ID>" }`                                       | `{ "_id": "...", "chat_history": [...] }`                                                            |

## Testing

You can use [MasihMeeting.postman_collection.json](MasihMeeting.postman_collection.json) with [Postman](https://www.postman.com/) to test all endpoints.

## Notes

- Audio and video files should be placed in the `audio_files/` and `video_files/` directories, respectively.
- Transcriptions and summaries are stored in `MONGODB_COLLECTION`.
- Requires FFmpeg installed for video-to-audio conversion.

## License

This project is for internal
