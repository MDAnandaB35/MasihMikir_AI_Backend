# MasihMeeting AI Backend

This is the backend service for MasihMeeting, providing APIs for transcribing, summarizing, and generating quizzes from YouTube videos, audio, and video files using AI models.

## Features

-   **Transcribe YouTube Videos:** Extracts subtitles/transcripts from YouTube videos and saves them to MongoDB.
-   **Transcribe Audio Files:** Uses OpenAI Whisper API to transcribe uploaded audio files.
-   **Transcribe Video Files:** Converts video files to audio, then transcribes using Whisper.
-   **Summarize Transcriptions:** Summarizes meeting transcripts using DeepSeek or OpenAI models.
-   **Generate Quizzes:** Generates multiple-choice questions (MCQs) from transcriptions using DeepSeek or OpenAI.

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

    ```sh
    pip install -r requirements.txt
    ```

3. **Environment Variables:**
   Create a `.env` file with the following variables:

    ```
    OPENAI_API_KEY=your_openai_api_key
    DEEPSEEK_API_KEY=your_deepseek_api_key
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

| Name                  | Method | Endpoint                       | Body                                                                       | Returns                                 |
| --------------------- | ------ | ------------------------------ | -------------------------------------------------------------------------- | --------------------------------------- |
| YouTube Transcription | `POST` | `/youtube_subtitle_transcribe` | `{ "url": "<YouTube URL>" }`                                               | Transcription, summary, and MongoDB ID. |
| Audio Transcription   | `POST` | `/whisper_file_transcribe`     | `{ "filename": "<audio file>", "language": "<lang>" }`                     | Transcription and summary.              |
| Video Transcription   | `POST` | `/video_to_audio_transcribe`   | `{ "filename": "<video file>", "language": "<lang>" }`                     | Transcription and summary.              |
| Request Summary       | `POST` | `/summarize_transcription`     | `{ "_id": "<MongoDB document ID>" }`                                       | Summary.                                |
| Generate Quiz         | `POST` | `/generate_quiz`               | `{ "_id": "<MongoDB document ID>", "quiz_level": "<easy, medium, hard>" }` | MCQs.                                   |

## Testing

You can use [MasihMeeting.postman_collection.json](MasihMeeting.postman_collection.json) with [Postman](https://www.postman.com/) to test all endpoints.

## Notes

-   Audio and video files should be placed in the `audio_files/` and `video_files/` directories, respectively.
-   Transcriptions and summaries are stored in MongoDB.
-   Requires ffmpeg installed for video-to-audio conversion.

## License

This project is for internal
