# AI Model Configuration Guide

This application now supports multiple AI models through a unified interface. You can easily switch between different AI providers by setting the `AI_MODEL_IN_USE` environment variable.

## Supported AI Models

### 1. OpenAI (Paid)

- **Environment Variable**: `AI_MODEL_IN_USE=openai`
- **Required API Keys**: `OPENAI_API_KEY`
- **Required Model**: `OPENAI_MODEL`
- **Best for**: High-quality, fast responses
- **Cost**: Pay-per-use

### 2. Helpy (Free/Paid)

- **Environment Variable**: `AI_MODEL_IN_USE=helpy`
- **Required API Keys**: `HELPY_API_KEY`
- **Model**: Automatically uses `helpy-v-reasoning-c`
- **Best for**: Free tier usage, good reasoning capabilities
- **Cost**: Free tier available

### 3. OpenRouter (Free/Paid)

- **Environment Variable**: `AI_MODEL_IN_USE=openrouter`
- **Required API Keys**: `AI_MODEL_API_KEY`
- **Required Model**: `AI_MODEL_NAME`
- **Best for**: Access to multiple models, free tier available
- **Cost**: Free tier available, pay-per-use for premium models

## Configuration

### Environment Variables Setup

Create a `.env` file in your project root with the following variables:

```env
# AI Model Selection (REQUIRED)
AI_MODEL_IN_USE=openrouter

# OpenAI Configuration (if using OpenAI)
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-3.5-turbo

# Helpy Configuration (if using Helpy)
HELPY_API_KEY=your_helpy_api_key_here

# OpenRouter Configuration (if using OpenRouter)
AI_MODEL_API_KEY=your_openrouter_api_key_here
AI_MODEL_NAME=anthropic/claude-3-haiku

# Other required variables...
MONGODB_URI=your_mongodb_uri
MONGODB_DB=your_database_name
MONGODB_COLLECTION=your_collection_name
```

### Switching Between Models

To switch between AI models, simply change the `AI_MODEL_IN_USE` value in your `.env` file:

```env
# For OpenAI
AI_MODEL_IN_USE=openai

# For Helpy
AI_MODEL_IN_USE=helpy

# For OpenRouter
AI_MODEL_IN_USE=openrouter
```

## API Usage

The application automatically uses the selected AI model for all operations:

- **Text Summarization**: `/summarize_transcription`
- **Quiz Generation**: `/generate_quiz`
- **Question Answering**: `/ask_question`

No changes are needed in your API calls - the system automatically routes to the correct AI provider based on your configuration.

## Error Handling

The system will throw clear error messages if:

1. **Invalid AI Model**: If `AI_MODEL_IN_USE` is set to an unsupported value
2. **Missing API Key**: If the required API key for the selected model is not provided
3. **API Errors**: If the AI service returns an error

## Recommendations

- **For Production**: Use OpenAI for the best quality and reliability
- **For Development/Testing**: Use OpenRouter or Helpy for cost-effective testing
- **For Free Usage**: Start with OpenRouter or Helpy free tiers

## Migration from Old System

If you were previously using the commented-out functions, simply:

1. Set the appropriate `AI_MODEL_IN_USE` value
2. Ensure the required API keys are set
3. No code changes needed - the unified functions handle everything automatically
