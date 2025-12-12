# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LLM_QA_Bot is a multi-modal research and productivity workspace that ingests documents, videos, audio, and web content, builds vector indexes for retrieval, and provides AI-powered tools through a Gradio interface. It features a "Memory Palace" for long-term knowledge storage (Supabase) and supports multiple LLM providers through a unified client interface.

## Core Architecture

### Unified LLM Client Pattern
The codebase uses a **unified client abstraction** (`helper_functions/llm_client.py`) that normalizes interactions across LiteLLM and Ollama providers. This is the most important architectural pattern to understand:

- **UnifiedLLMClient**: Factory pattern that creates OpenAI-compatible clients for LiteLLM or Ollama
- **Model Tiers**: Three tiers defined in config - "fast" (quick tasks), "smart" (balanced), "strategic" (complex reasoning)
- **Custom LLM Classes**: `CustomOpenAILLM` and `CustomOpenAIEmbedding` bypass LlamaIndex's model validation to support any model on LiteLLM/Ollama
- **Usage**: Always use `get_client(provider="litellm", model_tier="smart")` rather than directly instantiating OpenAI/Gemini/etc. clients

### Content Analysis Pipeline
The `helper_functions/analyzers.py` module orchestrates the entire content processing flow:

1. **Input Validation** → `fileformatvaliditycheck()` checks extensions
2. **Content Extraction** → Different handlers for PDFs, videos (YouTube), audio, articles
3. **Transcription** → Whisper for audio, YouTube Transcript API for videos (with fallback to download+Whisper)
4. **Index Building** → LlamaIndex creates both VectorStoreIndex (for Q&A) and SummaryIndex (for summaries)
5. **Summary Generation** → Uses the "fast" tier model with templates from `config/prompts.yml`
6. **Memory Palace** → Optional Supabase storage for long-term retrieval

### Configuration System
All runtime configuration lives in `config/config.yml` (template at `config/config.example.yml`):

- **LiteLLM Config**: `litellm_base_url`, `litellm_fast_llm`, `litellm_smart_llm`, `litellm_strategic_llm`, `litellm_embedding`
- **Ollama Config**: Parallel structure with `ollama_` prefix
- **Provider Keys**: `google_api_key`, `groq_api_key`, `cohere_api_key`, etc.
- **Paths**: `UPLOAD_FOLDER`, `SUMMARY_FOLDER`, `VECTOR_FOLDER`, `WEB_SEARCH_FOLDER`
- **Settings**: `temperature`, `max_tokens`, `context_window`, `default_chatbot_model`

Prompts and templates are in `config/prompts.yml` (sum_template, eg_template, ques_template, system_prompt_content).

### Entry Points
- **gradio_ui_full.py**: Full-featured UI with provider switchboard (main entry point)
- **misc_scripts/azure_gradioui.py**: Older Azure-focused UI (moved to misc_scripts)
- **year_progress_and_news_reporter_litellm.py**: Standalone script for generating news reports

## Common Development Commands

### Environment Setup
```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies (requires ffmpeg on system PATH)
pip install -r requirements.txt

# Copy and configure settings
cp config/config.example.yml config/config.yml
# Edit config/config.yml with your API keys
```

### Running the Application
```bash
# Run the main Gradio UI (default port 7860)
python gradio_ui_full.py

# Run with Docker
docker build -t llmqabot .
docker run --restart always -p 7860:7860 --name llmqabot llmqabot
```

### Testing
```bash
# Run all tests
pytest tests -v

# Run tests with coverage report
pytest tests --cov=helper_functions --cov-report=term-missing

# Run specific test module
pytest tests/test_analyzers.py -v
pytest tests/test_chat_generation.py -v
pytest tests/test_llm_client.py -v

# Generate HTML coverage report
pytest tests --cov=helper_functions --cov-report=html
# View at: htmlcov/index.html
```

### Development Tools
```bash
# Code formatting and linting
black helper_functions/
isort helper_functions/
mypy helper_functions/
flake8 helper_functions/
pylint helper_functions/

# Clear all local indexes and uploads (useful for testing)
rm -rf data/vector_index/* data/summary_index/* data/uploads/*
```

## Key Modules and Their Responsibilities

### helper_functions/llm_client.py
Unified LLM client for LiteLLM and Ollama. Provides:
- `UnifiedLLMClient`: Main client class with `chat_completion()`, `get_embedding()`, `get_llamaindex_llm()`, `get_llamaindex_embedding()`
- `CustomOpenAILLM`: LlamaIndex-compatible LLM that bypasses model validation
- `CustomOpenAIEmbedding`: Handles asymmetric embeddings (NVIDIA NIM models with input_type parameter)
- `get_client()`: Factory function - use this as the primary interface

**Important**: The client strips provider prefixes (e.g., "openai:" or "ollama:") from model names when making API calls.

### helper_functions/analyzers.py
Content processing and indexing. Key functions:
- `analyze_file(files, lite_mode_in)`: Process uploaded files (PDF, TXT, DOCX, images, audio)
- `analyze_article(url, lite_mode_in)`: Extract and analyze articles from URLs
- `analyze_ytvideo(url, lite_mode_in)`: Process YouTube videos (transcript or full download)
- `analyze_media(url, lite_mode_in)`: Generic media handler
- `build_index()`: Create VectorStoreIndex and SummaryIndex from documents
- `get_answer(question)`: Query the vector index with a question

**Lite Mode**: When `lite_mode=True`, skips video downloads for YouTube URLs without transcripts.

### helper_functions/chat_generation.py
Multi-provider chat interface using `generate_chat(query, selected_model, chat_history)`:
- Supports GEMINI, GEMINI_THINKING, GROQ variants, LITELLM tiers, OLLAMA tiers, COHERE
- Uses native SDKs (google.generativeai, groq, cohere) for provider-specific features
- Falls back to unified client for LiteLLM/Ollama
- Returns tuple: `(chat_history, chat_history)` for Gradio chatbot component

### helper_functions/chat_generation_with_internet.py
Internet-connected chatbot with web research capabilities:
- `internet_connected_chatbot()`: Main function with Firecrawl/Tavily integration
- `scrape_firecrawl()`, `query_firecrawl()`: Scrape and search using Firecrawl server
- `fetch_bing_news()`, `fetch_tavily_news()`: News retrieval
- `web_reader()`, `save_to_folder()`: Download and process web content

### helper_functions/query_supabasememory.py
Memory Palace interface for Supabase-backed long-term storage:
- Uses Azure OpenAI for embeddings (separate from main unified client)
- `generate_embeddings()`: Create embeddings for semantic search
- `query_memory()`: Stream results from Supabase RPC function `mp_search`
- Requires: `supabase_service_role_key`, `public_supabase_url`, Azure embedding credentials

### helper_functions/chat_gita.py
Bhagavad Gita chatbot using Pinecone vector database:
- `extract_context_frompinecone(query)`: Retrieve relevant verses
- `ask_bhagawatgeeta(query, chat_history)`: Query with context from Pinecone
- Requires: `pinecone_api_key`, `pinecone_environment` in config

### Specialized Tools
- **trip_planner.py**: `generate_trip_plan()` creates day-by-day itineraries with weather
- **food_planner.py**: `craving_satisfier()` suggests random food based on preferences
- **gptimage_tool.py**: OpenAI image generation/editing (uses direct OpenAI SDK)
- **nvidia_image_gen.py**: NVIDIA NIM image generation
- **researcher.py**: LlamaIndex-based web research agent
- **news_researcher.py**: Fetches and summarizes Bing/Tavily news

## Testing Architecture

The test suite uses pytest with extensive mocking (fixtures in `tests/conftest.py`):

### Common Fixtures
- `temp_upload_folder`, `temp_summary_folder`, `temp_vector_folder`: Temporary directories
- `mock_file`: Mock file objects for upload testing
- Extensive use of `@patch` decorators to mock external services (OpenAI, Supabase, Firecrawl, NVIDIA)

### Test Structure
Tests are organized by module with 80%+ code coverage target (currently ~80%):
- `test_analyzers.py`: 60+ tests, 92% coverage
- `test_chat_generation.py`: 25+ tests, 86% coverage
- `test_chat_generation_with_internet.py`: 45+ tests, 74% coverage
- `test_llm_client.py`: Tests unified client and custom LLM classes
- `test_*_planner.py`: Trip planner, food planner tests

**Testing Patterns**:
- Mock external API calls to avoid hitting live services
- Use temporary folders that are cleaned up after each test
- Verify both success and error handling paths
- Test edge cases (empty inputs, invalid formats, missing config)

## Important Gotchas

### ffmpeg Dependency
The application requires `ffmpeg` on the system PATH for audio/video processing (used by moviepy and Whisper). Install before running:
```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg

# Windows
# Download from https://ffmpeg.org and add to PATH
```

### Reasoning Models
Some models (DeepSeek R1, OpenAI o1, o3) return content in `reasoning_content` field instead of `content`. The unified client handles this automatically (see llm_client.py:123-130, 168-175, 289-296).

### Asymmetric Embeddings
NVIDIA NIM embedding models require `input_type` parameter ("query" vs "passage"). The `CustomOpenAIEmbedding` class detects these models and adds the parameter automatically (llm_client.py:33, 42, 53).

### Model Name Prefixes
LiteLLM and Ollama models may have provider prefixes (e.g., "openai:gpt-4"). The unified client strips these before making API calls, so the proxy sees the correct model name (llm_client.py:276-279).

### Index Persistence
Vector and summary indexes are persisted to disk in `VECTOR_FOLDER` and `SUMMARY_FOLDER`. If you encounter stale data or errors, clear these folders and rebuild indexes.

### Memory Palace Configuration
The Memory Palace feature requires **Azure OpenAI** credentials (separate from the main unified client) because it uses a specific Azure embedding deployment. This is hardcoded in `query_supabasememory.py:48-50` and cannot use the unified client.

## Adding New Features

### Adding a New LLM Provider
1. Add configuration keys to `config/config.yml`
2. Update `helper_functions/chat_generation.py` to handle the new provider
3. If the provider is OpenAI-compatible, consider routing through LiteLLM instead of native SDK

### Adding a New Content Type
1. Update `fileformatvaliditycheck()` in `analyzers.py` to accept the new extension
2. Add extraction logic (LlamaIndex has readers for many formats)
3. Follow the existing pattern: extract → build index → generate summary → return

### Adding a New Tool/Agent
1. Create a new module in `helper_functions/`
2. Follow the pattern of using `get_client()` for LLM calls
3. Add configuration keys to `config.yml` if needed
4. Wire it into `gradio_ui_full.py` as a new tab or component
5. Write tests in `tests/test_<module_name>.py`

## Configuration Best Practices

- Keep secrets in `config/.env` (not tracked in git)
- Use `config/config.yml` for non-secret configuration
- Test configurations in `config/config.example.yml` should be safe to commit
- The config system merges `.env` values into the config module (see `config/config.py:12-16`)
