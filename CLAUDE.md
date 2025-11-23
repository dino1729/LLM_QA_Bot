# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LLM_QA_Bot is a multi-modal research and productivity workspace built with Gradio, LlamaIndex, and provider-agnostic LLM clients. It ingests documents, articles, media, and YouTube videos, builds local vector indexes, and provides conversational Q&A, chat completions across multiple LLM providers, planning utilities, and image generation.

## Core Architecture

### LLM Provider Abstraction
- **Central LLM Client**: `helper_functions/llm_client.py` provides `UnifiedLLMClient` that abstracts LiteLLM and Ollama behind an OpenAI-compatible interface
- **Custom LlamaIndex Integration**: `CustomOpenAILLM` and `CustomOpenAIEmbedding` bypass LlamaIndex's model validation to support any model on LiteLLM/Ollama
- **Provider Selection**: The client supports three tiers: "fast", "smart", "strategic" models, configured in `config/config.yml`
- **Model Prefix Handling**: LiteLLM/Ollama model names use prefixes (e.g., `openai:gpt-4`, `ollama:llama2`) in config but strip them before API calls

### Content Processing Pipeline
1. **Ingestion** (`helper_functions/analyzers.py`):
   - Files (PDF, DOCX, TXT, images, audio) → `analyze_file()`
   - YouTube videos → `analyze_ytvideo()` (extracts transcripts or downloads via pytube)
   - Articles → `analyze_article()` (newspaper3k or BeautifulSoup fallback)
   - Media URLs → `analyze_media()` (downloads, extracts audio via moviepy, transcribes with Whisper)

2. **Indexing**:
   - `build_index()` creates both `VectorStoreIndex` and `SummaryIndex` from documents
   - Persisted to local folders: `VECTOR_FOLDER` and `SUMMARY_FOLDER` (configured in `config/config.yml`)
   - Uses LlamaIndex's `SimpleDirectoryReader` to process files from `UPLOAD_FOLDER`

3. **Query Processing**:
   - Vector retrieval: `VectorIndexRetriever` with similarity search (top-k=10)
   - Summary generation: `SummaryIndex` with tree summarization
   - Example question generation: Uses LLM to create 8 follow-up questions

### Configuration System
- **Main Config**: `config/config.yml` contains all API keys, model names, paths, and settings
- **Environment Overrides**: Optional `config/.env` for secrets (higher precedence)
- **Prompts**: `config/prompts.yml` stores templates for summaries, examples, and Q&A
- **Access Pattern**: Import via `from config import config`, then `config.litellm_base_url`, etc.

### Gradio UI Structure
- **Azure-focused**: `azure_gradioui.py` (simpler, Azure OpenAI-centric)
- **Full Provider Switchboard**: `gradio_ui_full.py` (Azure, Gemini, Cohere, Groq, LiteLLM, Ollama)
- **Tabs**:
  - Content Analysis (file/video/article/media upload)
  - Memory Palace (Supabase-backed semantic search via `query_supabasememory.py`)
  - Multi-provider Chat (with optional Firecrawl/Tavily web search)
  - Holy Book Chatbot (Pinecone-backed Bhagavad Gita Q&A)
  - Trip Planner / Food Planner / Weather
  - Image Studio (OpenAI Images or NVIDIA NIM)

### Helper Functions Organization
All in `helper_functions/`:
- `analyzers.py` - Document/video/article/media processing
- `llm_client.py` - Unified LLM client factory
- `chat_generation.py` - Multi-provider chat completions
- `chat_generation_with_internet.py` - Chat with Firecrawl/Tavily web search
- `firecrawl_researcher.py` - Firecrawl-based web research
- `researcher.py` - Tavily-based web research
- `query_supabasememory.py` - Memory Palace streaming queries
- `chat_gita.py` - Pinecone-backed Bhagavad Gita chatbot
- `trip_planner.py` / `food_planner.py` - Planning agents
- `gptimage_tool.py` - OpenAI Images generation/editing
- `nvidia_image_gen.py` - NVIDIA NIM image generation
- `audio_processors.py` - NVIDIA Riva ASR/TTS

## Common Development Commands

### Environment Setup
```bash
# Activate virtual environment
source venv/bin/activate
# Or on Windows:
# .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy and configure settings
cp config/config.example.yml config/config.yml
# Edit config/config.yml with your API keys and model preferences
```

### Running the Application
```bash
# Azure-focused UI
python azure_gradioui.py

# Full provider UI (recommended)
python gradio_ui_full.py

# Docker
docker build -t llmqabot .
docker run --restart always -p 7860:7860 --name llmqabot llmqabot
```

### Testing
```bash
# Run all tests (161 tests, 80% coverage)
pytest tests/ -v

# Run specific test file
pytest tests/test_analyzers.py -v
pytest tests/test_chat_generation.py -v
pytest tests/test_llm_client.py -v

# Run with coverage report
pytest tests/ --cov=helper_functions --cov-report=term-missing

# HTML coverage report
pytest tests/ --cov=helper_functions --cov-report=html
# View at htmlcov/index.html

# Stop on first failure
pytest tests/ -x

# Show print statements (useful for debugging)
pytest tests/ -v -s

# Run only failed tests from last run
pytest tests/ --lf
```

### Linting and Formatting
```bash
# Format code with Black
black .

# Sort imports with isort
isort .

# Type checking with mypy
mypy helper_functions/

# Lint with flake8
flake8 helper_functions/

# Lint with pylint
pylint helper_functions/
```

## Key Implementation Patterns

### Adding a New LLM Provider
1. Add provider credentials to `config/config.yml` (e.g., `new_provider_api_key`)
2. Update `config/config.py` to load the new config values
3. Modify `helper_functions/llm_client.py`:
   - Add provider logic in `UnifiedLLMClient.__init__()` (similar to litellm/ollama branches)
   - Set `base_url`, `api_key`, model names for fast/smart/strategic tiers
4. Update `helper_functions/chat_generation.py` to support the new provider in `generate_chat_completion()`
5. Add UI dropdown option in Gradio interface files
6. Write tests in `tests/test_llm_client.py` and `tests/test_chat_generation.py`

### Adding a New Content Analyzer
1. Implement analyzer function in `helper_functions/analyzers.py`:
   - Follow pattern: `analyze_<type>(url_or_files, memorize)`
   - Download/extract content → save to `UPLOAD_FOLDER`
   - Call `build_index()` → `summary_generator()` → `example_generator()`
   - Return dict: `{"message": ..., "summary": ..., "example_queries": ..., "<type>_title": ..., "<type>_memoryupload_status": ...}`
2. Add Gradio UI components in `azure_gradioui.py` or `gradio_ui_full.py`:
   - Create tab with input components
   - Wire up button click to call analyzer function
   - Display results in output components
3. Write comprehensive tests in `tests/test_analyzers.py` (follow existing patterns with mocking)

### Working with Vector Indexes
- Indexes are persisted to disk in folders defined by `config.VECTOR_FOLDER` and `config.SUMMARY_FOLDER`
- To rebuild: Delete folder contents or call `clearallfiles()` + re-upload content
- Query with `ask_query(question)` which loads the persisted `VectorStoreIndex`
- Summary-based queries use `ask_fromfullcontext(question, template)` with `SummaryIndex`

### Custom Models and Embeddings
- The `CustomOpenAILLM` and `CustomOpenAIEmbedding` classes in `llm_client.py` are designed to work with **any** model name
- They bypass LlamaIndex's built-in OpenAI model validation
- Support asymmetric embeddings (NVIDIA NV-Embed) with `input_type` parameter differentiation for queries vs passages

### Reasoning Model Support
- The `CustomOpenAILLM` class automatically handles reasoning models (o1, o3, gpt-oss-120b)
- Falls back to `reasoning_content` field when `content` is empty in the response

## Critical Dependencies

- **ffmpeg**: Required for audio/video processing (Whisper, moviepy). Install via system package manager.
- **LlamaIndex**: Core indexing and retrieval framework. Uses `Settings` singleton for LLM and embedding model configuration.
- **Whisper**: Audio transcription (base model). Loaded on-demand in `analyzers.py`.
- **pytube**: YouTube video downloading (fallback when transcripts unavailable).
- **newspaper3k**: Article extraction (with BeautifulSoup fallback).
- **Gradio**: Web UI framework. All UIs serve on port 7860.

## Testing Strategy

- **Fixtures**: Defined in `tests/conftest.py` for temp folders, mock clients, mock services
- **Coverage Goal**: 90%+ (currently 80% overall)
- **Mocking Approach**:
  - External APIs mocked with `unittest.mock` or `responses`
  - Supabase, OpenAI, Firecrawl, NVIDIA services all have mock fixtures
  - File system operations use temporary directories
- **Test Organization**: One file per module, classes group tests by function, methods cover scenarios
- **Key Test Files**:
  - `test_analyzers.py` (92% coverage, 60+ tests)
  - `test_llm_client.py` (tests UnifiedLLMClient, custom LLM/embedding classes)
  - `test_chat_generation.py` (86% coverage, multi-provider chat)
  - `test_chat_generation_with_internet.py` (74% coverage, Firecrawl integration)

## Troubleshooting

### Common Issues
- **"ffmpeg not found"**: Install ffmpeg via system package manager
- **Embeddings/chat fail**: Check API keys in `config/config.yml` or `config/.env`, restart app
- **Vector index errors**: Delete `VECTOR_FOLDER` and `SUMMARY_FOLDER`, re-upload content
- **LlamaIndex model validation errors**: Ensure you're using `CustomOpenAILLM` / `CustomOpenAIEmbedding` from `llm_client.py`, not stock LlamaIndex classes
- **Supabase RPC errors**: Verify `mp_search` stored procedure exists with correct permissions

### Debugging Analyzer Issues
1. Check `UPLOAD_FOLDER` for downloaded/uploaded files
2. Verify `VECTOR_FOLDER` and `SUMMARY_FOLDER` contain `docstore.json`, `index_store.json`, etc.
3. Enable debug logging in `analyzers.py` (change `logging.CRITICAL` to `logging.DEBUG`)
4. Run with `pytest -v -s` to see print statements

### LLM Client Debugging
- Check model names in `config/config.yml` (ensure correct tier: fast/smart/strategic)
- Verify base URL and API key for provider
- Test with direct `UnifiedLLMClient` instantiation in Python REPL
- Confirm model name prefix stripping (prefixes should be removed before API calls)

## Related Documentation

- README.md: User-facing setup and feature documentation
- tests/README.md: Comprehensive pytest usage guide
- tests/TEST_SUMMARY.md: Detailed coverage breakdown and failure analysis
- config/config.example.yml: Full configuration reference with comments
