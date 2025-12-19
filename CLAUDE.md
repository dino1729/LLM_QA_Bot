# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LLM_QA_Bot is a multi-modal research and productivity workspace with provider-agnostic LLM routing, document ingestion, Memory Palace integration, and specialized agents for planning, image generation, and news aggregation. The architecture decouples provider selection from core logic through a unified client interface.

## Core Architecture

### Multi-Provider LLM Abstraction (`helper_functions/llm_client.py`)

The **UnifiedLLMClient** provides the foundation for all LLM interactions:

- **Provider-agnostic interface**: Supports LiteLLM (gateway to 100+ models), Ollama (local models), Azure OpenAI, Gemini, Cohere, and Groq through a single API surface
- **Three-tier model selection**: `fast` (quick responses), `smart` (balanced), `strategic` (complex reasoning) - configured in `config/config.yml` per provider
- **LlamaIndex integration**: Custom `CustomOpenAILLM` and `CustomOpenAIEmbedding` classes bypass LlamaIndex's model validation, enabling any OpenAI-compatible model to work with document indexing
- **Asymmetric embedding support**: Handles NVIDIA NIM models that require `input_type` parameter ("query" vs "passage")

**Usage pattern**:
```python
from helper_functions.llm_client import get_client

# Get a client for a specific provider and tier
client = get_client(provider="litellm", model_tier="smart")
response = client.chat_completion(messages=[...])

# For LlamaIndex operations (document Q&A)
llm = client.get_llamaindex_llm()
embed_model = client.get_llamaindex_embedding()
```

**When adding new features**: Always use `get_client()` instead of directly importing OpenAI, Gemini, etc. This ensures consistent provider routing and configuration.

### Configuration System (`config/config.py` + `config/config.yml`)

All runtime settings live in `config/config.yml` with no hardcoded model defaults:

- **Provider sections**: Each provider (litellm, ollama, azure, gemini, cohere, groq) has dedicated keys for API credentials and model names
- **Tier mappings**: Models are mapped to `fast_llm`, `smart_llm`, `strategic_llm` per provider
- **Path configuration**: Upload folders, vector indexes, and summary stores are configurable
- **Prompts**: Separate `config/prompts.yml` for summarization, question generation, and system prompts
- **Secrets**: Use `config/.env` (git-ignored) for API keys; `config.py` loads both YAML and .env

**Key principle**: Never hardcode model names or API endpoints. Reference `config.model_name` or use the `get_client()` abstraction.

### Helper Functions Organization (`helper_functions/`)

Each module in `helper_functions/` is self-contained with a clear responsibility:

- **`analyzers.py`**: Document/video/article ingestion → LlamaIndex vector stores
- **`chat_generation.py`**: Multi-provider chat completions with streaming support
- **`chat_generation_with_internet.py`**: Internet-connected chat using Firecrawl/Tavily for web research
- **`firecrawl_researcher.py`**: Web scraping and research agent with tool calling
- **`llm_client.py`**: Provider abstraction layer (core architecture component)
- **`query_supabasememory.py`**: Memory Palace semantic search with streaming responses
- **`audio_processors.py`**: NVIDIA Riva ASR/TTS integration
- **`tts_vibevoice.py`**: On-device TTS using VibeVoice models (GPU-accelerated)
- **`newsletter_generation.py`**: News aggregation and summarization
- **`trip_planner.py`, `food_planner.py`**: Specialized planning agents
- **`gptimage_tool.py`, `nvidia_image_gen.py`**: Image generation/editing workflows

**Pattern**: When adding functionality, create a new helper module or extend an existing one. Each module should have a corresponding test file in `tests/`.

### Document Processing Flow (LlamaIndex Integration)

1. **Upload** → Saved to `UPLOAD_FOLDER` (configured in `config/config.yml`)
2. **Parse** → LlamaIndex readers extract text (PDF, DOCX, audio via Whisper, video via yt-dlp)
3. **Index** → Two indexes created:
   - **VectorStoreIndex**: Semantic search over document chunks (in `VECTOR_FOLDER`)
   - **SummaryIndex**: Full document summarization (in `SUMMARY_FOLDER`)
4. **Query** → User questions routed to vector store; responses generated via LlamaIndex chat engine with the configured LLM/embedding model

**Important**: Both indexes persist to disk. To rebuild indexes, delete the folders or use the "Clear" button in the UI.

### Memory Palace Integration (`helper_functions/query_supabasememory.py`)

The Memory Palace feature stores analyzed content in Supabase for long-term semantic search:

- **Storage**: Documents pushed to Supabase with Azure OpenAI embeddings
- **Retrieval**: `mp_search` RPC in Supabase performs similarity search
- **Streaming**: Results stream back to UI via Server-Sent Events (SSE)

**Prerequisites**: Requires `supabase_service_role_key` and `public_supabase_url` in `config/config.yml`, plus the `mp_search` stored procedure in Supabase.

### Testing Architecture (`tests/`)

- **161 total tests** with 80% coverage (targeting 90%+)
- **Shared fixtures** in `tests/conftest.py`: temp folders, mock clients (OpenAI, Riva, Supabase, Firecrawl)
- **Offline by default**: All tests use mocks to avoid hitting external APIs
- **Test naming**: `test_<module>.py` → `TestFunctionName` → `test_scenario_description`

**Test pattern**:
```python
@patch('helper_functions.module_name.get_client')
def test_feature(mock_get_client):
    mock_client = Mock()
    mock_client.chat_completion.return_value = "Expected output"
    mock_get_client.return_value = mock_client
    # Test the feature
```

## Common Development Tasks

### Running the Application

```bash
# Full multi-provider UI (primary interface)
python gradio_ui_full.py

# Azure-focused layout
python misc_scripts/azure_gradioui.py

# Frontend dev server (React/Vite)
cd frontend && npm install && npm run dev

# Docker
docker build -t llmqabot .
docker run -p 7860:7860 --name llmqabot llmqabot
```

**Note**: `gradio_ui_full.py` serves both Gradio UI and `/api/*` routes for the frontend on port 7860.

### Testing

```bash
# Run all tests
pytest tests/ -v

# Single test file
pytest tests/test_analyzers.py -v

# With coverage
pytest tests/ --cov=helper_functions --cov-report=term-missing

# Specific test
pytest tests/test_analyzers.py::TestClearAllFiles::test_clearallfiles_empty_folder -v

# Show print statements (useful for debugging)
pytest tests/ -v -s

# Stop on first failure
pytest tests/ -x
```

**Test markers**: Use `-m "not slow"` to skip slow tests, `-m "not integration"` for unit tests only.

### Adding a New LLM Provider

1. Add provider config to `config/config.yml`:
   ```yaml
   newprovider_api_key: ""
   newprovider_base_url: ""
   newprovider_fast_llm: "model-name"
   newprovider_smart_llm: "model-name"
   ```

2. Extend `UnifiedLLMClient.__init__()` in `helper_functions/llm_client.py`:
   ```python
   elif provider == "newprovider":
       self.base_url = config.newprovider_base_url
       self.api_key = config.newprovider_api_key
       # Map tiers to models
   ```

3. Add tests to `tests/test_llm_client.py`

4. Update README.md with provider-specific setup instructions

### Working with VibeVoice TTS (Local GPU Audio)

VibeVoice provides on-device text-to-speech as an alternative to NVIDIA Riva cloud API:

```bash
# Use local TTS instead of Riva
python year_progress_and_news_reporter_litellm.py --local-tts

# List available voices
python year_progress_and_news_reporter_litellm.py --list-voices

# Use specific voice
python year_progress_and_news_reporter_litellm.py --local-tts --voice en-mike_man
```

**Architecture**: `helper_functions/tts_vibevoice.py` wraps the VibeVoice model with caching (`_vibevoice_tts_cache`) to reuse instances per speaker. Supports expressive control via `temperature`, `top_p`, `cfg_scale`, and `excitement_level`.

**Troubleshooting CUDA errors with VibeVoice**:
If you see `CUDA error: no kernel image is available for execution on the device`:
1. Check your GPU compute capability: `python -c "import torch; print(torch.cuda.get_device_capability())"`
2. For Blackwell GPUs (RTX 5090, etc.) with compute capability 12.0:
   - Verify PyTorch version: `python -c "import torch; print(torch.__version__)"`
   - Should show `2.11.0.dev` or later with `+cu128`
   - If not, install nightly: `pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128 --upgrade`
3. Verify NVIDIA driver: `nvidia-smi --query-gpu=driver_version --format=csv,noheader`
   - Should be R570 or higher for Blackwell support
4. Test GPU access: `python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"`

### Newsletter Generation Workflow

The `year_progress_and_news_reporter_litellm.py` script demonstrates the daily bundle pattern:

1. **Gather data**: Weather, quote, lesson, news (tech/financial/India)
2. **Generate content**: Newsletter sections via LLM, voicebot script
3. **Bundle**: JSON structure (`build_daily_bundle`) with all data
4. **Render**: HTML templates for email + web viewing
5. **Output**: Audio (TTS), email, saved to `newsletter_research_data/`

**Testing modes**:
```bash
# Fast iteration (use cached news, no email/audio)
python year_progress_and_news_reporter_litellm.py --test

# Full cache mode (load all data from bundle, skip LLM generation, only generate audio)
python year_progress_and_news_reporter_litellm.py --full-cache --skip-email --local-tts

# HTML redesign (regenerate from latest bundle)
python year_progress_and_news_reporter_litellm.py --html-only

# Check cache status
python year_progress_and_news_reporter_litellm.py --cache-info
```

**Key files**:
- `helper_functions/news_cache.py`: 24-hour cache for news data
- `helper_functions/progress_bundle.py`: Bundle creation/loading
- `helper_functions/html_templates.py`: Email/web rendering

## Important Patterns and Conventions

### Provider Selection in Code

**Always** use the provider abstraction:
```python
# ✅ Correct
from helper_functions.llm_client import get_client
client = get_client(provider="litellm", model_tier="smart")

# ❌ Avoid - bypasses unified routing
from openai import OpenAI
client = OpenAI(...)
```

### Handling Reasoning Models (DeepSeek, o1, etc.)

Some models return content in `reasoning_content` instead of `content`. The `UnifiedLLMClient` handles this automatically:

```python
# In llm_client.py chat_completion()
content = message.content
if not content and hasattr(message, 'reasoning_content'):
    content = message.reasoning_content
```

### Streaming Responses

For real-time UI feedback, use streaming:
```python
# Chat streaming
for chunk in client.stream_chat_completion(messages=[...]):
    print(chunk, end="", flush=True)

# Memory Palace streaming (SSE)
from helper_functions.query_supabasememory import search_memorypalace_stream
for chunk in search_memorypalace_stream(query, client):
    yield chunk
```

### Error Handling Best Practices

- **Graceful degradation**: If Firecrawl/Tavily fails, return cached results or basic search
- **Mock in tests**: Use `@patch` and `Mock()` to simulate errors without network calls
- **Log clearly**: Use `logger.info()` / `logger.error()` with context
- **User feedback**: Return meaningful error messages to the UI (not stack traces)

### Configuration Validation

When adding new config keys, update `config/config.py` with safe defaults:
```python
# Provide fallback for optional features
new_feature_key = config_yaml.get("new_feature_key", "")
```

Then document in README.md under "Configuration Reference".

## Repository-Specific Context

### Git Status Note (as of session start)

The repository has unstaged changes:
- Modified: `helper_functions/tts_vibevoice.py`, `year_progress_and_news_reporter_litellm.py`
- Deleted: `CLAUDE.md`, `test_model_listing.py`
- Untracked: New test files in `tests/` (vibevoice, newsletter)

**Before committing**: Run `pytest tests/ -v` to ensure tests pass, and stage related test files with implementation changes.

### Known Dependencies

- **ffmpeg**: Required for Whisper and moviepy (audio/video processing)
- **NVIDIA Riva**: Optional cloud TTS/ASR service (can use VibeVoice locally instead)
- **Supabase**: Required only for Memory Palace feature
- **LiteLLM**: Primary LLM gateway (can run locally or use hosted proxy)
- **PyTorch with CUDA 12.8+**: Required for VibeVoice TTS on NVIDIA Blackwell GPUs (RTX 5090, etc.)

Install core deps: `pip install -r requirements.txt`

**Important for NVIDIA Blackwell GPUs (RTX 5090, 5080, etc.)**:
- Requires **PyTorch nightly** with CUDA 12.8 support
- Requires **NVIDIA Driver R570+**
- Standard PyTorch releases (as of Dec 2024) only support up to compute capability sm_90
- Blackwell architecture uses compute capability sm_120
- Install command: `pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128`
- See [NVIDIA Blackwell Software Migration Guide](https://forums.developer.nvidia.com/t/software-migration-guide-for-nvidia-blackwell-rtx-gpus-a-guide-to-cuda-12-8-pytorch-tensorrt-and-llama-cpp/321330) for details

### Frontend (React + Vite)

- Entry: `frontend/src/App.tsx` (tabbed interface)
- API client: `frontend/src/api.ts` (typed fetch helpers to `/api/*`)
- Backend integration: `gradio_ui_full.py` serves both Gradio UI and FastAPI routes

**Development**: Run backend first (`python gradio_ui_full.py`), then `cd frontend && npm run dev`.

## Testing Guardrails

- **Always add tests** for new helper functions (target 90%+ coverage)
- **Use fixtures** from `tests/conftest.py` for temp folders and mocks
- **Parameterize** edge cases instead of duplicating test code
- **Mock external APIs** (OpenAI, Supabase, Firecrawl, Riva) to keep tests offline
- **Run coverage** before PRs: `pytest tests/ --cov=helper_functions --cov-report=term-missing`

## Security Notes

- **Never commit** API keys or secrets (use `config/.env`)
- **Clear transient data** before committing (uploads, caches, vector indexes)
- **Validate inputs** in analyzers (file extensions, URL schemes)
- **Supabase RLS**: Memory Palace requires service role key; ensure proper RLS policies in production

## Quick Reference

| Task | Command |
|------|---------|
| Run main UI | `python gradio_ui_full.py` |
| Run tests | `pytest tests/ -v` |
| Coverage report | `pytest tests/ --cov=helper_functions --cov-report=term-missing` |
| Single test file | `pytest tests/test_analyzers.py -v` |
| Frontend dev | `cd frontend && npm run dev` |
| Newsletter (test mode) | `python year_progress_and_news_reporter_litellm.py --test` |
| Newsletter (local TTS) | `python year_progress_and_news_reporter_litellm.py --local-tts` |
| Newsletter (full cache) | `python year_progress_and_news_reporter_litellm.py --full-cache --skip-email --local-tts` |
| List LLM models | `python -c "from helper_functions.llm_client import list_available_models; print(list_available_models('litellm'))"` |
| Setup Blackwell GPU | See `docs/BLACKWELL_GPU_SETUP.md` |

## Architecture Decision Records

### Why UnifiedLLMClient instead of direct provider SDKs?

- **Consistency**: Single interface eliminates per-provider quirks
- **Testability**: Mock once at the client level instead of per-provider
- **Flexibility**: Switch providers via config without code changes
- **LlamaIndex compatibility**: Custom classes bypass model validation, allowing any model to work with document indexing

### Why separate vector and summary indexes?

- **VectorStoreIndex**: Optimized for semantic search over chunks (precise retrieval)
- **SummaryIndex**: Provides high-level document overview (context for follow-up questions)
- Both persist to disk for fast subsequent queries without re-indexing

### Why helper_functions/ pattern?

- **Modularity**: Each file has one responsibility (analyzers, chat, planners, etc.)
- **Testability**: Each module gets its own test file with focused coverage
- **Discoverability**: Clear naming (`trip_planner.py`, `news_researcher.py`) vs monolithic `utils.py`

### Why both Gradio and React frontends?

- **Gradio**: Rapid prototyping, ML-friendly interfaces, built-in authentication
- **React**: Custom UX, better state management, reusable components
- Both share the same FastAPI backend routes (`/api/*`), allowing parallel development
