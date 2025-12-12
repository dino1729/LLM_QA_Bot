# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LLM_QA_Bot is a multi-provider LLM application with document Q&A, web research, news aggregation, and specialized chatbots. It uses a FastAPI backend with a React frontend, supporting multiple LLM providers through a unified interface.

## Commands

### Running the Application
```bash
# Main application (port 7860)
python gradio_ui_full.py

# Newsletter generator
python year_progress_and_news_reporter_litellm.py
```

### Testing
```bash
# Run all tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=helper_functions --cov-report=term-missing

# Single test file
pytest tests/test_analyzers.py -v
```

### Frontend Development
```bash
cd frontend
npm install
npm run dev      # Development server
npm run build    # Production build
npm run lint     # ESLint
```

### Docker
```bash
docker build -t llmqabot .
docker run --restart always -p 7860:7860 --name llmqabot llmqabot
```

## Architecture

```
Frontend (React/Vite)  →  FastAPI (gradio_ui_full.py)  →  Helper Functions
                                    ↓
                    ┌───────────────┴───────────────┐
                    ↓                               ↓
              LLM Routing                   Content Processing
            (llm_client.py)                  (analyzers.py)
                    ↓                               ↓
    ┌───────────────┼───────────────┐        LlamaIndex RAG
    ↓               ↓               ↓
 LiteLLM         Ollama          Gemini/Groq/Cohere
```

### Key Components

**Entry Points:**
- `gradio_ui_full.py` - FastAPI server with REST API endpoints for all features
- `year_progress_and_news_reporter_litellm.py` - Standalone newsletter generator with multi-model strategy

**Helper Functions (`/helper_functions`):**
| File | Purpose |
|------|---------|
| `llm_client.py` | Unified LLM client for all providers (LiteLLM, Ollama, Gemini, Groq, Cohere) |
| `chat_generation.py` | Multi-provider chat completions |
| `chat_generation_with_internet.py` | Internet-connected chatbot using Firecrawl |
| `analyzers.py` | Content analysis (PDFs, articles, YouTube videos, media) |
| `news_researcher.py` | News aggregation with 6-hour cache TTL |
| `researcher.py` | GPT Researcher integration |
| `firecrawl_researcher.py` | Web scraping via Firecrawl |
| `query_supabasememory.py` | Memory Palace (Supabase) integration |
| `chat_gita.py` | Bhagavad Gita chatbot (Pinecone vector DB) |
| `trip_planner.py` | Travel itinerary generator |
| `food_planner.py` | Restaurant recommendations |
| `gptimage_tool.py` | OpenAI image generation/editing |
| `nvidia_image_gen.py` | NVIDIA NIM image generation |
| `audio_processors.py` | NVIDIA Riva audio transcription & TTS |

**Configuration:**
- `config/config.py` - Loads YAML config + .env overrides
- `config/config.example.yml` - Template with all settings
- `config/prompts.yml` - LLM prompt templates

## Design Patterns

1. **Provider-Agnostic LLM Interface:** `llm_client.py` abstracts all LLM providers behind a single interface

2. **Model Tiering:** Models organized as fast/smart/strategic for cost-performance optimization

3. **Local Vector Storage:** LlamaIndex manages vector/summary indexes locally for document Q&A

4. **Configuration Centralization:** All settings in `config/config.yml` with environment variable overrides

## Configuration

Copy `config/config.example.yml` to `config/config.yml` and configure:
- LLM provider API keys (OpenAI, Groq, Cohere, etc.)
- Supabase credentials (for Memory Palace)
- Pinecone credentials (for Gita chatbot)
- Firecrawl/Tavily keys (for web research)
- SMTP settings (for newsletter emails)

Environment variables can override config values via `.env` file.

## API Endpoints

Main endpoints in `gradio_ui_full.py`:
- `/api/analyze` - Content analysis (PDFs, articles, videos)
- `/api/chat` - Chat completions
- `/api/docqa` - Document Q&A with RAG
- `/api/models` - List available models
- Static files served from `/frontend/dist`

## Testing Notes

- Tests use mocked external services (LLMs, APIs)
- Coverage target: 80%
- Test files mirror source structure in `tests/`
