# LLM_QA_Bot Project Context

## Project Overview
LLM_QA_Bot is a comprehensive multi-modal research and productivity workspace. It serves as a central hub for ingesting various forms of content (documents, audio, video, URLs), indexing them for efficient retrieval, and providing a suite of AI-powered tools through a Gradio interface.

The system features a "Memory Palace" for long-term knowledge retention (backed by Supabase) and supports a wide array of LLM providers (Azure OpenAI, Gemini, Cohere, Groq, LiteLLM, Ollama) allowing for flexible model switching.

## Key Technologies
*   **Language**: Python 3.10+
*   **UI Framework**: Gradio
*   **Orchestration & Indexing**: LlamaIndex
*   **LLM Interaction**: LiteLLM (proxy), Ollama (local), Native Provider SDKs (OpenAI, Gemini, etc.)
*   **Vector/Memory Storage**: Local vector stores, Supabase (Memory Palace)
*   **Media Processing**: ffmpeg, Whisper (audio transcription)
*   **Web Research**: Firecrawl, Tavily

## Project Structure

### Core Directories
*   `config/`: Configuration files (`config.yml`, `prompts.yml`).
    *   `config.yml`: Central configuration for API keys, model selections, and paths.
    *   `prompts.yml`: Templates for system prompts, summaries, and Q&A.
*   `helper_functions/`: Contains the business logic and backend modules.
    *   `llm_client.py`: A unified client wrapper for LiteLLM and Ollama, abstracting model tiers ("fast", "smart", "strategic").
    *   `analyzers.py`: Logic for processing content (PDFs, videos, etc.) and building indexes.
    *   `query_supabasememory.py`: Interface for the Supabase Memory Palace.
    *   `chat_generation.py` & `chat_generation_with_internet.py`: Chat logic.
*   `tests/`: Comprehensive pytest suite (>150 tests) covering analyzers, planners, and integration points.
*   `data/`: Default location for local indexes (`vector_index`, `summary_index`).

### Entry Points
*   `azure_gradioui.py`: Launches the Gradio UI with an Azure-focused layout.
*   `gradio_ui_full.py`: Launches the full-featured Gradio UI with a switchboard for all supported providers.

## Setup and Configuration

1.  **Environment**: Python 3.10+ virtual environment recommended.
2.  **Dependencies**: `pip install -r requirements.txt`. (Requires `ffmpeg` installed on the system).
3.  **Configuration**:
    *   Copy `config/config.example.yml` to `config/config.yml`.
    *   Populate API keys for desired providers (OpenAI, Gemini, Cohere, etc.).
    *   Configure `paths` for local storage or accept defaults.
    *   Set up LLM tiers (Fast, Smart, Strategic) in the config for LiteLLM/Ollama routing.

## Building and Running

### Run the Application
*   **Full UI**: `python gradio_ui_full.py`
*   **Azure UI**: `python azure_gradioui.py`
*   **Docker**:
    ```bash
    docker build -t llmqabot .
    docker run --restart always -p 7860:7860 --name llmqabot llmqabot
    ```

### Testing
*   Run full suite: `pytest tests -v`
*   Run specific component: `pytest tests/test_analyzers.py`

## Development Conventions

*   **LLM Abstraction**: Use `helper_functions.llm_client.UnifiedLLMClient` (or `get_client`) for model interactions. It normalizes inputs for LiteLLM and Ollama, supporting "fast", "smart", and "strategic" model tiers defined in config.
*   **Testing**: Tests use `unittest.mock` extensively (see `tests/conftest.py`) to avoid hitting live APIs during routine testing. Ensure new features have corresponding tests.
*   **Configuration**: Avoid hardcoding values. Expose tunable parameters in `config/config.yml` and retrieve them via the `config` module.
