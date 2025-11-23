# LLM_QA_Bot

## Project Overview

**LLM_QA_Bot** is a comprehensive, multi-modal research and productivity workspace designed to centralize knowledge ingestion, retrieval, and synthesis. It acts as a "second brain," allowing users to analyze documents, videos, and web content through a unified interface.

### Key Capabilities
*   **Content Ingestion:** Drag-and-drop support for PDFs, DOCX, text files, images, and audio.
*   **Media Analysis:** Automatic transcription and summarization of YouTube videos and audio files using Whisper.
*   **Knowledge Retrieval:** Uses LlamaIndex to build local vector and summary indexes for instant Q&A against uploaded content.
*   **Memory Palace:** Integration with Supabase to persist and search long-term memories and insights.
*   **Multi-Provider LLM Support:** agnostic client switching between Azure OpenAI, Google Gemini, Cohere, Groq, LiteLLM, and local Ollama models.
*   **AI Agents:** Specialized tools for browsing (Firecrawl/Tavily), planning (Trip/Food planners), and spiritual guidance (Gita bot).

### Architecture
*   **Frontend:** Built with **Gradio**, offering two main entry points: `gradio_ui_full.py` (all providers) and `azure_gradioui.py` (Azure-centric).
*   **Core Logic:** encapsulated within the `helper_functions/` directory (e.g., `analyzers.py`, `llm_client.py`, `researcher.py`).
*   **Configuration:** managed via `config/config.yml` and environment variables.
*   **Storage:** Local file system for temporary vectors/summaries (`data/`) and Supabase for persistent memory.

## Building and Running

### Prerequisites
*   **Python:** 3.10 or higher.
*   **System Tools:** `ffmpeg` (required for audio/video processing).
*   **API Keys:** Access tokens for desired providers (OpenAI, Google, Groq, etc.) configured in `config/config.yml`.

### Installation

1.  **Environment Setup:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```

2.  **Configuration:**
    Copy the example configuration and populate it with your API keys:
    ```bash
    cp config/config.example.yml config/config.yml
    # Edit config/config.yml with your keys
    ```

### Running the Application

*   **Full Interface (Recommended):**
    ```bash
    python gradio_ui_full.py
    ```
    Access the UI at `http://localhost:7860`.

*   **Azure-Specific Interface:**
    ```bash
    python azure_gradioui.py
    ```

*   **Docker:**
    ```bash
    docker build -t llmqabot .
    docker run -p 7860:7860 --env-file config/.env -v $(pwd)/data:/app/data llmqabot
    ```

## Testing

The project maintains a robust test suite using **pytest**.

*   **Run All Tests:**
    ```bash
    pytest tests/ -v
    ```
*   **Run Specific Test Module:**
    ```bash
    pytest tests/test_analyzers.py -v
    ```
*   **Coverage:**
    The project aims for >90% code coverage.
    ```bash
    pytest tests/ --cov=helper_functions --cov-report=term-missing
    ```
    Refer to `tests/README.md` for detailed testing documentation.

## Development Conventions

*   **Modular Design:** Logic is strictly separated from the UI. New features should be implemented as independent modules in `helper_functions/` before being wired into the Gradio interface.
*   **Configuration:** Never hardcode API keys or secrets. Use `config/config.yml` or environment variables.
*   **Testing:** All new features must be accompanied by unit tests. External APIs (OpenAI, Supabase, etc.) must be mocked using the fixtures provided in `tests/conftest.py` to ensure tests run offline and without cost.
*   **Type Safety:** While not strictly enforced everywhere, type hinting is encouraged for core helper functions.
