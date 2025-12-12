# Repository Guidelines

## Project Structure & Module Organization
- Gradio entry points: `gradio_ui_full.py` (main) and `misc_scripts/azure_gradioui.py` (Azure-only). Support scripts live in `misc_scripts/`.
- Core logic sits in `helper_functions/` (`analyzers.py` for ingestion and indexing, `llm_client.py` for provider routing, `chat_generation*.py` for chat flows, planners, image tools, and research agents).
- Configuration and prompts are in `config/` (`config.example.yml`, `config.yml`, `prompts.yml`); secrets can live in `config/.env`.
- Tests cover nearly every module in `tests/`; screenshots for UI changes live in `screenshots/`; spiritual corpus is in `holybook/`.

## Build, Test, and Development Commands
- Bootstrap:  
  ```bash
  python -m venv .venv && source .venv/bin/activate
  pip install -r requirements.txt
  cp config/config.example.yml config/config.yml  # then add keys
  ```
- Run UI locally: `python gradio_ui_full.py` (defaults to port 7860).
- Full test suite: `pytest tests -v`; with coverage: `pytest tests --cov=helper_functions --cov-report=term-missing`.
- Lint/format (preferred): `black helper_functions`, `isort helper_functions`, and `flake8` before sending PRs.

## Coding Style & Naming Conventions
- Python 3.10+, 4-space indents, type hints for new or touched functions.
- Prefer the unified LLM client (`helper_functions/llm_client.py`) instead of raw SDK calls; use model tiers (`fast`, `smart`, `strategic`) rather than hardcoded names.
- Keep module-level functions small and composable; reuse helpers in `helper_functions/analyzers.py` and `helper_functions/research_agent.py` when adding new ingestion or research flows.
- File and directory names stay snake_case; UI-facing copy should be concise and user-friendly.

## Testing Guidelines
- Tests are pytest-based with rich fixtures in `tests/conftest.py` (mocking Supabase, OpenAI, Firecrawl, NVIDIA, temporary folders). Use them to avoid hitting real services.
- Name tests after the behavior under check (`test_chat_generation_handles_cohere_error`), and keep deterministic inputs.
- Target >=90% coverage for new or modified modules; add regression tests alongside fixes in the matching `tests/test_*.py`.

## Commit & Pull Request Guidelines
- Commit messages in this repo are short, present-tense, and task-focused (e.g., “Enhance lesson generation…”). Follow that style and scope each commit to one logical change.
- PRs should describe intent, major code paths touched, test commands run, and any config prerequisites. Include screenshots for UI or UX-affecting work (`screenshots/`), and link issues or tickets when available.
- Avoid committing secrets; use `config/.env` or environment variables instead, and keep `config/config.yml` keys redacted in examples.

## Security & Configuration Tips
- Never hardcode API keys; load them via `config/config.yml` or env vars. Confirm the `paths` entries exist or are empty so the app can create them.
- When adding new providers or tools, route credentials through the config file and surface toggles in the UI rather than enabling by default.
