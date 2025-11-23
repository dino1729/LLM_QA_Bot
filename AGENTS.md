# Repository Guidelines

## Project Structure & Module Organization
Interactive Gradio frontends live in `azure_gradioui.py` (Azure-focused) and `gradio_ui_full.py` (provider switchboard). Core analyzers, routers, planners, and utility agents sit inside `helper_functions/`, while config templates and prompt packs reside in `config/`. Uploaded assets and indexes follow the folders declared in `config/config.yml` (e.g., `data/`, `web_search_data/`). Tests stay in `tests/`, and reference media belongs in `screenshots/`.

## Build, Test, and Development Commands
- `python -m venv .venv && source .venv/bin/activate` — spin up the recommended Python 3.10+ environment.
- `pip install -r requirements.txt` — pull in runtime and testing dependencies.
- `python gradio_ui_full.py` or `python azure_gradioui.py` — launch the UI locally; use the full switchboard when hopping between providers.
- `pytest tests -v` — run the suite; add `--maxfail=1` when iterating quickly.
- `docker build -t llmqabot . && docker run -p 7860:7860 llmqabot` — containerize the app for parity with deployment targets.

## Coding Style & Naming Conventions
Follow PEP 8 with 4-space indentation, descriptive snake_case for functions/modules, and UpperCamelCase for classes. Keep configuration keys consistent with `config/config.yml`, avoid hidden globals, and favor explicit imports plus type hints so pytest fixtures can hook into functions. In UI scripts, wire Gradio components top-down and reserve comments for non-obvious state coordination.

## Testing Guidelines
Tests use `pytest` plus fixtures in `tests/conftest.py` to mock Supabase, Firecrawl, embeddings, and filesystem paths, so suites run offline. Name new files `test_<feature>.py`, drop sample payloads in `tests/data/`, and cover both happy and failure paths for analyzers, planners, and integrations. Run `pytest tests --cov=helper_functions --cov-report=term-missing` before submitting to watch coverage drift.

## Commit & Pull Request Guidelines
History favors short, imperative summaries such as “Update prompts and requirements …” or “Refactor project structure …”. Mirror that tone: start with a verb, keep the subject clear, and elaborate in the body only if needed. Pull requests should highlight user-visible impact, mention config or schema touches, attach relevant screenshots, and quote the exact test commands executed.

## Security & Configuration Tips
Never commit secrets; keep keys in `config/.env` or environment vars and use `config/config.example.yml` as the template. Ensure directories defined under `paths` exist (or let the app create them) before running ingestion agents. When adding providers, document the new keys in README and `config/config.yml`, and strip logs or notebooks of user data before publishing.
