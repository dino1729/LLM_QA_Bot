# Repository Guidelines

## Project Structure & Module Organization

- `gradio_ui_full.py`: main FastAPI/Gradio entrypoint (serves `/api/*` and optionally `frontend/dist`).
- `helper_functions/`: core backend modules (LLM routing, analyzers, web research, TTS, Memory Palace).
- `misc_scripts/`: one-off tools and experiments (keep changes isolated and well-named).
- `frontend/`: React + TypeScript + Vite UI (calls backend via `/api`).
- `tests/`: pytest suite and fixtures (see `tests/README.md`).
- `config/`: config templates and prompts (`config/config.example.yml`, `config/prompts.yml`).
- Runtime data is ignored by git: `data/`, `web_search_cache/`, `newsletter_research_data/`, etc.

## Build, Test, and Development Commands

- Backend setup: `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`
- Run backend (full UI): `python gradio_ui_full.py`
- Run backend (Azure-focused UI): `python misc_scripts/azure_gradioui.py`
- Fullstack bootstrap (builds frontend + runs backend): `./start_fullstack.sh`
- Tests: `pytest tests -v` (use `-m "not slow"` or `-m "not integration"` as needed)
- Frontend dev: `cd frontend && npm install && npm run dev`
- Frontend lint/build: `cd frontend && npm run lint && npm run build`
- Docker: `docker build -t llmqabot . && docker run -p 7860:7860 llmqabot`

## Coding Style & Naming Conventions

- Python: 4-space indentation, keep functions small and I/O boundaries clear (API calls, filesystem, ffmpeg).
- Prefer explicit names over abbreviations; mirror existing module names in `helper_functions/`.
- TypeScript/React: follow ESLint guidance (`frontend/eslint.config.js`); keep API shapes centralized in `frontend/src/api.ts`.
- Tests: name files `tests/test_*.py`, classes `Test*`, functions `test_*` (enforced in `pytest.ini`).

## Testing Guidelines

- Framework: pytest. Use fixtures in `tests/conftest.py` to mock external services (tests should run offline).
- When adding features, add/adjust tests in the matching `tests/test_<module>.py`.
- Coverage: aim for high coverage on core paths; see `tests/README.md` for `--cov` examples.

## Commit & Pull Request Guidelines

- Commits in history are plain-English, imperative summaries (e.g., “Refactor …”, “Add …”, “Update …”); keep the first line ≤72 chars and scope the change.
- PRs: include a short description, how to run relevant tests, and screenshots for UI changes (`screenshots/`).
- If a change touches config, update `config/config.example.yml` and document new env vars (never commit real keys).

## Security & Configuration Tips

- Secrets belong in `config/.env` or `.env` (both git-ignored); keep `config/config.yml` local.
- Avoid introducing network calls in tests; gate optional integrations behind configuration and provide mocks.
