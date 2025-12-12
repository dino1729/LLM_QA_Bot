# Repository Guidelines

This guide explains how to work in LLM_QA_Bot, keep changes consistent, and ship updates safely.

## Project Structure & Module Organization
- Backend entry points: `gradio_ui_full.py` (main) and `misc_scripts/azure_gradioui.py` (Azure-first layout).
- Core logic in `helper_functions/` (analyzers, chat, planners, image tools, Memory Palace connectors) and `config/` for runtime settings and prompts.
- Tests live in `tests/` with shared fixtures in `tests/conftest.py`; see `tests/README.md` for commands and coverage notes.
- Frontend experiments sit in `frontend/` (Vite + React); assets and reference screenshots live in `screenshots/` and `holybook/`.

## Build, Test, and Development Commands
- Python setup: `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`.
- Run the primary UI: `python gradio_ui_full.py`; Azure-focused variant: `python misc_scripts/azure_gradioui.py`.
- Docker: `docker build -t llmqabot . && docker run -p 7860:7860 --name llmqabot llmqabot`.
- Frontend: `cd frontend && npm install && npm run dev` (Vite dev server).
- Tests: `pytest tests -v`; add `--cov=helper_functions --cov-report=term-missing` before opening a PR.

## Coding Style & Naming Conventions
- Python: follow PEP 8 with 4-space indents, prefer type hints, f-strings, and small, single-purpose functions. Keep module-level configuration in `config/` rather than scattering constants.
- React/TypeScript: use PascalCase for components, camelCase for variables, and keep hooks near the feature they serve. Run `npm run lint` in `frontend/` before committing UI changes.
- Naming: functions describe intent (`generate_trip_plan`, `query_memorypalace_stream`); tests mirror the function under test (`test_clearallfiles_with_files`).

## Testing Guidelines
- Add or update tests alongside any behavioral change. Favor unit tests in `tests/` that rely on existing fixtures (mocked OpenAI, Supabase, Firecrawl, NVIDIA services) to stay offline.
- Name tests `test_<module>.py` and functions `test_<scenario>`. Use parameterization for edge cases instead of duplicating code.
- Aim to maintain or increase current coverage (~80% overall, targeting 90%+). Run focused commands from `tests/README.md` for coverage or slow-test triage.

## Commit & Pull Request Guidelines
- Commits use imperative mood and present tense (e.g., `Add supabase streaming guard`, `Refine planner prompts`); keep them scoped and readable.
- PRs include: summary of changes, steps to reproduce/run, tests executed (`pytest ...`), config keys touched, and screenshots for UI-visible updates.
- Link related issues or TODOs in the description; note any follow-up work so reviewers can plan sequencing.

## Security & Configuration Tips
- Never commit secrets; populate `config/config.yml` or `config/.env` locally and redact keys from logs and screenshots.
- If adding new providers, gate credentials behind env vars and document the required keys in `config/config.example.yml`.
- Clear or ignore transient artifacts (`web_search_cache`, `UPLOAD_FOLDER`, generated indexes) before committing to keep the repo clean.
