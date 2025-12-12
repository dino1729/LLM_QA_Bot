# LLM_QA_Bot Frontend

React + TypeScript + Vite UI for the LLM_QA_Bot FastAPI/Gradio backend. This layer calls `/api/*` routes exposed by `gradio_ui_full.py` (or `misc_scripts/azure_gradioui.py`) and renders tabs for Document Q&A, AI chat, fun tools, and the Image Studio.

## Quick Start
- Install deps: `cd frontend && npm install`.
- Run the backend (from repo root): `python gradio_ui_full.py` (ensures `/api` is available).
- Start the dev server: `npm run dev` and open the shown localhost URL. Vite serves the frontend; API calls expect the backend on the same origin or behind a proxy to `/api`.
- Build for prod: `npm run build`; preview locally with `npm run preview`.
- Lint: `npm run lint` (ESLint 9 + TypeScript rules).

## Project Structure
- `src/main.tsx` boots React; `src/App.tsx` defines the tabbed layout and routing state.
- `src/components/` holds feature panels:
  - `DocumentQA` for uploads and retrieval Q&A
  - `AIAssistant` for chat
  - `FunTools` for trip/food planners
  - `ImageStudio` for gen/edit/enhance workflows
- `src/api.ts` centralizes typed fetch helpers with abort + timeout handling against `/api`.
- `src/styles`, `src/App.css`, and `src/index.css` define the theme variables and layout.
- Assets and logos live under `src/assets/`.

## Development Notes
- API base is hardcoded to `/api`; if the backend runs elsewhere, proxy requests or adjust `API_BASE` in `src/api.ts`.
- Keep components small and state-local; prefer lifting shared fetch logic into `api.ts` rather than reimplementing `fetch`.
- Use TypeScript interfaces in `api.ts` as the single source of truth for response shapes; update them alongside backend changes.
- CSS variables in `index.css`/`App.css` control typography and accent colors—tweak there before editing component styles.
- Before committing UI changes, run `npm run lint` and, if applicable, include screenshots in the PR to reflect visual updates.
