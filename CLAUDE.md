# CLAUDE.md

LLM_QA_Bot is a multi-modal research and productivity workspace: provider-agnostic
LLM routing, document ingestion (LlamaIndex), Memory Palace (Supabase), and
specialized agents for planning, image generation, and news aggregation.

`gradio_ui_full.py` is the primary entrypoint — it serves both the Gradio UI and
the `/api/*` routes the React frontend consumes, on port 7860.

## Gotchas

**Always route LLM calls through `get_client()`** from
`helper_functions/llm_client.py` — never import OpenAI/Gemini/etc. directly.
That abstraction is what makes provider switching config-only, and it is where
per-provider quirks are absorbed:

- Reasoning models (DeepSeek, o1) return text in `reasoning_content`, not
  `content`. `chat_completion()` already falls back.
- NVIDIA NIM embedding models need an asymmetric `input_type` ("query" vs
  "passage").
- `CustomOpenAILLM` / `CustomOpenAIEmbedding` deliberately bypass LlamaIndex's
  model validation so any OpenAI-compatible model works with document indexing.

**Never hardcode model names or endpoints.** Everything lives in
`config/config.yml` under three tiers (`fast_llm`, `smart_llm`, `strategic_llm`)
per provider. Secrets go in `config/.env` (gitignored). New config keys need a
safe default in `config/config.py`.

**Document indexes persist to disk** — both the VectorStoreIndex
(`VECTOR_FOLDER`) and SummaryIndex (`SUMMARY_FOLDER`). To rebuild, delete the
folders or use the UI's Clear button; re-running ingestion alone won't refresh
them.

**Tests are offline by default** and must stay that way — mock OpenAI, Supabase,
Perplexity, and Riva rather than hitting them. Shared fixtures live in
`tests/conftest.py`. Use `-m "not slow"` / `-m "not integration"` to narrow.

**VibeVoice on Blackwell GPUs (RTX 5090, sm_120)** needs PyTorch nightly with
CUDA 12.8 and driver R570+. `CUDA error: no kernel image is available` means the
stock PyTorch wheel (sm_90 max) is installed:

```bash
pip install --pre torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/nightly/cu128
```

See `docs/BLACKWELL_GPU_SETUP.md`.

**EDITH runs as a systemd *user* service** (`edith-bot`) whose unit lives
outside the repo at `~/.config/systemd/user/edith-bot.service`. It needs pyenv
Python 3.11.9 (`~/.pyenv/versions/3.11.9/bin/python`) — the project venv is
3.10, which lacks `StrEnum`. Linger is enabled so it survives logout; it
auto-restarts on failure. Logs at `edith-bot.log`.

```bash
systemctl --user restart edith-bot     # status / stop / start likewise
python -m helper_functions.memory_palace_bot --discover   # find your TG user ID
```

**Memory Palace needs the `mp_search` stored procedure in Supabase**, plus
`supabase_service_role_key` and `public_supabase_url` in config. It streams via
SSE. The service role key bypasses RLS — check policies before production use.

**`vault/`** is a gitignored, *generated* llm-wiki knowledge base with its own
`CLAUDE.md` schema doc. Don't hand-edit its pages; regenerate them.

## Conventions

- Each `helper_functions/` module has one responsibility and a matching
  `tests/test_<module>.py`. Extend an existing module or add a new one; don't
  grow a `utils.py`.
- Newsletter flow (`year_progress_and_news_reporter_litellm.py`): gather →
  generate → `build_daily_bundle` → render HTML → output audio/email. Iterate
  with `--test` (cached news, no email/audio), `--full-cache`, or `--html-only`
  rather than doing full runs.
- Graceful degradation on search-provider failure: fall back to cached or basic
  results, and return readable errors to the UI rather than stack traces.

## Architecture decisions

- **UnifiedLLMClient over direct SDKs** — one mocking point in tests, config-only
  provider switches, and one place to absorb per-provider quirks.
- **Separate vector and summary indexes** — the vector store does precise chunk
  retrieval, the summary index supplies whole-document context for follow-ups.
- **Gradio and React share one FastAPI backend** — Gradio for rapid ML-facing
  prototyping, React for custom UX, both against `/api/*`.
