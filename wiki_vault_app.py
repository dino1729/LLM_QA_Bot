"""Standalone Memory Wiki Vault web app.

This app intentionally exposes only the read-only wiki atlas API and the
React atlas UI. It does not mount the broader Nexus Mind API surface.
"""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse

from wiki.web_atlas import create_wiki_router


BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIST = BASE_DIR / "frontend" / "dist"

app = FastAPI(title="Memory Wiki Vault")
app.include_router(create_wiki_router())


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "app": "memory-wiki-vault"}


@app.get("/")
async def wiki_root():
    return RedirectResponse("/wiki")


@app.get("/{full_path:path}")
async def serve_wiki_app(full_path: str):
    if full_path.startswith("api/"):
        return JSONResponse({"error": "API route not found"}, status_code=404)

    if not FRONTEND_DIST.exists():
        return JSONResponse(
            {"error": "Frontend not built. Run 'npm run build' in frontend directory."},
            status_code=500,
        )

    requested = (FRONTEND_DIST / full_path).resolve()
    frontend_root = FRONTEND_DIST.resolve()

    try:
        is_frontend_file = (
            requested.is_file()
            and Path(os.path.commonpath([frontend_root, requested])) == frontend_root
        )
    except ValueError:
        is_frontend_file = False

    if is_frontend_file:
        return FileResponse(requested)

    if full_path == "wiki" or full_path.startswith("wiki/"):
        index_path = FRONTEND_DIST / "index.html"
        if index_path.exists():
            return FileResponse(index_path)

    return RedirectResponse("/wiki")
