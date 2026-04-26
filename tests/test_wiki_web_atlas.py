from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from wiki.web_atlas import WikiAtlas, create_wiki_router


CONCEPT_PAGE = """---
title: Compound Interest
type: concept
category: economics
tags: [finance, patience]
sources:
  - migration:Rules of Life.pdf
last_updated: '2026-04-14'
confidence: medium
---

# Compound Interest

## Core Idea
The exponential growth of returns reinvested over time.

## Connections
- Related to [[warren-buffett|Warren Buffett]].
"""


PERSON_PAGE = """---
title: Warren Buffett
type: person
category: investing
tags: [finance]
sources: 0
confidence: low
---

# Warren Buffett

American investor known for value investing.

## See Also
- [[compound-interest|Compound Interest]]
"""


THIN_PAGE = """---
title: Empty Note
type: concept
---

# Empty Note
"""


@pytest.fixture
def vault(tmp_path: Path) -> Path:
    root = tmp_path / "vault"
    for subdir in ["concepts", "entities", "summaries"]:
        (root / "wiki" / subdir).mkdir(parents=True)
    (root / "wiki" / "concepts" / "compound-interest.md").write_text(CONCEPT_PAGE, encoding="utf-8")
    (root / "wiki" / "entities" / "warren-buffett.md").write_text(PERSON_PAGE, encoding="utf-8")
    (root / "wiki" / "concepts" / "empty-note.md").write_text(THIN_PAGE, encoding="utf-8")
    return root


def test_summary_reports_live_vault_stats(vault: Path):
    atlas = WikiAtlas(vault)

    summary = atlas.summary()

    assert summary["vault_exists"] is True
    assert summary["total_pages"] == 3
    assert summary["sections"]["concepts"] == 2
    assert summary["sections"]["entities"] == 1
    assert summary["total_source_count"] == 1
    assert summary["total_backlinks"] == 2
    assert summary["weak_page_count"] == 2


def test_list_pages_searches_metadata_and_flags_quality(vault: Path):
    atlas = WikiAtlas(vault)

    results = atlas.list_pages(query="finance", category="economics")

    assert [page["title"] for page in results["pages"]] == ["Compound Interest"]
    page = results["pages"][0]
    assert page["summary"] == "The exponential growth of returns reinvested over time."
    assert page["source_count"] == 1
    assert page["quality_flags"] == []


def test_page_detail_strips_frontmatter_and_resolves_links(vault: Path):
    atlas = WikiAtlas(vault)

    detail = atlas.page_detail("concepts", "compound-interest")

    assert detail["title"] == "Compound Interest"
    assert "---" not in detail["render_markdown"]
    assert "[Warren Buffett](/wiki/entities/warren-buffett)" in detail["render_markdown"]
    assert detail["outgoing_links"][0]["resolved"] is True
    assert detail["outgoing_links"][0]["url"] == "/wiki/entities/warren-buffett"
    assert detail["backlinks"][0]["title"] == "Warren Buffett"


def test_missing_vault_returns_actionable_empty_state(tmp_path: Path):
    atlas = WikiAtlas(tmp_path / "missing-vault")

    summary = atlas.summary()
    listing = atlas.list_pages()

    assert summary["vault_exists"] is False
    assert summary["total_pages"] == 0
    assert "No generated wiki pages" in summary["message"]
    assert listing["pages"] == []


def test_invalid_section_and_slug_are_rejected(vault: Path):
    atlas = WikiAtlas(vault)

    with pytest.raises(ValueError):
        atlas.page_detail("raw", "refs")

    with pytest.raises(ValueError):
        atlas.page_detail("concepts", "../secrets")


def test_api_rejects_path_traversal(vault: Path):
    app = FastAPI()
    router = create_wiki_router(lambda: vault)
    app.include_router(router)
    isolated = TestClient(app)

    response = isolated.get("/api/wiki/pages/concepts/..%2Fsecrets")

    assert response.status_code == 404
