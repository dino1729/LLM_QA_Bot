"""
LLM Wiki - Karpathy-style persistent knowledge wiki for Memory Palace.

Uses a multi-model LLM Council to build and maintain an interlinked
Obsidian vault from Memory Palace entries. The wiki compiles knowledge
once and maintains it incrementally, rather than re-deriving on every query.

Three layers (llm-wiki skill architecture):
1. Raw sources: vault/raw/ (immutable article text)
2. Wiki: vault/wiki/{concepts,entities,summaries}/ (LLM-generated, interlinked)
3. Schema: vault/CLAUDE.md
"""
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Lazy singleton for WikiBuilder
_wiki_builder: Optional["WikiBuilder"] = None


def get_wiki_builder() -> Optional["WikiBuilder"]:
    """Get the singleton WikiBuilder, or None if wiki is disabled."""
    global _wiki_builder

    from config import config

    wiki_config = getattr(config, "wiki_config_data", None)
    if wiki_config is None:
        # Load from config module
        try:
            from config import wiki_config as wc
            if not wc.wiki_enabled:
                return None
        except (ImportError, AttributeError):
            return None

    if _wiki_builder is None:
        try:
            from config import wiki_config as wc
            from wiki.builder import WikiBuilder

            vault_path = Path(wc.wiki_vault_path)
            _wiki_builder = WikiBuilder(vault_root=vault_path, config=wc)
            logger.info("WikiBuilder initialized, vault at %s", vault_path)
        except Exception:
            logger.exception("Failed to initialize WikiBuilder")
            return None

    return _wiki_builder
