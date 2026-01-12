"""
Knowledge Archive Database - LlamaIndex VectorStoreIndex wrapper for storing indexed articles.

This module provides:
- Pydantic models for article entries with rich metadata
- LlamaIndex-based storage with semantic search
- URL-based deduplication
- Recent entries listing for API browsing
- Domain and tag statistics
"""
import json
import logging
import uuid
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from config import config
from helper_functions.llm_client import get_client

logger = logging.getLogger(__name__)


class KnowledgeArchiveMetadata(BaseModel):
    """Metadata stored with each indexed article."""
    url: str  # Original source URL (deduplication key)
    title: str
    indexed_at: datetime = Field(default_factory=datetime.now)
    word_count: int
    takeaway_count: int  # 3 for < 2000 words, up to 8 for >= 2000 words
    author: Optional[str] = None  # Extracted if available
    publish_date: Optional[datetime] = None  # Extracted if available
    source_domain: str  # e.g., "paulgraham.com", "stratechery.com"
    estimated_read_time: int  # minutes, calculated from word_count

    # LLM extraction metadata
    distilled_by_model: str  # Model used for extraction
    tags: List[str] = Field(default_factory=list)  # Auto-generated tags (pass 2)

    # Archive.org fallback tracking
    archive_org_fallback: bool = False  # True if content was from archive.org
    original_url_failed: bool = False  # True if primary URL scrape failed


class KnowledgeArchiveEntry(BaseModel):
    """Complete Knowledge Archive entry."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))

    # Two-pass LLM extraction results
    summary: str  # Concise summary (pass 1)
    takeaways: str  # Monolithic text blob of key takeaways (pass 1)

    # Full content for search (chunked and indexed)
    content_preview: str  # First 500 chars for display

    metadata: KnowledgeArchiveMetadata

    model_config = {"use_enum_values": True}


class SimilarArchiveEntry(BaseModel):
    """Archive entry with similarity score for retrieval."""
    entry: KnowledgeArchiveEntry
    similarity_score: float


def extract_domain(url: str) -> str:
    """Extract source domain from URL."""
    parsed = urlparse(url)
    domain = parsed.netloc
    if domain.startswith("www."):
        domain = domain[4:]
    return domain


class KnowledgeArchiveDB:
    """
    Database wrapper for Knowledge Archive using LlamaIndex VectorStoreIndex.

    Stores indexed articles permanently with rich metadata.
    Separate from MemoryPalaceDB (user wisdom) and WebKnowledgeDB (ephemeral).
    """

    def __init__(
        self,
        provider: str = None,
        model_tier: str = None,
        index_folder: str = None,
    ):
        """
        Initialize the Knowledge Archive database.

        Args:
            provider: LLM provider for embeddings (default from config)
            model_tier: Model tier for embeddings (default from config)
            index_folder: Path to LlamaIndex index (default: ./memory_palace/archive_index)
        """
        self.provider = provider or getattr(config, "memory_palace_provider", "litellm")
        self.model_tier = model_tier or getattr(config, "memory_palace_model_tier", "fast")

        base_folder = getattr(config, "MEMORY_PALACE_FOLDER", None) or "./memory_palace"
        default_index = getattr(config, "knowledge_archive_index_folder", f"{base_folder}/archive_index")
        self.index_folder = index_folder or default_index

        self.index_path = Path(self.index_folder)
        self._index = None
        self._ensure_index()

    def _get_embed_model(self):
        """Get the LlamaIndex-compatible embedding model from llm_client."""
        client = get_client(provider=self.provider, model_tier=self.model_tier)
        return client.get_llamaindex_embedding()

    def _ensure_index(self):
        """Load or create the vector index."""
        from llama_index.core import (
            StorageContext,
            load_index_from_storage,
        )
        from llama_index.core import Settings

        # Set the embedding model for LlamaIndex operations
        self._embed_model = self._get_embed_model()
        Settings.embed_model = self._embed_model

        self.index_path.mkdir(parents=True, exist_ok=True)

        if self.index_path.exists() and any(self.index_path.iterdir()):
            try:
                storage_context = StorageContext.from_defaults(persist_dir=str(self.index_path))
                self._index = load_index_from_storage(
                    storage_context,
                    embed_model=self._embed_model
                )
                logger.info(f"Loaded existing Knowledge Archive index from {self.index_path}")
            except Exception as e:
                logger.warning(f"Failed to load index: {e}. Creating new empty index.")
                self._create_empty_index()
        else:
            self._create_empty_index()

    def _create_empty_index(self):
        """Create a new empty vector index."""
        from llama_index.core import VectorStoreIndex

        self._index = VectorStoreIndex.from_documents(
            [],
            embed_model=self._embed_model
        )
        self._index.storage_context.persist(persist_dir=str(self.index_path))
        logger.info(f"Created new Knowledge Archive index at {self.index_path}")

    def add_entry(self, entry: KnowledgeArchiveEntry) -> str:
        """
        Add an entry to the index.

        Args:
            entry: KnowledgeArchiveEntry object to store

        Returns:
            Entry ID
        """
        from llama_index.core import Document

        # Create searchable text combining summary and takeaways
        searchable_text = f"{entry.metadata.title}\n\n{entry.summary}\n\n{entry.takeaways}"

        # Create LlamaIndex Document with metadata
        doc = Document(
            text=searchable_text,
            metadata={
                "id": entry.id,
                "url": entry.metadata.url,
                "title": entry.metadata.title,
                "indexed_at": entry.metadata.indexed_at.isoformat(),
                "word_count": entry.metadata.word_count,
                "takeaway_count": entry.metadata.takeaway_count,
                "author": entry.metadata.author or "",
                "publish_date": entry.metadata.publish_date.isoformat() if entry.metadata.publish_date else "",
                "source_domain": entry.metadata.source_domain,
                "estimated_read_time": entry.metadata.estimated_read_time,
                "distilled_by_model": entry.metadata.distilled_by_model,
                "tags": ",".join(entry.metadata.tags),
                "archive_org_fallback": str(entry.metadata.archive_org_fallback),
                "original_url_failed": str(entry.metadata.original_url_failed),
                # Store full summary and takeaways for retrieval
                "summary": entry.summary,
                "takeaways": entry.takeaways,
                "content_preview": entry.content_preview,
            },
            excluded_embed_metadata_keys=[
                "id", "indexed_at", "distilled_by_model", "archive_org_fallback",
                "original_url_failed", "summary", "takeaways", "content_preview"
            ],
        )

        self._index.insert(doc)
        self._index.storage_context.persist(persist_dir=str(self.index_path))
        logger.info(f"Added entry {entry.id[:8]}... to Knowledge Archive: {entry.metadata.title}")
        return entry.id

    def url_exists(self, url: str) -> bool:
        """
        Check if a URL already exists in the archive.

        Args:
            url: URL to check

        Returns:
            True if URL exists, False otherwise
        """
        return self.get_entry_by_url(url) is not None

    def get_entry_by_url(self, url: str) -> Optional[KnowledgeArchiveEntry]:
        """
        Get an entry by its URL.

        Args:
            url: URL to look up

        Returns:
            KnowledgeArchiveEntry if found, None otherwise
        """
        if self._index is None:
            return None

        try:
            docstore = self._index.docstore
            for doc_id in docstore.docs:
                doc = docstore.get_document(doc_id)
                if doc and doc.metadata.get("url") == url:
                    return self._doc_to_entry(doc)
        except Exception as e:
            logger.error(f"Error looking up URL: {e}")

        return None

    def get_entry_by_id(self, entry_id: str) -> Optional[KnowledgeArchiveEntry]:
        """
        Get an entry by its ID.

        Args:
            entry_id: Entry ID to look up

        Returns:
            KnowledgeArchiveEntry if found, None otherwise
        """
        if self._index is None:
            return None

        try:
            docstore = self._index.docstore
            for doc_id in docstore.docs:
                doc = docstore.get_document(doc_id)
                if doc and doc.metadata.get("id") == entry_id:
                    return self._doc_to_entry(doc)
        except Exception as e:
            logger.error(f"Error looking up entry: {e}")

        return None

    def _doc_to_entry(self, doc) -> KnowledgeArchiveEntry:
        """Convert a LlamaIndex Document to KnowledgeArchiveEntry."""
        meta = doc.metadata

        # Parse tags
        tags_str = meta.get("tags", "")
        tags = [t.strip() for t in tags_str.split(",") if t.strip()]

        # Parse publish_date
        publish_date = None
        publish_date_str = meta.get("publish_date", "")
        if publish_date_str:
            try:
                publish_date = datetime.fromisoformat(publish_date_str)
            except ValueError:
                pass

        # Parse indexed_at
        indexed_at_str = meta.get("indexed_at", datetime.now().isoformat())
        try:
            indexed_at = datetime.fromisoformat(indexed_at_str)
        except ValueError:
            indexed_at = datetime.now()

        # Parse boolean fields
        archive_org_fallback = meta.get("archive_org_fallback", "False")
        if isinstance(archive_org_fallback, str):
            archive_org_fallback = archive_org_fallback.lower() == "true"

        original_url_failed = meta.get("original_url_failed", "False")
        if isinstance(original_url_failed, str):
            original_url_failed = original_url_failed.lower() == "true"

        return KnowledgeArchiveEntry(
            id=meta.get("id", str(uuid.uuid4())),
            summary=meta.get("summary", ""),
            takeaways=meta.get("takeaways", ""),
            content_preview=meta.get("content_preview", ""),
            metadata=KnowledgeArchiveMetadata(
                url=meta.get("url", ""),
                title=meta.get("title", "Unknown Title"),
                indexed_at=indexed_at,
                word_count=int(meta.get("word_count", 0)),
                takeaway_count=int(meta.get("takeaway_count", 0)),
                author=meta.get("author") or None,
                publish_date=publish_date,
                source_domain=meta.get("source_domain", ""),
                estimated_read_time=int(meta.get("estimated_read_time", 0)),
                distilled_by_model=meta.get("distilled_by_model", "unknown"),
                tags=tags,
                archive_org_fallback=archive_org_fallback,
                original_url_failed=original_url_failed,
            )
        )

    def find_similar(self, text: str, top_k: int = 5) -> List[SimilarArchiveEntry]:
        """
        Find entries similar to the given text with similarity scores.

        Args:
            text: Text to search for
            top_k: Maximum number of results

        Returns:
            List of SimilarArchiveEntry objects sorted by similarity
        """
        from llama_index.core.retrievers import VectorIndexRetriever

        if self._index is None or self.get_entry_count() == 0:
            return []

        retriever = VectorIndexRetriever(index=self._index, similarity_top_k=top_k)
        nodes = retriever.retrieve(text)

        results = []
        for node in nodes:
            try:
                entry = self._doc_to_entry(node)
                results.append(SimilarArchiveEntry(entry=entry, similarity_score=node.score or 0.0))
            except Exception as e:
                logger.warning(f"Error parsing node: {e}")
                continue

        return results

    def get_all_entries(self) -> List[KnowledgeArchiveEntry]:
        """
        Get all entries from the index.

        Returns:
            List of KnowledgeArchiveEntry objects
        """
        if self._index is None:
            return []

        entries = []
        try:
            docstore = self._index.docstore
            for doc_id in docstore.docs:
                doc = docstore.get_document(doc_id)
                if doc:
                    try:
                        entry = self._doc_to_entry(doc)
                        entries.append(entry)
                    except Exception as e:
                        logger.warning(f"Error parsing document {doc_id}: {e}")
                        continue
        except Exception as e:
            logger.error(f"Error enumerating docstore: {e}")

        return entries

    def get_recent(self, n: int = 10) -> List[KnowledgeArchiveEntry]:
        """
        Get the most recently indexed entries.

        Args:
            n: Number of entries to return

        Returns:
            List of KnowledgeArchiveEntry objects sorted by indexed_at descending
        """
        all_entries = self.get_all_entries()
        sorted_entries = sorted(
            all_entries,
            key=lambda e: e.metadata.indexed_at,
            reverse=True
        )
        return sorted_entries[:n]

    def get_entry_count(self) -> int:
        """Get the total number of entries in the database."""
        if self._index is None:
            return 0
        try:
            return len(self._index.docstore.docs)
        except Exception:
            return 0

    def get_domain_stats(self) -> Dict[str, int]:
        """Get entry count per source domain."""
        all_entries = self.get_all_entries()
        return dict(Counter(e.metadata.source_domain for e in all_entries))

    def get_tag_stats(self) -> Dict[str, int]:
        """Get entry count per tag."""
        all_entries = self.get_all_entries()
        tag_counts: Dict[str, int] = {}
        for entry in all_entries:
            for tag in entry.metadata.tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
        return tag_counts

    def search_by_domain(self, domain: str, top_k: int = 10) -> List[KnowledgeArchiveEntry]:
        """
        Get entries from a specific domain.

        Args:
            domain: Source domain to filter by
            top_k: Maximum number of results

        Returns:
            List of KnowledgeArchiveEntry objects
        """
        all_entries = self.get_all_entries()
        filtered = [e for e in all_entries if e.metadata.source_domain == domain]
        sorted_entries = sorted(filtered, key=lambda e: e.metadata.indexed_at, reverse=True)
        return sorted_entries[:top_k]

    def search_by_tags(self, tags: List[str], top_k: int = 10) -> List[KnowledgeArchiveEntry]:
        """
        Get entries matching any of the given tags.

        Args:
            tags: List of tags to match
            top_k: Maximum number of results

        Returns:
            List of KnowledgeArchiveEntry objects matching any tag
        """
        all_entries = self.get_all_entries()
        tags_set = set(t.lower() for t in tags)

        filtered = [
            e for e in all_entries
            if any(t.lower() in tags_set for t in e.metadata.tags)
        ]
        sorted_entries = sorted(filtered, key=lambda e: e.metadata.indexed_at, reverse=True)
        return sorted_entries[:top_k]

    def delete_entry(self, entry_id: str) -> bool:
        """
        Delete an entry by ID.

        Args:
            entry_id: ID of the entry to delete

        Returns:
            True if deleted, False if not found
        """
        if self._index is None:
            return False

        try:
            docstore = self._index.docstore
            doc_id_to_delete = None

            # Find the document with matching entry ID
            for doc_id in list(docstore.docs.keys()):
                doc = docstore.get_document(doc_id)
                if doc and doc.metadata.get("id") == entry_id:
                    doc_id_to_delete = doc_id
                    break

            if doc_id_to_delete is None:
                return False

            # Delete from docstore
            docstore.delete_document(doc_id_to_delete)

            # Delete from vector store (if present)
            try:
                vector_store = self._index._vector_store
                if hasattr(vector_store, 'delete'):
                    vector_store.delete(doc_id_to_delete)
            except Exception:
                pass  # Vector store deletion is best-effort

            # Persist changes
            self._index.storage_context.persist(persist_dir=str(self.index_path))
            logger.info(f"Deleted entry {entry_id[:8]}... from Knowledge Archive")
            return True

        except Exception as e:
            logger.error(f"Error deleting entry: {e}")

        return False
