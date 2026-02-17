"""
Web Knowledge Database - SimpleVectorStore for auto-learned web content.

This module provides:
- Pydantic models for web knowledge with TTL (time-to-live)
- Separate storage from user wisdom (Memory Palace lessons)
- Automatic expiration after 7 days (configurable)
- Confidence tier tracking based on source agreement
"""

import json
import logging
import uuid
from datetime import datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from config import config
from helper_functions.llm_client import get_client
from helper_functions.vector_store import SimpleVectorStore, Document

logger = logging.getLogger(__name__)


class ConfidenceTier(StrEnum):
    """Qualitative confidence tiers for web knowledge."""
    VERY_CONFIDENT = "very_confident"
    FAIRLY_SURE = "fairly_sure"
    UNCERTAIN = "uncertain"


class WebKnowledgeMetadata(BaseModel):
    """Metadata stored with each web knowledge entry."""
    source_urls: List[str] = Field(default_factory=list)
    original_query: str  # Question that triggered the research
    created_at: datetime = Field(default_factory=datetime.now)
    expires_at: datetime  # TTL expiration (default: created_at + 7 days)
    confidence_tier: ConfidenceTier = ConfidenceTier.FAIRLY_SURE
    distilled_by_model: str = "unknown"
    source_count: int = 0  # How many sources were used


class WebKnowledge(BaseModel):
    """Complete web knowledge representation."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    distilled_text: str  # Single-line insight (same format as user lessons)
    metadata: WebKnowledgeMetadata

    model_config = {"use_enum_values": True}


class SimilarWebKnowledge(BaseModel):
    """Web knowledge with similarity score for retrieval."""
    knowledge: WebKnowledge
    similarity_score: float


class WebKnowledgeDB:
    """
    Database wrapper for web-learned knowledge using SimpleVectorStore.

    Separate from MemoryPalaceDB to maintain distinction between:
    - User wisdom (permanent, user-curated)
    - Web knowledge (ephemeral, auto-learned, 7-day TTL)
    """

    DEFAULT_TTL_DAYS = 7

    def __init__(
        self,
        provider: str = None,
        model_tier: str = None,
        index_folder: str = None,
        ttl_days: int = None,
    ):
        """
        Initialize the Web Knowledge database.

        Args:
            provider: LLM provider for embeddings (default from config)
            model_tier: Model tier for embeddings (default from config)
            index_folder: Path to vector store (default: ./memory_palace/web_knowledge_index)
            ttl_days: Days before knowledge expires (default: 7)
        """
        self.provider = provider or config.memory_palace_provider
        self.model_tier = model_tier or config.memory_palace_model_tier
        self.ttl_days = ttl_days or self.DEFAULT_TTL_DAYS

        # Use separate index folder for web knowledge
        base_folder = config.MEMORY_PALACE_FOLDER or "./memory_palace"
        self.index_folder = index_folder or f"{base_folder}/web_knowledge_index"

        self.index_path = Path(self.index_folder)
        self._store = None
        self._ensure_index()

    def _get_embed_fn(self):
        """Get an embedding function from llm_client."""
        client = get_client(provider=self.provider, model_tier=self.model_tier)
        return client.get_embedding

    def _ensure_index(self):
        """Load or create the vector store."""
        embed_fn = self._get_embed_fn()
        self.index_path.mkdir(parents=True, exist_ok=True)
        self._store = SimpleVectorStore(
            persist_dir=str(self.index_path),
            embed_fn=embed_fn,
        )
        logger.info(f"Initialized Web Knowledge store at {self.index_path}")

    def _parse_knowledge_from_data(
        self, doc_id: str, metadata: dict, text: str
    ) -> Optional[WebKnowledge]:
        """
        Parse a WebKnowledge object from raw store data.

        Args:
            doc_id: Document/knowledge ID
            metadata: Metadata dict from the store
            text: The distilled text content

        Returns:
            WebKnowledge object, or None if parsing fails
        """
        try:
            # Parse expires_at
            expires_at_str = metadata.get(
                "expires_at",
                (datetime.now() + timedelta(days=self.ttl_days)).isoformat()
            )
            try:
                expires_at = datetime.fromisoformat(expires_at_str)
            except ValueError:
                expires_at = datetime.now() + timedelta(days=self.ttl_days)

            # Parse created_at
            created_at_str = metadata.get("created_at", datetime.now().isoformat())
            try:
                created_at = datetime.fromisoformat(created_at_str)
            except ValueError:
                created_at = datetime.now()

            # Parse source_urls from JSON string
            source_urls_str = metadata.get("source_urls", "[]")
            try:
                source_urls = json.loads(source_urls_str)
            except json.JSONDecodeError:
                source_urls = []

            # Parse confidence_tier
            confidence_str = metadata.get("confidence_tier", "fairly_sure")
            try:
                confidence_tier = ConfidenceTier(confidence_str)
            except ValueError:
                confidence_tier = ConfidenceTier.FAIRLY_SURE

            return WebKnowledge(
                id=metadata.get("id", doc_id),
                distilled_text=text,
                metadata=WebKnowledgeMetadata(
                    source_urls=source_urls,
                    original_query=metadata.get("original_query", ""),
                    created_at=created_at,
                    expires_at=expires_at,
                    confidence_tier=confidence_tier,
                    distilled_by_model=metadata.get("distilled_by_model", "unknown"),
                    source_count=metadata.get("source_count", 0),
                )
            )
        except Exception as e:
            logger.warning(f"Error parsing knowledge from data (doc_id={doc_id}): {e}")
            return None

    def add_knowledge(self, knowledge: WebKnowledge) -> str:
        """
        Add web knowledge to the store.

        Args:
            knowledge: WebKnowledge object to store

        Returns:
            Knowledge ID
        """
        doc = Document(
            text=knowledge.distilled_text,
            metadata={
                "id": knowledge.id,
                "source_urls": json.dumps(knowledge.metadata.source_urls),
                "original_query": knowledge.metadata.original_query,
                "created_at": knowledge.metadata.created_at.isoformat(),
                "expires_at": knowledge.metadata.expires_at.isoformat(),
                "confidence_tier": knowledge.metadata.confidence_tier,
                "distilled_by_model": knowledge.metadata.distilled_by_model,
                "source_count": knowledge.metadata.source_count,
            },
            doc_id=knowledge.id,
        )

        self._store.insert(doc)
        logger.info(
            f"Added web knowledge {knowledge.id[:8]}... "
            f"(confidence: {knowledge.metadata.confidence_tier}, "
            f"expires: {knowledge.metadata.expires_at.date()})"
        )
        return knowledge.id

    def find_similar(
        self,
        text: str,
        top_k: int = 5,
        include_expired: bool = False
    ) -> List[SimilarWebKnowledge]:
        """
        Find web knowledge similar to the given text.

        Args:
            text: Text to search for
            top_k: Maximum number of results
            include_expired: If True, include expired knowledge

        Returns:
            List of SimilarWebKnowledge objects sorted by similarity
        """
        if self._store is None or self.get_knowledge_count() == 0:
            return []

        nodes = self._store.search(text, top_k=top_k)

        results = []
        now = datetime.now()

        for node in nodes:
            try:
                knowledge = self._parse_knowledge_from_data(
                    node.doc_id, node.metadata, node.get_content()
                )
                if knowledge is None:
                    continue

                # Skip expired unless explicitly requested
                if not include_expired and knowledge.metadata.expires_at < now:
                    continue

                results.append(SimilarWebKnowledge(
                    knowledge=knowledge,
                    similarity_score=node.score or 0.0
                ))
            except Exception as e:
                logger.warning(f"Error parsing node: {e}")
                continue

        return results

    def is_stale(self, knowledge_id: str) -> bool:
        """
        Check if a knowledge entry is stale (expired).

        Args:
            knowledge_id: ID of the knowledge to check

        Returns:
            True if expired, False if still valid
        """
        if self._store is None:
            return True

        try:
            for doc_id, data in self._store.docs.items():
                if data["metadata"].get("id") == knowledge_id:
                    expires_at_str = data["metadata"].get("expires_at")
                    if expires_at_str:
                        expires_at = datetime.fromisoformat(expires_at_str)
                        return expires_at < datetime.now()
            return True  # Not found = stale
        except Exception as e:
            logger.warning(f"Error checking staleness: {e}")
            return True

    def get_expired(self) -> List[WebKnowledge]:
        """
        Get all expired knowledge entries.

        Returns:
            List of expired WebKnowledge objects
        """
        all_knowledge = self.get_all_knowledge()
        now = datetime.now()
        return [k for k in all_knowledge if k.metadata.expires_at < now]

    def delete_expired(self) -> int:
        """
        Delete all expired knowledge entries.

        Returns:
            Number of entries deleted
        """
        if self._store is None:
            return 0

        now = datetime.now()
        to_delete = []

        for doc_id, data in self._store.docs.items():
            expires_at_str = data["metadata"].get("expires_at")
            if expires_at_str:
                try:
                    expires_at = datetime.fromisoformat(expires_at_str)
                    if expires_at < now:
                        to_delete.append(doc_id)
                except ValueError:
                    continue

        for doc_id in to_delete:
            self._store.delete_document(doc_id)

        if to_delete:
            logger.info(f"Deleted {len(to_delete)} expired web knowledge entries")

        return len(to_delete)

    def url_exists(self, url: str) -> bool:
        """
        Check if knowledge from a specific URL exists.

        Args:
            url: URL to check

        Returns:
            True if URL exists in any entry's source_urls
        """
        if self._store is None:
            return False

        for doc_id, data in self._store.docs.items():
            source_urls_str = data["metadata"].get("source_urls", "[]")
            try:
                source_urls = json.loads(source_urls_str)
                if url in source_urls:
                    return True
            except json.JSONDecodeError:
                continue

        return False

    def get_all_knowledge(self) -> List[WebKnowledge]:
        """Get all knowledge entries from the store."""
        if self._store is None:
            return []

        knowledge_list = []
        try:
            for doc_id, data in self._store.docs.items():
                try:
                    knowledge = self._parse_knowledge_from_data(
                        doc_id, data["metadata"], data["text"]
                    )
                    if knowledge is not None:
                        knowledge_list.append(knowledge)
                except Exception as e:
                    logger.warning(f"Error parsing document {doc_id}: {e}")
                    continue
        except Exception as e:
            logger.error(f"Error enumerating store: {e}")

        return knowledge_list

    def get_knowledge_count(self) -> int:
        """Get the total number of knowledge entries."""
        if self._store is None:
            return 0
        return len(self._store)

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about web knowledge."""
        all_knowledge = self.get_all_knowledge()
        now = datetime.now()

        expired_count = sum(1 for k in all_knowledge if k.metadata.expires_at < now)
        valid_count = len(all_knowledge) - expired_count

        confidence_counts = {}
        for k in all_knowledge:
            tier = k.metadata.confidence_tier
            confidence_counts[tier] = confidence_counts.get(tier, 0) + 1

        return {
            "total": len(all_knowledge),
            "valid": valid_count,
            "expired": expired_count,
            "by_confidence": confidence_counts,
        }


def calculate_confidence_tier(
    source_count: int,
    sources_agree: bool = True
) -> ConfidenceTier:
    """
    Calculate confidence tier based on source agreement.

    Args:
        source_count: Number of sources used
        sources_agree: Whether sources generally agree

    Returns:
        ConfidenceTier enum value
    """
    if not sources_agree:
        return ConfidenceTier.UNCERTAIN

    if source_count >= 5:
        return ConfidenceTier.VERY_CONFIDENT
    elif source_count >= 2:
        return ConfidenceTier.FAIRLY_SURE
    else:
        return ConfidenceTier.UNCERTAIN
