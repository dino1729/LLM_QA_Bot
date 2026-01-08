"""
Web Knowledge Database - LlamaIndex VectorStoreIndex for auto-learned web content.

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
    Database wrapper for web-learned knowledge using LlamaIndex VectorStoreIndex.

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
            index_folder: Path to LlamaIndex index (default: ./memory_palace/web_knowledge_index)
            ttl_days: Days before knowledge expires (default: 7)
        """
        self.provider = provider or config.memory_palace_provider
        self.model_tier = model_tier or config.memory_palace_model_tier
        self.ttl_days = ttl_days or self.DEFAULT_TTL_DAYS

        # Use separate index folder for web knowledge
        base_folder = config.MEMORY_PALACE_FOLDER or "./memory_palace"
        self.index_folder = index_folder or f"{base_folder}/web_knowledge_index"

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
            VectorStoreIndex,
            StorageContext,
            load_index_from_storage,
        )
        from llama_index.core import Settings

        self._embed_model = self._get_embed_model()
        Settings.embed_model = self._embed_model

        self.index_path.mkdir(parents=True, exist_ok=True)

        if self.index_path.exists() and any(self.index_path.iterdir()):
            try:
                storage_context = StorageContext.from_defaults(
                    persist_dir=str(self.index_path)
                )
                self._index = load_index_from_storage(
                    storage_context,
                    embed_model=self._embed_model
                )
                logger.info(f"Loaded Web Knowledge index from {self.index_path}")
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
        logger.info(f"Created new Web Knowledge index at {self.index_path}")

    def add_knowledge(self, knowledge: WebKnowledge) -> str:
        """
        Add web knowledge to the index.

        Args:
            knowledge: WebKnowledge object to store

        Returns:
            Knowledge ID
        """
        from llama_index.core import Document

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
            excluded_embed_metadata_keys=[
                "id", "source_urls", "distilled_by_model", "source_count"
            ],
        )

        self._index.insert(doc)
        self._index.storage_context.persist(persist_dir=str(self.index_path))
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
        from llama_index.core.retrievers import VectorIndexRetriever

        if self._index is None or self.get_knowledge_count() == 0:
            return []

        retriever = VectorIndexRetriever(index=self._index, similarity_top_k=top_k)
        nodes = retriever.retrieve(text)

        results = []
        now = datetime.now()

        for node in nodes:
            try:
                # Parse expiration
                expires_at_str = node.metadata.get(
                    "expires_at",
                    (datetime.now() + timedelta(days=self.ttl_days)).isoformat()
                )
                try:
                    expires_at = datetime.fromisoformat(expires_at_str)
                except ValueError:
                    expires_at = datetime.now() + timedelta(days=self.ttl_days)

                # Skip expired unless explicitly requested
                if not include_expired and expires_at < now:
                    continue

                # Parse other metadata
                created_at_str = node.metadata.get("created_at", datetime.now().isoformat())
                try:
                    created_at = datetime.fromisoformat(created_at_str)
                except ValueError:
                    created_at = datetime.now()

                source_urls_str = node.metadata.get("source_urls", "[]")
                try:
                    source_urls = json.loads(source_urls_str)
                except json.JSONDecodeError:
                    source_urls = []

                confidence_str = node.metadata.get("confidence_tier", "fairly_sure")
                try:
                    confidence_tier = ConfidenceTier(confidence_str)
                except ValueError:
                    confidence_tier = ConfidenceTier.FAIRLY_SURE

                knowledge = WebKnowledge(
                    id=node.metadata.get("id", str(uuid.uuid4())),
                    distilled_text=node.get_content(),
                    metadata=WebKnowledgeMetadata(
                        source_urls=source_urls,
                        original_query=node.metadata.get("original_query", ""),
                        created_at=created_at,
                        expires_at=expires_at,
                        confidence_tier=confidence_tier,
                        distilled_by_model=node.metadata.get("distilled_by_model", "unknown"),
                        source_count=node.metadata.get("source_count", 0),
                    )
                )
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
        # Get the document from docstore
        if self._index is None:
            return True

        try:
            docstore = self._index.docstore
            for doc_id in docstore.docs:
                doc = docstore.get_document(doc_id)
                if doc and doc.metadata.get("id") == knowledge_id:
                    expires_at_str = doc.metadata.get("expires_at")
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
        # Note: LlamaIndex doesn't have a clean delete API, so we rebuild the index
        # without expired entries. This is a workaround.
        expired = self.get_expired()
        if not expired:
            return 0

        expired_ids = {k.id for k in expired}
        valid_knowledge = [
            k for k in self.get_all_knowledge()
            if k.id not in expired_ids
        ]

        # Rebuild index with valid entries only
        self._rebuild_index(valid_knowledge)
        logger.info(f"Deleted {len(expired)} expired web knowledge entries")
        return len(expired)

    def _rebuild_index(self, knowledge_list: List[WebKnowledge]):
        """Rebuild the index with the given knowledge list."""
        from llama_index.core import VectorStoreIndex, Document

        documents = []
        for k in knowledge_list:
            doc = Document(
                text=k.distilled_text,
                metadata={
                    "id": k.id,
                    "source_urls": json.dumps(k.metadata.source_urls),
                    "original_query": k.metadata.original_query,
                    "created_at": k.metadata.created_at.isoformat(),
                    "expires_at": k.metadata.expires_at.isoformat(),
                    "confidence_tier": k.metadata.confidence_tier,
                    "distilled_by_model": k.metadata.distilled_by_model,
                    "source_count": k.metadata.source_count,
                },
                excluded_embed_metadata_keys=[
                    "id", "source_urls", "distilled_by_model", "source_count"
                ],
            )
            documents.append(doc)

        self._index = VectorStoreIndex.from_documents(
            documents,
            embed_model=self._embed_model
        )
        self._index.storage_context.persist(persist_dir=str(self.index_path))

    def get_all_knowledge(self) -> List[WebKnowledge]:
        """Get all knowledge entries from the index."""
        if self._index is None:
            return []

        knowledge_list = []
        try:
            docstore = self._index.docstore
            for doc_id in docstore.docs:
                doc = docstore.get_document(doc_id)
                if doc:
                    try:
                        created_at_str = doc.metadata.get(
                            "created_at", datetime.now().isoformat()
                        )
                        expires_at_str = doc.metadata.get(
                            "expires_at",
                            (datetime.now() + timedelta(days=self.ttl_days)).isoformat()
                        )
                        source_urls_str = doc.metadata.get("source_urls", "[]")

                        try:
                            created_at = datetime.fromisoformat(created_at_str)
                        except ValueError:
                            created_at = datetime.now()

                        try:
                            expires_at = datetime.fromisoformat(expires_at_str)
                        except ValueError:
                            expires_at = datetime.now() + timedelta(days=self.ttl_days)

                        try:
                            source_urls = json.loads(source_urls_str)
                        except json.JSONDecodeError:
                            source_urls = []

                        confidence_str = doc.metadata.get("confidence_tier", "fairly_sure")
                        try:
                            confidence_tier = ConfidenceTier(confidence_str)
                        except ValueError:
                            confidence_tier = ConfidenceTier.FAIRLY_SURE

                        knowledge = WebKnowledge(
                            id=doc.metadata.get("id", doc_id),
                            distilled_text=doc.get_content(),
                            metadata=WebKnowledgeMetadata(
                                source_urls=source_urls,
                                original_query=doc.metadata.get("original_query", ""),
                                created_at=created_at,
                                expires_at=expires_at,
                                confidence_tier=confidence_tier,
                                distilled_by_model=doc.metadata.get(
                                    "distilled_by_model", "unknown"
                                ),
                                source_count=doc.metadata.get("source_count", 0),
                            )
                        )
                        knowledge_list.append(knowledge)
                    except Exception as e:
                        logger.warning(f"Error parsing document {doc_id}: {e}")
                        continue
        except Exception as e:
            logger.error(f"Error enumerating docstore: {e}")

        return knowledge_list

    def get_knowledge_count(self) -> int:
        """Get the total number of knowledge entries."""
        if self._index is None:
            return 0
        try:
            return len(self._index.docstore.docs)
        except Exception:
            return 0

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
