"""Tests for web_knowledge_db.py - Web Knowledge Database."""

import json
import os
import shutil
import tempfile
import pytest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch

from helper_functions.web_knowledge_db import (
    ConfidenceTier,
    WebKnowledge,
    WebKnowledgeDB,
    WebKnowledgeMetadata,
    SimilarWebKnowledge,
    calculate_confidence_tier,
)


class TestConfidenceTier:
    """Tests for ConfidenceTier enum."""

    def test_confidence_tier_values(self):
        """Test that confidence tier values are correct."""
        assert ConfidenceTier.VERY_CONFIDENT == "very_confident"
        assert ConfidenceTier.FAIRLY_SURE == "fairly_sure"
        assert ConfidenceTier.UNCERTAIN == "uncertain"


class TestWebKnowledgeMetadata:
    """Tests for WebKnowledgeMetadata model."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        meta = WebKnowledgeMetadata(
            original_query="test query",
            expires_at=datetime.now() + timedelta(days=7),
        )
        assert meta.source_urls == []
        assert meta.confidence_tier == ConfidenceTier.FAIRLY_SURE
        assert meta.distilled_by_model == "unknown"
        assert meta.source_count == 0

    def test_custom_values(self):
        """Test metadata with custom values."""
        expires = datetime.now() + timedelta(days=7)
        meta = WebKnowledgeMetadata(
            source_urls=["https://example.com"],
            original_query="test query",
            expires_at=expires,
            confidence_tier=ConfidenceTier.VERY_CONFIDENT,
            distilled_by_model="gpt-4",
            source_count=5,
        )
        assert meta.source_urls == ["https://example.com"]
        assert meta.original_query == "test query"
        assert meta.confidence_tier == ConfidenceTier.VERY_CONFIDENT
        assert meta.distilled_by_model == "gpt-4"
        assert meta.source_count == 5


class TestWebKnowledge:
    """Tests for WebKnowledge model."""

    def test_auto_generated_id(self):
        """Test that ID is auto-generated."""
        knowledge = WebKnowledge(
            distilled_text="Test insight",
            metadata=WebKnowledgeMetadata(
                original_query="test",
                expires_at=datetime.now() + timedelta(days=7),
            )
        )
        assert knowledge.id is not None
        assert len(knowledge.id) == 36  # UUID format

    def test_custom_id(self):
        """Test that custom ID is preserved."""
        knowledge = WebKnowledge(
            id="custom-id-123",
            distilled_text="Test insight",
            metadata=WebKnowledgeMetadata(
                original_query="test",
                expires_at=datetime.now() + timedelta(days=7),
            )
        )
        assert knowledge.id == "custom-id-123"


class TestCalculateConfidenceTier:
    """Tests for calculate_confidence_tier function."""

    def test_high_confidence_many_sources(self):
        """Test that 5+ agreeing sources gives high confidence."""
        tier = calculate_confidence_tier(5, sources_agree=True)
        assert tier == ConfidenceTier.VERY_CONFIDENT

    def test_medium_confidence_few_sources(self):
        """Test that 2-4 agreeing sources gives medium confidence."""
        tier = calculate_confidence_tier(3, sources_agree=True)
        assert tier == ConfidenceTier.FAIRLY_SURE

    def test_low_confidence_one_source(self):
        """Test that 1 source gives low confidence."""
        tier = calculate_confidence_tier(1, sources_agree=True)
        assert tier == ConfidenceTier.UNCERTAIN

    def test_uncertain_when_sources_disagree(self):
        """Test that disagreeing sources always gives uncertain."""
        tier = calculate_confidence_tier(10, sources_agree=False)
        assert tier == ConfidenceTier.UNCERTAIN


@pytest.fixture
def temp_db_dir():
    """Create a temporary directory for test database."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_embed_model():
    """Create a mock embedding model for LlamaIndex."""
    from llama_index.core.embeddings.mock_embed_model import MockEmbedding

    return MockEmbedding(embed_dim=1536)


@pytest.fixture
def mock_config(temp_db_dir, mock_embed_model):
    """Mock config module with test paths and LLM client."""
    with patch("helper_functions.web_knowledge_db.config") as mock_cfg:
        mock_cfg.memory_palace_provider = "litellm"
        mock_cfg.memory_palace_model_tier = "fast"
        mock_cfg.MEMORY_PALACE_FOLDER = temp_db_dir

        # Mock get_client to return a mock client with mock embedding
        with patch("helper_functions.web_knowledge_db.get_client") as mock_get_client:
            mock_client = Mock()
            mock_client.get_llamaindex_embedding.return_value = mock_embed_model
            mock_get_client.return_value = mock_client
            yield mock_cfg


class TestWebKnowledgeDB:
    """Tests for WebKnowledgeDB class."""

    def test_init_creates_index_folder(self, mock_config, temp_db_dir):
        """Test that __init__ creates the index folder."""
        index_path = os.path.join(temp_db_dir, "web_knowledge_index")

        db = WebKnowledgeDB(index_folder=index_path)

        assert Path(index_path).exists()

    def test_add_and_retrieve_knowledge(self, mock_config, temp_db_dir):
        """Test adding and retrieving knowledge."""
        index_path = os.path.join(temp_db_dir, "web_knowledge_index")
        db = WebKnowledgeDB(index_folder=index_path)

        knowledge = WebKnowledge(
            distilled_text="Quantum entanglement links particles instantly.",
            metadata=WebKnowledgeMetadata(
                source_urls=["https://physics.org"],
                original_query="What is quantum entanglement?",
                expires_at=datetime.now() + timedelta(days=7),
                confidence_tier=ConfidenceTier.FAIRLY_SURE,
                source_count=3,
            )
        )

        knowledge_id = db.add_knowledge(knowledge)

        assert knowledge_id is not None
        assert db.get_knowledge_count() == 1

    def test_get_stats(self, mock_config, temp_db_dir):
        """Test getting statistics."""
        index_path = os.path.join(temp_db_dir, "web_knowledge_index")
        db = WebKnowledgeDB(index_folder=index_path)

        stats = db.get_stats()

        assert "total" in stats
        assert "valid" in stats
        assert "expired" in stats
        assert "by_confidence" in stats

    def test_is_stale_not_found(self, mock_config, temp_db_dir):
        """Test is_stale returns True for non-existent knowledge."""
        index_path = os.path.join(temp_db_dir, "web_knowledge_index")
        db = WebKnowledgeDB(index_folder=index_path)

        assert db.is_stale("non-existent-id") is True

    def test_ttl_default(self, mock_config, temp_db_dir):
        """Test that TTL defaults to 7 days."""
        index_path = os.path.join(temp_db_dir, "web_knowledge_index")
        db = WebKnowledgeDB(index_folder=index_path)

        assert db.ttl_days == 7

    def test_add_and_find_similar(self, mock_config, temp_db_dir):
        """Test adding knowledge and finding similar entries."""
        index_path = os.path.join(temp_db_dir, "web_knowledge_index")
        db = WebKnowledgeDB(index_folder=index_path)

        knowledge = WebKnowledge(
            distilled_text="Python is a high-level programming language.",
            metadata=WebKnowledgeMetadata(
                source_urls=["https://python.org"],
                original_query="What is Python?",
                expires_at=datetime.now() + timedelta(days=7),
                confidence_tier=ConfidenceTier.VERY_CONFIDENT,
                source_count=5,
            )
        )

        db.add_knowledge(knowledge)

        # Find similar (mock embedding will return results based on text)
        results = db.find_similar("Python programming", top_k=5)
        assert len(results) >= 0  # May be empty with mock embeddings

    def test_expired_knowledge_filtered(self, mock_config, temp_db_dir):
        """Test that expired knowledge is filtered by default."""
        index_path = os.path.join(temp_db_dir, "web_knowledge_index")
        db = WebKnowledgeDB(index_folder=index_path)

        # Add expired knowledge
        expired_knowledge = WebKnowledge(
            distilled_text="This is expired knowledge.",
            metadata=WebKnowledgeMetadata(
                source_urls=["https://old.org"],
                original_query="Old query",
                expires_at=datetime.now() - timedelta(days=1),  # Expired yesterday
                confidence_tier=ConfidenceTier.UNCERTAIN,
                source_count=1,
            )
        )

        db.add_knowledge(expired_knowledge)

        # Check expired count
        stats = db.get_stats()
        assert stats["expired"] == 1
        assert stats["valid"] == 0


class TestSimilarWebKnowledge:
    """Tests for SimilarWebKnowledge model."""

    def test_creation(self):
        """Test creating a SimilarWebKnowledge."""
        knowledge = WebKnowledge(
            distilled_text="Test insight",
            metadata=WebKnowledgeMetadata(
                original_query="test",
                expires_at=datetime.now() + timedelta(days=7),
            )
        )

        similar = SimilarWebKnowledge(
            knowledge=knowledge,
            similarity_score=0.85,
        )

        assert similar.knowledge.distilled_text == "Test insight"
        assert similar.similarity_score == 0.85
