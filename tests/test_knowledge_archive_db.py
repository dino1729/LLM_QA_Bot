"""
Tests for the Knowledge Archive Database module.

Tests cover:
- Pydantic model validation
- LlamaIndex CRUD operations
- URL-based deduplication
- Semantic search
- Statistics and listing
"""
import os
import shutil
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from helper_functions.knowledge_archive_db import (
    KnowledgeArchiveMetadata,
    KnowledgeArchiveEntry,
    SimilarArchiveEntry,
    KnowledgeArchiveDB,
    extract_domain,
)


class TestExtractDomain:
    """Tests for the extract_domain utility function."""

    def test_basic_domain(self):
        """Test basic domain extraction."""
        assert extract_domain("https://paulgraham.com/ds.html") == "paulgraham.com"

    def test_www_prefix_removed(self):
        """Test that www. prefix is removed."""
        assert extract_domain("https://www.stratechery.com/article") == "stratechery.com"

    def test_subdomain_preserved(self):
        """Test that non-www subdomains are preserved."""
        assert extract_domain("https://blog.example.com/post") == "blog.example.com"

    def test_http_url(self):
        """Test HTTP URL extraction."""
        assert extract_domain("http://example.org/page") == "example.org"

    def test_url_with_port(self):
        """Test URL with port number."""
        assert extract_domain("https://localhost:8080/api") == "localhost:8080"

    def test_url_with_query_params(self):
        """Test URL with query parameters."""
        assert extract_domain("https://example.com/search?q=test&page=1") == "example.com"


class TestKnowledgeArchiveMetadata:
    """Tests for KnowledgeArchiveMetadata Pydantic model."""

    def test_required_fields(self):
        """Test creation with required fields only."""
        meta = KnowledgeArchiveMetadata(
            url="https://example.com/article",
            title="Test Article",
            word_count=500,
            takeaway_count=3,
            source_domain="example.com",
            estimated_read_time=2,
            distilled_by_model="test-model",
        )
        assert meta.url == "https://example.com/article"
        assert meta.title == "Test Article"
        assert meta.word_count == 500
        assert meta.takeaway_count == 3

    def test_default_values(self):
        """Test default field values."""
        meta = KnowledgeArchiveMetadata(
            url="https://example.com",
            title="Title",
            word_count=100,
            takeaway_count=3,
            source_domain="example.com",
            estimated_read_time=1,
            distilled_by_model="model",
        )
        assert meta.author is None
        assert meta.publish_date is None
        assert meta.tags == []
        assert meta.archive_org_fallback is False
        assert meta.original_url_failed is False
        assert isinstance(meta.indexed_at, datetime)

    def test_all_fields(self):
        """Test creation with all fields populated."""
        custom_time = datetime(2024, 6, 15, 12, 0, 0)
        publish_time = datetime(2024, 6, 1)
        meta = KnowledgeArchiveMetadata(
            url="https://paulgraham.com/ds.html",
            title="Do Things that Don't Scale",
            indexed_at=custom_time,
            word_count=2500,
            takeaway_count=6,
            author="Paul Graham",
            publish_date=publish_time,
            source_domain="paulgraham.com",
            estimated_read_time=10,
            distilled_by_model="gpt-4",
            tags=["startups", "growth", "strategy"],
            archive_org_fallback=True,
            original_url_failed=True,
        )
        assert meta.indexed_at == custom_time
        assert meta.author == "Paul Graham"
        assert meta.publish_date == publish_time
        assert "startups" in meta.tags
        assert meta.archive_org_fallback is True
        assert meta.original_url_failed is True


class TestKnowledgeArchiveEntry:
    """Tests for KnowledgeArchiveEntry Pydantic model."""

    def test_auto_generated_id(self):
        """Test that ID is auto-generated."""
        entry = KnowledgeArchiveEntry(
            summary="Article summary",
            takeaways="Key takeaway 1. Key takeaway 2.",
            content_preview="First 500 chars...",
            metadata=KnowledgeArchiveMetadata(
                url="https://example.com/article",
                title="Test",
                word_count=500,
                takeaway_count=2,
                source_domain="example.com",
                estimated_read_time=2,
                distilled_by_model="test",
            )
        )
        assert entry.id is not None
        assert len(entry.id) == 36  # UUID format

    def test_custom_id(self):
        """Test custom ID assignment."""
        custom_id = "custom-entry-id-123"
        entry = KnowledgeArchiveEntry(
            id=custom_id,
            summary="Summary",
            takeaways="Takeaways",
            content_preview="Preview",
            metadata=KnowledgeArchiveMetadata(
                url="https://example.com",
                title="Title",
                word_count=100,
                takeaway_count=3,
                source_domain="example.com",
                estimated_read_time=1,
                distilled_by_model="test",
            )
        )
        assert entry.id == custom_id

    def test_entry_fields(self):
        """Test entry field access."""
        entry = KnowledgeArchiveEntry(
            summary="This is the summary",
            takeaways="1. First takeaway\n2. Second takeaway",
            content_preview="Content preview text...",
            metadata=KnowledgeArchiveMetadata(
                url="https://example.com/article",
                title="Article Title",
                word_count=1000,
                takeaway_count=2,
                source_domain="example.com",
                estimated_read_time=4,
                distilled_by_model="gpt-4",
            )
        )
        assert entry.summary == "This is the summary"
        assert "First takeaway" in entry.takeaways
        assert entry.content_preview == "Content preview text..."
        assert entry.metadata.title == "Article Title"


class TestSimilarArchiveEntry:
    """Tests for SimilarArchiveEntry model."""

    def test_creation(self):
        """Test creating a SimilarArchiveEntry with score."""
        entry = KnowledgeArchiveEntry(
            summary="Test summary",
            takeaways="Test takeaways",
            content_preview="Preview",
            metadata=KnowledgeArchiveMetadata(
                url="https://example.com",
                title="Test",
                word_count=500,
                takeaway_count=3,
                source_domain="example.com",
                estimated_read_time=2,
                distilled_by_model="test",
            )
        )
        similar = SimilarArchiveEntry(entry=entry, similarity_score=0.85)
        assert similar.entry.summary == "Test summary"
        assert similar.similarity_score == 0.85


@pytest.fixture
def temp_db_dir():
    """Create a temporary directory for test database."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_config(temp_db_dir):
    """Mock config module with test paths and LLM client."""
    with patch("helper_functions.knowledge_archive_db.config") as mock_cfg:
        mock_cfg.memory_palace_provider = "litellm"
        mock_cfg.memory_palace_model_tier = "fast"
        mock_cfg.MEMORY_PALACE_FOLDER = temp_db_dir
        mock_cfg.knowledge_archive_index_folder = os.path.join(temp_db_dir, "archive_index")

        with patch("helper_functions.knowledge_archive_db.get_client") as mock_get_client:
            mock_client = Mock()
            mock_client.get_embedding.return_value = [0.1] * 1536
            mock_get_client.return_value = mock_client
            yield mock_cfg


@pytest.fixture
def sample_entry():
    """Create a sample KnowledgeArchiveEntry for testing."""
    return KnowledgeArchiveEntry(
        summary="Do things that don't scale to find what works, then scale it.",
        takeaways=(
            "1. Manual work in early stages helps understand customer needs.\n"
            "2. Recruiting users individually builds strong foundation.\n"
            "3. Delight a small group before trying to please everyone."
        ),
        content_preview="When we launched Airbnb, we had to do everything manually...",
        metadata=KnowledgeArchiveMetadata(
            url="https://paulgraham.com/ds.html",
            title="Do Things that Don't Scale",
            word_count=2500,
            takeaway_count=3,
            author="Paul Graham",
            source_domain="paulgraham.com",
            estimated_read_time=10,
            distilled_by_model="gpt-4",
            tags=["startups", "growth", "strategy"],
        )
    )


@pytest.fixture
def sample_entry_2():
    """Create a second sample entry for testing."""
    return KnowledgeArchiveEntry(
        summary="Aggregation theory explains how the internet changes industry structure.",
        takeaways=(
            "1. Aggregators win by controlling demand.\n"
            "2. Commoditizing supply benefits aggregators.\n"
            "3. User experience is the key differentiator."
        ),
        content_preview="The value chain for any given consumer market...",
        metadata=KnowledgeArchiveMetadata(
            url="https://stratechery.com/aggregation-theory",
            title="Aggregation Theory",
            word_count=3200,
            takeaway_count=3,
            author="Ben Thompson",
            source_domain="stratechery.com",
            estimated_read_time=13,
            distilled_by_model="gpt-4",
            tags=["technology", "business", "strategy"],
        )
    )


class TestKnowledgeArchiveDB:
    """Tests for KnowledgeArchiveDB class."""

    def test_init_creates_index_folder(self, mock_config, temp_db_dir):
        """Test that initialization creates the index folder."""
        index_path = os.path.join(temp_db_dir, "archive_index")
        assert not os.path.exists(index_path)

        db = KnowledgeArchiveDB()
        assert os.path.exists(index_path)

    def test_add_and_retrieve_entry(self, mock_config, sample_entry):
        """Test adding and retrieving an entry."""
        db = KnowledgeArchiveDB()

        entry_id = db.add_entry(sample_entry)
        assert entry_id == sample_entry.id

        # Verify retrieval
        all_entries = db.get_all_entries()
        assert len(all_entries) == 1
        assert all_entries[0].summary == sample_entry.summary
        assert all_entries[0].metadata.title == sample_entry.metadata.title

    def test_get_entry_count(self, mock_config, sample_entry, sample_entry_2):
        """Test entry counting."""
        db = KnowledgeArchiveDB()
        assert db.get_entry_count() == 0

        db.add_entry(sample_entry)
        assert db.get_entry_count() == 1

        db.add_entry(sample_entry_2)
        assert db.get_entry_count() == 2

    def test_url_exists_true(self, mock_config, sample_entry):
        """Test URL exists returns True for existing URL."""
        db = KnowledgeArchiveDB()
        db.add_entry(sample_entry)

        assert db.url_exists("https://paulgraham.com/ds.html") is True

    def test_url_exists_false(self, mock_config, sample_entry):
        """Test URL exists returns False for non-existing URL."""
        db = KnowledgeArchiveDB()
        db.add_entry(sample_entry)

        assert db.url_exists("https://nonexistent.com/article") is False

    def test_url_exists_empty_db(self, mock_config):
        """Test URL exists on empty database."""
        db = KnowledgeArchiveDB()
        assert db.url_exists("https://example.com") is False

    def test_get_entry_by_url(self, mock_config, sample_entry):
        """Test retrieving entry by URL."""
        db = KnowledgeArchiveDB()
        db.add_entry(sample_entry)

        retrieved = db.get_entry_by_url("https://paulgraham.com/ds.html")
        assert retrieved is not None
        assert retrieved.id == sample_entry.id
        assert retrieved.metadata.title == "Do Things that Don't Scale"

    def test_get_entry_by_url_not_found(self, mock_config, sample_entry):
        """Test retrieving non-existent URL returns None."""
        db = KnowledgeArchiveDB()
        db.add_entry(sample_entry)

        retrieved = db.get_entry_by_url("https://nonexistent.com")
        assert retrieved is None

    def test_get_entry_by_id(self, mock_config, sample_entry):
        """Test retrieving entry by ID."""
        db = KnowledgeArchiveDB()
        entry_id = db.add_entry(sample_entry)

        retrieved = db.get_entry_by_id(entry_id)
        assert retrieved is not None
        assert retrieved.metadata.url == sample_entry.metadata.url

    def test_get_entry_by_id_not_found(self, mock_config, sample_entry):
        """Test retrieving non-existent ID returns None."""
        db = KnowledgeArchiveDB()
        db.add_entry(sample_entry)

        retrieved = db.get_entry_by_id("non-existent-id")
        assert retrieved is None

    def test_find_similar_empty_db(self, mock_config):
        """Test similarity search on empty database."""
        db = KnowledgeArchiveDB()
        results = db.find_similar("test query")
        assert results == []

    def test_get_all_entries(self, mock_config, sample_entry, sample_entry_2):
        """Test getting all entries."""
        db = KnowledgeArchiveDB()
        db.add_entry(sample_entry)
        db.add_entry(sample_entry_2)

        all_entries = db.get_all_entries()
        assert len(all_entries) == 2
        titles = [e.metadata.title for e in all_entries]
        assert "Do Things that Don't Scale" in titles
        assert "Aggregation Theory" in titles

    def test_get_recent(self, mock_config):
        """Test getting recent entries."""
        db = KnowledgeArchiveDB()

        # Add entries with different indexed_at times
        for i in range(5):
            entry = KnowledgeArchiveEntry(
                summary=f"Summary {i}",
                takeaways=f"Takeaway {i}",
                content_preview=f"Preview {i}",
                metadata=KnowledgeArchiveMetadata(
                    url=f"https://example.com/article{i}",
                    title=f"Article {i}",
                    indexed_at=datetime.now() - timedelta(days=i),
                    word_count=500,
                    takeaway_count=3,
                    source_domain="example.com",
                    estimated_read_time=2,
                    distilled_by_model="test",
                )
            )
            db.add_entry(entry)

        recent = db.get_recent(3)
        assert len(recent) == 3
        # Most recent should be first (Article 0 has newest indexed_at)
        assert recent[0].metadata.title == "Article 0"

    def test_get_domain_stats(self, mock_config, sample_entry, sample_entry_2):
        """Test domain statistics."""
        db = KnowledgeArchiveDB()
        db.add_entry(sample_entry)
        db.add_entry(sample_entry_2)

        # Add another entry from paulgraham.com
        entry3 = KnowledgeArchiveEntry(
            summary="Another PG essay",
            takeaways="Takeaways",
            content_preview="Preview",
            metadata=KnowledgeArchiveMetadata(
                url="https://paulgraham.com/other.html",
                title="Another Essay",
                word_count=1000,
                takeaway_count=3,
                source_domain="paulgraham.com",
                estimated_read_time=4,
                distilled_by_model="test",
            )
        )
        db.add_entry(entry3)

        stats = db.get_domain_stats()
        assert stats.get("paulgraham.com") == 2
        assert stats.get("stratechery.com") == 1

    def test_get_tag_stats(self, mock_config, sample_entry, sample_entry_2):
        """Test tag statistics."""
        db = KnowledgeArchiveDB()
        db.add_entry(sample_entry)
        db.add_entry(sample_entry_2)

        stats = db.get_tag_stats()
        # Both entries have "strategy" tag
        assert stats.get("strategy") == 2
        assert stats.get("startups") == 1
        assert stats.get("technology") == 1

    def test_search_by_domain(self, mock_config, sample_entry, sample_entry_2):
        """Test filtering by domain."""
        db = KnowledgeArchiveDB()
        db.add_entry(sample_entry)
        db.add_entry(sample_entry_2)

        pg_entries = db.search_by_domain("paulgraham.com")
        assert len(pg_entries) == 1
        assert pg_entries[0].metadata.title == "Do Things that Don't Scale"

        stratechery_entries = db.search_by_domain("stratechery.com")
        assert len(stratechery_entries) == 1
        assert stratechery_entries[0].metadata.title == "Aggregation Theory"

    def test_search_by_domain_not_found(self, mock_config, sample_entry):
        """Test domain search with no matches."""
        db = KnowledgeArchiveDB()
        db.add_entry(sample_entry)

        results = db.search_by_domain("nonexistent.com")
        assert results == []

    def test_search_by_tags(self, mock_config, sample_entry, sample_entry_2):
        """Test filtering by tags."""
        db = KnowledgeArchiveDB()
        db.add_entry(sample_entry)
        db.add_entry(sample_entry_2)

        # Search for "startups" - only sample_entry has it
        startup_entries = db.search_by_tags(["startups"])
        assert len(startup_entries) == 1
        assert startup_entries[0].metadata.title == "Do Things that Don't Scale"

        # Search for "strategy" - both entries have it
        strategy_entries = db.search_by_tags(["strategy"])
        assert len(strategy_entries) == 2

    def test_search_by_tags_case_insensitive(self, mock_config, sample_entry):
        """Test tag search is case-insensitive."""
        db = KnowledgeArchiveDB()
        db.add_entry(sample_entry)

        results = db.search_by_tags(["STARTUPS"])
        assert len(results) == 1

    def test_search_by_tags_multiple(self, mock_config, sample_entry, sample_entry_2):
        """Test searching with multiple tags (OR logic)."""
        db = KnowledgeArchiveDB()
        db.add_entry(sample_entry)
        db.add_entry(sample_entry_2)

        # "startups" OR "technology" should match both
        results = db.search_by_tags(["startups", "technology"])
        assert len(results) == 2

    def test_delete_entry(self, mock_config, sample_entry, sample_entry_2):
        """Test deleting an entry."""
        db = KnowledgeArchiveDB()
        entry_id = db.add_entry(sample_entry)
        db.add_entry(sample_entry_2)

        assert db.get_entry_count() == 2

        # Delete first entry
        result = db.delete_entry(entry_id)
        assert result is True
        assert db.get_entry_count() == 1

        # Verify it's gone
        assert db.get_entry_by_id(entry_id) is None
        assert db.url_exists(sample_entry.metadata.url) is False

    def test_delete_entry_not_found(self, mock_config, sample_entry):
        """Test deleting non-existent entry returns False."""
        db = KnowledgeArchiveDB()
        db.add_entry(sample_entry)

        result = db.delete_entry("non-existent-id")
        assert result is False
        assert db.get_entry_count() == 1


class TestKnowledgeArchiveDBMetadataParsing:
    """Tests for metadata parsing edge cases."""

    def test_entry_with_empty_author(self, mock_config):
        """Test handling entry with empty author string."""
        db = KnowledgeArchiveDB()

        entry = KnowledgeArchiveEntry(
            summary="Summary",
            takeaways="Takeaways",
            content_preview="Preview",
            metadata=KnowledgeArchiveMetadata(
                url="https://example.com/article",
                title="Test Article",
                word_count=500,
                takeaway_count=3,
                author=None,  # No author
                source_domain="example.com",
                estimated_read_time=2,
                distilled_by_model="test",
            )
        )
        db.add_entry(entry)

        retrieved = db.get_entry_by_url("https://example.com/article")
        assert retrieved.metadata.author is None

    def test_entry_with_empty_tags(self, mock_config):
        """Test handling entry with no tags."""
        db = KnowledgeArchiveDB()

        entry = KnowledgeArchiveEntry(
            summary="Summary",
            takeaways="Takeaways",
            content_preview="Preview",
            metadata=KnowledgeArchiveMetadata(
                url="https://example.com/article",
                title="Test",
                word_count=500,
                takeaway_count=3,
                source_domain="example.com",
                estimated_read_time=2,
                distilled_by_model="test",
                tags=[],
            )
        )
        db.add_entry(entry)

        retrieved = db.get_entry_by_url("https://example.com/article")
        assert retrieved.metadata.tags == []

    def test_entry_preserves_archive_org_flags(self, mock_config):
        """Test that archive.org fallback flags are preserved."""
        db = KnowledgeArchiveDB()

        entry = KnowledgeArchiveEntry(
            summary="Summary from archive.org",
            takeaways="Takeaways",
            content_preview="Preview",
            metadata=KnowledgeArchiveMetadata(
                url="https://paywalled-site.com/article",
                title="Paywalled Article",
                word_count=1000,
                takeaway_count=3,
                source_domain="paywalled-site.com",
                estimated_read_time=4,
                distilled_by_model="test",
                archive_org_fallback=True,
                original_url_failed=True,
            )
        )
        db.add_entry(entry)

        retrieved = db.get_entry_by_url("https://paywalled-site.com/article")
        assert retrieved.metadata.archive_org_fallback is True
        assert retrieved.metadata.original_url_failed is True


class TestKnowledgeArchiveDBIndex:
    """Tests for index persistence and loading."""

    def test_index_persists_between_instances(self, mock_config, temp_db_dir, sample_entry):
        """Test that index data persists when creating a new DB instance."""
        # Create first instance and add entry
        db1 = KnowledgeArchiveDB()
        db1.add_entry(sample_entry)
        assert db1.get_entry_count() == 1

        # Create second instance pointing to same directory
        db2 = KnowledgeArchiveDB()
        assert db2.get_entry_count() == 1

        # Verify data is accessible
        retrieved = db2.get_entry_by_url(sample_entry.metadata.url)
        assert retrieved is not None
        assert retrieved.metadata.title == sample_entry.metadata.title

    def test_empty_index_creation(self, mock_config, temp_db_dir):
        """Test creating a new empty index."""
        index_path = os.path.join(temp_db_dir, "archive_index")
        assert not os.path.exists(index_path)

        db = KnowledgeArchiveDB()
        assert os.path.exists(index_path)
        assert db.get_entry_count() == 0
