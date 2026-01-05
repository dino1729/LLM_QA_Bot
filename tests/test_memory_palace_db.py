"""
Tests for the Memory Palace Database module.

Tests cover:
- Pydantic model validation
- LlamaIndex CRUD operations
- Duplicate detection
- Recency tracking
- LLM distillation
"""
import json
import os
import shutil
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from helper_functions.memory_palace_db import (
    CATEGORIES,
    Lesson,
    LessonCategory,
    LessonDistillationResult,
    LessonMetadata,
    MemoryPalaceDB,
    SimilarLesson,
    distill_lesson,
    suggest_category,
)


class TestLessonCategory:
    """Tests for the LessonCategory enum."""

    def test_all_categories_exist(self):
        """Verify all 10 categories are defined."""
        expected = {
            "strategy", "psychology", "history", "science", "technology",
            "economics", "engineering", "biology", "leadership", "observations"
        }
        actual = {c.value for c in LessonCategory}
        assert actual == expected

    def test_category_enum_values(self):
        """Test enum value access."""
        assert LessonCategory.STRATEGY.value == "strategy"
        assert LessonCategory.PSYCHOLOGY.value == "psychology"

    def test_category_from_string(self):
        """Test creating category from string."""
        assert LessonCategory("strategy") == LessonCategory.STRATEGY
        with pytest.raises(ValueError):
            LessonCategory("invalid_category")


class TestCategoriesDict:
    """Tests for the CATEGORIES metadata dictionary."""

    def test_all_categories_have_metadata(self):
        """Verify all enum categories have metadata entries."""
        for category in LessonCategory:
            assert category.value in CATEGORIES
            assert "display" in CATEGORIES[category.value]
            assert "keywords" in CATEGORIES[category.value]

    def test_category_display_names(self):
        """Test display name format."""
        assert CATEGORIES["strategy"]["display"] == "Strategy & Decision Making"
        assert CATEGORIES["psychology"]["display"] == "Psychology & Cognitive Science"

    def test_category_keywords_not_empty(self):
        """Verify all categories have keywords."""
        for category, meta in CATEGORIES.items():
            assert len(meta["keywords"]) > 0, f"Category {category} has no keywords"


class TestLessonMetadata:
    """Tests for LessonMetadata Pydantic model."""

    def test_default_values(self):
        """Test default field values."""
        meta = LessonMetadata(
            category=LessonCategory.STRATEGY,
            original_input="test input",
            distilled_by_model="test-model"
        )
        assert meta.source == "telegram"
        assert meta.tags == []
        assert isinstance(meta.created_at, datetime)

    def test_custom_values(self):
        """Test custom field values."""
        custom_time = datetime(2024, 1, 1, 12, 0, 0)
        meta = LessonMetadata(
            category=LessonCategory.ECONOMICS,
            created_at=custom_time,
            source="migration",
            original_input="original",
            distilled_by_model="gpt-4",
            tags=["finance", "investing"]
        )
        assert meta.category == LessonCategory.ECONOMICS
        assert meta.created_at == custom_time
        assert meta.source == "migration"
        assert "finance" in meta.tags


class TestLesson:
    """Tests for Lesson Pydantic model."""

    def test_auto_generated_id(self):
        """Test that ID is auto-generated."""
        lesson = Lesson(
            distilled_text="Test insight",
            metadata=LessonMetadata(
                category=LessonCategory.HISTORY,
                original_input="original",
                distilled_by_model="test"
            )
        )
        assert lesson.id is not None
        assert len(lesson.id) == 36  # UUID format

    def test_custom_id(self):
        """Test custom ID assignment."""
        custom_id = "custom-lesson-id-123"
        lesson = Lesson(
            id=custom_id,
            distilled_text="Test insight",
            metadata=LessonMetadata(
                category=LessonCategory.SCIENCE,
                original_input="original",
                distilled_by_model="test"
            )
        )
        assert lesson.id == custom_id


class TestSimilarLesson:
    """Tests for SimilarLesson model."""

    def test_similar_lesson_creation(self):
        """Test creating a SimilarLesson with score."""
        lesson = Lesson(
            distilled_text="Test",
            metadata=LessonMetadata(
                category=LessonCategory.BIOLOGY,
                original_input="test",
                distilled_by_model="test"
            )
        )
        similar = SimilarLesson(lesson=lesson, similarity_score=0.85)
        assert similar.lesson.distilled_text == "Test"
        assert similar.similarity_score == 0.85


class TestSuggestCategory:
    """Tests for keyword-based category suggestion."""

    def test_strategy_keywords(self):
        """Test strategy category detection."""
        assert suggest_category("game theory and negotiation tactics") == "strategy"
        assert suggest_category("OODA loop decision making") == "strategy"

    def test_psychology_keywords(self):
        """Test psychology category detection."""
        assert suggest_category("cognitive bias and heuristics") == "psychology"
        assert suggest_category("Dunning-Kruger effect") == "psychology"

    def test_technology_keywords(self):
        """Test technology category detection."""
        assert suggest_category("AI algorithms and computing") == "technology"

    def test_fallback_to_observations(self):
        """Test fallback when no keywords match."""
        assert suggest_category("random unrelated text xyz") == "observations"

    def test_case_insensitive(self):
        """Test case-insensitive matching."""
        assert suggest_category("GAME THEORY STRATEGY") == "strategy"


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
    with patch("helper_functions.memory_palace_db.config") as mock_cfg:
        mock_cfg.memory_palace_provider = "litellm"
        mock_cfg.memory_palace_model_tier = "fast"
        mock_cfg.memory_palace_primary_model = "test-model"
        mock_cfg.memory_palace_fallback_model = "fallback-model"
        mock_cfg.memory_palace_similarity_threshold = 0.75
        mock_cfg.memory_palace_recency_window_days = 30
        mock_cfg.memory_palace_index_folder = os.path.join(temp_db_dir, "lessons_index")
        mock_cfg.MEMORY_PALACE_FOLDER = temp_db_dir

        # Mock get_client to return a mock client with mock embedding
        with patch("helper_functions.memory_palace_db.get_client") as mock_get_client:
            mock_client = Mock()
            mock_client.get_llamaindex_embedding.return_value = mock_embed_model
            mock_get_client.return_value = mock_client
            yield mock_cfg


class TestMemoryPalaceDB:
    """Tests for MemoryPalaceDB class."""

    def test_init_creates_index_folder(self, mock_config, temp_db_dir):
        """Test that initialization creates the index folder."""
        index_path = os.path.join(temp_db_dir, "lessons_index")
        assert not os.path.exists(index_path)

        db = MemoryPalaceDB()
        assert os.path.exists(index_path)

    def test_add_and_retrieve_lesson(self, mock_config):
        """Test adding and retrieving a lesson."""
        db = MemoryPalaceDB()

        lesson = Lesson(
            distilled_text="The Tit-for-Tat strategy balances cooperation and retaliation.",
            metadata=LessonMetadata(
                category=LessonCategory.STRATEGY,
                original_input="Long text about game theory...",
                distilled_by_model="test-model"
            )
        )

        lesson_id = db.add_lesson(lesson)
        assert lesson_id == lesson.id

        # Verify retrieval
        all_lessons = db.get_all_lessons()
        assert len(all_lessons) == 1
        assert all_lessons[0].distilled_text == lesson.distilled_text

    def test_get_lesson_count(self, mock_config):
        """Test lesson counting."""
        db = MemoryPalaceDB()
        assert db.get_lesson_count() == 0

        for i in range(3):
            lesson = Lesson(
                distilled_text=f"Lesson {i}",
                metadata=LessonMetadata(
                    category=LessonCategory.HISTORY,
                    original_input=f"Input {i}",
                    distilled_by_model="test"
                )
            )
            db.add_lesson(lesson)

        assert db.get_lesson_count() == 3

    def test_get_lessons_by_category(self, mock_config):
        """Test filtering by category."""
        db = MemoryPalaceDB()

        # Add lessons in different categories
        categories = [LessonCategory.STRATEGY, LessonCategory.PSYCHOLOGY, LessonCategory.STRATEGY]
        for i, cat in enumerate(categories):
            lesson = Lesson(
                distilled_text=f"Lesson {i}",
                metadata=LessonMetadata(
                    category=cat,
                    original_input=f"Input {i}",
                    distilled_by_model="test"
                )
            )
            db.add_lesson(lesson)

        strategy_lessons = db.get_lessons_by_category(LessonCategory.STRATEGY)
        assert len(strategy_lessons) == 2

        psychology_lessons = db.get_lessons_by_category(LessonCategory.PSYCHOLOGY)
        assert len(psychology_lessons) == 1

    def test_get_category_stats(self, mock_config):
        """Test category statistics."""
        db = MemoryPalaceDB()

        # Add lessons
        for cat in [LessonCategory.STRATEGY, LessonCategory.STRATEGY, LessonCategory.HISTORY]:
            lesson = Lesson(
                distilled_text="Test",
                metadata=LessonMetadata(
                    category=cat,
                    original_input="test",
                    distilled_by_model="test"
                )
            )
            db.add_lesson(lesson)

        stats = db.get_category_stats()
        assert stats.get("strategy") == 2
        assert stats.get("history") == 1

    def test_find_similar_empty_db(self, mock_config):
        """Test similarity search on empty database."""
        db = MemoryPalaceDB()
        results = db.find_similar("test query")
        assert results == []

    def test_check_duplicate_below_threshold(self, mock_config):
        """Test duplicate check when similarity is below threshold."""
        db = MemoryPalaceDB()

        lesson = Lesson(
            distilled_text="Compound interest is powerful for wealth building.",
            metadata=LessonMetadata(
                category=LessonCategory.ECONOMICS,
                original_input="original",
                distilled_by_model="test"
            )
        )
        db.add_lesson(lesson)

        # Mock find_similar to return low similarity (below 0.75 threshold)
        # This is needed because LlamaIndex's test embedding model returns
        # constant vectors, resulting in similarity scores of 1.0
        with patch.object(db, 'find_similar') as mock_find_similar:
            mock_find_similar.return_value = [
                SimilarLesson(lesson=lesson, similarity_score=0.4)
            ]
            result = db.check_duplicate("Quantum physics explains particle behavior.")
            assert result is None

    def test_check_duplicate_above_threshold(self, mock_config):
        """Test duplicate check when similarity is above threshold."""
        db = MemoryPalaceDB()

        lesson = Lesson(
            distilled_text="Compound interest is powerful for wealth building.",
            metadata=LessonMetadata(
                category=LessonCategory.ECONOMICS,
                original_input="original",
                distilled_by_model="test"
            )
        )
        db.add_lesson(lesson)

        # Mock find_similar to return high similarity (above 0.75 threshold)
        with patch.object(db, 'find_similar') as mock_find_similar:
            mock_find_similar.return_value = [
                SimilarLesson(lesson=lesson, similarity_score=0.85)
            ]
            result = db.check_duplicate("Compound interest accelerates wealth.")
            assert result is not None
            assert result.similarity_score == 0.85
            assert result.lesson.id == lesson.id

    def test_get_random_lesson_empty_db(self, mock_config):
        """Test random selection on empty database."""
        db = MemoryPalaceDB()
        result = db.get_random_lesson()
        assert result is None

    def test_get_random_lesson(self, mock_config):
        """Test random lesson selection."""
        db = MemoryPalaceDB()

        lessons = []
        for i in range(5):
            lesson = Lesson(
                distilled_text=f"Lesson {i}",
                metadata=LessonMetadata(
                    category=LessonCategory.LEADERSHIP,
                    original_input=f"Input {i}",
                    distilled_by_model="test"
                )
            )
            db.add_lesson(lesson)
            lessons.append(lesson)

        random_lesson = db.get_random_lesson(exclude_recent=False)
        assert random_lesson is not None
        assert random_lesson.distilled_text in [l.distilled_text for l in lessons]

    def test_get_few_shot_examples_empty_db(self, mock_config):
        """Test few-shot examples fallback on empty database."""
        db = MemoryPalaceDB()
        examples = db.get_few_shot_examples()
        assert len(examples) == 3  # Default fallback examples
        assert any("Tit-for-Tat" in ex for ex in examples)


class TestShownHistory:
    """Tests for recency tracking functionality."""

    def test_mark_as_shown_creates_file(self, mock_config, temp_db_dir):
        """Test that marking as shown creates the history file."""
        db = MemoryPalaceDB()
        history_file = Path(temp_db_dir) / "shown_history.json"

        assert not history_file.exists()
        db.mark_as_shown("test-lesson-id")
        assert history_file.exists()

    def test_shown_history_format(self, mock_config, temp_db_dir):
        """Test the format of shown history entries."""
        db = MemoryPalaceDB()
        db.mark_as_shown("lesson-123")

        history_file = Path(temp_db_dir) / "shown_history.json"
        with open(history_file) as f:
            history = json.load(f)

        assert len(history) == 1
        assert history[0]["id"] == "lesson-123"
        assert "shown_at" in history[0]

    def test_recent_shown_ids(self, mock_config, temp_db_dir):
        """Test filtering recent shown IDs."""
        db = MemoryPalaceDB()

        # Mark some lessons as shown
        db.mark_as_shown("recent-1")
        db.mark_as_shown("recent-2")

        recent_ids = db._get_recent_shown_ids()
        assert "recent-1" in recent_ids
        assert "recent-2" in recent_ids

    def test_reset_shown_history(self, mock_config, temp_db_dir):
        """Test resetting shown history."""
        db = MemoryPalaceDB()
        history_file = Path(temp_db_dir) / "shown_history.json"

        db.mark_as_shown("test-id")
        assert history_file.exists()

        db._reset_shown_history()
        assert not history_file.exists()

    def test_random_lesson_excludes_recent(self, mock_config):
        """Test that random selection excludes recently shown lessons."""
        db = MemoryPalaceDB()

        # Add lessons
        lesson_ids = []
        for i in range(3):
            lesson = Lesson(
                distilled_text=f"Unique lesson text {i}",
                metadata=LessonMetadata(
                    category=LessonCategory.TECHNOLOGY,
                    original_input=f"Input {i}",
                    distilled_by_model="test"
                )
            )
            db.add_lesson(lesson)
            lesson_ids.append(lesson.id)

        # Mark first two as shown
        db.mark_as_shown(lesson_ids[0])
        db.mark_as_shown(lesson_ids[1])

        # Random should return the unshown one
        random_lesson = db.get_random_lesson(exclude_recent=True)
        assert random_lesson is not None
        assert random_lesson.id == lesson_ids[2]


class TestDistillLesson:
    """Tests for LLM-based lesson distillation."""

    @patch("helper_functions.memory_palace_db.get_client")
    @patch("helper_functions.memory_palace_db.config")
    def test_distill_lesson_success(self, mock_config, mock_get_client):
        """Test successful lesson distillation."""
        mock_config.memory_palace_provider = "litellm"
        mock_config.memory_palace_model_tier = "fast"
        mock_config.memory_palace_primary_model = "test-model"

        mock_client = Mock()
        mock_client.chat_completion.return_value = json.dumps({
            "distilled_text": "Compound interest accelerates wealth accumulation exponentially.",
            "suggested_category": "economics",
            "suggested_tags": ["finance", "investing"]
        })
        mock_get_client.return_value = mock_client

        result = distill_lesson("Long text about compound interest and its effects on wealth...")

        assert isinstance(result, LessonDistillationResult)
        assert "Compound interest" in result.distilled_text
        assert result.suggested_category == "economics"
        assert "finance" in result.suggested_tags

    @patch("helper_functions.memory_palace_db.get_client")
    @patch("helper_functions.memory_palace_db.config")
    def test_distill_lesson_invalid_category_fallback(self, mock_config, mock_get_client):
        """Test fallback when LLM suggests invalid category."""
        mock_config.memory_palace_provider = "litellm"
        mock_config.memory_palace_model_tier = "fast"
        mock_config.memory_palace_primary_model = "test-model"

        mock_client = Mock()
        mock_client.chat_completion.return_value = json.dumps({
            "distilled_text": "Test insight",
            "suggested_category": "invalid_category",
            "suggested_tags": []
        })
        mock_get_client.return_value = mock_client

        result = distill_lesson("Test input")

        assert result.suggested_category == "observations"  # Fallback

    @patch("helper_functions.memory_palace_db.get_client")
    @patch("helper_functions.memory_palace_db.config")
    def test_distill_lesson_json_parse_error_fallback(self, mock_config, mock_get_client):
        """Test fallback when LLM returns invalid JSON."""
        mock_config.memory_palace_provider = "litellm"
        mock_config.memory_palace_model_tier = "fast"
        mock_config.memory_palace_primary_model = "test-model"

        mock_client = Mock()
        mock_client.chat_completion.return_value = "This is not valid JSON"
        mock_get_client.return_value = mock_client

        result = distill_lesson("Test input text")

        assert isinstance(result, LessonDistillationResult)
        assert result.suggested_category == "observations"
        assert result.distilled_text == "Test input text"

    @patch("helper_functions.memory_palace_db.get_client")
    @patch("helper_functions.memory_palace_db.config")
    def test_distill_lesson_with_few_shot(self, mock_config, mock_get_client):
        """Test distillation with custom few-shot examples."""
        mock_config.memory_palace_provider = "litellm"
        mock_config.memory_palace_model_tier = "fast"
        mock_config.memory_palace_primary_model = "test-model"

        mock_client = Mock()
        mock_client.chat_completion.return_value = json.dumps({
            "distilled_text": "Test insight matching examples",
            "suggested_category": "strategy",
            "suggested_tags": ["example"]
        })
        mock_get_client.return_value = mock_client

        custom_examples = ["Custom example 1", "Custom example 2"]
        result = distill_lesson("Input", few_shot_examples=custom_examples)

        # Verify the prompt included custom examples
        call_args = mock_client.chat_completion.call_args
        prompt = call_args[1]["messages"][0]["content"]
        assert "Custom example 1" in prompt
        assert "Custom example 2" in prompt

    @patch("helper_functions.memory_palace_db.get_client")
    @patch("helper_functions.memory_palace_db.config")
    def test_distill_lesson_handles_markdown_code_block(self, mock_config, mock_get_client):
        """Test handling of JSON wrapped in markdown code blocks."""
        mock_config.memory_palace_provider = "litellm"
        mock_config.memory_palace_model_tier = "fast"
        mock_config.memory_palace_primary_model = "test-model"

        mock_client = Mock()
        mock_client.chat_completion.return_value = """```json
{"distilled_text": "Test insight", "suggested_category": "history", "suggested_tags": ["test"]}
```"""
        mock_get_client.return_value = mock_client

        result = distill_lesson("Test input")

        assert result.distilled_text == "Test insight"
        assert result.suggested_category == "history"


class TestLessonDistillationResult:
    """Tests for LessonDistillationResult model."""

    def test_creation(self):
        """Test basic creation."""
        result = LessonDistillationResult(
            distilled_text="Test insight",
            suggested_category="strategy",
            suggested_tags=["tag1"]
        )
        assert result.distilled_text == "Test insight"
        assert result.suggested_category == "strategy"
        assert result.suggested_tags == ["tag1"]

    def test_default_tags(self):
        """Test default empty tags."""
        result = LessonDistillationResult(
            distilled_text="Test",
            suggested_category="history"
        )
        assert result.suggested_tags == []
