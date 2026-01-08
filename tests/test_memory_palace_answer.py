"""Tests for memory_palace_answer.py - Answer Engine."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from helper_functions.memory_palace_answer import (
    AnswerEngine,
    AnswerResult,
    SourceType,
    format_answer_for_telegram,
)
from helper_functions.memory_palace_db import (
    Lesson,
    LessonCategory,
    LessonMetadata,
    SimilarLesson,
)
from helper_functions.web_knowledge_db import (
    ConfidenceTier,
    SimilarWebKnowledge,
    WebKnowledge,
    WebKnowledgeMetadata,
)


class TestSourceType:
    """Tests for SourceType enum."""

    def test_source_type_values(self):
        """Test that source type values are correct."""
        assert SourceType.USER_WISDOM == "user_wisdom"
        assert SourceType.WEB_KNOWLEDGE == "web_knowledge"
        assert SourceType.BOTH == "both"
        assert SourceType.NONE == "none"


class TestAnswerResult:
    """Tests for AnswerResult dataclass."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        result = AnswerResult(
            answer_text="Test answer",
            confidence_tier=ConfidenceTier.FAIRLY_SURE,
            source_type=SourceType.USER_WISDOM,
        )
        assert result.related_topics == []
        assert result.offer_to_research is False
        assert result.reasoning_prefix == ""
        assert result.wisdom_matches == []
        assert result.knowledge_matches == []

    def test_custom_values(self):
        """Test answer result with custom values."""
        result = AnswerResult(
            answer_text="Custom answer",
            confidence_tier=ConfidenceTier.VERY_CONFIDENT,
            source_type=SourceType.BOTH,
            related_topics=["Psychology", "Strategy"],
            offer_to_research=True,
            reasoning_prefix="[From memory: Very confident]",
        )
        assert result.answer_text == "Custom answer"
        assert result.confidence_tier == ConfidenceTier.VERY_CONFIDENT
        assert result.source_type == SourceType.BOTH
        assert len(result.related_topics) == 2


class TestAnswerEngineConfidenceCalculation:
    """Tests for AnswerEngine confidence calculation."""

    @pytest.fixture
    def mock_wisdom_db(self):
        """Create a mock wisdom database."""
        return MagicMock()

    @pytest.fixture
    def mock_knowledge_db(self):
        """Create a mock knowledge database."""
        return MagicMock()

    @pytest.fixture
    def answer_engine(self, mock_wisdom_db, mock_knowledge_db):
        """Create an answer engine with mocked databases."""
        with patch("helper_functions.memory_palace_answer.MemoryPalaceDB", return_value=mock_wisdom_db):
            with patch("helper_functions.memory_palace_answer.WebKnowledgeDB", return_value=mock_knowledge_db):
                return AnswerEngine(
                    wisdom_db=mock_wisdom_db,
                    knowledge_db=mock_knowledge_db,
                )

    def _create_test_lesson(self, text="Test lesson", category=LessonCategory.STRATEGY):
        """Helper to create a test lesson with required fields."""
        return Lesson(
            distilled_text=text,
            metadata=LessonMetadata(
                category=category,
                original_input="Original user input for testing",
                distilled_by_model="test-model",
            )
        )

    def test_high_wisdom_confidence(self, answer_engine):
        """Test that high wisdom score gives VERY_CONFIDENT from USER_WISDOM."""
        lesson = self._create_test_lesson()
        best_wisdom = SimilarLesson(lesson=lesson, similarity_score=0.9)

        confidence, source_type, offer = answer_engine._calculate_confidence(
            best_wisdom, None
        )

        assert confidence == ConfidenceTier.VERY_CONFIDENT
        assert source_type == SourceType.USER_WISDOM
        assert offer is False

    def test_medium_wisdom_only(self, answer_engine):
        """Test medium wisdom score without knowledge gives FAIRLY_SURE."""
        lesson = self._create_test_lesson()
        best_wisdom = SimilarLesson(lesson=lesson, similarity_score=0.7)

        confidence, source_type, offer = answer_engine._calculate_confidence(
            best_wisdom, None
        )

        assert confidence == ConfidenceTier.FAIRLY_SURE
        assert source_type == SourceType.USER_WISDOM
        assert offer is True  # Offer to research for partial match

    def test_both_sources_combined(self, answer_engine):
        """Test medium wisdom + medium knowledge gives VERY_CONFIDENT from BOTH."""
        lesson = self._create_test_lesson()
        best_wisdom = SimilarLesson(lesson=lesson, similarity_score=0.7)

        knowledge = WebKnowledge(
            distilled_text="Test knowledge",
            metadata=WebKnowledgeMetadata(
                original_query="test",
                expires_at=datetime.now() + timedelta(days=7),
            )
        )
        best_knowledge = SimilarWebKnowledge(knowledge=knowledge, similarity_score=0.7)

        confidence, source_type, offer = answer_engine._calculate_confidence(
            best_wisdom, best_knowledge
        )

        assert confidence == ConfidenceTier.VERY_CONFIDENT
        assert source_type == SourceType.BOTH
        assert offer is False

    def test_knowledge_only_high(self, answer_engine):
        """Test high knowledge score without wisdom gives FAIRLY_SURE."""
        knowledge = WebKnowledge(
            distilled_text="Test knowledge",
            metadata=WebKnowledgeMetadata(
                original_query="test",
                expires_at=datetime.now() + timedelta(days=7),
            )
        )
        best_knowledge = SimilarWebKnowledge(knowledge=knowledge, similarity_score=0.9)

        confidence, source_type, offer = answer_engine._calculate_confidence(
            None, best_knowledge
        )

        assert confidence == ConfidenceTier.FAIRLY_SURE
        assert source_type == SourceType.WEB_KNOWLEDGE
        assert offer is False

    def test_no_matches(self, answer_engine):
        """Test no matches gives UNCERTAIN from NONE."""
        confidence, source_type, offer = answer_engine._calculate_confidence(
            None, None
        )

        assert confidence == ConfidenceTier.UNCERTAIN
        assert source_type == SourceType.NONE
        assert offer is True


class TestAnswerEngineConflictCheck:
    """Tests for AnswerEngine conflict detection."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock LLM client."""
        mock = MagicMock()
        mock.chat_completion.return_value = "no"
        return mock

    def _create_test_lesson(self, text="Test lesson", category=LessonCategory.STRATEGY):
        """Helper to create a test lesson with required fields."""
        return Lesson(
            distilled_text=text,
            metadata=LessonMetadata(
                category=category,
                original_input="Original user input for testing",
                distilled_by_model="test-model",
            )
        )

    def test_conflict_detected(self, mock_client):
        """Test that conflict is detected when LLM says yes."""
        mock_client.chat_completion.return_value = "yes"

        with patch("helper_functions.memory_palace_answer.get_client", return_value=mock_client):
            engine = AnswerEngine(wisdom_db=MagicMock(), knowledge_db=MagicMock())

            lesson = self._create_test_lesson(
                text="Intermittent fasting is the best diet",
                category=LessonCategory.BIOLOGY
            )
            wisdom = SimilarLesson(lesson=lesson, similarity_score=0.8)

            has_conflict = engine.check_for_conflict(
                wisdom, "Mediterranean diet is best for longevity"
            )

            assert has_conflict is True

    def test_no_conflict_detected(self, mock_client):
        """Test that no conflict is detected when LLM says no."""
        mock_client.chat_completion.return_value = "no"

        with patch("helper_functions.memory_palace_answer.get_client", return_value=mock_client):
            engine = AnswerEngine(wisdom_db=MagicMock(), knowledge_db=MagicMock())

            lesson = self._create_test_lesson(
                text="Exercise improves mood",
                category=LessonCategory.BIOLOGY
            )
            wisdom = SimilarLesson(lesson=lesson, similarity_score=0.8)

            has_conflict = engine.check_for_conflict(
                wisdom, "Physical activity releases endorphins"
            )

            assert has_conflict is False

    def test_conflict_check_handles_error(self, mock_client):
        """Test that conflict check returns False on error."""
        mock_client.chat_completion.side_effect = Exception("API error")

        with patch("helper_functions.memory_palace_answer.get_client", return_value=mock_client):
            engine = AnswerEngine(wisdom_db=MagicMock(), knowledge_db=MagicMock())

            lesson = self._create_test_lesson(
                text="Test wisdom",
                category=LessonCategory.OBSERVATIONS
            )
            wisdom = SimilarLesson(lesson=lesson, similarity_score=0.8)

            has_conflict = engine.check_for_conflict(wisdom, "Test info")

            assert has_conflict is False


class TestFormatAnswerForTelegram:
    """Tests for format_answer_for_telegram function."""

    def _create_test_lesson(self, text="Test lesson", category=LessonCategory.STRATEGY):
        """Helper to create a test lesson with required fields."""
        return Lesson(
            distilled_text=text,
            metadata=LessonMetadata(
                category=category,
                original_input="Original user input for testing",
                distilled_by_model="test-model",
            )
        )

    def test_format_no_match(self):
        """Test formatting when no match is found."""
        result = AnswerResult(
            answer_text="",
            confidence_tier=ConfidenceTier.UNCERTAIN,
            source_type=SourceType.NONE,
            related_topics=["Psychology", "Strategy"],
        )

        formatted = format_answer_for_telegram(result)

        assert "I don't have any knowledge" in formatted
        assert "Psychology" in formatted
        assert "Strategy" in formatted

    def test_format_with_wisdom_match(self):
        """Test formatting with user wisdom match."""
        lesson = self._create_test_lesson()
        wisdom_match = SimilarLesson(lesson=lesson, similarity_score=0.85)

        result = AnswerResult(
            answer_text="Your lesson suggests X",
            confidence_tier=ConfidenceTier.VERY_CONFIDENT,
            source_type=SourceType.USER_WISDOM,
            reasoning_prefix="[From memory: Very confident] (85% match)",
            wisdom_matches=[wisdom_match],
        )

        formatted = format_answer_for_telegram(result)

        assert "[From memory: Very confident]" in formatted
        assert "Your lesson suggests X" in formatted
        assert "Strategy" in formatted

    def test_format_with_knowledge_match(self):
        """Test formatting with web knowledge match."""
        result = AnswerResult(
            answer_text="Research indicates X",
            confidence_tier=ConfidenceTier.FAIRLY_SURE,
            source_type=SourceType.WEB_KNOWLEDGE,
            reasoning_prefix="[From research: Fairly sure] (3 sources)",
        )

        formatted = format_answer_for_telegram(result)

        assert "[From research: Fairly sure]" in formatted
        assert "Research indicates X" in formatted
