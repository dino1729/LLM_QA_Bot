"""
Tests for the Memory Palace Telegram Bot.

Tests cover:
- Access control (authorized_only decorator)
- Command handlers (start, add, search, random, stats, cancel)
- Conversation flow and state transitions
- Distillation and confirmation workflow
"""

import asyncio
from datetime import timedelta
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from helper_functions.memory_palace_bot import (
    IntentResult,
    MemoryPalaceBot,
    State,
    UserIntent,
    authorized_only,
)
from helper_functions.memory_palace_db import (
    Lesson,
    LessonCategory,
    LessonDistillationResult,
    LessonMetadata,
    SimilarLesson,
)


@pytest.fixture
def mock_config():
    """Mock config module."""
    with patch("helper_functions.memory_palace_bot.config") as mock_cfg:
        mock_cfg.telegram_bot_token = "test-token-12345"
        mock_cfg.memory_palace_telegram_user_id = 123456789
        mock_cfg.memory_palace_provider = "litellm"
        mock_cfg.memory_palace_model_tier = "fast"
        mock_cfg.memory_palace_primary_model = "test-model"
        yield mock_cfg


@pytest.fixture
def mock_db():
    """Mock MemoryPalaceDB."""
    mock = MagicMock()
    mock.get_lesson_count.return_value = 100
    mock.get_category_stats.return_value = {
        "strategy": 20,
        "psychology": 15,
        "history": 10,
        "technology": 25,
        "observations": 30,
    }
    mock.find_similar.return_value = []
    mock.check_duplicate.return_value = None
    mock.get_few_shot_examples.return_value = [
        "Example lesson 1",
        "Example lesson 2",
        "Example lesson 3",
    ]
    mock.add_lesson.return_value = "test-lesson-id-12345"
    return mock


@pytest.fixture
def mock_update():
    """Create a mock Telegram Update object."""
    update = MagicMock()
    update.effective_user.id = 123456789  # Authorized user
    update.message.text = "Test message"
    update.message.reply_text = AsyncMock()
    update.callback_query = None
    return update


@pytest.fixture
def mock_context():
    """Create a mock Telegram context."""
    context = MagicMock()
    context.user_data = {}
    context.args = []
    return context


class TestAccessControl:
    """Tests for the authorized_only decorator."""

    @pytest.mark.asyncio
    @patch("helper_functions.memory_palace_bot.config")
    async def test_authorized_user_passes(self, mock_cfg):
        """Test that authorized user can access handlers."""
        mock_cfg.memory_palace_telegram_user_id = 123456789

        @authorized_only
        async def handler(update, context):
            return "success"

        update = MagicMock()
        update.effective_user.id = 123456789
        context = MagicMock()

        result = await handler(update, context)
        assert result == "success"

    @pytest.mark.asyncio
    @patch("helper_functions.memory_palace_bot.config")
    async def test_unauthorized_user_blocked(self, mock_cfg):
        """Test that unauthorized user is blocked."""
        mock_cfg.memory_palace_telegram_user_id = 123456789

        @authorized_only
        async def handler(update, context):
            return "success"

        update = MagicMock()
        update.effective_user.id = 987654321  # Different user
        update.message.reply_text = AsyncMock()
        context = MagicMock()

        from telegram.ext import ConversationHandler
        result = await handler(update, context)
        assert result == ConversationHandler.END
        update.message.reply_text.assert_called_once()

    @pytest.mark.asyncio
    @patch("helper_functions.memory_palace_bot.config")
    async def test_discovery_mode_shows_user_id(self, mock_cfg):
        """Test discovery mode shows user ID when telegram_user_id is None."""
        mock_cfg.memory_palace_telegram_user_id = None

        @authorized_only
        async def handler(update, context):
            return "success"

        update = MagicMock()
        update.effective_user.id = 555555555
        update.message.reply_text = AsyncMock()
        context = MagicMock()

        from telegram.ext import ConversationHandler
        result = await handler(update, context)
        assert result == ConversationHandler.END

        # Should show the user ID
        call_args = update.message.reply_text.call_args
        assert "555555555" in call_args[0][0]


class TestBotInitialization:
    """Tests for bot initialization."""

    @patch("helper_functions.memory_palace_bot.MemoryPalaceDB")
    @patch("helper_functions.memory_palace_bot.config")
    def test_init_with_token(self, mock_cfg, mock_db_class):
        """Test bot initializes with token."""
        mock_cfg.telegram_bot_token = "test-token"
        mock_db_class.return_value = MagicMock()

        bot = MemoryPalaceBot()
        assert bot.token == "test-token"

    @patch("helper_functions.memory_palace_bot.MemoryPalaceDB")
    @patch("helper_functions.memory_palace_bot.config")
    def test_init_without_token_raises(self, mock_cfg, mock_db_class):
        """Test bot raises error without token."""
        mock_cfg.telegram_bot_token = None
        mock_db_class.return_value = MagicMock()

        with pytest.raises(ValueError, match="telegram_bot_token"):
            MemoryPalaceBot()


class TestCommandHandlers:
    """Tests for command handlers."""

    @pytest.mark.asyncio
    @patch("helper_functions.memory_palace_bot.config")
    @patch("helper_functions.memory_palace_bot.MemoryPalaceDB")
    async def test_start_command(self, mock_db_class, mock_cfg):
        """Test /start command response."""
        mock_cfg.telegram_bot_token = "test-token"
        mock_cfg.memory_palace_telegram_user_id = 123456789
        mock_cfg.memory_palace_primary_model = "test-model"

        mock_db_class.return_value = MagicMock()
        mock_db_class.return_value.get_category_stats.return_value = {"strategy": 10}

        # Create mock update and context
        mock_update = MagicMock()
        mock_update.effective_user.id = 123456789
        mock_update.message.reply_text = AsyncMock()

        mock_context = MagicMock()
        mock_context.user_data = {}

        bot = MemoryPalaceBot()
        result = await bot.start_command(mock_update, mock_context)

        mock_update.message.reply_text.assert_called_once()
        call_text = mock_update.message.reply_text.call_args[0][0]
        assert "Welcome to Memory Palace" in call_text
        assert result == State.AWAITING_LESSON

    @pytest.mark.asyncio
    @patch("helper_functions.memory_palace_bot.config")
    @patch("helper_functions.memory_palace_bot.MemoryPalaceDB")
    async def test_add_command(self, mock_db_class, mock_cfg):
        """Test /add command response."""
        mock_cfg.telegram_bot_token = "test-token"
        mock_cfg.memory_palace_telegram_user_id = 123456789
        mock_cfg.memory_palace_primary_model = "test-model"

        mock_db_class.return_value = MagicMock()

        mock_update = MagicMock()
        mock_update.effective_user.id = 123456789
        mock_update.message.reply_text = AsyncMock()

        mock_context = MagicMock()
        mock_context.user_data = {}

        bot = MemoryPalaceBot()
        result = await bot.add_command(mock_update, mock_context)

        mock_update.message.reply_text.assert_called_once()
        call_text = mock_update.message.reply_text.call_args[0][0]
        assert "Send me the text" in call_text
        assert result == State.AWAITING_LESSON

    @pytest.mark.asyncio
    @patch("helper_functions.memory_palace_bot.config")
    @patch("helper_functions.memory_palace_bot.MemoryPalaceDB")
    async def test_stats_command(self, mock_db_class, mock_cfg):
        """Test /stats command response."""
        mock_cfg.telegram_bot_token = "test-token"
        mock_cfg.memory_palace_telegram_user_id = 123456789
        mock_cfg.memory_palace_primary_model = "test-model"

        mock_db_class.return_value = MagicMock()
        mock_db_class.return_value.get_category_stats.return_value = {
            "strategy": 20,
            "psychology": 15,
        }

        mock_update = MagicMock()
        mock_update.effective_user.id = 123456789
        mock_update.message.reply_text = AsyncMock()

        mock_context = MagicMock()
        mock_context.user_data = {}

        bot = MemoryPalaceBot()
        result = await bot.stats_command(mock_update, mock_context)

        mock_update.message.reply_text.assert_called_once()
        call_text = mock_update.message.reply_text.call_args[0][0]
        assert "Statistics" in call_text
        assert "35" in call_text  # Total
        assert result == State.AWAITING_LESSON

    @pytest.mark.asyncio
    @patch("helper_functions.memory_palace_bot.config")
    @patch("helper_functions.memory_palace_bot.MemoryPalaceDB")
    async def test_random_command(self, mock_db_class, mock_cfg):
        """Test /random command response."""
        mock_cfg.telegram_bot_token = "test-token"
        mock_cfg.memory_palace_telegram_user_id = 123456789
        mock_cfg.memory_palace_primary_model = "test-model"

        mock_db_class.return_value = MagicMock()
        mock_lesson = Lesson(
            distilled_text="Test lesson content",
            metadata=LessonMetadata(
                category=LessonCategory.STRATEGY,
                original_input="test",
                distilled_by_model="test"
            )
        )
        mock_db_class.return_value.get_random_lesson.return_value = mock_lesson

        mock_update = MagicMock()
        mock_update.effective_user.id = 123456789
        mock_update.message.reply_text = AsyncMock()

        mock_context = MagicMock()
        mock_context.user_data = {}

        bot = MemoryPalaceBot()
        result = await bot.random_command(mock_update, mock_context)

        mock_update.message.reply_text.assert_called_once()
        call_text = mock_update.message.reply_text.call_args[0][0]
        assert "Test lesson content" in call_text
        assert result == State.AWAITING_LESSON

    @pytest.mark.asyncio
    @patch("helper_functions.memory_palace_bot.config")
    @patch("helper_functions.memory_palace_bot.MemoryPalaceDB")
    async def test_random_command_empty_db(self, mock_db_class, mock_cfg):
        """Test /random on empty database."""
        mock_cfg.telegram_bot_token = "test-token"
        mock_cfg.memory_palace_telegram_user_id = 123456789
        mock_cfg.memory_palace_primary_model = "test-model"

        mock_db_class.return_value = MagicMock()
        mock_db_class.return_value.get_random_lesson.return_value = None

        mock_update = MagicMock()
        mock_update.effective_user.id = 123456789
        mock_update.message.reply_text = AsyncMock()

        mock_context = MagicMock()
        mock_context.user_data = {}

        bot = MemoryPalaceBot()
        result = await bot.random_command(mock_update, mock_context)

        call_text = mock_update.message.reply_text.call_args[0][0]
        assert "No lessons" in call_text

    @pytest.mark.asyncio
    @patch("helper_functions.memory_palace_bot.config")
    @patch("helper_functions.memory_palace_bot.MemoryPalaceDB")
    async def test_search_command_with_query(self, mock_db_class, mock_cfg):
        """Test /search with query."""
        mock_cfg.telegram_bot_token = "test-token"
        mock_cfg.memory_palace_telegram_user_id = 123456789
        mock_cfg.memory_palace_primary_model = "test-model"

        mock_db_class.return_value = MagicMock()
        mock_lesson = Lesson(
            distilled_text="Found lesson",
            metadata=LessonMetadata(
                category=LessonCategory.PSYCHOLOGY,
                original_input="test",
                distilled_by_model="test"
            )
        )
        mock_db_class.return_value.find_similar.return_value = [
            SimilarLesson(lesson=mock_lesson, similarity_score=0.85)
        ]

        mock_update = MagicMock()
        mock_update.effective_user.id = 123456789
        mock_update.message.reply_text = AsyncMock()

        mock_context = MagicMock()
        mock_context.user_data = {}
        mock_context.args = ["game", "theory"]

        bot = MemoryPalaceBot()
        result = await bot.search_command(mock_update, mock_context)

        call_text = mock_update.message.reply_text.call_args[0][0]
        assert "Found lesson" in call_text
        assert "0.85" in call_text

    @pytest.mark.asyncio
    @patch("helper_functions.memory_palace_bot.config")
    @patch("helper_functions.memory_palace_bot.MemoryPalaceDB")
    async def test_search_command_no_query(self, mock_db_class, mock_cfg):
        """Test /search without query shows usage."""
        mock_cfg.telegram_bot_token = "test-token"
        mock_cfg.memory_palace_telegram_user_id = 123456789
        mock_cfg.memory_palace_primary_model = "test-model"

        mock_db_class.return_value = MagicMock()

        mock_update = MagicMock()
        mock_update.effective_user.id = 123456789
        mock_update.message.reply_text = AsyncMock()

        mock_context = MagicMock()
        mock_context.user_data = {}
        mock_context.args = []

        bot = MemoryPalaceBot()
        result = await bot.search_command(mock_update, mock_context)

        call_text = mock_update.message.reply_text.call_args[0][0]
        assert "Usage:" in call_text

    @pytest.mark.asyncio
    @patch("helper_functions.memory_palace_bot.config")
    @patch("helper_functions.memory_palace_bot.MemoryPalaceDB")
    async def test_cancel_command(self, mock_db_class, mock_cfg):
        """Test /cancel command clears state."""
        mock_cfg.telegram_bot_token = "test-token"
        mock_cfg.memory_palace_telegram_user_id = 123456789
        mock_cfg.memory_palace_primary_model = "test-model"

        mock_db_class.return_value = MagicMock()

        mock_update = MagicMock()
        mock_update.message.reply_text = AsyncMock()

        mock_context = MagicMock()
        mock_context.user_data = {"some": "data"}

        bot = MemoryPalaceBot()
        result = await bot.cancel_command(mock_update, mock_context)

        assert mock_context.user_data == {}
        from telegram.ext import ConversationHandler
        assert result == ConversationHandler.END


class TestDistillationFlow:
    """Tests for distillation conversation flow."""

    @pytest.mark.asyncio
    @patch("helper_functions.memory_palace_bot.detect_intent")
    @patch("helper_functions.memory_palace_bot.AnswerEngine")
    @patch("helper_functions.memory_palace_bot.WebKnowledgeDB")
    @patch("helper_functions.memory_palace_bot.config")
    @patch("helper_functions.memory_palace_bot.distill_lesson")
    @patch("helper_functions.memory_palace_bot.MemoryPalaceDB")
    async def test_receive_lesson_distills(
        self, mock_db_class, mock_distill, mock_cfg, mock_web_kb, mock_answer_engine, mock_detect
    ):
        """Test receiving lesson text triggers distillation."""
        mock_cfg.telegram_bot_token = "test-token"
        mock_cfg.memory_palace_telegram_user_id = 123456789
        mock_cfg.memory_palace_primary_model = "test-model"
        mock_cfg.memory_palace_provider = "litellm"
        mock_cfg.memory_palace_model_tier = "fast"

        mock_db_class.return_value = MagicMock()
        mock_db_class.return_value.check_duplicate.return_value = None
        mock_db_class.return_value.get_few_shot_examples.return_value = []

        mock_web_kb.return_value = MagicMock()
        mock_answer_engine.return_value = MagicMock()

        # Mock intent detection to return ADD_LESSON
        mock_detect.return_value = IntentResult(
            intent=UserIntent.ADD_LESSON,
            confidence=0.9
        )

        mock_distill.return_value = LessonDistillationResult(
            distilled_text="Distilled insight",
            suggested_category="strategy",
            suggested_tags=["test"]
        )

        mock_update = MagicMock()
        mock_update.effective_user.id = 123456789
        mock_update.message.text = "Long input text to distill..."
        mock_update.message.reply_text = AsyncMock()

        mock_context = MagicMock()
        mock_context.user_data = {}

        bot = MemoryPalaceBot()
        result = await bot.receive_lesson_text(mock_update, mock_context)

        assert mock_context.user_data["distilled_text"] == "Distilled insight"
        assert mock_context.user_data["suggested_category"] == "strategy"
        assert result == State.CONFIRMING_DISTILLED

    @pytest.mark.asyncio
    @patch("helper_functions.memory_palace_bot.detect_intent")
    @patch("helper_functions.memory_palace_bot.AnswerEngine")
    @patch("helper_functions.memory_palace_bot.WebKnowledgeDB")
    @patch("helper_functions.memory_palace_bot.config")
    @patch("helper_functions.memory_palace_bot.distill_lesson")
    @patch("helper_functions.memory_palace_bot.MemoryPalaceDB")
    async def test_duplicate_detection_triggers_warning(
        self, mock_db_class, mock_distill, mock_cfg, mock_web_kb, mock_answer_engine, mock_detect
    ):
        """Test duplicate detection shows warning."""
        mock_cfg.telegram_bot_token = "test-token"
        mock_cfg.memory_palace_telegram_user_id = 123456789
        mock_cfg.memory_palace_primary_model = "test-model"
        mock_cfg.memory_palace_provider = "litellm"
        mock_cfg.memory_palace_model_tier = "fast"

        mock_db = MagicMock()
        mock_db_class.return_value = mock_db
        mock_db.get_few_shot_examples.return_value = []

        mock_web_kb.return_value = MagicMock()
        mock_answer_engine.return_value = MagicMock()

        # Mock intent detection to return ADD_LESSON
        mock_detect.return_value = IntentResult(
            intent=UserIntent.ADD_LESSON,
            confidence=0.9
        )

        # Create existing lesson that will be flagged as duplicate
        existing_lesson = Lesson(
            distilled_text="Similar existing lesson",
            metadata=LessonMetadata(
                category=LessonCategory.STRATEGY,
                original_input="test",
                distilled_by_model="test"
            )
        )
        mock_db.check_duplicate.return_value = SimilarLesson(
            lesson=existing_lesson,
            similarity_score=0.85
        )

        mock_distill.return_value = LessonDistillationResult(
            distilled_text="New similar lesson",
            suggested_category="strategy",
            suggested_tags=[]
        )

        mock_update = MagicMock()
        mock_update.effective_user.id = 123456789
        mock_update.message.text = "Input text"
        mock_update.message.reply_text = AsyncMock()

        mock_context = MagicMock()
        mock_context.user_data = {}

        bot = MemoryPalaceBot()
        result = await bot.receive_lesson_text(mock_update, mock_context)

        assert result == State.CONFIRMING_DUPLICATE
        call_text = mock_update.message.reply_text.call_args[0][0]
        assert "Similar lesson found" in call_text


class TestCallbackHandlers:
    """Tests for callback query handlers."""

    @pytest.mark.asyncio
    @patch("helper_functions.memory_palace_bot.MemoryPalaceDB")
    async def test_approve_goes_to_category_selection(
        self, mock_db_class, mock_config, mock_context
    ):
        """Test approving distilled text goes to category selection."""
        mock_db_class.return_value = MagicMock()

        mock_query = AsyncMock()
        mock_query.data = "approve"
        mock_query.answer = AsyncMock()
        mock_query.edit_message_text = AsyncMock()

        update = MagicMock()
        update.callback_query = mock_query

        mock_context.user_data = {
            "distilled_text": "Test lesson",
            "suggested_category": "strategy"
        }

        bot = MemoryPalaceBot()
        result = await bot.handle_distillation_response(update, mock_context)

        assert result == State.CONFIRMING_CATEGORY
        mock_query.edit_message_text.assert_called_once()

    @pytest.mark.asyncio
    @patch("helper_functions.memory_palace_bot.MemoryPalaceDB")
    async def test_reject_clears_state(self, mock_db_class, mock_config, mock_context):
        """Test rejecting distilled text clears state."""
        mock_db_class.return_value = MagicMock()

        mock_query = AsyncMock()
        mock_query.data = "reject"
        mock_query.answer = AsyncMock()
        mock_query.edit_message_text = AsyncMock()

        update = MagicMock()
        update.callback_query = mock_query

        mock_context.user_data = {"some": "data"}

        bot = MemoryPalaceBot()
        result = await bot.handle_distillation_response(update, mock_context)

        from telegram.ext import ConversationHandler
        assert result == ConversationHandler.END
        assert mock_context.user_data == {}

    @pytest.mark.asyncio
    @patch("helper_functions.memory_palace_bot.MemoryPalaceDB")
    async def test_edit_triggers_editing_state(self, mock_db_class, mock_config, mock_context):
        """Test edit option triggers editing state."""
        mock_db_class.return_value = MagicMock()

        mock_query = AsyncMock()
        mock_query.data = "edit"
        mock_query.answer = AsyncMock()
        mock_query.edit_message_text = AsyncMock()

        update = MagicMock()
        update.callback_query = mock_query

        mock_context.user_data = {"distilled_text": "Test"}

        bot = MemoryPalaceBot()
        result = await bot.handle_distillation_response(update, mock_context)

        assert result == State.EDITING_LESSON

    @pytest.mark.asyncio
    @patch("helper_functions.memory_palace_bot.MemoryPalaceDB")
    async def test_category_selection_saves_lesson(
        self, mock_db_class, mock_config, mock_context
    ):
        """Test category selection saves the lesson."""
        mock_db = MagicMock()
        mock_db_class.return_value = mock_db
        mock_db.add_lesson.return_value = "new-lesson-id-123"

        mock_query = AsyncMock()
        mock_query.data = "cat_strategy"
        mock_query.answer = AsyncMock()
        mock_query.edit_message_text = AsyncMock()

        update = MagicMock()
        update.callback_query = mock_query

        mock_context.user_data = {
            "distilled_text": "Test lesson",
            "original_input": "Original input",
            "suggested_tags": ["tag1"]
        }

        bot = MemoryPalaceBot()
        result = await bot.handle_category_selection(update, mock_context)

        from telegram.ext import ConversationHandler
        assert result == ConversationHandler.END
        mock_db.add_lesson.assert_called_once()

        # Verify lesson was created correctly
        saved_lesson = mock_db.add_lesson.call_args[0][0]
        assert saved_lesson.distilled_text == "Test lesson"
        assert saved_lesson.metadata.category == LessonCategory.STRATEGY


class TestKeyboards:
    """Tests for keyboard generation."""

    @patch("helper_functions.memory_palace_bot.MemoryPalaceDB")
    def test_confirmation_keyboard(self, mock_db_class, mock_config):
        """Test confirmation keyboard has correct buttons."""
        mock_db_class.return_value = MagicMock()

        bot = MemoryPalaceBot()
        keyboard = bot._get_confirmation_keyboard()

        # Check keyboard structure
        buttons = []
        for row in keyboard.inline_keyboard:
            for btn in row:
                buttons.append(btn.callback_data)

        assert "approve" in buttons
        assert "edit" in buttons
        assert "reject" in buttons

    @patch("helper_functions.memory_palace_bot.MemoryPalaceDB")
    def test_category_keyboard(self, mock_db_class, mock_config):
        """Test category keyboard includes all categories."""
        mock_db_class.return_value = MagicMock()

        bot = MemoryPalaceBot()
        keyboard = bot._get_category_keyboard()

        buttons = []
        for row in keyboard.inline_keyboard:
            for btn in row:
                buttons.append(btn.callback_data)

        # Check all categories are present
        from helper_functions.memory_palace_db import CATEGORIES
        for cat in CATEGORIES.keys():
            assert f"cat_{cat}" in buttons

    @patch("helper_functions.memory_palace_bot.MemoryPalaceDB")
    def test_duplicate_keyboard(self, mock_db_class, mock_config):
        """Test duplicate keyboard has correct options."""
        mock_db_class.return_value = MagicMock()

        bot = MemoryPalaceBot()
        keyboard = bot._get_duplicate_keyboard()

        buttons = []
        for row in keyboard.inline_keyboard:
            for btn in row:
                buttons.append(btn.callback_data)

        assert "add_anyway" in buttons
        assert "cancel_dup" in buttons


class TestBotApplication:
    """Tests for bot application building."""

    @patch("helper_functions.memory_palace_bot.MemoryPalaceDB")
    def test_build_application(self, mock_db_class, mock_config):
        """Test application builds successfully with handlers."""
        mock_db_class.return_value = MagicMock()

        bot = MemoryPalaceBot()
        app = bot.build_application()

        # Application should be built successfully
        assert app is not None
        # Should have handlers registered
        assert len(app.handlers) > 0


class TestScheduledReminders:
    """Tests for periodic random memory reminders."""

    @patch("helper_functions.memory_palace_bot.MemoryPalaceDB")
    def test_schedule_memory_reminder_registers_8h_job(self, mock_db_class, mock_config):
        """Test reminder scheduler registers an 8-hour repeating job."""
        mock_db_class.return_value = MagicMock()

        bot = MemoryPalaceBot()
        app = MagicMock()
        app.job_queue = MagicMock()
        app.job_queue.get_jobs_by_name.return_value = []

        bot._schedule_memory_reminder(app)

        app.job_queue.run_repeating.assert_called_once()
        call_kwargs = app.job_queue.run_repeating.call_args.kwargs
        assert call_kwargs["interval"] == timedelta(hours=8)
        assert call_kwargs["first"] == timedelta(hours=8)
        assert call_kwargs["chat_id"] == mock_config.memory_palace_telegram_user_id

    @patch("helper_functions.memory_palace_bot.MemoryPalaceDB")
    def test_schedule_memory_reminder_skips_when_user_missing(self, mock_db_class, mock_config):
        """Test scheduler is skipped when no authorized user is configured."""
        mock_db_class.return_value = MagicMock()
        mock_config.memory_palace_telegram_user_id = None

        bot = MemoryPalaceBot()
        app = MagicMock()
        app.job_queue = MagicMock()

        bot._schedule_memory_reminder(app)

        app.job_queue.run_repeating.assert_not_called()

    @patch("helper_functions.memory_palace_bot.MemoryPalaceDB")
    def test_schedule_memory_reminder_falls_back_without_job_queue(self, mock_db_class, mock_config):
        """Test scheduler uses async fallback when PTB job queue is unavailable."""
        mock_db_class.return_value = MagicMock()

        bot = MemoryPalaceBot()
        app = MagicMock()
        app.job_queue = None
        app.create_task = MagicMock(side_effect=lambda coro, **kwargs: coro.close())

        bot._schedule_memory_reminder(app)

        app.create_task.assert_called_once()
        assert bot._reminder_task_started is True

    @pytest.mark.asyncio
    @patch("helper_functions.memory_palace_bot.MemoryPalaceDB")
    async def test_random_reminder_sends_lesson_and_marks_shown(self, mock_db_class, mock_config):
        """Test reminder sends a lesson and updates shown history."""
        mock_db = MagicMock()
        mock_db_class.return_value = mock_db

        lesson = Lesson(
            distilled_text="Reminder lesson",
            metadata=LessonMetadata(
                category=LessonCategory.STRATEGY,
                original_input="input",
                distilled_by_model="test-model",
            ),
        )
        mock_db.get_random_lesson.return_value = lesson

        context = MagicMock()
        context.bot.send_message = AsyncMock()
        context.job = MagicMock()
        context.job.chat_id = mock_config.memory_palace_telegram_user_id

        bot = MemoryPalaceBot()
        await bot._send_random_memory_reminder(context)

        mock_db.get_random_lesson.assert_called_once_with(exclude_recent=True)
        mock_db.mark_as_shown.assert_called_once_with(lesson.id)
        context.bot.send_message.assert_called_once()
        sent_text = context.bot.send_message.call_args.kwargs["text"]
        assert "Memory reminder" in sent_text
        assert "Reminder lesson" in sent_text

    @pytest.mark.asyncio
    @patch("helper_functions.memory_palace_bot.MemoryPalaceDB")
    async def test_random_reminder_skips_non_objective_lesson_then_sends_objective(
        self, mock_db_class, mock_config
    ):
        """Reminder selection should skip source-referential lessons."""
        mock_db = MagicMock()
        mock_db_class.return_value = mock_db

        bad_lesson = Lesson(
            distilled_text="The author predicts that domain-specific chatbots will emerge as a game-changing technology.",
            metadata=LessonMetadata(
                category=LessonCategory.TECHNOLOGY,
                original_input="input",
                distilled_by_model="test-model",
            ),
        )
        good_lesson = Lesson(
            distilled_text="Domain-specific chatbots can be game-changing in narrow workflows with clear feedback loops.",
            metadata=LessonMetadata(
                category=LessonCategory.TECHNOLOGY,
                original_input="input",
                distilled_by_model="test-model",
            ),
        )
        mock_db.get_random_lesson.side_effect = [bad_lesson, good_lesson]

        context = MagicMock()
        context.bot.send_message = AsyncMock()
        context.job = MagicMock()
        context.job.chat_id = mock_config.memory_palace_telegram_user_id

        bot = MemoryPalaceBot()
        await bot._send_random_memory_reminder(context)

        assert mock_db.get_random_lesson.call_count == 2
        mock_db.mark_as_shown.assert_called_once_with(good_lesson.id)
        sent_text = context.bot.send_message.call_args.kwargs["text"]
        assert "Domain-specific chatbots" in sent_text
        assert "author predicts" not in sent_text.lower()

    @pytest.mark.asyncio
    @patch("helper_functions.memory_palace_bot.MemoryPalaceDB")
    async def test_random_reminder_no_lessons_sends_nothing(self, mock_db_class, mock_config):
        """Test reminder does nothing when lesson store is empty."""
        mock_db = MagicMock()
        mock_db.get_random_lesson.return_value = None
        mock_db_class.return_value = mock_db

        context = MagicMock()
        context.bot.send_message = AsyncMock()
        context.job = MagicMock()
        context.job.chat_id = mock_config.memory_palace_telegram_user_id

        bot = MemoryPalaceBot()
        await bot._send_random_memory_reminder(context)

        mock_db.get_random_lesson.assert_called_once_with(exclude_recent=True)
        mock_db.mark_as_shown.assert_not_called()
        context.bot.send_message.assert_not_called()

    @pytest.mark.asyncio
    @patch("helper_functions.memory_palace_bot.MemoryPalaceDB")
    async def test_random_reminder_send_failure_does_not_mark_shown(
        self, mock_db_class, mock_config
    ):
        """Test failed send does not consume reminder recency slot."""
        mock_db = MagicMock()
        mock_db_class.return_value = mock_db

        lesson = Lesson(
            distilled_text="Reminder lesson",
            metadata=LessonMetadata(
                category=LessonCategory.STRATEGY,
                original_input="input",
                distilled_by_model="test-model",
            ),
        )
        mock_db.get_random_lesson.return_value = lesson

        context = MagicMock()
        context.bot.send_message = AsyncMock(side_effect=RuntimeError("telegram error"))
        context.job = MagicMock()
        context.job.chat_id = mock_config.memory_palace_telegram_user_id

        bot = MemoryPalaceBot()
        with pytest.raises(RuntimeError, match="telegram error"):
            await bot._send_random_memory_reminder(context)

        mock_db.mark_as_shown.assert_not_called()
