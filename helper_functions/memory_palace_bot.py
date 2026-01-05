"""
Memory Palace Telegram Bot

A Telegram bot for adding, searching, and managing lessons in the Memory Palace.

Features:
- Natural language lesson input
- LLM-powered distillation into single-line insights
- Approve/Edit/Reject confirmation workflow
- Category suggestion with user override
- Duplicate detection with warning
- Search and random lesson retrieval
- Access control (single user ID)

Usage:
    # Discover your Telegram user ID
    python -m helper_functions.memory_palace_bot --discover

    # Run the bot (foreground)
    python -m helper_functions.memory_palace_bot

    # Run in background
    nohup python -m helper_functions.memory_palace_bot > mp_bot.log 2>&1 &
"""

import argparse
import asyncio
import logging
from enum import IntEnum, auto
from functools import wraps
from typing import Callable, TypeVar, ParamSpec

from telegram import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Update,
)
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    ConversationHandler,
    MessageHandler,
    filters,
)

from config import config
from helper_functions.memory_palace_db import (
    CATEGORIES,
    Lesson,
    LessonCategory,
    LessonMetadata,
    MemoryPalaceDB,
    distill_lesson,
)

logger = logging.getLogger(__name__)

# Type hints for decorator
P = ParamSpec("P")
R = TypeVar("R")


class State(IntEnum):
    """Conversation states for the bot."""

    AWAITING_LESSON = auto()
    CONFIRMING_DISTILLED = auto()
    CONFIRMING_CATEGORY = auto()
    EDITING_LESSON = auto()
    CONFIRMING_DUPLICATE = auto()


def authorized_only(func: Callable[P, R]) -> Callable[P, R]:
    """Decorator to restrict access to authorized user only.

    Works with both standalone functions and instance methods by using
    duck typing to detect whether the first argument is an Update object.
    """

    @wraps(func)
    async def wrapper(*args, **kwargs):
        # Handle both methods (self, update, context) and functions (update, context)
        # by checking if first arg has 'effective_user' attribute (Update objects do)
        if args and hasattr(args[0], 'effective_user'):
            # Standalone function: (update, context, ...)
            update = args[0]
            remaining_args = args
        else:
            # Instance method: (self, update, context, ...)
            update = args[1] if len(args) > 1 else kwargs.get('update')
            remaining_args = args

        user_id = update.effective_user.id
        authorized_id = config.memory_palace_telegram_user_id

        if authorized_id is None:
            # Discovery mode - accept all users and show their ID
            await update.message.reply_text(
                f"Your Telegram User ID: `{user_id}`\n\n"
                "Add this to your config.yml under memory_palace.telegram_user_id "
                "to enable access control.",
                parse_mode="Markdown"
            )
            return ConversationHandler.END

        if user_id != authorized_id:
            logger.warning(f"Unauthorized access attempt from user {user_id}")
            await update.message.reply_text("You are not authorized to use this bot.")
            return ConversationHandler.END

        return await func(*remaining_args, **kwargs)

    return wrapper


class MemoryPalaceBot:
    """Telegram bot for Memory Palace interactions."""

    def __init__(self):
        """Initialize the bot."""
        self.db = MemoryPalaceDB()
        self.token = config.telegram_bot_token

        if not self.token:
            raise ValueError("telegram_bot_token not set in config.yml")

    def _get_confirmation_keyboard(self) -> InlineKeyboardMarkup:
        """Get keyboard for distillation confirmation."""
        keyboard = [
            [
                InlineKeyboardButton("✅ Approve", callback_data="approve"),
                InlineKeyboardButton("✏️ Edit", callback_data="edit"),
            ],
            [InlineKeyboardButton("❌ Reject", callback_data="reject")],
        ]
        return InlineKeyboardMarkup(keyboard)

    def _get_category_keyboard(self) -> InlineKeyboardMarkup:
        """Get keyboard for category selection."""
        keyboard = []
        row = []

        for cat_value, cat_meta in CATEGORIES.items():
            btn = InlineKeyboardButton(
                cat_meta["display"],
                callback_data=f"cat_{cat_value}"
            )
            row.append(btn)

            if len(row) == 2:  # 2 buttons per row
                keyboard.append(row)
                row = []

        if row:  # Add remaining buttons
            keyboard.append(row)

        return InlineKeyboardMarkup(keyboard)

    def _get_duplicate_keyboard(self) -> InlineKeyboardMarkup:
        """Get keyboard for duplicate confirmation."""
        keyboard = [
            [
                InlineKeyboardButton("✅ Add Anyway", callback_data="add_anyway"),
                InlineKeyboardButton("❌ Cancel", callback_data="cancel_dup"),
            ],
        ]
        return InlineKeyboardMarkup(keyboard)

    @authorized_only
    async def start_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> int:
        """Handle /start command."""
        stats = self.db.get_category_stats()
        total = sum(stats.values())

        await update.message.reply_text(
            f"Welcome to Memory Palace!\n\n"
            f"You have {total} lessons stored.\n\n"
            f"Commands:\n"
            f"/add - Add a new lesson\n"
            f"/search <query> - Search lessons\n"
            f"/random - Get a random lesson\n"
            f"/stats - View statistics\n"
            f"/cancel - Cancel current operation\n\n"
            f"Or just send me some text to distill into a lesson!"
        )
        return State.AWAITING_LESSON

    @authorized_only
    async def add_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> int:
        """Handle /add command."""
        await update.message.reply_text(
            "Send me the text you want to distill into a lesson.\n\n"
            "This can be a long paragraph, an observation, or any insight you want to remember."
        )
        return State.AWAITING_LESSON

    @authorized_only
    async def search_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> int:
        """Handle /search command."""
        query = " ".join(context.args) if context.args else ""

        if not query:
            await update.message.reply_text(
                "Usage: /search <query>\n\n"
                "Example: /search game theory strategy"
            )
            return State.AWAITING_LESSON

        results = self.db.find_similar(query, top_k=5)

        if not results:
            await update.message.reply_text("No lessons found matching your query.")
            return State.AWAITING_LESSON

        response = f"Found {len(results)} lessons:\n\n"
        for i, result in enumerate(results, 1):
            cat_display = CATEGORIES.get(
                result.lesson.metadata.category.value, {}
            ).get("display", result.lesson.metadata.category.value)
            response += (
                f"{i}. [{cat_display}]\n"
                f"   {result.lesson.distilled_text}\n"
                f"   (similarity: {result.similarity_score:.2f})\n\n"
            )

        await update.message.reply_text(response)
        return State.AWAITING_LESSON

    @authorized_only
    async def random_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> int:
        """Handle /random command."""
        lesson = self.db.get_random_lesson(exclude_recent=False)

        if not lesson:
            await update.message.reply_text("No lessons in the database yet!")
            return State.AWAITING_LESSON

        cat_display = CATEGORIES.get(
            lesson.metadata.category.value, {}
        ).get("display", lesson.metadata.category.value)

        await update.message.reply_text(
            f"📚 Random Lesson\n\n"
            f"Category: {cat_display}\n\n"
            f"{lesson.distilled_text}"
        )
        return State.AWAITING_LESSON

    @authorized_only
    async def stats_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> int:
        """Handle /stats command."""
        stats = self.db.get_category_stats()
        total = sum(stats.values())

        response = f"📊 Memory Palace Statistics\n\n"
        response += f"Total lessons: {total}\n\n"
        response += "By category:\n"

        for cat_value, count in sorted(stats.items(), key=lambda x: -x[1]):
            cat_display = CATEGORIES.get(cat_value, {}).get("display", cat_value)
            pct = (count / total * 100) if total > 0 else 0
            response += f"  • {cat_display}: {count} ({pct:.1f}%)\n"

        await update.message.reply_text(response)
        return State.AWAITING_LESSON

    async def cancel_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> int:
        """Handle /cancel command."""
        context.user_data.clear()
        await update.message.reply_text(
            "Operation cancelled. Send /add to start again."
        )
        return ConversationHandler.END

    @authorized_only
    async def receive_lesson_text(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> int:
        """Handle incoming lesson text for distillation."""
        text = update.message.text
        context.user_data["original_input"] = text

        await update.message.reply_text("Distilling your insight... ✨")

        # Get few-shot examples from database
        few_shot_examples = self.db.get_few_shot_examples()

        # Distill the lesson
        try:
            result = distill_lesson(text, few_shot_examples=few_shot_examples)
        except Exception as e:
            logger.error(f"Distillation failed: {e}")
            await update.message.reply_text(
                f"Failed to distill: {e}\n\nPlease try again or rephrase."
            )
            return State.AWAITING_LESSON

        context.user_data["distilled_text"] = result.distilled_text
        context.user_data["suggested_category"] = result.suggested_category
        context.user_data["suggested_tags"] = result.suggested_tags

        # Check for duplicate
        duplicate = self.db.check_duplicate(result.distilled_text)
        if duplicate:
            context.user_data["duplicate_match"] = duplicate

            await update.message.reply_text(
                f"⚠️ Similar lesson found!\n\n"
                f"Existing:\n{duplicate.lesson.distilled_text}\n\n"
                f"Similarity: {duplicate.similarity_score:.2f}\n\n"
                f"New:\n{result.distilled_text}\n\n"
                f"Do you want to add anyway?",
                reply_markup=self._get_duplicate_keyboard()
            )
            return State.CONFIRMING_DUPLICATE

        # Show distilled result for confirmation
        cat_display = CATEGORIES.get(
            result.suggested_category, {}
        ).get("display", result.suggested_category)

        tags_str = ", ".join(result.suggested_tags) if result.suggested_tags else "none"

        await update.message.reply_text(
            f"📝 Distilled Lesson:\n\n"
            f"{result.distilled_text}\n\n"
            f"Suggested category: {cat_display}\n"
            f"Suggested tags: {tags_str}\n\n"
            f"Is this correct?",
            reply_markup=self._get_confirmation_keyboard()
        )
        return State.CONFIRMING_DISTILLED

    async def handle_duplicate_response(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> int:
        """Handle duplicate confirmation response."""
        query = update.callback_query
        await query.answer()

        if query.data == "cancel_dup":
            context.user_data.clear()
            await query.edit_message_text("Cancelled. Send /add to try again.")
            return ConversationHandler.END

        # User chose to add anyway
        cat_display = CATEGORIES.get(
            context.user_data["suggested_category"], {}
        ).get("display", context.user_data["suggested_category"])

        tags_str = ", ".join(context.user_data.get("suggested_tags", [])) or "none"

        await query.edit_message_text(
            f"📝 Distilled Lesson:\n\n"
            f"{context.user_data['distilled_text']}\n\n"
            f"Suggested category: {cat_display}\n"
            f"Suggested tags: {tags_str}\n\n"
            f"Is this correct?",
            reply_markup=self._get_confirmation_keyboard()
        )
        return State.CONFIRMING_DISTILLED

    async def handle_distillation_response(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> int:
        """Handle confirmation of distilled lesson."""
        query = update.callback_query
        await query.answer()

        if query.data == "reject":
            context.user_data.clear()
            await query.edit_message_text("Rejected. Send /add to try again.")
            return ConversationHandler.END

        if query.data == "edit":
            await query.edit_message_text(
                "Please provide feedback or a corrected version.\n\n"
                "I'll re-distill based on your input."
            )
            return State.EDITING_LESSON

        # Approved - proceed to category selection
        await query.edit_message_text(
            f"Great! Select a category for this lesson:\n\n"
            f"{context.user_data['distilled_text']}",
            reply_markup=self._get_category_keyboard()
        )
        return State.CONFIRMING_CATEGORY

    async def handle_category_selection(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> int:
        """Handle category selection."""
        query = update.callback_query
        await query.answer()

        category = query.data.replace("cat_", "")

        # Create and save the lesson
        try:
            lesson = Lesson(
                distilled_text=context.user_data["distilled_text"],
                metadata=LessonMetadata(
                    category=LessonCategory(category),
                    source="telegram",
                    original_input=context.user_data["original_input"],
                    distilled_by_model=config.memory_palace_primary_model or "unknown",
                    tags=context.user_data.get("suggested_tags", []),
                )
            )
            lesson_id = self.db.add_lesson(lesson)

            cat_display = CATEGORIES.get(category, {}).get("display", category)
            await query.edit_message_text(
                f"✅ Lesson saved!\n\n"
                f"Category: {cat_display}\n"
                f"ID: {lesson_id[:8]}...\n\n"
                f"Send /add to add another lesson."
            )
        except Exception as e:
            logger.error(f"Failed to save lesson: {e}")
            await query.edit_message_text(
                f"Failed to save: {e}\n\nPlease try again with /add"
            )

        context.user_data.clear()
        return ConversationHandler.END

    @authorized_only
    async def handle_edit_feedback(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> int:
        """Handle edit feedback and re-distill."""
        feedback = update.message.text
        original = context.user_data.get("original_input", "")

        await update.message.reply_text("Re-distilling with your feedback... ✨")

        # Combine original input with feedback for re-distillation
        combined = f"Original: {original}\n\nUser feedback/correction: {feedback}"

        try:
            few_shot_examples = self.db.get_few_shot_examples()
            result = distill_lesson(combined, few_shot_examples=few_shot_examples)
        except Exception as e:
            logger.error(f"Re-distillation failed: {e}")
            await update.message.reply_text(
                f"Failed: {e}\n\nPlease try again or send /cancel"
            )
            return State.EDITING_LESSON

        context.user_data["distilled_text"] = result.distilled_text
        context.user_data["suggested_category"] = result.suggested_category
        context.user_data["suggested_tags"] = result.suggested_tags

        cat_display = CATEGORIES.get(
            result.suggested_category, {}
        ).get("display", result.suggested_category)

        tags_str = ", ".join(result.suggested_tags) if result.suggested_tags else "none"

        await update.message.reply_text(
            f"📝 Updated Lesson:\n\n"
            f"{result.distilled_text}\n\n"
            f"Suggested category: {cat_display}\n"
            f"Suggested tags: {tags_str}\n\n"
            f"Is this correct?",
            reply_markup=self._get_confirmation_keyboard()
        )
        return State.CONFIRMING_DISTILLED

    def build_application(self) -> Application:
        """Build the Telegram application with handlers."""
        application = Application.builder().token(self.token).build()

        # Conversation handler for lesson adding workflow
        conv_handler = ConversationHandler(
            entry_points=[
                CommandHandler("start", self.start_command),
                CommandHandler("add", self.add_command),
                MessageHandler(
                    filters.TEXT & ~filters.COMMAND,
                    self.receive_lesson_text
                ),
            ],
            states={
                State.AWAITING_LESSON: [
                    MessageHandler(
                        filters.TEXT & ~filters.COMMAND,
                        self.receive_lesson_text
                    ),
                ],
                State.CONFIRMING_DUPLICATE: [
                    CallbackQueryHandler(
                        self.handle_duplicate_response,
                        pattern="^(add_anyway|cancel_dup)$"
                    ),
                ],
                State.CONFIRMING_DISTILLED: [
                    CallbackQueryHandler(
                        self.handle_distillation_response,
                        pattern="^(approve|edit|reject)$"
                    ),
                ],
                State.CONFIRMING_CATEGORY: [
                    CallbackQueryHandler(
                        self.handle_category_selection,
                        pattern="^cat_"
                    ),
                ],
                State.EDITING_LESSON: [
                    MessageHandler(
                        filters.TEXT & ~filters.COMMAND,
                        self.handle_edit_feedback
                    ),
                ],
            },
            fallbacks=[
                CommandHandler("cancel", self.cancel_command),
            ],
            allow_reentry=True,
        )

        application.add_handler(conv_handler)

        # Standalone command handlers (outside conversation)
        application.add_handler(CommandHandler("search", self.search_command))
        application.add_handler(CommandHandler("random", self.random_command))
        application.add_handler(CommandHandler("stats", self.stats_command))

        return application

    def run(self) -> None:
        """Run the bot using long-polling."""
        application = self.build_application()
        logger.info("Starting Memory Palace bot (long-polling mode)...")
        application.run_polling(allowed_updates=Update.ALL_TYPES)


def main():
    """CLI entry point for the bot."""
    parser = argparse.ArgumentParser(
        description="Memory Palace Telegram Bot"
    )
    parser.add_argument(
        "--discover",
        action="store_true",
        help="Run in discovery mode to find your Telegram user ID"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    if args.discover:
        print("\n=== Discovery Mode ===")
        print("Bot will reply with your Telegram user ID when you message it.")
        print("After finding your ID, add it to config.yml:")
        print("  memory_palace:")
        print("    telegram_user_id: <your_id>")
        print("\nPress Ctrl+C to stop.\n")

    try:
        bot = MemoryPalaceBot()
        bot.run()
    except KeyboardInterrupt:
        print("\nBot stopped.")
    except Exception as e:
        logger.exception("Bot failed")
        print(f"\nBot failed: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
