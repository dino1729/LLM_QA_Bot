"""
Memory Palace Telegram Bot

A Telegram bot for adding, searching, and managing lessons in the Memory Palace.

Features:
- Natural language intent detection (NL agent routes to appropriate action)
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
import json
import logging
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import IntEnum, StrEnum, auto
from functools import wraps
from typing import Callable, Dict, List, Optional, TypeVar, ParamSpec

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
from helper_functions.llm_client import get_client
from helper_functions.memory_palace_db import (
    CATEGORIES,
    Lesson,
    LessonCategory,
    LessonMetadata,
    MemoryPalaceDB,
    distill_lesson,
)
from helper_functions.web_knowledge_db import (
    ConfidenceTier,
    WebKnowledge,
    WebKnowledgeDB,
    WebKnowledgeMetadata,
    calculate_confidence_tier,
)
from helper_functions.memory_palace_answer import (
    AnswerEngine,
    AnswerResult,
    SourceType,
    format_answer_for_telegram,
)

logger = logging.getLogger(__name__)


class UserIntent(StrEnum):
    """Possible user intents detected from natural language."""

    ADD_LESSON = "add_lesson"
    ANSWER_QUESTION = "answer_question"  # NEW: Ask EDITH a question
    GET_RANDOM = "get_random"
    SEARCH = "search"  # Explicit search request
    GET_STATS = "get_stats"
    HELP = "help"
    FORGET = "forget"  # NEW: Forget specific knowledge


@dataclass
class IntentResult:
    """Result of intent detection."""

    intent: UserIntent
    search_query: Optional[str] = None  # Extracted search query if intent is SEARCH
    confidence: float = 1.0


def detect_intent(user_message: str) -> IntentResult:
    """
    Use LLM to detect user intent from natural language message.

    This enables the bot to understand requests like:
    - "What is game theory?" -> ANSWER_QUESTION
    - "Tell me a random lesson" -> GET_RANDOM
    - "Find lessons about X" -> SEARCH
    - "Show me my stats" -> GET_STATS
    - "I learned that X today" -> ADD_LESSON
    - "Forget everything about X" -> FORGET

    Args:
        user_message: The user's raw message text

    Returns:
        IntentResult with detected intent and optional search query
    """
    provider = config.memory_palace_provider
    model_name = config.memory_palace_primary_model

    client = get_client(provider=provider, model_name=model_name)

    prompt = f"""You are an intent classifier for "EDITH", a personal knowledge assistant.

Classify the user's message into ONE of these intents:
- answer_question: User is ASKING A QUESTION to get an answer (What is X? How does Y work? Explain Z)
- add_lesson: User wants to SAVE a new insight, lesson, or wisdom they learned
- get_random: User wants to RETRIEVE a random lesson from their collection
- search: User wants to explicitly SEARCH/LIST lessons (Find lessons about X, Show me what I know about Y)
- get_stats: User wants to see STATISTICS about their lessons
- help: User is asking for HELP or how to use the bot
- forget: User wants to DELETE/FORGET specific knowledge (Forget about X, Delete X)

KEY DISTINCTIONS:
- Questions ending in "?" that ask for information = answer_question
- "What is X?" / "How does Y work?" / "Explain Z" = answer_question (ASKING)
- "Find lessons about X" / "Show me what I know about Y" = search (LISTING)
- "Tell me a lesson" / "Give me something I learned" = get_random (RANDOM RETRIEVAL)
- "I learned that X" / "Here's an insight: X" / "Save this: X" = add_lesson (SAVING)
- "Forget about X" / "Delete lessons about Y" = forget (DELETING)

Respond with JSON only:
{{"intent": "<intent>", "search_query": "<query if search/answer_question/forget, else null>"}}

User message: {user_message}"""

    try:
        response = client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )

        # Parse JSON response
        content = response.strip()
        # Handle markdown code blocks if present
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        content = content.strip()

        result = json.loads(content)
        intent_str = result.get("intent", "add_lesson").lower()
        search_query = result.get("search_query")

        # Map to enum
        intent_map = {
            "add_lesson": UserIntent.ADD_LESSON,
            "answer_question": UserIntent.ANSWER_QUESTION,
            "get_random": UserIntent.GET_RANDOM,
            "search": UserIntent.SEARCH,
            "get_stats": UserIntent.GET_STATS,
            "help": UserIntent.HELP,
            "forget": UserIntent.FORGET,
        }

        intent = intent_map.get(intent_str, UserIntent.ADD_LESSON)
        return IntentResult(intent=intent, search_query=search_query)

    except Exception as e:
        logger.warning(f"Intent detection failed: {e}, defaulting to ANSWER_QUESTION")
        return IntentResult(intent=UserIntent.ANSWER_QUESTION)

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
    # New states for EDITH
    AWAITING_RESEARCH_CONFIRM = auto()  # User decides whether to research
    RESEARCHING = auto()                 # Progress updates during research
    RESOLVING_CONFLICT = auto()          # Memory vs web conflict
    AWAITING_SOCRATIC = auto()           # After Socratic question
    CONFIRMING_FORGET = auto()           # Confirm before forgetting


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
                f"Your Telegram User ID: {user_id}\n\n"
                "Add this to your config.yml under memory_palace.telegram_user_id "
                "to enable access control."
            )
            return ConversationHandler.END

        if user_id != authorized_id:
            logger.warning(f"Unauthorized access attempt from user {user_id}")
            await update.message.reply_text("You are not authorized to use this bot.")
            return ConversationHandler.END

        return await func(*remaining_args, **kwargs)

    return wrapper


class MemoryPalaceBot:
    """Telegram bot for Memory Palace interactions - powered by EDITH."""

    MAX_CONTEXT_TURNS = 10  # Session context window

    def __init__(self):
        """Initialize the bot with dual knowledge stores and answer engine."""
        # Core databases
        self.db = MemoryPalaceDB()  # User wisdom
        self.web_knowledge_db = WebKnowledgeDB()  # Web-learned knowledge

        # Answer engine for question handling
        self.answer_engine = AnswerEngine(
            wisdom_db=self.db,
            knowledge_db=self.web_knowledge_db,
        )

        # Session context tracking (in-memory, lost on restart)
        self.session_contexts: Dict[int, List[dict]] = {}

        # Telegram token
        self.token = config.telegram_bot_token

        if not self.token:
            raise ValueError("telegram_bot_token not set in config.yml")

    def _add_to_context(self, user_id: int, role: str, content: str):
        """Add a message to the session context."""
        if user_id not in self.session_contexts:
            self.session_contexts[user_id] = []

        self.session_contexts[user_id].append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
        })

        # Keep only last N turns (N*2 messages for user+assistant)
        max_messages = self.MAX_CONTEXT_TURNS * 2
        if len(self.session_contexts[user_id]) > max_messages:
            self.session_contexts[user_id] = self.session_contexts[user_id][-max_messages:]

    def _get_context(self, user_id: int) -> List[dict]:
        """Get the session context for a user."""
        return self.session_contexts.get(user_id, [])

    def _clear_context(self, user_id: int):
        """Clear session context for a user."""
        if user_id in self.session_contexts:
            del self.session_contexts[user_id]

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

    def _get_research_keyboard(self) -> InlineKeyboardMarkup:
        """Get keyboard for research confirmation."""
        keyboard = [
            [
                InlineKeyboardButton("🔍 Research more", callback_data="research_yes"),
                InlineKeyboardButton("📚 Answer from memory", callback_data="research_no"),
            ],
        ]
        return InlineKeyboardMarkup(keyboard)

    def _get_conflict_keyboard(self) -> InlineKeyboardMarkup:
        """Get keyboard for conflict resolution."""
        keyboard = [
            [
                InlineKeyboardButton("Personal preference", callback_data="conflict_keep"),
                InlineKeyboardButton("Outdated fact", callback_data="conflict_update"),
            ],
        ]
        return InlineKeyboardMarkup(keyboard)

    def _get_forget_keyboard(self) -> InlineKeyboardMarkup:
        """Get keyboard for forget confirmation."""
        keyboard = [
            [
                InlineKeyboardButton("Yes, forget", callback_data="forget_yes"),
                InlineKeyboardButton("No, keep", callback_data="forget_no"),
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
        """Handle incoming text with intent detection.

        Routes to appropriate handler based on detected intent:
        - ANSWER_QUESTION: Ask EDITH a question
        - ADD_LESSON: Distill and save a new lesson
        - GET_RANDOM: Return a random lesson
        - SEARCH: Search lessons by query
        - GET_STATS: Show statistics
        - HELP: Show help message
        - FORGET: Forget specific knowledge
        """
        text = update.message.text
        user_id = update.effective_user.id

        # Add user message to session context
        self._add_to_context(user_id, "user", text)

        # Detect user intent
        logger.info(f"Detecting intent for: {text[:50]}...")
        intent_result = detect_intent(text)
        logger.info(f"Detected intent: {intent_result.intent}")

        # Route based on intent
        if intent_result.intent == UserIntent.ANSWER_QUESTION:
            query = intent_result.search_query or text
            return await self._handle_answer_question(update, context, query)

        elif intent_result.intent == UserIntent.GET_RANDOM:
            return await self._handle_get_random(update, context)

        elif intent_result.intent == UserIntent.SEARCH:
            query = intent_result.search_query or text
            return await self._handle_search(update, context, query)

        elif intent_result.intent == UserIntent.GET_STATS:
            return await self._handle_stats(update, context)

        elif intent_result.intent == UserIntent.HELP:
            return await self._handle_help(update, context)

        elif intent_result.intent == UserIntent.FORGET:
            query = intent_result.search_query or text
            return await self._handle_forget(update, context, query)

        # Default: ADD_LESSON - distill and save
        return await self._handle_add_lesson(update, context, text)

    async def _handle_get_random(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> int:
        """Handle get_random intent."""
        lesson = self.db.get_random_lesson(exclude_recent=False)

        if not lesson:
            await update.message.reply_text("No lessons in your Memory Palace yet!")
            return State.AWAITING_LESSON

        cat_display = CATEGORIES.get(
            lesson.metadata.category.value, {}
        ).get("display", lesson.metadata.category.value)

        await update.message.reply_text(
            f"📚 Here's a lesson from your Memory Palace:\n\n"
            f"[{cat_display}]\n"
            f"{lesson.distilled_text}"
        )
        return State.AWAITING_LESSON

    async def _handle_search(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE, query: str
    ) -> int:
        """Handle search intent."""
        results = self.db.find_similar(query, top_k=5)

        if not results:
            await update.message.reply_text(
                f"No lessons found matching: {query}"
            )
            return State.AWAITING_LESSON

        response = f"🔍 Found {len(results)} lessons:\n\n"
        for i, result in enumerate(results, 1):
            cat_display = CATEGORIES.get(
                result.lesson.metadata.category.value, {}
            ).get("display", result.lesson.metadata.category.value)
            response += (
                f"{i}. [{cat_display}]\n"
                f"   {result.lesson.distilled_text}\n\n"
            )

        await update.message.reply_text(response)
        return State.AWAITING_LESSON

    async def _handle_stats(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> int:
        """Handle get_stats intent."""
        stats = self.db.get_category_stats()
        total = sum(stats.values())

        response = f"📊 Memory Palace Statistics\n\n"
        response += f"Total lessons: {total}\n\n"
        response += "By category:\n"

        for cat_value, count in sorted(stats.items(), key=lambda x: -x[1]):
            cat_display = CATEGORIES.get(cat_value, {}).get("display", cat_value)
            pct = (count / total * 100) if total > 0 else 0
            response += f"  {cat_display}: {count} ({pct:.1f}%)\n"

        await update.message.reply_text(response)
        return State.AWAITING_LESSON

    async def _handle_help(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> int:
        """Handle help intent."""
        await update.message.reply_text(
            "🏛️ EDITH - Your Knowledge Assistant\n\n"
            "I understand natural language! Try:\n\n"
            "❓ To ask a question:\n"
            '   "What is game theory?"\n'
            '   "How does compound interest work?"\n\n'
            "📥 To add a lesson:\n"
            '   "I learned that X today"\n'
            '   "Here\'s an insight: X"\n\n'
            "🎲 To get a random lesson:\n"
            '   "Tell me something I learned"\n\n'
            "🔍 To search:\n"
            '   "Find lessons about Y"\n\n'
            "🗑️ To forget:\n"
            '   "Forget everything about X"\n\n'
            "Or use commands:\n"
            "/add - Add a lesson\n"
            "/random - Random lesson\n"
            "/search <query> - Search\n"
            "/stats - Statistics"
        )
        return State.AWAITING_LESSON

    async def _handle_answer_question(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE, question: str
    ) -> int:
        """Handle ANSWER_QUESTION intent - ask EDITH a question."""
        user_id = update.effective_user.id
        session_context = self._get_context(user_id)

        # Get answer from the engine
        result = self.answer_engine.answer(
            question=question,
            session_context=session_context,
        )

        # Store result for potential research follow-up
        context.user_data["last_question"] = question
        context.user_data["last_answer_result"] = result

        # Format and send response
        if result.source_type == SourceType.NONE:
            # No knowledge found - offer to research
            msg = (
                f"{result.reasoning_prefix}\n\n"
                "I don't have any knowledge about this topic."
            )
            if result.related_topics:
                msg += "\n\nRelated topics I know about:\n"
                for topic in result.related_topics:
                    msg += f"  - {topic}\n"
            msg += "\n\nWould you like me to research this?"

            await update.message.reply_text(
                msg,
                reply_markup=self._get_research_keyboard()
            )
            return State.AWAITING_RESEARCH_CONFIRM

        # We have an answer
        response = format_answer_for_telegram(result)

        # Add EDITH response to context
        self._add_to_context(user_id, "assistant", result.answer_text)

        # Check if we should offer to research more (partial match)
        if result.offer_to_research:
            response += "\n\nWould you like me to research this further?"
            await update.message.reply_text(
                response,
                reply_markup=self._get_research_keyboard()
            )
            return State.AWAITING_RESEARCH_CONFIRM

        # 50% chance of Socratic follow-up for good matches
        if result.confidence_tier == ConfidenceTier.VERY_CONFIDENT and random.random() < 0.5:
            response += "\n\nDoes this align with what you expected?"
            context.user_data["awaiting_socratic"] = True

        await update.message.reply_text(response)
        return State.AWAITING_LESSON

    async def _handle_forget(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE, query: str
    ) -> int:
        """Handle FORGET intent - soft delete knowledge."""
        # Find matching lessons
        similar = self.db.find_similar(query, top_k=5, include_forgotten=False)

        if not similar:
            await update.message.reply_text(
                "I couldn't find any lessons matching that topic."
            )
            return State.AWAITING_LESSON

        # Show what will be forgotten
        context.user_data["forget_query"] = query
        context.user_data["forget_matches"] = [s.lesson.id for s in similar if s.similarity_score >= 0.7]

        matching = [s for s in similar if s.similarity_score >= 0.7]
        if not matching:
            await update.message.reply_text(
                f"I found some related lessons, but none are close enough matches:\n\n"
                + "\n".join([f"  - {s.lesson.distilled_text[:60]}..." for s in similar[:3]])
                + "\n\nBe more specific about what to forget."
            )
            return State.AWAITING_LESSON

        msg = f"I found {len(matching)} lessons to forget:\n\n"
        for s in matching[:5]:
            msg += f"  - {s.lesson.distilled_text[:80]}...\n"
        msg += "\nAre you sure you want to forget these?"

        await update.message.reply_text(
            msg,
            reply_markup=self._get_forget_keyboard()
        )
        return State.CONFIRMING_FORGET

    async def handle_research_confirmation(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> int:
        """Handle research confirmation response."""
        query = update.callback_query
        await query.answer()

        if query.data == "research_no":
            # User doesn't want research - just acknowledge
            last_result = context.user_data.get("last_answer_result")
            if last_result and last_result.answer_text:
                await query.edit_message_text(
                    format_answer_for_telegram(last_result)
                )
            else:
                await query.edit_message_text("Understood. Let me know if you have other questions!")
            return State.AWAITING_LESSON

        # User wants research
        question = context.user_data.get("last_question", "")
        if not question:
            await query.edit_message_text("I've lost track of the question. Please ask again.")
            return State.AWAITING_LESSON

        await query.edit_message_text("🔍 Researching: Searching for sources...")

        # Do the research (this will be async in Phase 4)
        try:
            from helper_functions.firecrawl_researcher import conduct_research_firecrawl

            # Update progress
            await query.edit_message_text("🔍 Researching: Scraping sources...")

            report = conduct_research_firecrawl(
                query=question,
                provider=config.memory_palace_provider,
                max_sources=8,
            )

            await query.edit_message_text("🔍 Researching: Synthesizing answer...")

            # Distill the research into a single-line insight
            distilled = distill_lesson(report)

            # Calculate confidence based on report quality
            source_count = report.count("Source") if report else 0
            confidence = calculate_confidence_tier(source_count, sources_agree=True)

            # Save to web knowledge store
            knowledge = WebKnowledge(
                distilled_text=distilled.distilled_text,
                metadata=WebKnowledgeMetadata(
                    source_urls=[],  # Would be extracted from report
                    original_query=question,
                    created_at=datetime.now(),
                    expires_at=datetime.now() + timedelta(days=7),
                    confidence_tier=confidence,
                    distilled_by_model=config.memory_palace_primary_model or "unknown",
                    source_count=source_count,
                )
            )
            self.web_knowledge_db.add_knowledge(knowledge)

            # Check for conflict with stored user wisdom
            wisdom_matches = self.db.find_similar(question, top_k=1)

            if wisdom_matches and wisdom_matches[0].similarity_score >= 0.6:
                # Check if the research conflicts with stored wisdom
                best_wisdom = wisdom_matches[0]
                has_conflict = self.answer_engine.check_for_conflict(
                    best_wisdom, distilled.distilled_text
                )

                if has_conflict:
                    # Store conflict context for resolution
                    context.user_data["conflict_wisdom"] = best_wisdom
                    context.user_data["conflict_new_info"] = distilled.distilled_text
                    context.user_data["conflict_knowledge_id"] = knowledge.id

                    conflict_msg = (
                        "I found a potential conflict:\n\n"
                        f"**Your stored wisdom:**\n{best_wisdom.lesson.distilled_text}\n\n"
                        f"**New research:**\n{distilled.distilled_text}\n\n"
                        "Is your stored wisdom a personal preference (keep it) "
                        "or an outdated fact (update it)?"
                    )
                    await query.edit_message_text(
                        conflict_msg,
                        reply_markup=self._get_conflict_keyboard()
                    )
                    return State.RESOLVING_CONFLICT

            # Format the teaching response
            confidence_display = {
                ConfidenceTier.VERY_CONFIDENT: "Very confident",
                ConfidenceTier.FAIRLY_SURE: "Fairly sure",
                ConfidenceTier.UNCERTAIN: "Uncertain",
            }

            response = (
                f"EDITH learned something new!\n\n"
                f"[Researched: {confidence_display[confidence]}]\n\n"
                f"**Key Insight:**\n{distilled.distilled_text}\n\n"
            )

            # Add supporting context from the report (truncated)
            if len(report) > 500:
                response += f"**Summary:**\n{report[:400]}...\n\n"

            # 50% chance of Socratic follow-up
            if random.random() < 0.5:
                response += "Does this align with what you expected?"
                context.user_data["awaiting_socratic"] = True

            await query.edit_message_text(response)

        except Exception as e:
            logger.error(f"Research failed: {e}")
            await query.edit_message_text(
                f"Research failed: {e}\n\n"
                "I'll answer from what I know instead."
            )
            # Fallback to memory-only answer
            last_result = context.user_data.get("last_answer_result")
            if last_result and last_result.answer_text:
                await update.effective_message.reply_text(
                    format_answer_for_telegram(last_result)
                )

        return State.AWAITING_LESSON

    async def handle_forget_confirmation(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> int:
        """Handle forget confirmation response."""
        query = update.callback_query
        await query.answer()

        if query.data == "forget_no":
            context.user_data.pop("forget_query", None)
            context.user_data.pop("forget_matches", None)
            await query.edit_message_text("Cancelled. Your lessons are safe.")
            return State.AWAITING_LESSON

        # User confirmed forget
        forget_query = context.user_data.get("forget_query", "")
        count = self.db.forget_by_query(forget_query)

        context.user_data.pop("forget_query", None)
        context.user_data.pop("forget_matches", None)

        await query.edit_message_text(
            f"Done. Marked {count} lessons as forgotten.\n\n"
            "(They can be recovered if needed.)"
        )
        return State.AWAITING_LESSON

    async def handle_socratic_response(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> int:
        """Handle Socratic follow-up response (just acknowledge, don't store)."""
        _ = update.message.text  # Acknowledge but don't store
        context.user_data["awaiting_socratic"] = False

        # Thoughtful acknowledgment
        await update.message.reply_text(
            "That's a great reflection. These connections help solidify understanding."
        )
        return State.AWAITING_LESSON

    async def handle_conflict_resolution(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> int:
        """Handle conflict resolution response."""
        query = update.callback_query
        await query.answer()

        conflict_wisdom = context.user_data.get("conflict_wisdom")
        new_info = context.user_data.get("conflict_new_info")

        if not conflict_wisdom or not new_info:
            await query.edit_message_text(
                "I've lost track of the conflict context. Please continue."
            )
            return State.AWAITING_LESSON

        if query.data == "conflict_keep":
            # User says it's a personal preference - keep existing wisdom
            await query.edit_message_text(
                "Understood! I'll keep your stored wisdom as a personal preference.\n\n"
                f"**Your wisdom:** {conflict_wisdom.lesson.distilled_text}\n\n"
                "The new research has still been saved to my web knowledge store "
                "for reference, but your personal wisdom takes priority."
            )

        elif query.data == "conflict_update":
            # User says it's outdated - soft delete the old wisdom
            if conflict_wisdom.lesson.id:
                self.db._mark_forgotten(conflict_wisdom.lesson.id)
                await query.edit_message_text(
                    "Got it! I've marked your previous wisdom as outdated.\n\n"
                    f"**Old (forgotten):** {conflict_wisdom.lesson.distilled_text}\n\n"
                    f"**Updated:** {new_info}\n\n"
                    "The new research is now your primary reference for this topic."
                )
            else:
                await query.edit_message_text(
                    "I couldn't update the old wisdom, but the new research "
                    f"has been saved:\n\n{new_info}"
                )

        # Clean up conflict context
        context.user_data.pop("conflict_wisdom", None)
        context.user_data.pop("conflict_new_info", None)
        context.user_data.pop("conflict_knowledge_id", None)

        return State.AWAITING_LESSON

    async def _handle_add_lesson(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE, text: str
    ) -> int:
        """Handle add_lesson intent - distill and save."""
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

    async def _send_startup_message(self, application: Application) -> None:
        """Send a welcome message to the authorized user on bot startup."""
        user_id = config.memory_palace_telegram_user_id
        if not user_id:
            logger.info("No telegram_user_id configured, skipping startup message")
            return

        try:
            stats = self.db.get_category_stats()
            total = sum(stats.values())
            web_stats = self.web_knowledge_db.get_stats()
            web_total = web_stats.get("valid", 0)

            await application.bot.send_message(
                chat_id=user_id,
                text=(
                    f"EDITH is online!\n\n"
                    f"Your Memory Palace: {total} lessons\n"
                    f"Web Knowledge: {web_total} facts\n\n"
                    f"Ask me anything or send a lesson to save."
                ),
            )
            logger.info(f"Sent startup message to user {user_id}")
        except Exception as e:
            logger.warning(f"Failed to send startup message: {e}")

    def build_application(self) -> Application:
        """Build the Telegram application with handlers."""
        application = (
            Application.builder()
            .token(self.token)
            .post_init(self._send_startup_message)
            .build()
        )

        # Conversation handler for EDITH interactions
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
                # New states for EDITH
                State.AWAITING_RESEARCH_CONFIRM: [
                    CallbackQueryHandler(
                        self.handle_research_confirmation,
                        pattern="^(research_yes|research_no)$"
                    ),
                ],
                State.CONFIRMING_FORGET: [
                    CallbackQueryHandler(
                        self.handle_forget_confirmation,
                        pattern="^(forget_yes|forget_no)$"
                    ),
                ],
                State.AWAITING_SOCRATIC: [
                    MessageHandler(
                        filters.TEXT & ~filters.COMMAND,
                        self.handle_socratic_response
                    ),
                ],
                State.RESOLVING_CONFLICT: [
                    CallbackQueryHandler(
                        self.handle_conflict_resolution,
                        pattern="^(conflict_keep|conflict_update)$"
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
