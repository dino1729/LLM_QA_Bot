# Memory Palace System - Technical Documentation

## Overview

The Memory Palace is a curated knowledge management system that stores distilled insights from various learning sources (books, articles, podcasts, etc.) and integrates them into the daily newsletter workflow. It uses LlamaIndex VectorStoreIndex for semantic search and retrieval, with a Telegram bot interface (EDITH) for adding lessons, answering questions, and autonomous web research.

**Key Features:**
- LlamaIndex-based vector database for semantic search
- **EDITH Persona** - Telegram bot with autonomous learning capabilities
- **Dual-Store Architecture** - User Wisdom (permanent) + Web Knowledge (7-day TTL)
- **Question Answering** - Answer questions from stored knowledge with confidence tiers
- **Autonomous Web Research** - Firecrawl integration for learning unknown topics
- **Conflict Resolution** - Detect and resolve contradictions between stored wisdom and new research
- LLM-powered distillation of verbose content into single-line insights
- Recency tracking to prevent repeated lessons in newsletters
- 10 consolidated categories with automatic LLM-suggested categorization
- Duplicate detection with configurable similarity threshold
- **Soft Delete** - Mark lessons as forgotten (recoverable)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Memory Palace System (EDITH)                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                         DUAL-STORE ARCHITECTURE                         │ │
│  ├──────────────────────────────┬─────────────────────────────────────────┤ │
│  │      USER WISDOM (Primary)   │     WEB KNOWLEDGE (Secondary)           │ │
│  │   ┌─────────────────────┐    │    ┌─────────────────────┐              │ │
│  │   │   MemoryPalaceDB    │    │    │   WebKnowledgeDB    │              │ │
│  │   │   - Permanent       │    │    │   - 7-day TTL       │              │ │
│  │   │   - User-curated    │    │    │   - Auto-learned    │              │ │
│  │   │   - Soft delete     │    │    │   - Auto-expires    │              │ │
│  │   └─────────────────────┘    │    └─────────────────────┘              │ │
│  └──────────────────────────────┴─────────────────────────────────────────┘ │
│                                    │                                         │
│                                    ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                          ANSWER ENGINE                                   ││
│  │  ┌─────────────┐  ┌─────────────────┐  ┌────────────────────────────┐  ││
│  │  │  Retrieval  │──│ Confidence Calc │──│    Answer Synthesis        │  ││
│  │  │ (wisdom     │  │ - Very confident│  │ - EDITH persona            │  ││
│  │  │  first)     │  │ - Fairly sure   │  │ - Source attribution       │  ││
│  │  │             │  │ - Uncertain     │  │ - Conflict detection       │  ││
│  │  └─────────────┘  └─────────────────┘  └────────────────────────────┘  ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                    │                                         │
│                                    ▼                                         │
│  ┌──────────────────┐    ┌───────────────────┐    ┌──────────────────────┐ │
│  │   Telegram Bot   │    │   Firecrawl       │    │   Session Context    │ │
│  │   (EDITH)        │◀───│   Researcher      │    │   (5-10 turns)       │ │
│  │   - Q&A          │    │   - Web scraping  │    │   - In-memory        │ │
│  │   - Lessons      │    │   - Progress CB   │    │   - Per-user         │ │
│  │   - Forget       │    │   - 5-10 sources  │    │                      │ │
│  └──────────────────┘    └───────────────────┘    └──────────────────────┘ │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                    Newsletter Integration                                ││
│  │  year_progress_and_news_reporter_litellm.py                             ││
│  │  └─▶ get_memory_palace_lesson() ─▶ HTML rendering                       ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## File Structure

```
LLM_QA_Bot/
├── helper_functions/
│   ├── memory_palace_db.py         # User wisdom database (permanent)
│   ├── web_knowledge_db.py         # Web knowledge database (7-day TTL) [NEW]
│   ├── memory_palace_answer.py     # Answer engine with confidence tiers [NEW]
│   ├── memory_palace_bot.py        # Telegram bot (EDITH)
│   ├── memory_palace_migration.py  # One-time migration script
│   ├── firecrawl_researcher.py     # Web research integration
│   └── html_templates.py           # Newsletter HTML (includes MP section)
├── memory_palace/
│   ├── lessons_index/              # User wisdom LlamaIndex folder
│   │   ├── docstore.json           # Document storage
│   │   ├── index_store.json        # Index metadata
│   │   ├── default__vector_store.json  # Vector embeddings
│   │   └── graph_store.json        # Graph relationships
│   ├── web_knowledge_index/        # Web knowledge LlamaIndex folder [NEW]
│   │   └── (same structure)
│   ├── shown_history.json          # Recency tracking
│   └── *.json                      # Legacy lesson files (migrated)
├── config/
│   ├── config.yml                  # Memory Palace configuration
│   └── config.py                   # Config accessors
├── tests/
│   ├── test_memory_palace_db.py        # 39 tests
│   ├── test_memory_palace_bot.py       # 23 tests
│   ├── test_memory_palace_migration.py # 26 tests
│   ├── test_web_knowledge_db.py        # 17 tests [NEW]
│   └── test_memory_palace_answer.py    # 14 tests [NEW]
└── year_progress_and_news_reporter_litellm.py  # Newsletter script
```

---

## Core Module: `memory_palace_db.py`

### Location
`helper_functions/memory_palace_db.py`

### Dependencies
```python
from llama_index.core import VectorStoreIndex, Document, StorageContext
from llama_index.core import load_index_from_storage
from pydantic import BaseModel, Field
from helper_functions.llm_client import get_client
```

### Data Models

#### LessonCategory (Enum)
10 consolidated categories mapped from the original 23:

| Category | Display Name | Keywords |
|----------|--------------|----------|
| `strategy` | Strategy & Game Theory | strategy, negotiation, decision, game theory |
| `psychology` | Psychology & Cognitive Science | psychology, cognitive, bias, behavior |
| `history` | History & Civilization | history, civilization, empire, war |
| `science` | Science & Physics | physics, chemistry, quantum, relativity |
| `technology` | Technology & AI | technology, AI, computing, software |
| `economics` | Economics & Finance | economics, finance, markets, investing |
| `engineering` | Engineering & Systems | engineering, systems, design, architecture |
| `biology` | Biology & Health | biology, health, medicine, genetics |
| `leadership` | Leadership & Growth | leadership, management, growth, career |
| `observations` | Observations & Ideas | miscellaneous, observations, facts |

#### Lesson (Pydantic Model)
```python
class Lesson(BaseModel):
    id: str                      # UUID for the lesson
    distilled_text: str          # Single-line insight (< 200 chars ideal)
    metadata: LessonMetadata     # Category, source, timestamps, etc.

    model_config = {"use_enum_values": True}  # Pydantic v2 syntax
```

#### LessonMetadata (Pydantic Model)
```python
class LessonMetadata(BaseModel):
    category: LessonCategory
    created_at: datetime
    source: str | None = None         # Book title, article URL, etc.
    original_input: str                # Raw user input before distillation
    distilled_by_model: str            # Model used for distillation
    tags: list[str] = []
    # Soft delete support [NEW]
    is_forgotten: bool = False         # Mark lesson as forgotten (recoverable)
    forgotten_at: datetime | None = None  # When it was forgotten
```

### MemoryPalaceDB Class

#### Initialization
```python
db = MemoryPalaceDB(
    index_folder="memory_palace/lessons_index",  # Default from config
    provider="litellm",                          # LLM provider for embeddings
    model_tier="fast"                            # Model tier for embeddings
)
```

**Key Implementation Detail:** The embedding model is obtained via `get_client()` to ensure proper routing through LiteLLM proxy:

```python
def _get_embed_model(self):
    """Get LlamaIndex-compatible embedding model from llm_client."""
    client = get_client(provider=self.provider, model_tier=self.model_tier)
    return client.get_llamaindex_embedding()
```

#### Core Methods

| Method | Description | Returns |
|--------|-------------|---------|
| `add_lesson(text, category, source, tags)` | Add a new lesson to the index | `Lesson` |
| `find_similar(query, top_k)` | Semantic search for similar lessons | `list[SimilarLesson]` |
| `check_duplicate(text, threshold)` | Check if lesson exists (default 0.75) | `SimilarLesson \| None` |
| `get_random_lesson(exclude_recent)` | Get random lesson with recency exclusion | `Lesson \| None` |
| `mark_as_shown(lesson_id)` | Mark lesson as shown for recency tracking | `None` |
| `get_all_lessons()` | Enumerate all lessons from docstore | `list[Lesson]` |
| `get_lessons_by_category(category)` | Filter lessons by category | `list[Lesson]` |
| `delete_lesson(lesson_id)` | Remove lesson from index | `bool` |

#### LlamaIndex Document Structure
Each lesson is stored as a LlamaIndex Document with metadata:

```python
doc = Document(
    doc_id=lesson.id,
    text=lesson.distilled_text,
    metadata={
        "category": lesson.metadata.category.value,
        "created_at": lesson.metadata.created_at.isoformat(),
        "source": lesson.metadata.source,
        "original_input": lesson.metadata.original_input,
        "distilled_by_model": lesson.metadata.distilled_by_model,
        "tags": json.dumps(lesson.metadata.tags),
    }
)
```

### Distillation Function

```python
def distill_lesson(
    raw_input: str,
    provider: str = "litellm",
    model_tier: str = "fast",
    few_shot_examples: list[str] | None = None
) -> LessonDistillationResult
```

**System Prompt Pattern:**
```
You are a knowledge distiller. Your task is to extract the core insight
from the user's input and condense it into a single, memorable sentence.

Rules:
1. Maximum 200 characters
2. Focus on the actionable or surprising insight
3. Remove filler words and redundancy
4. Preserve the essence, not the details

[Few-shot examples from database if provided]
```

**Returns:**
```python
class LessonDistillationResult(BaseModel):
    distilled_text: str           # The condensed insight
    suggested_category: LessonCategory  # LLM-suggested category
    suggested_tags: list[str]     # Extracted keywords
    confidence: float             # 0-1 confidence score
```

---

## Web Knowledge Module: `web_knowledge_db.py` [NEW]

### Location
`helper_functions/web_knowledge_db.py`

### Purpose
Stores auto-learned knowledge from web research with a 7-day TTL. Separate from user wisdom to maintain distinction between:
- **User Wisdom** - Permanent, user-curated lessons
- **Web Knowledge** - Ephemeral, auto-learned, expires after 7 days

### Data Models

#### ConfidenceTier (Enum)
```python
class ConfidenceTier(StrEnum):
    VERY_CONFIDENT = "very_confident"  # 5+ agreeing sources
    FAIRLY_SURE = "fairly_sure"        # 2-4 agreeing sources
    UNCERTAIN = "uncertain"            # 1 source or sources disagree
```

#### WebKnowledgeMetadata (Pydantic Model)
```python
class WebKnowledgeMetadata(BaseModel):
    source_urls: list[str] = []       # URLs where info came from
    original_query: str               # Question that triggered research
    created_at: datetime
    expires_at: datetime              # created_at + 7 days
    confidence_tier: ConfidenceTier = ConfidenceTier.FAIRLY_SURE
    distilled_by_model: str = "unknown"
    source_count: int = 0             # How many sources were used
```

#### WebKnowledge (Pydantic Model)
```python
class WebKnowledge(BaseModel):
    id: str                           # UUID
    distilled_text: str               # Same format as user lessons
    metadata: WebKnowledgeMetadata
```

### WebKnowledgeDB Class

#### Core Methods
| Method | Description | Returns |
|--------|-------------|---------|
| `add_knowledge(knowledge)` | Add web knowledge to index | `str` (ID) |
| `find_similar(text, top_k, include_expired)` | Semantic search | `list[SimilarWebKnowledge]` |
| `is_stale(knowledge_id)` | Check if expired | `bool` |
| `get_expired()` | Get all expired entries | `list[WebKnowledge]` |
| `delete_expired()` | Remove expired entries | `int` (count) |
| `get_stats()` | Get statistics | `dict` |

---

## Answer Engine: `memory_palace_answer.py` [NEW]

### Location
`helper_functions/memory_palace_answer.py`

### Purpose
Synthesizes answers from stored knowledge with confidence tiers and source attribution. Searches user wisdom first (priority), then web knowledge.

### Data Models

#### SourceType (Enum)
```python
class SourceType(StrEnum):
    USER_WISDOM = "user_wisdom"       # Answer from user's lessons
    WEB_KNOWLEDGE = "web_knowledge"   # Answer from web research
    BOTH = "both"                     # Combined sources
    NONE = "none"                     # No matching knowledge
```

#### AnswerResult (Dataclass)
```python
@dataclass
class AnswerResult:
    answer_text: str                  # Synthesized answer
    confidence_tier: ConfidenceTier   # Very confident / Fairly sure / Uncertain
    source_type: SourceType           # Where the answer came from
    related_topics: list[str] = []    # For empty results fallback
    offer_to_research: bool = False   # Should we offer web research?
    reasoning_prefix: str = ""        # E.g., "[From memory: 85% match]"
    wisdom_matches: list[SimilarLesson] = []
    knowledge_matches: list[SimilarWebKnowledge] = []
```

### AnswerEngine Class

#### Confidence Thresholds
```python
HIGH_CONFIDENCE_THRESHOLD = 0.85   # Very confident
MEDIUM_CONFIDENCE_THRESHOLD = 0.6  # Fairly sure
PARTIAL_MATCH_THRESHOLD = 0.4      # Uncertain (offer research)
```

#### Core Methods
| Method | Description | Returns |
|--------|-------------|---------|
| `answer(question, session_context)` | Answer from stored knowledge | `AnswerResult` |
| `check_for_conflict(wisdom, web_info)` | Detect contradictions | `bool` |

#### Confidence Calculation Logic
```
1. Wisdom score >= 0.85 → VERY_CONFIDENT from USER_WISDOM
2. Wisdom 0.6-0.85 + Knowledge 0.6+ → VERY_CONFIDENT from BOTH
3. Wisdom 0.6-0.85 alone → FAIRLY_SURE from USER_WISDOM (offer research)
4. Knowledge >= 0.85 → FAIRLY_SURE from WEB_KNOWLEDGE
5. Knowledge 0.6-0.85 → FAIRLY_SURE from WEB_KNOWLEDGE (offer research)
6. Either 0.4-0.6 → UNCERTAIN (offer research)
7. Neither >= 0.4 → NONE (offer research)
```

### EDITH Persona
```python
EDITH_SYSTEM_PROMPT = """You are EDITH, a personal knowledge assistant for Dinesh.
You help recall lessons from his Memory Palace and research new topics.
Speak in first person ("I remember...", "I found that...").
Be warm but concise. Focus on insights, not fluff.
When synthesizing answers, prioritize Dinesh's personal lessons over generic web knowledge."""
```

---

## Telegram Bot (EDITH): `memory_palace_bot.py`

### Location
`helper_functions/memory_palace_bot.py`

### Dependencies
```python
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    CallbackQueryHandler, ConversationHandler, ContextTypes
)
from helper_functions.memory_palace_answer import AnswerEngine, AnswerResult
from helper_functions.web_knowledge_db import WebKnowledgeDB
```

### Intent Detection [NEW]

EDITH uses LLM-powered intent detection to route messages:

```python
class UserIntent(StrEnum):
    ADD_LESSON = "add_lesson"         # "I learned that...", "Save this:"
    ANSWER_QUESTION = "answer_question"  # "What is X?", "How does Y work?"
    GET_RANDOM = "get_random"         # "Random", "Surprise me"
    SEARCH = "search"                 # "Find lessons about X"
    GET_STATS = "get_stats"           # "Stats", "How many lessons?"
    HELP = "help"                     # "Help", "What can you do?"
    FORGET = "forget"                 # "Forget about X", "Remove Y"
```

### State Machine (Updated)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    EDITH Telegram Bot State Machine                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│    User sends message                                                        │
│              │                                                               │
│              ▼                                                               │
│    ┌─────────────────────────────────────────────────────────────┐          │
│    │               INTENT DETECTION (LLM)                         │          │
│    └──────┬────────┬────────┬────────┬────────┬────────┬────────┘          │
│           │        │        │        │        │        │                    │
│    ADD_LESSON  ANSWER_Q  SEARCH  RANDOM  STATS  FORGET                     │
│           │        │        │        │        │        │                    │
│           ▼        │        │        │        │        │                    │
│    ┌──────────┐    │        │        │        │        │                    │
│    │ Distill  │    │        │        │        │        │                    │
│    │   LLM    │    │        │        │        │        │                    │
│    └────┬─────┘    │        │        │        │        │                    │
│         │          │        │        │        │        │                    │
│         ▼          │        │        │        │        │                    │
│    ┌──────────┐    │        │        │        │        │                    │
│    │CONFIRMING│    │        │        │        │        │                    │
│    │DISTILLED │    │        │        │        │        │                    │
│    └────┬─────┘    │        │        │        │        │                    │
│         │          │        │        │        │        │                    │
│         ▼          ▼        │        │        │        ▼                    │
│    ┌──────────┐ ┌────────┐  │        │        │  ┌──────────┐              │
│    │CONFIRMING│ │ Answer │  │        │        │  │CONFIRMING│              │
│    │ CATEGORY │ │ Engine │  │        │        │  │  FORGET  │              │
│    └────┬─────┘ └────┬───┘  │        │        │  └────┬─────┘              │
│         │            │      │        │        │       │                     │
│         │            ▼      │        │        │       ▼                     │
│         │     ┌───────────┐ │        │        │  ┌──────────┐              │
│         │     │ Partial   │ │        │        │  │Soft Delete│              │
│         │     │ Match?    │ │        │        │  └──────────┘              │
│         │     └─────┬─────┘ │        │        │                             │
│         │           │       │        │        │                             │
│         │      [Research?]  │        │        │                             │
│         │           │       │        │        │                             │
│         │           ▼       │        │        │                             │
│         │     ┌───────────┐ │        │        │                             │
│         │     │RESEARCHING│ │        │        │                             │
│         │     │(Firecrawl)│ │        │        │                             │
│         │     └─────┬─────┘ │        │        │                             │
│         │           │       │        │        │                             │
│         │           ▼       │        │        │                             │
│         │     ┌───────────┐ │        │        │                             │
│         │     │ Conflict? │ │        │        │                             │
│         │     └─────┬─────┘ │        │        │                             │
│         │           │       │        │        │                             │
│         │      [Keep/Update]│        │        │                             │
│         │           │       │        │        │                             │
│         │           ▼       │        │        │                             │
│         │     ┌───────────┐ │        │        │                             │
│         │     │ Socratic? │ │        │        │                             │
│         │     │ (50%)     │ │        │        │                             │
│         │     └───────────┘ │        │        │                             │
│         │                   │        │        │                             │
│         ▼                   ▼        ▼        ▼                             │
│    ┌─────────────────────────────────────────────────────────────┐          │
│    │                    AWAITING_LESSON                           │          │
│    └─────────────────────────────────────────────────────────────┘          │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### User Flow Examples [NEW]

#### Flow 1: Question with Good Memory Match
```
User: What is the Dunning-Kruger effect?
EDITH: [From memory: Very confident] (87% match)

The Dunning-Kruger Effect causes people with low ability to
overestimate their competence.

(From your Psychology lessons)
```

#### Flow 2: Question with Partial Match - Offer Research
```
User: How does quantum entanglement work?
EDITH: [From memory: Uncertain] (42% match)

I have some related knowledge about quantum physics, but nothing
specific on entanglement.

[Research more?] [Answer from what I know]
```

#### Flow 3: Research and Learn
```
User: [taps "Research more?"]
EDITH: Researching: Searching for sources...
EDITH: Researching: Scraping source 3/8...
EDITH: Researching: Synthesizing answer...

EDITH learned something new!

[Researched: Fairly sure]

**Key Insight:**
Quantum entanglement links particles so measuring one instantly
affects the other regardless of distance.

**Summary:**
Einstein called it "spooky action at a distance"...

Does this align with what you expected?
```

#### Flow 4: Conflict Resolution
```
User: What's the best diet for longevity?

EDITH: I found a potential conflict:

**Your stored wisdom:**
Intermittent fasting is the key to longevity.

**New research:**
Recent studies show caloric restriction matters more than timing,
and Mediterranean diet shows strongest longevity benefits.

Is your stored wisdom a personal preference (keep it) or an
outdated fact (update it)?

[Personal preference] [Outdated fact]
```

#### Flow 5: Forget a Lesson
```
User: Forget everything about my old productivity system
EDITH: I found 3 lessons matching "productivity system":

1. GTD contexts improve task switching efficiency
2. Pomodoro technique works better for deep work
3. Weekly reviews prevent task buildup

Forget these 3 lessons? (They can be recovered if needed.)

[Forget] [Cancel]
```

### Commands

| Command | Description |
|---------|-------------|
| `/start` | Welcome message and usage instructions |
| `/add` | Start adding a new lesson |
| `/search <query>` | Semantic search for lessons |
| `/random` | Get a random lesson |
| `/stats` | Show database statistics |
| `/cancel` | Cancel current operation |

### Access Control

The `@authorized_only` decorator restricts access to a single Telegram user ID:

```python
def authorized_only(func):
    """Decorator using duck typing to handle both functions and methods."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        # Duck typing: check if first arg has 'effective_user' (Update does)
        if args and hasattr(args[0], 'effective_user'):
            update = args[0]  # Standalone function
        else:
            update = args[1]  # Instance method (self, update, context)

        user_id = update.effective_user.id
        authorized_id = config.memory_palace_telegram_user_id

        if authorized_id is None:
            # Discovery mode - show user their ID
            await update.message.reply_text(f"Your User ID: {user_id}")
            return ConversationHandler.END

        if user_id != authorized_id:
            return ConversationHandler.END

        return await func(*args, **kwargs)
    return wrapper
```

**Important:** The decorator uses duck typing to support both standalone functions and class methods. This was a key fix during development.

### Running the Bot

```bash
# Discovery mode (find your Telegram user ID)
python -m helper_functions.memory_palace_bot --discover

# Normal operation
python -m helper_functions.memory_palace_bot

# Background with nohup
nohup python -m helper_functions.memory_palace_bot > mp_bot.log 2>&1 &

# Background with screen
screen -dmS mp_bot python -m helper_functions.memory_palace_bot
```

---

## Migration Script: `memory_palace_migration.py`

### Location
`helper_functions/memory_palace_migration.py`

### Purpose
One-time import of existing JSON lesson files from `memory_palace/` folder into the LlamaIndex database.

### Category Mapping (23 -> 10)

```python
CATEGORY_MAPPING = {
    # Strategy consolidation
    "strategy_and_game_theory": "strategy",
    "strategy_and_decision_making": "strategy",
    "negotiation_and_persuasion": "strategy",

    # Psychology consolidation
    "psychology_and_cognitive_biases": "psychology",
    "psychology_and_philosophy": "psychology",
    "philosophy_and_psychology": "psychology",

    # History consolidation
    "history_and_civilization": "history",
    "history_and_society": "history",

    # ... (see full mapping in source)

    # Catch-all
    "miscellaneous_facts": "observations",
    "miscellaneous_observations": "observations",
}
```

### Files Skipped
- `itl.json` - Duplicate content from another source

### Usage

```bash
# Dry run (preview without changes)
python -m helper_functions.memory_palace_migration --dry-run

# Actual migration
python -m helper_functions.memory_palace_migration

# Force re-migration (clears existing data)
python -m helper_functions.memory_palace_migration --force
```

### Migration Stats (Initial Run)
- **163 lessons** imported
- **10 categories** populated
- **0 duplicates** (text hash deduplication)

---

## Newsletter Integration

### Location
`year_progress_and_news_reporter_litellm.py`

### Integration Points

#### 1. Import with Graceful Fallback
```python
try:
    from helper_functions.memory_palace_db import MemoryPalaceDB, CATEGORIES
    MEMORY_PALACE_AVAILABLE = True
except ImportError:
    MEMORY_PALACE_AVAILABLE = False
```

#### 2. Lesson Fetching Function
```python
def get_memory_palace_lesson(mark_shown: bool = True) -> dict | None:
    """Fetch random lesson with recency exclusion."""
    if not MEMORY_PALACE_AVAILABLE:
        return None

    db = MemoryPalaceDB()
    lesson = db.get_random_lesson(exclude_recent=True)

    if lesson and mark_shown:
        db.mark_as_shown(lesson.id)

    return {
        "topic": CATEGORIES[lesson.metadata.category.value]["display"],
        "key_insight": lesson.distilled_text,
        "category": lesson.metadata.category.value,
        "source": lesson.metadata.source,
        "id": lesson.id,
    }
```

#### 3. Blend Strategy
Based on `config.memory_palace_blend_with_generated`:

- **Blend mode (True):** Both generated lesson AND Memory Palace wisdom included
- **Replace mode (False):** Memory Palace lesson replaces generated lesson

```python
if mp_lesson and config.memory_palace_blend_with_generated:
    # Use both
    lesson_dict["mp_insight"] = mp_lesson["key_insight"]
    lesson_dict["mp_topic"] = mp_lesson["topic"]
elif mp_lesson:
    # Memory Palace only
    lesson_dict = {
        "topic": mp_lesson["topic"],
        "key_insight": mp_lesson["key_insight"],
        "mp_id": mp_lesson["id"],
    }
```

#### 4. CLI Flag
```bash
# Skip Memory Palace (use generated lesson only)
python year_progress_and_news_reporter_litellm.py --no-memory-palace
```

### HTML Rendering

In `helper_functions/html_templates.py`, the `render_lesson_html()` function now includes:

```python
# Memory Palace wisdom section
mp_insight = lesson.get("mp_insight", "")
mp_topic = lesson.get("mp_topic", "")
if mp_insight:
    html_parts.append(f'''
    <div class="wisdom-section wisdom-memory-palace"
         style="border-left: 3px solid #9333EA; margin-top: var(--space-md);">
        <div class="wisdom-label" style="color: #9333EA;">
            🏛️ MEMORY PALACE — {safe_mp_topic}
        </div>
        <div class="wisdom-text">{safe_mp_insight}</div>
    </div>''')
```

**Visual Styling:**
- Purple accent (#9333EA) to distinguish from gold-themed generated content
- Same `wisdom-section` styling for visual consistency
- Topic displayed in label (e.g., "MEMORY PALACE - Engineering & Systems")

---

## Configuration

### config/config.yml

```yaml
# Memory Palace Configuration
memory_palace:
  telegram_user_id: 123456789    # Your Telegram numeric user ID
  primary_model: "nemotron-3-nano-30b-a3b"  # For distillation
  fallback_model: "gemini-2.5-flash-lite"
  provider: "litellm"
  model_tier: "fast"
  similarity_threshold: 0.75     # Duplicate detection threshold
  recency_window_days: 30        # Days before lesson can repeat
  blend_with_generated: true     # Include with or replace generated
  index_folder: "./memory_palace/lessons_index"
```

### config/config.py Accessors

```python
memory_palace_config = config_yaml.get("memory_palace", {})
memory_palace_telegram_user_id = memory_palace_config.get("telegram_user_id")
memory_palace_provider = memory_palace_config.get("provider", "litellm")
memory_palace_model_tier = memory_palace_config.get("model_tier", "fast")
memory_palace_similarity_threshold = memory_palace_config.get("similarity_threshold", 0.75)
memory_palace_recency_window_days = memory_palace_config.get("recency_window_days", 30)
memory_palace_blend_with_generated = memory_palace_config.get("blend_with_generated", True)
memory_palace_index_folder = memory_palace_config.get("index_folder", "./memory_palace/lessons_index")
```

---

## Testing

### Test Files
| File | Tests | Coverage |
|------|-------|----------|
| `tests/test_memory_palace_db.py` | 39 | Database CRUD, distillation, recency, soft delete |
| `tests/test_memory_palace_bot.py` | 23 | Commands, state machine, callbacks |
| `tests/test_memory_palace_migration.py` | 26 | Category mapping, file parsing, dedup |
| `tests/test_web_knowledge_db.py` | 17 | Web knowledge CRUD, TTL, expiration [NEW] |
| `tests/test_memory_palace_answer.py` | 14 | Answer engine, confidence, conflicts [NEW] |

**Total: 119 tests** across 5 test files (70 core Memory Palace tests)

### Running Tests

```bash
# All Memory Palace tests (including new modules)
pytest tests/test_memory_palace_*.py tests/test_web_knowledge_db.py -v

# With coverage
pytest tests/test_memory_palace_*.py tests/test_web_knowledge_db.py --cov=helper_functions --cov-report=term-missing

# Single test file
pytest tests/test_memory_palace_db.py -v

# Run all new EDITH tests
pytest tests/test_web_knowledge_db.py tests/test_memory_palace_answer.py -v
```

### Key Test Fixtures

```python
@pytest.fixture
def mock_embed_model():
    """Mock embedding model for offline testing."""
    from llama_index.core.embeddings.mock_embed_model import MockEmbedding
    return MockEmbedding(embed_dim=1536)

@pytest.fixture
def mock_config(temp_db_dir, mock_embed_model):
    """Mock config with patched get_client."""
    with patch("helper_functions.memory_palace_db.config") as mock_cfg:
        mock_cfg.memory_palace_index_folder = str(temp_db_dir)
        # ... other config values ...

        with patch("helper_functions.memory_palace_db.get_client") as mock_get_client:
            mock_client = Mock()
            mock_client.get_llamaindex_embedding.return_value = mock_embed_model
            mock_get_client.return_value = mock_client
            yield mock_cfg
```

---

## Recency Tracking

### File Location
`memory_palace/shown_history.json`

### Format
```json
{
  "shown": {
    "uuid-1234-5678": "2026-01-04T10:30:00",
    "uuid-9abc-defg": "2026-01-03T08:15:00"
  }
}
```

### Logic
```python
def get_random_lesson(self, exclude_recent: bool = True) -> Lesson | None:
    all_lessons = self.get_all_lessons()

    if exclude_recent:
        shown = self._load_shown_history()
        cutoff = datetime.now() - timedelta(days=config.memory_palace_recency_window_days)

        # Filter out recently shown lessons
        available = [
            lesson for lesson in all_lessons
            if lesson.id not in shown or
               datetime.fromisoformat(shown[lesson.id]) < cutoff
        ]
    else:
        available = all_lessons

    return random.choice(available) if available else None
```

---

## Future Improvements

### High Priority
1. **Spaced Repetition Algorithm** - Replace random selection with SM-2 or similar algorithm for optimal recall timing
2. **Lesson Relationships** - Track connections between lessons (e.g., "builds on", "contradicts", "applies to")
3. **Search Filters** - Add category and date range filters to `/search` command
4. **Bulk Import** - Support importing from Readwise, Kindle highlights, etc.

### Medium Priority
5. **Web UI** - Gradio interface for browsing and editing lessons
6. **Export Functionality** - Export lessons to Anki, Notion, or other tools
7. **Multi-User Support** - Per-user databases for shared bot instances
8. **Lesson Versioning** - Track edits and allow reverting to previous versions

### Low Priority
9. **Voice Input** - Accept voice messages and transcribe via Whisper
10. **Image Lessons** - Store and retrieve visual insights (diagrams, infographics)
11. **Lesson Chains** - Group related lessons into learning paths
12. **Analytics Dashboard** - Track learning patterns and category coverage

---

## Troubleshooting

### Common Issues

#### 1. "OpenAI 401 Unauthorized" during operations
**Cause:** LlamaIndex defaulting to OpenAI embeddings instead of LiteLLM proxy.
**Fix:** Ensure `_get_embed_model()` uses `get_client()` from `llm_client.py`.

#### 2. Bot decorator receiving wrong arguments
**Cause:** `@authorized_only` on instance methods receives `self` first.
**Fix:** Use duck typing to detect Update objects via `hasattr(args[0], 'effective_user')`.

#### 3. Memory Palace section not appearing in newsletter
**Cause:** `render_lesson_html()` in `html_templates.py` missing MP section.
**Fix:** Add check for `mp_insight` and `mp_topic` fields in lesson dict.

#### 4. All lessons marked as "recently shown"
**Cause:** `shown_history.json` has entries for all lessons.
**Fix:** Delete or clear the file, or increase `recency_window_days`.

### Debug Commands

```bash
# Check index contents
python -c "
from helper_functions.memory_palace_db import MemoryPalaceDB
db = MemoryPalaceDB()
lessons = db.get_all_lessons()
print(f'Total lessons: {len(lessons)}')
for cat in set(l.metadata.category.value for l in lessons):
    count = len([l for l in lessons if l.metadata.category.value == cat])
    print(f'  {cat}: {count}')
"

# Check shown history
cat memory_palace/shown_history.json | python -m json.tool

# Test lesson retrieval
python -c "
from year_progress_and_news_reporter_litellm import get_memory_palace_lesson
lesson = get_memory_palace_lesson(mark_shown=False)
print(f'Topic: {lesson[\"topic\"]}')
print(f'Insight: {lesson[\"key_insight\"]}')
"
```

---

## Version History

| Date | Version | Changes |
|------|---------|---------|
| 2026-01-07 | 2.0.0 | EDITH Autonomous Learning - Dual-store architecture, question answering with confidence tiers, Firecrawl web research, conflict resolution, soft delete, session context, EDITH persona |
| 2026-01-04 | 1.0.0 | Initial implementation with all 5 phases complete |

---

## Contributors

- Implementation: Claude Code (Opus 4.5)
- Architecture Design: Collaborative (22-question interview for v1.0, 12-question interview for v2.0 EDITH features)
- Testing: 119 tests across 5 test files
