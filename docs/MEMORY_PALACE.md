# Memory Palace System - Technical Documentation

## Overview

The Memory Palace is a curated knowledge management system that stores distilled insights from various learning sources (books, articles, podcasts, etc.) and integrates them into the daily newsletter workflow. It uses LlamaIndex VectorStoreIndex for semantic search and retrieval, with a Telegram bot interface for adding new lessons.

**Key Features:**
- LlamaIndex-based vector database for semantic search
- Telegram bot with conversational state machine for lesson ingestion
- LLM-powered distillation of verbose content into single-line insights
- Recency tracking to prevent repeated lessons in newsletters
- 10 consolidated categories with automatic LLM-suggested categorization
- Duplicate detection with configurable similarity threshold

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Memory Palace System                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐     ┌──────────────────┐     ┌─────────────┐ │
│  │ Telegram Bot │────▶│  MemoryPalaceDB  │────▶│  LlamaIndex │ │
│  │ (Ingestion)  │     │  (Core Module)   │     │  VectorStore│ │
│  └──────────────┘     └──────────────────┘     └─────────────┘ │
│         │                     │                       │         │
│         │                     │                       │         │
│         ▼                     ▼                       ▼         │
│  ┌──────────────┐     ┌──────────────────┐     ┌─────────────┐ │
│  │  LLM Client  │     │  shown_history   │     │   docstore  │ │
│  │ (Distillation│     │     .json        │     │   .json     │ │
│  └──────────────┘     └──────────────────┘     └─────────────┘ │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              Newsletter Integration                        │   │
│  │  year_progress_and_news_reporter_litellm.py               │   │
│  │  └─▶ get_memory_palace_lesson() ─▶ HTML rendering         │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## File Structure

```
LLM_QA_Bot/
├── helper_functions/
│   ├── memory_palace_db.py      # Core database module
│   ├── memory_palace_bot.py     # Telegram bot
│   ├── memory_palace_migration.py  # One-time migration script
│   └── html_templates.py        # Newsletter HTML (includes MP section)
├── memory_palace/
│   ├── lessons_index/           # LlamaIndex persistence folder
│   │   ├── docstore.json        # Document storage
│   │   ├── index_store.json     # Index metadata
│   │   ├── default__vector_store.json  # Vector embeddings
│   │   └── graph_store.json     # Graph relationships
│   ├── shown_history.json       # Recency tracking
│   └── *.json                   # Legacy lesson files (migrated)
├── config/
│   ├── config.yml               # Memory Palace configuration
│   └── config.py                # Config accessors
├── tests/
│   ├── test_memory_palace_db.py       # 39 tests
│   ├── test_memory_palace_bot.py      # 23 tests
│   └── test_memory_palace_migration.py # 26 tests
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
    original_input: str | None = None # Raw user input before distillation
    distilled_by_model: str | None = None  # Model used for distillation
    tags: list[str] = []
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

## Telegram Bot: `memory_palace_bot.py`

### Location
`helper_functions/memory_palace_bot.py`

### Dependencies
```python
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    CallbackQueryHandler, ConversationHandler, ContextTypes
)
```

### State Machine

```
┌─────────────────────────────────────────────────────────────────┐
│                    Telegram Bot State Machine                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│    User sends /add or message                                    │
│              │                                                   │
│              ▼                                                   │
│    ┌─────────────────┐                                          │
│    │ AWAITING_LESSON │◀──────────────────────────────┐          │
│    └────────┬────────┘                               │          │
│             │ (receive text)                         │          │
│             ▼                                        │          │
│    ┌─────────────────┐                               │          │
│    │  Distill with   │                               │          │
│    │      LLM        │                               │          │
│    └────────┬────────┘                               │          │
│             │                                        │          │
│             ▼                                        │          │
│    ┌─────────────────┐      ┌────────────────┐      │          │
│    │ Check Duplicate │─Yes─▶│  CONFIRMING_   │      │          │
│    │ (sim > 0.75?)   │      │   DUPLICATE    │      │          │
│    └────────┬────────┘      └───────┬────────┘      │          │
│             │ No                    │               │          │
│             ▼                       │ "Add Anyway"  │          │
│    ┌─────────────────┐              │               │          │
│    │   CONFIRMING_   │◀─────────────┘               │          │
│    │   DISTILLED     │                              │          │
│    └────────┬────────┘                              │          │
│             │                                       │          │
│     ┌───────┼───────┐                               │          │
│     │       │       │                               │          │
│  Approve  Edit   Reject ────────────────────────────┘          │
│     │       │                                                   │
│     │       ▼                                                   │
│     │  ┌─────────────────┐                                      │
│     │  │ EDITING_LESSON  │                                      │
│     │  └────────┬────────┘                                      │
│     │           │ (re-distill with feedback)                    │
│     │           └────────────────────────────┐                  │
│     ▼                                        │                  │
│    ┌─────────────────┐                       │                  │
│    │   CONFIRMING_   │◀──────────────────────┘                  │
│    │    CATEGORY     │                                          │
│    └────────┬────────┘                                          │
│             │ (select from keyboard)                            │
│             ▼                                                   │
│    ┌─────────────────┐                                          │
│    │  SAVE TO DB     │                                          │
│    │  END CONVERSATION│                                          │
│    └─────────────────┘                                          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
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
| `tests/test_memory_palace_db.py` | 39 | Database CRUD, distillation, recency |
| `tests/test_memory_palace_bot.py` | 23 | Commands, state machine, callbacks |
| `tests/test_memory_palace_migration.py` | 26 | Category mapping, file parsing, dedup |

### Running Tests

```bash
# All Memory Palace tests
pytest tests/test_memory_palace_*.py -v

# With coverage
pytest tests/test_memory_palace_*.py --cov=helper_functions --cov-report=term-missing

# Single test file
pytest tests/test_memory_palace_db.py -v
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
| 2026-01-04 | 1.0.0 | Initial implementation with all 5 phases complete |

---

## Contributors

- Implementation: Claude Code (Opus 4.5)
- Architecture Design: Collaborative (22-question interview)
- Testing: 114 tests across 3 test files
