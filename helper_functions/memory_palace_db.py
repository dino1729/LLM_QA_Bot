"""
Memory Palace Database - LlamaIndex VectorStoreIndex wrapper for storing lessons.

This module provides:
- Pydantic models for strict lesson validation
- LlamaIndex-based storage with semantic search
- Duplicate detection via similarity threshold
- Recency tracking for daily newsletter integration
- LLM-powered lesson distillation
"""
import json
import logging
import os
import random
import re
import uuid
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from config import config
from helper_functions.llm_client import get_client

logger = logging.getLogger(__name__)


class LessonCategory(str, Enum):
    """10 consolidated categories for Memory Palace lessons."""
    STRATEGY = "strategy"
    PSYCHOLOGY = "psychology"
    HISTORY = "history"
    SCIENCE = "science"
    TECHNOLOGY = "technology"
    ECONOMICS = "economics"
    ENGINEERING = "engineering"
    BIOLOGY = "biology"
    LEADERSHIP = "leadership"
    OBSERVATIONS = "observations"


# Category metadata with display names and keywords for classification
CATEGORIES: Dict[str, Dict[str, Any]] = {
    "strategy": {
        "display": "Strategy & Decision Making",
        "keywords": ["game theory", "negotiation", "decision", "tit-for-tat", "nash", "ooda", "strategy", "tactics"],
    },
    "psychology": {
        "display": "Psychology & Cognitive Science",
        "keywords": ["bias", "cognitive", "heuristic", "dunning", "anchoring", "effect", "psychology", "mental"],
    },
    "history": {
        "display": "History & Civilization",
        "keywords": ["empire", "war", "civilization", "ancient", "revolution", "history", "historical"],
    },
    "science": {
        "display": "Science & Physics",
        "keywords": ["physics", "quantum", "relativity", "particle", "force", "science", "scientific"],
    },
    "technology": {
        "display": "Technology & Computing",
        "keywords": ["ai", "computing", "algorithm", "software", "hardware", "technology", "digital", "tech"],
    },
    "economics": {
        "display": "Economics & Finance",
        "keywords": ["market", "investment", "inflation", "compound", "money", "economics", "finance", "wealth"],
    },
    "engineering": {
        "display": "Engineering & Systems",
        "keywords": ["design", "system", "process", "verification", "manufacturing", "engineering", "silicon"],
    },
    "biology": {
        "display": "Biology & Health",
        "keywords": ["health", "brain", "cell", "disease", "evolution", "biology", "biological", "body"],
    },
    "leadership": {
        "display": "Leadership & Career",
        "keywords": ["leadership", "career", "growth", "mindset", "success", "management", "team"],
    },
    "observations": {
        "display": "Observations & Ideas",
        "keywords": ["observation", "idea", "prediction", "future", "philosophy", "life", "wisdom"],
    },
}


class LessonMetadata(BaseModel):
    """Metadata stored with each lesson in LlamaIndex."""
    category: LessonCategory
    created_at: datetime = Field(default_factory=datetime.now)
    source: str = "telegram"  # telegram, migration, manual
    original_input: str  # User's raw input before distillation
    distilled_by_model: str  # Model used for distillation (e.g., "litellm:nemotron-3-nano-30b-a3b")
    tags: List[str] = Field(default_factory=list)


class Lesson(BaseModel):
    """Complete lesson representation."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    distilled_text: str  # Single-line insight (the core lesson)
    metadata: LessonMetadata

    model_config = {"use_enum_values": True}


class SimilarLesson(BaseModel):
    """Lesson with similarity score for duplicate detection and retrieval."""
    lesson: Lesson
    similarity_score: float


class LessonDistillationResult(BaseModel):
    """Result from LLM distillation."""
    distilled_text: str
    suggested_category: str
    suggested_tags: List[str] = Field(default_factory=list)


class MemoryPalaceDB:
    """
    Database wrapper for Memory Palace lessons using LlamaIndex VectorStoreIndex.

    Handles storage, retrieval, duplicate detection, and daily selection with recency tracking.
    """

    def __init__(
        self,
        provider: str = None,
        model_tier: str = None,
        index_folder: str = None,
    ):
        """
        Initialize the Memory Palace database.

        Args:
            provider: LLM provider for distillation (default from config)
            model_tier: Model tier for distillation (default from config)
            index_folder: Path to LlamaIndex index (default from config)
        """
        self.provider = provider or config.memory_palace_provider
        self.model_tier = model_tier or config.memory_palace_model_tier
        self.index_folder = index_folder or config.memory_palace_index_folder
        self.similarity_threshold = config.memory_palace_similarity_threshold
        self.recency_window_days = config.memory_palace_recency_window_days

        # Paths
        self.index_path = Path(self.index_folder)
        self.shown_history_file = Path(config.MEMORY_PALACE_FOLDER or "./memory_palace") / "shown_history.json"

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

        # Set the embedding model for LlamaIndex operations
        self._embed_model = self._get_embed_model()
        Settings.embed_model = self._embed_model

        self.index_path.mkdir(parents=True, exist_ok=True)

        if self.index_path.exists() and any(self.index_path.iterdir()):
            try:
                storage_context = StorageContext.from_defaults(persist_dir=str(self.index_path))
                self._index = load_index_from_storage(
                    storage_context,
                    embed_model=self._embed_model
                )
                logger.info(f"Loaded existing Memory Palace index from {self.index_path}")
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
        logger.info(f"Created new Memory Palace index at {self.index_path}")

    def add_lesson(self, lesson: Lesson) -> str:
        """
        Add a lesson to the index.

        Args:
            lesson: Lesson object to store

        Returns:
            Lesson ID
        """
        from llama_index.core import Document

        # Create LlamaIndex Document with metadata
        doc = Document(
            text=lesson.distilled_text,
            metadata={
                "id": lesson.id,
                "category": lesson.metadata.category if isinstance(lesson.metadata.category, str) else lesson.metadata.category.value,
                "created_at": lesson.metadata.created_at.isoformat(),
                "source": lesson.metadata.source,
                "original_input": lesson.metadata.original_input,
                "distilled_by_model": lesson.metadata.distilled_by_model,
                "tags": ",".join(lesson.metadata.tags),
            },
            excluded_embed_metadata_keys=["original_input", "id", "distilled_by_model"],
        )

        self._index.insert(doc)
        self._index.storage_context.persist(persist_dir=str(self.index_path))
        logger.info(f"Added lesson {lesson.id[:8]}... to Memory Palace (category: {lesson.metadata.category})")
        return lesson.id

    def find_similar(self, text: str, top_k: int = 5) -> List[SimilarLesson]:
        """
        Find lessons similar to the given text with similarity scores.

        Args:
            text: Text to search for
            top_k: Maximum number of results

        Returns:
            List of SimilarLesson objects sorted by similarity
        """
        from llama_index.core.retrievers import VectorIndexRetriever

        if self._index is None or self.get_lesson_count() == 0:
            return []

        retriever = VectorIndexRetriever(index=self._index, similarity_top_k=top_k)
        nodes = retriever.retrieve(text)

        results = []
        for node in nodes:
            try:
                category_str = node.metadata.get("category", "observations")
                category = LessonCategory(category_str) if category_str in [c.value for c in LessonCategory] else LessonCategory.OBSERVATIONS

                tags_str = node.metadata.get("tags", "")
                tags = tags_str.split(",") if tags_str else []

                created_at_str = node.metadata.get("created_at", datetime.now().isoformat())
                try:
                    created_at = datetime.fromisoformat(created_at_str)
                except ValueError:
                    created_at = datetime.now()

                lesson = Lesson(
                    id=node.metadata.get("id", str(uuid.uuid4())),
                    distilled_text=node.get_content(),
                    metadata=LessonMetadata(
                        category=category,
                        created_at=created_at,
                        source=node.metadata.get("source", "unknown"),
                        original_input=node.metadata.get("original_input", ""),
                        distilled_by_model=node.metadata.get("distilled_by_model", "unknown"),
                        tags=tags,
                    )
                )
                results.append(SimilarLesson(lesson=lesson, similarity_score=node.score or 0.0))
            except Exception as e:
                logger.warning(f"Error parsing node: {e}")
                continue

        return results

    def check_duplicate(self, text: str) -> Optional[SimilarLesson]:
        """
        Check if text is semantically similar to an existing lesson.

        Args:
            text: Text to check for duplicates

        Returns:
            Most similar lesson if above threshold, None otherwise
        """
        similar = self.find_similar(text, top_k=1)
        if similar and similar[0].similarity_score >= self.similarity_threshold:
            return similar[0]
        return None

    def get_random_lesson(self, exclude_recent: bool = True) -> Optional[Lesson]:
        """
        Get a random lesson, optionally excluding recently shown ones.

        Args:
            exclude_recent: If True, exclude lessons shown in the recency window

        Returns:
            Random Lesson or None if database is empty
        """
        all_lessons = self.get_all_lessons()
        if not all_lessons:
            return None

        if exclude_recent:
            recent_ids = self._get_recent_shown_ids()
            eligible = [l for l in all_lessons if l.id not in recent_ids]
            if not eligible:
                # All lessons have been shown, reset history
                self._reset_shown_history()
                eligible = all_lessons
        else:
            eligible = all_lessons

        return random.choice(eligible) if eligible else None

    def mark_as_shown(self, lesson_id: str):
        """
        Mark a lesson as shown for recency tracking.

        Args:
            lesson_id: ID of the lesson that was shown
        """
        history = self._load_shown_history()
        history.append({
            "id": lesson_id,
            "shown_at": datetime.now().isoformat()
        })

        self.shown_history_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.shown_history_file, "w") as f:
            json.dump(history, f, indent=2)

        logger.debug(f"Marked lesson {lesson_id[:8]}... as shown")

    def _load_shown_history(self) -> List[Dict]:
        """Load the shown history from file."""
        if not self.shown_history_file.exists():
            return []

        try:
            with open(self.shown_history_file, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return []

    def _get_recent_shown_ids(self) -> set:
        """Get IDs of lessons shown within the recency window."""
        history = self._load_shown_history()
        cutoff = datetime.now() - timedelta(days=self.recency_window_days)

        recent_ids = set()
        for entry in history:
            try:
                shown_at = datetime.fromisoformat(entry["shown_at"])
                if shown_at > cutoff:
                    recent_ids.add(entry["id"])
            except (KeyError, ValueError):
                continue

        return recent_ids

    def _reset_shown_history(self):
        """Reset the shown history when all lessons have been shown."""
        if self.shown_history_file.exists():
            self.shown_history_file.unlink()
        logger.info("Reset Memory Palace shown history")

    def get_all_lessons(self) -> List[Lesson]:
        """Get all lessons from the index."""
        if self._index is None:
            return []

        lessons = []
        try:
            docstore = self._index.docstore
            for doc_id in docstore.docs:
                doc = docstore.get_document(doc_id)
                if doc:
                    try:
                        category_str = doc.metadata.get("category", "observations")
                        category = LessonCategory(category_str) if category_str in [c.value for c in LessonCategory] else LessonCategory.OBSERVATIONS

                        tags_str = doc.metadata.get("tags", "")
                        tags = tags_str.split(",") if tags_str else []

                        created_at_str = doc.metadata.get("created_at", datetime.now().isoformat())
                        try:
                            created_at = datetime.fromisoformat(created_at_str)
                        except ValueError:
                            created_at = datetime.now()

                        lesson = Lesson(
                            id=doc.metadata.get("id", doc_id),
                            distilled_text=doc.get_content(),
                            metadata=LessonMetadata(
                                category=category,
                                created_at=created_at,
                                source=doc.metadata.get("source", "unknown"),
                                original_input=doc.metadata.get("original_input", ""),
                                distilled_by_model=doc.metadata.get("distilled_by_model", "unknown"),
                                tags=tags,
                            )
                        )
                        lessons.append(lesson)
                    except Exception as e:
                        logger.warning(f"Error parsing document {doc_id}: {e}")
                        continue
        except Exception as e:
            logger.error(f"Error enumerating docstore: {e}")

        return lessons

    def get_lessons_by_category(self, category: LessonCategory) -> List[Lesson]:
        """Get all lessons in a specific category."""
        all_lessons = self.get_all_lessons()
        category_value = category.value if isinstance(category, LessonCategory) else category
        return [l for l in all_lessons if l.metadata.category == category_value or
                (isinstance(l.metadata.category, LessonCategory) and l.metadata.category.value == category_value)]

    def get_lesson_count(self) -> int:
        """Get the total number of lessons in the database."""
        if self._index is None:
            return 0
        try:
            return len(self._index.docstore.docs)
        except Exception:
            return 0

    def get_category_stats(self) -> Dict[str, int]:
        """Get lesson count per category."""
        all_lessons = self.get_all_lessons()
        stats = {}
        for lesson in all_lessons:
            cat = lesson.metadata.category if isinstance(lesson.metadata.category, str) else lesson.metadata.category.value
            stats[cat] = stats.get(cat, 0) + 1
        return stats

    def get_few_shot_examples(self, count: int = 3) -> List[str]:
        """
        Get random lessons as few-shot examples for style consistency.

        Args:
            count: Number of examples to return

        Returns:
            List of distilled lesson texts
        """
        all_lessons = self.get_all_lessons()
        if not all_lessons:
            # Return default examples if database is empty
            return [
                "The 'Tit-for-Tat' strategy succeeds because it balances cooperation, immediate retaliation, and forgiveness.",
                "The Dunning-Kruger Effect causes people with low ability to overestimate their competence.",
                "Conway's Law states that organizations design systems mirroring their communication structures.",
            ]

        samples = random.sample(all_lessons, min(count, len(all_lessons)))
        return [l.distilled_text for l in samples]


def distill_lesson(
    raw_input: str,
    provider: str = None,
    model_tier: str = None,
    model_name: str = None,
    few_shot_examples: List[str] = None,
) -> LessonDistillationResult:
    """
    Use LLM to distill user input into a single-line insight.

    Args:
        raw_input: User's raw lesson text
        provider: LLM provider (default from config)
        model_tier: Model tier (default from config)
        model_name: Explicit model name (overrides tier)
        few_shot_examples: Optional examples for style consistency

    Returns:
        LessonDistillationResult with distilled text, category, and tags
    """
    provider = provider or config.memory_palace_provider
    model_tier = model_tier or config.memory_palace_model_tier
    model_name = model_name or config.memory_palace_primary_model

    client = get_client(provider=provider, model_tier=model_tier, model_name=model_name)

    # Build examples section
    if few_shot_examples:
        examples_text = "\n".join([f'- "{ex}"' for ex in few_shot_examples])
    else:
        examples_text = """- "The 'Tit-for-Tat' strategy succeeds because it balances cooperation, immediate retaliation, and forgiveness."
- "The Dunning-Kruger Effect causes people with low ability to overestimate their competence."
- "Conway's Law states that organizations design systems mirroring their communication structures."
- "Compound interest is the eighth wonder; those who understand it earn it, those who don't pay it."
- "The Pareto Principle suggests 80% of consequences come from 20% of causes."
"""

    # Build category options
    category_options = ", ".join([c.value for c in LessonCategory])

    prompt = f"""You are a wisdom curator. Distill the following input into a single, memorable insight.

EXAMPLES of well-distilled insights (match this style - concise, standalone, quotable):
{examples_text}

USER INPUT:
{raw_input}

INSTRUCTIONS:
1. Extract the core wisdom into ONE sentence (max 200 characters)
2. Make it standalone and quotable (no "this shows" or "the user learned")
3. Use active voice and present tense
4. Suggest the most fitting category from: {category_options}
5. Suggest 1-3 relevant tags (lowercase, no spaces)

Respond in JSON format only:
{{"distilled_text": "Your single-sentence insight here", "suggested_category": "category_name", "suggested_tags": ["tag1", "tag2"]}}

JSON RESPONSE:"""

    try:
        response = client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=500
        )

        # Parse JSON response
        # Handle potential markdown code blocks
        response_text = response.strip()
        if response_text.startswith("```"):
            # Remove markdown code fence
            lines = response_text.split("\n")
            response_text = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])

        result = json.loads(response_text)

        # Validate category
        suggested_category = result.get("suggested_category", "observations")
        if suggested_category not in [c.value for c in LessonCategory]:
            suggested_category = "observations"

        return LessonDistillationResult(
            distilled_text=result.get("distilled_text", raw_input[:200]),
            suggested_category=suggested_category,
            suggested_tags=result.get("suggested_tags", []),
        )

    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse LLM JSON response: {e}. Using fallback.")
        return LessonDistillationResult(
            distilled_text=raw_input[:200] if len(raw_input) > 200 else raw_input,
            suggested_category="observations",
            suggested_tags=[],
        )
    except Exception as e:
        logger.error(f"Distillation failed: {e}")
        raise


def suggest_category(text: str) -> str:
    """
    Suggest a category based on keyword matching (fast, no LLM).

    Args:
        text: Text to categorize

    Returns:
        Category name string
    """
    text_lower = text.lower()

    scores = {}
    for category, meta in CATEGORIES.items():
        score = sum(1 for kw in meta["keywords"] if kw in text_lower)
        if score > 0:
            scores[category] = score

    if scores:
        return max(scores, key=scores.get)
    return "observations"
