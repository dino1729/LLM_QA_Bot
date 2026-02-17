"""
Memory Palace Database - SimpleVectorStore wrapper for storing lessons.

This module provides:
- Pydantic models for strict lesson validation
- Vector-based storage with semantic search
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
from helper_functions.vector_store import SimpleVectorStore, Document

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
    """Metadata stored with each lesson."""
    category: LessonCategory
    created_at: datetime = Field(default_factory=datetime.now)
    source: str = "telegram"  # telegram, migration, manual
    original_input: str  # User's raw input before distillation
    distilled_by_model: str  # Model used for distillation
    tags: List[str] = Field(default_factory=list)
    # Soft delete support
    is_forgotten: bool = False
    forgotten_at: Optional[datetime] = None


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
    Database wrapper for Memory Palace lessons using SimpleVectorStore.
    """

    def __init__(
        self,
        provider: str = None,
        model_tier: str = None,
        index_folder: str = None,
    ):
        self.provider = provider or config.memory_palace_provider
        self.model_tier = model_tier or config.memory_palace_model_tier
        self.index_folder = index_folder or config.memory_palace_index_folder
        self.similarity_threshold = config.memory_palace_similarity_threshold
        self.recency_window_days = config.memory_palace_recency_window_days

        self.index_path = Path(self.index_folder)
        self.shown_history_file = Path(config.MEMORY_PALACE_FOLDER or "./memory_palace") / "shown_history.json"

        self._store = None
        self._ensure_store()

    def _get_embed_fn(self):
        """Get the embedding function from the LLM client."""
        client = get_client(provider=self.provider, model_tier=self.model_tier)
        return client.get_embedding

    def _ensure_store(self):
        """Load or create the vector store."""
        embed_fn = self._get_embed_fn()
        self.index_path.mkdir(parents=True, exist_ok=True)
        self._store = SimpleVectorStore(persist_dir=str(self.index_path), embed_fn=embed_fn)
        logger.info(f"Loaded Memory Palace store from {self.index_path} ({len(self._store)} docs)")

    def _parse_lesson_from_doc(self, doc_id: str, meta: dict, text: str) -> Optional[Lesson]:
        """Parse a Lesson from raw metadata dict and text.

        Args:
            doc_id: Document ID (used as fallback if metadata lacks "id").
            meta: Metadata dictionary (e.g. from node.metadata or data["metadata"]).
            text: The distilled lesson text.

        Returns:
            A Lesson object, or None if parsing fails.
        """
        try:
            is_forgotten = meta.get("is_forgotten", False)
            if isinstance(is_forgotten, str):
                is_forgotten = is_forgotten.lower() == "true"

            category_str = meta.get("category", "observations")
            category = LessonCategory(category_str) if category_str in [c.value for c in LessonCategory] else LessonCategory.OBSERVATIONS

            tags_str = meta.get("tags", "")
            tags = tags_str.split(",") if tags_str else []

            created_at_str = meta.get("created_at", datetime.now().isoformat())
            try:
                created_at = datetime.fromisoformat(created_at_str)
            except ValueError:
                created_at = datetime.now()

            forgotten_at_str = meta.get("forgotten_at")
            forgotten_at = None
            if forgotten_at_str:
                try:
                    forgotten_at = datetime.fromisoformat(forgotten_at_str)
                except ValueError:
                    pass

            return Lesson(
                id=meta.get("id", doc_id),
                distilled_text=text,
                metadata=LessonMetadata(
                    category=category,
                    created_at=created_at,
                    source=meta.get("source", "unknown"),
                    original_input=meta.get("original_input", ""),
                    distilled_by_model=meta.get("distilled_by_model", "unknown"),
                    tags=tags,
                    is_forgotten=is_forgotten,
                    forgotten_at=forgotten_at,
                )
            )
        except Exception as e:
            logger.warning(f"Error parsing lesson from doc {doc_id}: {e}")
            return None

    def add_lesson(self, lesson: Lesson) -> str:
        """Add a lesson to the store."""
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
                "is_forgotten": lesson.metadata.is_forgotten,
                "forgotten_at": lesson.metadata.forgotten_at.isoformat() if lesson.metadata.forgotten_at else None,
            },
            doc_id=lesson.id,
        )

        self._store.insert(doc)
        logger.info(f"Added lesson {lesson.id[:8]}... to Memory Palace (category: {lesson.metadata.category})")
        return lesson.id

    def find_similar(
        self, text: str, top_k: int = 5, include_forgotten: bool = False
    ) -> List[SimilarLesson]:
        """Find lessons similar to the given text with similarity scores."""
        if self._store is None or len(self._store) == 0:
            return []

        request_k = top_k * 2 if not include_forgotten else top_k
        nodes = self._store.search(text, top_k=request_k)

        results = []
        for node in nodes:
            meta = node.metadata
            is_forgotten = meta.get("is_forgotten", False)
            if isinstance(is_forgotten, str):
                is_forgotten = is_forgotten.lower() == "true"

            if is_forgotten and not include_forgotten:
                continue

            doc_id = meta.get("id", str(uuid.uuid4()))
            lesson = self._parse_lesson_from_doc(doc_id, meta, node.get_content())
            if lesson is None:
                continue

            results.append(SimilarLesson(lesson=lesson, similarity_score=node.score or 0.0))

            if len(results) >= top_k:
                break

        return results

    def check_duplicate(self, text: str) -> Optional[SimilarLesson]:
        """Check if text is semantically similar to an existing lesson."""
        similar = self.find_similar(text, top_k=1)
        if similar and similar[0].similarity_score >= self.similarity_threshold:
            return similar[0]
        return None

    def get_random_lesson(self, exclude_recent: bool = True) -> Optional[Lesson]:
        """Get a random lesson, optionally excluding recently shown ones."""
        all_lessons = self.get_all_lessons()
        if not all_lessons:
            return None

        if exclude_recent:
            recent_ids = self._get_recent_shown_ids()
            eligible = [l for l in all_lessons if l.id not in recent_ids]
            if not eligible:
                self._reset_shown_history()
                eligible = all_lessons
        else:
            eligible = all_lessons

        return random.choice(eligible) if eligible else None

    def mark_as_shown(self, lesson_id: str):
        """Mark a lesson as shown for recency tracking."""
        history = self._load_shown_history()
        history.append({
            "id": lesson_id,
            "shown_at": datetime.now().isoformat()
        })

        self.shown_history_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.shown_history_file, "w") as f:
            json.dump(history, f, indent=2)

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

    def get_all_lessons(self, include_forgotten: bool = False) -> List[Lesson]:
        """Get all lessons from the store."""
        if self._store is None:
            return []

        lessons = []
        try:
            for doc_id, data in self._store.docs.items():
                meta = data.get("metadata", {})

                is_forgotten = meta.get("is_forgotten", False)
                if isinstance(is_forgotten, str):
                    is_forgotten = is_forgotten.lower() == "true"

                if is_forgotten and not include_forgotten:
                    continue

                lesson = self._parse_lesson_from_doc(doc_id, meta, data.get("text", ""))
                if lesson is not None:
                    lessons.append(lesson)
        except Exception as e:
            logger.error(f"Error enumerating store: {e}")

        return lessons

    def get_lessons_by_category(self, category: LessonCategory) -> List[Lesson]:
        """Get all lessons in a specific category."""
        all_lessons = self.get_all_lessons()
        category_value = category.value if isinstance(category, LessonCategory) else category
        return [l for l in all_lessons if l.metadata.category == category_value or
                (isinstance(l.metadata.category, LessonCategory) and l.metadata.category.value == category_value)]

    def get_lesson_count(self) -> int:
        """Get the total number of lessons in the database."""
        if self._store is None:
            return 0
        return len(self._store)

    def get_category_stats(self) -> Dict[str, int]:
        """Get lesson count per category."""
        all_lessons = self.get_all_lessons()
        stats = {}
        for lesson in all_lessons:
            cat = lesson.metadata.category if isinstance(lesson.metadata.category, str) else lesson.metadata.category.value
            stats[cat] = stats.get(cat, 0) + 1
        return stats

    def forget_by_query(self, query: str, threshold: float = 0.7) -> int:
        """Soft delete lessons matching the query."""
        similar = self.find_similar(query, top_k=10, include_forgotten=True)
        count = 0
        for s in similar:
            if s.similarity_score >= threshold and not s.lesson.metadata.is_forgotten:
                self._mark_forgotten(s.lesson.id)
                count += 1
        return count

    def _mark_forgotten(self, lesson_id: str) -> bool:
        """Mark a lesson as forgotten (soft delete)."""
        if self._store is None:
            return False

        try:
            for doc_id, data in self._store._documents.items():
                meta = data.get("metadata", {})
                if meta.get("id") == lesson_id:
                    meta["is_forgotten"] = True
                    meta["forgotten_at"] = datetime.now().isoformat()
                    self._store.persist()
                    logger.info(f"Marked lesson {lesson_id[:8]}... as forgotten")
                    return True
            return False
        except Exception as e:
            logger.error(f"Error marking lesson as forgotten: {e}")
            return False

    def restore_forgotten(self, lesson_id: str) -> bool:
        """Restore a forgotten lesson."""
        if self._store is None:
            return False

        try:
            for doc_id, data in self._store.docs.items():
                meta = data.get("metadata", {})
                if meta.get("id") == lesson_id:
                    meta["is_forgotten"] = False
                    meta["forgotten_at"] = None
                    self._store.persist()
                    logger.info(f"Restored forgotten lesson {lesson_id[:8]}...")
                    return True
            return False
        except Exception as e:
            logger.error(f"Error restoring forgotten lesson: {e}")
            return False

    def get_forgotten_lessons(self) -> List[Lesson]:
        """Get all forgotten lessons (for potential recovery)."""
        all_lessons = self.get_all_lessons(include_forgotten=True)
        return [l for l in all_lessons if l.metadata.is_forgotten]

    def get_few_shot_examples(self, count: int = 3) -> List[str]:
        """Get random lessons as few-shot examples for style consistency."""
        all_lessons = self.get_all_lessons()
        if not all_lessons:
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
    """Use LLM to distill user input into a single-line insight."""
    provider = provider or config.memory_palace_provider
    model_tier = model_tier or config.memory_palace_model_tier
    model_name = model_name or config.memory_palace_primary_model

    client = get_client(provider=provider, model_tier=model_tier, model_name=model_name)

    if few_shot_examples:
        examples_text = "\n".join([f'- "{ex}"' for ex in few_shot_examples])
    else:
        examples_text = """- "The 'Tit-for-Tat' strategy succeeds because it balances cooperation, immediate retaliation, and forgiveness."
- "The Dunning-Kruger Effect causes people with low ability to overestimate their competence."
- "Conway's Law states that organizations design systems mirroring their communication structures."
- "Compound interest is the eighth wonder; those who understand it earn it, those who don't pay it."
- "The Pareto Principle suggests 80% of consequences come from 20% of causes."
"""

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

        response_text = response.strip()
        if response_text.startswith("```"):
            lines = response_text.split("\n")
            response_text = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])

        result = json.loads(response_text)

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
    """Suggest a category based on keyword matching (fast, no LLM)."""
    text_lower = text.lower()

    scores = {}
    for category, meta in CATEGORIES.items():
        score = sum(1 for kw in meta["keywords"] if kw in text_lower)
        if score > 0:
            scores[category] = score

    if scores:
        return max(scores, key=scores.get)
    return "observations"
