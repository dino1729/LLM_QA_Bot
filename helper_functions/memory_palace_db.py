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


def _extract_first_json_object(text: str) -> Optional[str]:
    """Extract the first balanced JSON object from text."""
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape_next = False

    for i in range(start, len(text)):
        ch = text[i]

        if escape_next:
            escape_next = False
            continue

        if ch == "\\" and in_string:
            escape_next = True
            continue

        if ch == '"':
            in_string = not in_string
            continue

        if in_string:
            continue

        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]

    return None


def _build_fallback_distilled_text(raw_input: str, max_chars: int = 400) -> str:
    """Build a readable fallback summary when LLM JSON parsing fails."""
    normalized = re.sub(r"\s+", " ", raw_input).strip()
    normalized = re.sub(r"^(save this lesson)\s*:\s*", "", normalized, flags=re.IGNORECASE)

    if not normalized:
        return "No lesson content provided."

    if len(normalized) <= max_chars:
        return normalized

    candidate = normalized[:max_chars + 1]
    sentence_cutoffs = [candidate.rfind(". "), candidate.rfind("! "), candidate.rfind("? ")]
    sentence_cutoff = max(sentence_cutoffs)

    if sentence_cutoff >= int(max_chars * 0.6):
        trimmed = candidate[:sentence_cutoff + 1].strip()
    else:
        word_cutoff = candidate.rfind(" ")
        trimmed = candidate[:word_cutoff].strip() if word_cutoff > 0 else candidate[:max_chars].strip()

    if len(trimmed) > max_chars - 3:
        trimmed = trimmed[: max_chars - 3].rstrip()

    return f"{trimmed}..."


def _parse_distillation_response(response_text: str) -> Dict[str, Any]:
    """Parse distillation response JSON, tolerating wrapper text."""
    cleaned = (response_text or "").strip()

    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        extracted = _extract_first_json_object(cleaned)
        if not extracted:
            raise
        return json.loads(extracted)


def _normalize_distilled_text(text: str, max_chars: int = 220) -> str:
    """Normalize and cap distilled text length."""
    normalized = re.sub(r"\s+", " ", (text or "")).strip().strip('"').strip("'")
    if not normalized:
        return ""

    if len(normalized) <= max_chars:
        return normalized

    # Prefer cutting at sentence boundary for readability.
    candidate = normalized[: max_chars + 1]
    sentence_cutoffs = [candidate.rfind(". "), candidate.rfind("! "), candidate.rfind("? ")]
    sentence_cutoff = max(sentence_cutoffs)

    if sentence_cutoff >= int(max_chars * 0.6):
        trimmed = candidate[:sentence_cutoff + 1].strip()
    else:
        word_cutoff = candidate.rfind(" ")
        trimmed = candidate[:word_cutoff].strip() if word_cutoff > 0 else candidate[:max_chars].strip()

    if len(trimmed) > max_chars:
        trimmed = trimmed[:max_chars].rstrip()

    return trimmed


def _normalize_suggested_tags(tags: Any) -> List[str]:
    """Normalize tag list into lowercase slug-style values."""
    if isinstance(tags, str):
        raw_tags = [t.strip() for t in tags.split(",") if t.strip()]
    elif isinstance(tags, list):
        raw_tags = [str(t).strip() for t in tags if str(t).strip()]
    else:
        raw_tags = []

    normalized: List[str] = []
    for tag in raw_tags:
        tag_norm = re.sub(r"\s+", "-", tag.lower())
        tag_norm = re.sub(r"[^a-z0-9-]", "", tag_norm)
        tag_norm = re.sub(r"-{2,}", "-", tag_norm).strip("-")
        if not tag_norm:
            continue
        if tag_norm not in normalized:
            normalized.append(tag_norm)
        if len(normalized) >= 3:
            break

    return normalized


OBJECTIVE_LESSON_META_PATTERNS: Tuple[Tuple[str, re.Pattern], ...] = (
    (
        "author_reference",
        re.compile(
            r"\b(?:the\s+)?author\s+(?:predicts|observes|argues|suggests|states|notes|believes|expresses|poses)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "article_reference",
        re.compile(r"\b(?:this|the)\s+(?:article|essay|post|piece|chapter)\b", re.IGNORECASE),
    ),
    (
        "source_reference",
        re.compile(r"\b(?:based on|from)\s+(?:the\s+)?(?:video|podcast|article|book)\b", re.IGNORECASE),
    ),
    (
        "speaker_reference",
        re.compile(r"\b(?:the\s+)?(?:speaker|presenter)\b", re.IGNORECASE),
    ),
    (
        "takeaway_preamble",
        re.compile(
            r"^\s*(?:here are|these are|the following|key takeaways|takeaways from)\b",
            re.IGNORECASE,
        ),
    ),
)


def _ensure_sentence_ending(text: str) -> str:
    """Ensure rewritten lessons end with sentence punctuation."""
    if not text:
        return text
    if text.endswith((".", "!", "?")):
        return text
    return f"{text}."


def _unique_non_empty(items: List[str]) -> List[str]:
    """Return unique non-empty values preserving order."""
    seen = set()
    deduped: List[str] = []
    for item in items:
        item_norm = item.strip()
        if not item_norm or item_norm in seen:
            continue
        seen.add(item_norm)
        deduped.append(item_norm)
    return deduped


def is_objective_lesson_text(text: str) -> Tuple[bool, List[str]]:
    """Validate whether lesson text is objective and source-agnostic."""
    normalized = re.sub(r"\s+", " ", (text or "")).strip()
    if not normalized:
        return False, ["empty_text"]

    reasons: List[str] = []
    for reason, pattern in OBJECTIVE_LESSON_META_PATTERNS:
        if pattern.search(normalized):
            reasons.append(reason)

    return len(reasons) == 0, reasons


def _apply_deterministic_objective_rewrite(raw_text: str) -> str:
    """
    Best-effort non-LLM rewrite for obvious meta phrasing.
    Returns empty string when no clean deterministic rewrite is available.
    """
    text = _normalize_distilled_text(raw_text, max_chars=260)
    if not text:
        return ""

    replacements = (
        (
            r"^\s*(?:based on|from)\s+(?:the\s+)?(?:video|podcast|article|book)[^:]*:\s*",
            "",
        ),
        (
            r"^\s*(?:based on|from)\s+(?:the\s+)?(?:video|podcast|article|book)\s*,\s*",
            "",
        ),
        (
            r"^\s*(?:the\s+)?author\s+expresses\s+a\s+desire\s+for\s+",
            "",
        ),
        (
            r"^\s*(?:the\s+)?author\s+poses\s+(?:theoretical\s+)?question\s+of\s+whether\s+",
            "Whether ",
        ),
        (
            r"^\s*(?:the\s+)?author\s+(?:predicts|observes|argues|suggests|states|notes|believes|highlights|discusses|explores)\s+that\s+",
            "",
        ),
        (
            r"^\s*(?:the\s+)?author\s+(?:predicts|observes|argues|suggests|states|notes|believes|highlights|discusses|explores)\s+",
            "",
        ),
        (
            r"^\s*(?:the\s+)?speaker\s+(?:argues|explains|notes|says|suggests|states)\s+that\s+",
            "",
        ),
        (
            r"^\s*(?:the\s+)?speaker\s+(?:argues|explains|notes|says|suggests|states)\s+",
            "",
        ),
    )

    rewritten = text
    for pattern, replacement in replacements:
        rewritten = re.sub(pattern, replacement, rewritten, flags=re.IGNORECASE)

    rewritten = _normalize_distilled_text(rewritten, max_chars=220)
    if rewritten:
        rewritten = rewritten[0].upper() + rewritten[1:] if len(rewritten) > 1 else rewritten.upper()
        rewritten = _ensure_sentence_ending(rewritten)
    return rewritten


def _parse_rewrite_response(response_text: str) -> Optional[str]:
    """Extract rewritten lesson text from a JSON response."""
    try:
        parsed = _parse_distillation_response(response_text)
    except Exception:
        return None

    if not isinstance(parsed, dict):
        return None

    for key in ("distilled_text", "rewritten_text", "lesson", "text"):
        value = parsed.get(key)
        if isinstance(value, str) and value.strip():
            return _normalize_distilled_text(value, max_chars=220)

    return None


def _rewrite_to_objective_with_client(
    client: Any,
    raw_text: str,
    max_tokens: int = 220,
) -> Optional[str]:
    """Call model once to rewrite text into objective lesson form."""
    prompt = f"""Rewrite the following lesson into ONE objective standalone sentence.

Rules:
- Keep the same core idea.
- Do not mention author, speaker, article, video, podcast, or source.
- No preamble, no list markers, no commentary.
- 90-220 characters.
- Active voice, present tense.
- Return JSON only: {{"distilled_text":"..."}}

INPUT:
{raw_text}
"""

    try:
        response = _chat_distillation_prompt(client, prompt, temperature=0.1, max_tokens=max_tokens)
    except Exception:
        return None

    rewritten = _parse_rewrite_response(response)
    if not rewritten:
        return None

    rewritten = _ensure_sentence_ending(rewritten)
    return rewritten


def rewrite_to_objective_lesson(
    raw_text: str,
    provider: str = None,
    model_tier: str = None,
    model_name: str = None,
    client: Any = None,
    fallback_model_name: str = None,
) -> str:
    """Rewrite lesson text into objective style using deterministic+LLM fallback."""
    normalized = _normalize_distilled_text(raw_text, max_chars=260)
    if not normalized:
        return ""

    is_valid, _ = is_objective_lesson_text(normalized)
    if is_valid:
        return _ensure_sentence_ending(normalized)

    deterministic = _apply_deterministic_objective_rewrite(normalized)
    if deterministic:
        deterministic_valid, _ = is_objective_lesson_text(deterministic)
        if deterministic_valid:
            return deterministic

    provider = provider or config.memory_palace_provider
    model_tier = model_tier or config.memory_palace_model_tier
    model_name = model_name or config.memory_palace_primary_model
    fallback_model_name = (
        fallback_model_name
        if fallback_model_name is not None
        else _coerce_fallback_model(getattr(config, "memory_palace_fallback_model", None))
    )

    primary_client = client or get_client(
        provider=provider,
        model_tier=model_tier,
        model_name=model_name,
    )
    rewritten = _rewrite_to_objective_with_client(primary_client, normalized)
    if rewritten:
        rewritten_valid, _ = is_objective_lesson_text(rewritten)
        if rewritten_valid:
            return rewritten

    if fallback_model_name and fallback_model_name != model_name:
        try:
            fallback_client = get_client(
                provider=provider,
                model_tier=model_tier,
                model_name=fallback_model_name,
            )
            fallback_rewrite = _rewrite_to_objective_with_client(fallback_client, normalized)
            if fallback_rewrite:
                fallback_valid, _ = is_objective_lesson_text(fallback_rewrite)
                if fallback_valid:
                    return fallback_rewrite
        except Exception:
            logger.warning("Fallback rewrite model '%s' failed", fallback_model_name)

    return deterministic or normalized


def _coerce_fallback_model(model_name: Any) -> Optional[str]:
    """Return fallback model only when configured as a non-empty string."""
    if not isinstance(model_name, str):
        return None
    model_name = model_name.strip()
    return model_name or None


def _chat_distillation_prompt(
    client: Any,
    prompt: str,
    temperature: float = 0.15,
    max_tokens: int = 320,
) -> str:
    """
    Request distillation output and prefer strict JSON mode when supported.
    """
    try:
        return client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
    except Exception:
        return client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )


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

    def _lesson_to_document(self, lesson: Lesson) -> Document:
        """Convert Lesson model into vector store document."""
        return Document(
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

    def _insert_lesson(self, lesson: Lesson, persist: bool = True) -> str:
        """Insert or overwrite a lesson by ID."""
        doc = self._lesson_to_document(lesson)
        try:
            self._store.insert(doc, persist=persist)
        except TypeError:
            # Backwards compatibility in case insert() signature lacks persist.
            self._store.insert(doc)
            if not persist:
                logger.warning("Vector store insert() does not support deferred persistence")
        return lesson.id

    def add_lesson(self, lesson: Lesson) -> str:
        """Add a lesson to the store."""
        self._insert_lesson(lesson, persist=True)
        logger.info(f"Added lesson {lesson.id[:8]}... to Memory Palace (category: {lesson.metadata.category})")
        return lesson.id

    def persist(self):
        """Persist store state to disk."""
        if self._store is not None:
            self._store.persist()

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
            logger.exception(f"Error enumerating store: {e}")

        return lessons

    def get_lesson_by_id(
        self,
        lesson_id: str,
        include_forgotten: bool = True,
    ) -> Optional[Lesson]:
        """Get a lesson by ID."""
        if self._store is None:
            return None

        for doc_id, data in self._store.docs.items():
            meta = data.get("metadata", {})
            entry_id = meta.get("id", doc_id)
            if entry_id != lesson_id and doc_id != lesson_id:
                continue

            is_forgotten = meta.get("is_forgotten", False)
            if isinstance(is_forgotten, str):
                is_forgotten = is_forgotten.lower() == "true"
            if is_forgotten and not include_forgotten:
                return None

            return self._parse_lesson_from_doc(entry_id, meta, data.get("text", ""))

        return None

    def update_lesson_text(
        self,
        lesson_id: str,
        new_text: str,
        *,
        rewritten_by_model: str,
        preserve_category: bool = True,
        append_tags: Optional[List[str]] = None,
        persist: bool = True,
    ) -> bool:
        """
        Update lesson text while preserving lesson identity and metadata.
        """
        lesson = self.get_lesson_by_id(lesson_id, include_forgotten=True)
        if lesson is None:
            return False

        normalized_text = _normalize_distilled_text(new_text, max_chars=220)
        if not normalized_text:
            return False

        category_value = (
            lesson.metadata.category.value
            if isinstance(lesson.metadata.category, LessonCategory)
            else str(lesson.metadata.category)
        )
        if not preserve_category:
            category_value = suggest_category(normalized_text)

        merged_tags = list(lesson.metadata.tags)
        if append_tags:
            merged_tags.extend(_normalize_suggested_tags(append_tags))
        merged_tags = _unique_non_empty(merged_tags)

        updated = Lesson(
            id=lesson.id,
            distilled_text=normalized_text,
            metadata=LessonMetadata(
                category=LessonCategory(category_value),
                created_at=lesson.metadata.created_at,
                source=lesson.metadata.source,
                original_input=lesson.metadata.original_input,
                distilled_by_model=rewritten_by_model,
                tags=merged_tags,
                is_forgotten=lesson.metadata.is_forgotten,
                forgotten_at=lesson.metadata.forgotten_at,
            ),
        )

        self._insert_lesson(updated, persist=persist)
        return True

    def get_lessons_by_category(self, category: LessonCategory) -> List[Lesson]:
        """Get all lessons in a specific category."""
        all_lessons = self.get_all_lessons()
        category_value = category.value if isinstance(category, LessonCategory) else category
        return [l for l in all_lessons if l.metadata.category == category_value or
                (isinstance(l.metadata.category, LessonCategory) and l.metadata.category.value == category_value)]

    def get_lesson_count(self) -> int:
        """Get the total number of active (non-forgotten) lessons."""
        if self._store is None:
            return 0
        count = 0
        for data in self._store._documents.values():
            is_forgotten = data.get("metadata", {}).get("is_forgotten", False)
            if isinstance(is_forgotten, str):
                is_forgotten = is_forgotten.lower() == "true"
            if not is_forgotten:
                count += 1
        return count

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
            for doc_id, data in self._store._documents.items():
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
    fallback_model_name = _coerce_fallback_model(
        getattr(config, "memory_palace_fallback_model", None)
    )

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
    raw_input_compact = re.sub(r"\s+", " ", raw_input).strip()
    raw_input_for_prompt = raw_input_compact[:3000] if len(raw_input_compact) > 3000 else raw_input_compact

    prompt = f"""You are a strict JSON generator for lesson distillation.
Return exactly ONE JSON object and nothing else.

EXAMPLES of well-distilled insights (match this style - concise, standalone, quotable):
{examples_text}

USER INPUT:
{raw_input_for_prompt}

INSTRUCTIONS:
1. Extract the core wisdom into ONE sentence (90-220 characters)
2. Make it standalone and quotable (no "this shows" or "the user learned")
3. Use active voice and present tense
4. Suggest the most fitting category from: {category_options}
5. Suggest 1-3 relevant tags (lowercase, no spaces)
6. Do not copy full paragraphs, compress to one sharp lesson
7. Never mention "the author", "the speaker", "this article", "the video", or source framing
8. No markdown, no code fences, no commentary

Respond in JSON format only:
{{"distilled_text": "Your single-sentence insight here", "suggested_category": "category_name", "suggested_tags": ["tag1", "tag2"]}}

JSON RESPONSE:"""

    def _build_result(parsed: Dict[str, Any]) -> LessonDistillationResult:
        suggested_category = str(parsed.get("suggested_category", "observations")).strip().lower()
        if suggested_category not in [c.value for c in LessonCategory]:
            suggested_category = "observations"

        distilled_text = _normalize_distilled_text(parsed.get("distilled_text", ""))
        if not distilled_text:
            distilled_text = _build_fallback_distilled_text(raw_input)

        valid_text, _ = is_objective_lesson_text(distilled_text)
        if not valid_text:
            rewritten = rewrite_to_objective_lesson(
                distilled_text or raw_input,
                provider=provider,
                model_tier=model_tier,
                model_name=model_name,
                client=client,
                fallback_model_name=fallback_model_name,
            )
            rewritten = _normalize_distilled_text(rewritten, max_chars=220)
            if rewritten:
                distilled_text = rewritten

        valid_text, _ = is_objective_lesson_text(distilled_text)
        if not valid_text:
            deterministic = _apply_deterministic_objective_rewrite(distilled_text or raw_input)
            deterministic = _normalize_distilled_text(deterministic, max_chars=220)
            if deterministic:
                distilled_text = deterministic

        tags = _normalize_suggested_tags(parsed.get("suggested_tags", []))

        return LessonDistillationResult(
            distilled_text=distilled_text,
            suggested_category=suggested_category,
            suggested_tags=tags,
        )

    last_parse_error = None
    primary_response = ""
    try:
        primary_response = _chat_distillation_prompt(client, prompt)
        parsed = _parse_distillation_response(primary_response)
        return _build_result(parsed)
    except json.JSONDecodeError as e:
        last_parse_error = e
        logger.warning(
            "Failed to parse distillation JSON from primary model '%s': %s",
            model_name,
            e,
        )
    except Exception as e:
        logger.warning(
            "Primary distillation call failed for model '%s': %s",
            model_name,
            e,
        )

    # Repair attempt: ask same model to convert its prior response into strict JSON.
    repair_prompt = f"""Convert the following text into a valid JSON object with this exact schema:
{{"distilled_text":"...", "suggested_category":"...", "suggested_tags":["tag1","tag2"]}}

Allowed categories: {category_options}
Rules:
- distilled_text must be one sentence (90-220 chars)
- suggested_tags must be 1-3 lowercase tokens
- Return JSON only

TEXT TO CONVERT:
{primary_response}
"""
    try:
        repair_response = _chat_distillation_prompt(client, repair_prompt, temperature=0.0, max_tokens=250)
        parsed = _parse_distillation_response(repair_response)
        return _build_result(parsed)
    except Exception as e:
        logger.warning("Repair distillation attempt failed for model '%s': %s", model_name, e)

    # Final model fallback attempt if configured.
    if fallback_model_name and fallback_model_name != model_name:
        try:
            fallback_client = get_client(
                provider=provider,
                model_tier=model_tier,
                model_name=fallback_model_name,
            )
            fallback_response = _chat_distillation_prompt(fallback_client, prompt)
            parsed = _parse_distillation_response(fallback_response)
            return _build_result(parsed)
        except Exception as e:
            logger.warning(
                "Fallback distillation model '%s' failed: %s",
                fallback_model_name,
                e,
            )

    if last_parse_error:
        logger.warning("Falling back to local distilled text after JSON parse failures")

    fallback_text = _build_fallback_distilled_text(raw_input)
    return LessonDistillationResult(
        distilled_text=fallback_text,
        suggested_category=suggest_category(fallback_text),
        suggested_tags=[],
    )


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
