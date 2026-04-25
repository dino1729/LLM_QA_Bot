"""
Entity Registry for the LLM Wiki.

Maintains a canonical mapping of entity names to prevent duplicates
when multiple LLM council models extract entities with different
surface forms (e.g., "Second-Order Thinking" vs "Second-Order Effects").

Uses Levenshtein distance + token Jaccard similarity for near-duplicate
detection - no LLM calls needed for name resolution.
"""
import json
import logging
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

logger = logging.getLogger(__name__)

# Articles and common words to strip during slug generation
_STRIP_WORDS = frozenset({"the", "a", "an", "of", "and", "in", "on", "for", "to", "with"})

EntityType = Literal["concept", "person", "source", "analysis"]


@dataclass(frozen=True)
class EntityEntry:
    """A registered entity with canonical slug and metadata."""

    canonical_slug: str
    display_name: str
    entity_type: EntityType
    aliases: Tuple[str, ...] = ()
    category: str = ""


@dataclass
class EntityRegistry:
    """
    Persistent mapping of entity names to canonical slugs.

    Stored at vault/raw/refs/legacy-schema/entity_registry.json. Checked before every
    page create to prevent duplicates. Uses string similarity (no LLM)
    for near-duplicate detection.
    """

    registry_path: Path
    _entries: Dict[str, EntityEntry] = field(default_factory=dict, repr=False)
    _alias_index: Dict[str, str] = field(default_factory=dict, repr=False)

    def load(self) -> None:
        """Load registry from JSON file."""
        if not self.registry_path.exists():
            self._entries = {}
            self._alias_index = {}
            return

        try:
            data = json.loads(self.registry_path.read_text(encoding="utf-8"))
            self._entries = {}
            self._alias_index = {}
            for slug, entry_data in data.items():
                entry = EntityEntry(
                    canonical_slug=slug,
                    display_name=entry_data["display_name"],
                    entity_type=entry_data["entity_type"],
                    aliases=tuple(entry_data.get("aliases", [])),
                    category=entry_data.get("category", ""),
                )
                self._entries[slug] = entry
                # Index all aliases for fast lookup
                for alias in entry.aliases:
                    self._alias_index[_normalize_for_lookup(alias)] = slug
                self._alias_index[_normalize_for_lookup(entry.display_name)] = slug
            logger.info("Loaded entity registry: %d entities", len(self._entries))
        except (json.JSONDecodeError, KeyError):
            logger.exception("Failed to load entity registry from %s", self.registry_path)
            self._entries = {}
            self._alias_index = {}

    def save(self) -> None:
        """Persist registry to JSON file (atomic write)."""
        data = {}
        for slug, entry in sorted(self._entries.items()):
            data[slug] = {
                "display_name": entry.display_name,
                "entity_type": entry.entity_type,
                "aliases": list(entry.aliases),
                "category": entry.category,
            }

        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.registry_path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp_path.replace(self.registry_path)

    def resolve(
        self,
        raw_name: str,
        entity_type: EntityType = "concept",
        category: str = "",
    ) -> str:
        """
        Return canonical slug for an entity name.

        If the name (or a near-duplicate) exists, returns the existing slug.
        Otherwise registers a new entity and returns the new slug.
        """
        normalized = _normalize_for_lookup(raw_name)

        # Exact alias match
        if normalized in self._alias_index:
            return self._alias_index[normalized]

        # Check slug-level match
        slug = slugify(raw_name)
        if slug in self._entries:
            return slug

        # Near-duplicate detection
        similar = self.find_similar_entities(raw_name)
        if similar:
            best_slug, best_score = similar[0]
            if best_score >= 0.85:
                # Close enough - map as alias
                self.register_alias(best_slug, raw_name)
                return best_slug

        # Genuinely new entity
        entry = EntityEntry(
            canonical_slug=slug,
            display_name=raw_name,
            entity_type=entity_type,
            aliases=(),
            category=category,
        )
        self._entries[slug] = entry
        self._alias_index[normalized] = slug
        logger.info("Registered new entity: %s -> %s", raw_name, slug)
        return slug

    def register_alias(self, canonical_slug: str, alias: str) -> None:
        """Add a new alias for an existing entity."""
        if canonical_slug not in self._entries:
            logger.warning("Cannot register alias for unknown entity: %s", canonical_slug)
            return

        entry = self._entries[canonical_slug]
        if alias not in entry.aliases:
            new_aliases = entry.aliases + (alias,)
            self._entries[canonical_slug] = EntityEntry(
                canonical_slug=entry.canonical_slug,
                display_name=entry.display_name,
                entity_type=entry.entity_type,
                aliases=new_aliases,
                category=entry.category,
            )
        self._alias_index[_normalize_for_lookup(alias)] = canonical_slug

    def get_entry(self, slug: str) -> Optional[EntityEntry]:
        """Get entity entry by canonical slug."""
        return self._entries.get(slug)

    def get_all_slugs(self) -> List[str]:
        """Return all registered canonical slugs."""
        return list(self._entries.keys())

    def get_page_path(self, canonical_slug: str, vault_root: Path) -> Path:
        """Return vault path for the entity's wiki page."""
        entry = self._entries.get(canonical_slug)
        if entry is None:
            # Default to concepts
            return vault_root / "wiki" / "concepts" / f"{canonical_slug}.md"

        type_to_dir = {
            "concept": ("wiki", "concepts"),
            "person": ("wiki", "entities"),
            "source": ("wiki", "summaries"),
            "analysis": ("outputs", "queries"),
        }
        directory = type_to_dir.get(entry.entity_type, ("wiki", "concepts"))
        return vault_root.joinpath(*directory) / f"{canonical_slug}.md"

    def find_similar_entities(
        self, name: str, threshold: float = 0.5
    ) -> List[Tuple[str, float]]:
        """
        Find entities with similar names using combined string similarity.

        Returns list of (slug, score) pairs sorted by score descending.
        Score is average of normalized Levenshtein similarity and token Jaccard.
        No LLM call - fast and deterministic.
        """
        target_normalized = _normalize_for_lookup(name)
        target_tokens = _tokenize(name)
        results = []

        for slug, entry in self._entries.items():
            # Check against display name and all aliases
            candidates = [entry.display_name] + list(entry.aliases)
            best_score = 0.0
            for candidate in candidates:
                cand_normalized = _normalize_for_lookup(candidate)
                cand_tokens = _tokenize(candidate)

                lev_sim = _levenshtein_similarity(target_normalized, cand_normalized)
                jaccard = _token_jaccard(target_tokens, cand_tokens)
                score = (lev_sim + jaccard) / 2.0
                best_score = max(best_score, score)

            if best_score >= threshold:
                results.append((slug, best_score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results

    @property
    def entity_count(self) -> int:
        return len(self._entries)


def slugify(name: str) -> str:
    """
    Convert a display name to a canonical slug.

    Rules: lowercase, strip articles, replace spaces with hyphens,
    remove special chars except hyphens.
    """
    # Normalize unicode
    text = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    text = text.lower().strip()

    # Remove possessives
    text = re.sub(r"'s\b", "", text)

    # Split into words, remove articles/common words
    words = text.split()
    words = [w for w in words if w not in _STRIP_WORDS]

    # Join and clean
    slug = "-".join(words)
    slug = re.sub(r"[^a-z0-9-]", "", slug)
    slug = re.sub(r"-+", "-", slug)
    slug = slug.strip("-")

    return slug or "unnamed"


def _normalize_for_lookup(name: str) -> str:
    """Normalize a name for index lookups (lowercase, stripped, no punctuation)."""
    text = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9 ]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def _tokenize(name: str) -> frozenset:
    """Split a name into lowercase token set for Jaccard comparison."""
    text = _normalize_for_lookup(name)
    tokens = {w for w in text.split() if w not in _STRIP_WORDS}
    return frozenset(tokens)


def _token_jaccard(a: frozenset, b: frozenset) -> float:
    """Jaccard similarity between two token sets."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _levenshtein_similarity(a: str, b: str) -> float:
    """Normalized Levenshtein similarity (1.0 = identical, 0.0 = completely different)."""
    if a == b:
        return 1.0
    max_len = max(len(a), len(b))
    if max_len == 0:
        return 1.0
    distance = _levenshtein_distance(a, b)
    return 1.0 - (distance / max_len)


def _levenshtein_distance(a: str, b: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    if len(a) < len(b):
        return _levenshtein_distance(b, a)

    if len(b) == 0:
        return len(a)

    previous_row = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        current_row = [i + 1]
        for j, cb in enumerate(b):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (ca != cb)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]
