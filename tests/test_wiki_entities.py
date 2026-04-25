"""Tests for wiki.entities - EntityRegistry and slug utilities."""
import json
import pytest
from pathlib import Path

from wiki.entities import (
    EntityEntry,
    EntityRegistry,
    slugify,
    _levenshtein_similarity,
    _token_jaccard,
    _tokenize,
    _normalize_for_lookup,
)


class TestSlugify:
    """Test the slugify function."""

    def test_basic_name(self):
        assert slugify("Compound Interest") == "compound-interest"

    def test_strips_articles(self):
        assert slugify("The Art of War") == "art-war"

    def test_removes_possessives(self):
        # Possessive 's is stripped entirely by the regex
        assert slugify("Munger's Mental Models") == "munger-mental-models"

    def test_removes_special_chars(self):
        assert slugify("Nash Equilibrium (Game Theory)") == "nash-equilibrium-game-theory"

    def test_handles_unicode(self):
        # Possessive 's stripped, unicode normalized
        assert slugify("Schrodinger's Cat") == "schrodinger-cat"

    def test_collapses_hyphens(self):
        assert slugify("  first - second  ") == "first-second"

    def test_empty_becomes_unnamed(self):
        assert slugify("") == "unnamed"
        assert slugify("the a an") == "unnamed"

    def test_person_name(self):
        assert slugify("Charlie Munger") == "charlie-munger"

    def test_no_trailing_hyphens(self):
        assert slugify("--test--") == "test"


class TestNormalize:
    """Test normalization helpers."""

    def test_normalize_for_lookup(self):
        assert _normalize_for_lookup("Compound Interest!") == "compound interest"
        assert _normalize_for_lookup("  Second-Order  Thinking  ") == "secondorder thinking"

    def test_tokenize(self):
        tokens = _tokenize("Second Order Thinking")
        assert "second" in tokens
        assert "order" in tokens
        assert "thinking" in tokens
        # Common words stripped
        assert "the" not in tokens


class TestSimilarity:
    """Test similarity functions."""

    def test_levenshtein_identical(self):
        assert _levenshtein_similarity("test", "test") == 1.0

    def test_levenshtein_completely_different(self):
        sim = _levenshtein_similarity("abc", "xyz")
        assert sim == 0.0

    def test_levenshtein_similar(self):
        sim = _levenshtein_similarity("compound interest", "compound interests")
        assert sim > 0.9

    def test_token_jaccard_identical(self):
        a = frozenset({"compound", "interest"})
        assert _token_jaccard(a, a) == 1.0

    def test_token_jaccard_partial(self):
        a = frozenset({"compound", "interest"})
        b = frozenset({"compound", "growth"})
        assert _token_jaccard(a, b) == pytest.approx(1 / 3)

    def test_token_jaccard_empty(self):
        assert _token_jaccard(frozenset(), frozenset()) == 1.0
        assert _token_jaccard(frozenset({"a"}), frozenset()) == 0.0


class TestEntityRegistry:
    """Test EntityRegistry with a temp file."""

    @pytest.fixture
    def registry_path(self, tmp_path):
        return tmp_path / "entity_registry.json"

    @pytest.fixture
    def registry(self, registry_path):
        reg = EntityRegistry(registry_path=registry_path)
        reg.load()
        return reg

    def test_resolve_new_entity(self, registry):
        slug = registry.resolve("Compound Interest", entity_type="concept")
        assert slug == "compound-interest"
        assert registry.entity_count == 1

    def test_resolve_existing_exact(self, registry):
        registry.resolve("Compound Interest", entity_type="concept")
        slug = registry.resolve("Compound Interest")
        assert slug == "compound-interest"
        assert registry.entity_count == 1  # No duplicate

    def test_resolve_near_duplicate(self, registry):
        registry.resolve("Compound Interest", entity_type="concept")
        # Very similar name should map to existing
        slug = registry.resolve("compound interest")
        assert slug == "compound-interest"
        assert registry.entity_count == 1

    def test_resolve_different_entity(self, registry):
        registry.resolve("Compound Interest")
        registry.resolve("Game Theory")
        assert registry.entity_count == 2

    def test_register_alias(self, registry):
        registry.resolve("Compound Interest", entity_type="concept")
        registry.register_alias("compound-interest", "Compounding")

        # Alias should resolve to the same slug
        slug = registry.resolve("Compounding")
        assert slug == "compound-interest"
        assert registry.entity_count == 1

    def test_save_and_load(self, registry, registry_path):
        registry.resolve("Compound Interest", entity_type="concept", category="economics")
        registry.resolve("Game Theory", entity_type="concept", category="strategy")
        registry.register_alias("compound-interest", "Compounding")
        registry.save()

        # Load into a new registry
        new_registry = EntityRegistry(registry_path=registry_path)
        new_registry.load()
        assert new_registry.entity_count == 2

        # Alias should still work
        slug = new_registry.resolve("Compounding")
        assert slug == "compound-interest"

    def test_get_page_path(self, registry, tmp_path):
        registry.resolve("Compound Interest", entity_type="concept")
        registry.resolve("Charlie Munger", entity_type="person")
        vault_root = tmp_path / "vault"

        concept_path = registry.get_page_path("compound-interest", vault_root)
        assert concept_path == vault_root / "wiki" / "concepts" / "compound-interest.md"

        person_path = registry.get_page_path("charlie-munger", vault_root)
        assert person_path == vault_root / "wiki" / "entities" / "charlie-munger.md"

    def test_find_similar_entities(self, registry):
        registry.resolve("Compound Interest")
        registry.resolve("Second Order Thinking")

        similar = registry.find_similar_entities("Compound Interests")
        assert len(similar) >= 1
        assert similar[0][0] == "compound-interest"
        # Combined Levenshtein + Jaccard score - moderate similarity
        assert similar[0][1] > 0.5

    def test_load_nonexistent_file(self, tmp_path):
        reg = EntityRegistry(registry_path=tmp_path / "nonexistent.json")
        reg.load()
        assert reg.entity_count == 0

    def test_load_corrupt_file(self, tmp_path):
        path = tmp_path / "corrupt.json"
        path.write_text("not valid json{{{", encoding="utf-8")
        reg = EntityRegistry(registry_path=path)
        reg.load()
        assert reg.entity_count == 0
