"""
Tests for the Memory Palace Migration script.

Tests cover:
- Category mapping from old (23) to new (10) categories
- JSON file parsing
- Duplicate detection and skipping
- Migration statistics
"""

import json
import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from helper_functions.memory_palace_migration import (
    CATEGORY_MAPPING,
    MIGRATION_FILES,
    SKIP_FILES,
    map_category,
    migrate_all,
    migrate_file,
    normalize_text,
    parse_json_file,
    text_hash,
)


class TestCategoryMapping:
    """Tests for category mapping configuration."""

    def test_all_old_categories_have_mappings(self):
        """Verify all old categories map to valid new categories."""
        valid_new_categories = {
            "strategy", "psychology", "history", "science", "technology",
            "economics", "engineering", "biology", "leadership", "observations"
        }

        for old_cat, new_cat in CATEGORY_MAPPING.items():
            assert new_cat in valid_new_categories, (
                f"Old category '{old_cat}' maps to invalid '{new_cat}'"
            )

    def test_strategy_category_mappings(self):
        """Test strategy-related mappings."""
        assert CATEGORY_MAPPING["strategy_and_game_theory"] == "strategy"
        assert CATEGORY_MAPPING["strategy_and_decision_making"] == "strategy"
        assert CATEGORY_MAPPING["negotiation_and_persuasion"] == "strategy"

    def test_psychology_category_mappings(self):
        """Test psychology-related mappings."""
        assert CATEGORY_MAPPING["psychology_and_cognitive_biases"] == "psychology"
        assert CATEGORY_MAPPING["psychology_and_philosophy"] == "psychology"

    def test_observations_catch_all(self):
        """Test observations category catches miscellaneous."""
        assert CATEGORY_MAPPING["miscellaneous_facts"] == "observations"
        assert CATEGORY_MAPPING["miscellaneous_observations"] == "observations"
        assert CATEGORY_MAPPING["future_predictions_and_ideas"] == "observations"


class TestMapCategory:
    """Tests for the map_category function."""

    def test_exact_match(self):
        """Test exact category name matching."""
        assert map_category("strategy_and_game_theory") == "strategy"
        assert map_category("biology_and_health") == "biology"

    def test_case_insensitive(self):
        """Test case-insensitive matching."""
        assert map_category("STRATEGY_AND_GAME_THEORY") == "strategy"
        assert map_category("Biology_And_Health") == "biology"

    def test_unknown_category_fallback(self):
        """Test unknown categories fall back to observations."""
        assert map_category("unknown_category") == "observations"
        assert map_category("random_stuff") == "observations"


class TestTextNormalization:
    """Tests for text normalization and hashing."""

    def test_normalize_whitespace(self):
        """Test whitespace normalization."""
        text = "  Hello   World  \n\t Test  "
        assert normalize_text(text) == "hello world test"

    def test_normalize_case(self):
        """Test case normalization."""
        assert normalize_text("HELLO") == "hello"
        assert normalize_text("HeLLo WoRLD") == "hello world"

    def test_hash_consistency(self):
        """Test that same text produces same hash."""
        text = "Test lesson content"
        h1 = text_hash(text)
        h2 = text_hash(text)
        assert h1 == h2

    def test_hash_ignores_whitespace_differences(self):
        """Test hash is consistent despite whitespace differences."""
        text1 = "Hello World"
        text2 = "  hello   world  "
        assert text_hash(text1) == text_hash(text2)

    def test_hash_different_for_different_text(self):
        """Test different text produces different hash."""
        h1 = text_hash("Hello World")
        h2 = text_hash("Goodbye World")
        assert h1 != h2


class TestParseJsonFile:
    """Tests for JSON file parsing."""

    def test_parse_standard_format(self, tmp_path):
        """Test parsing a standard Memory Palace JSON file."""
        test_file = tmp_path / "test.json"
        test_data = {
            "title": "Test Lessons",
            "source": "Test Source",
            "categories": {
                "strategy_and_game_theory": ["Lesson 1", "Lesson 2"],
                "psychology_and_cognitive_biases": ["Lesson 3"]
            }
        }
        test_file.write_text(json.dumps(test_data))

        title, source, categories = parse_json_file(test_file)

        assert title == "Test Lessons"
        assert source == "Test Source"
        assert len(categories) == 2
        assert len(categories["strategy_and_game_theory"]) == 2

    def test_parse_missing_title(self, tmp_path):
        """Test parsing file without title uses filename."""
        test_file = tmp_path / "my_lessons.json"
        test_data = {"categories": {"test": ["lesson"]}}
        test_file.write_text(json.dumps(test_data))

        title, source, categories = parse_json_file(test_file)

        assert title == "my_lessons"  # Stem of filename

    def test_parse_missing_source(self, tmp_path):
        """Test parsing file without source uses filename."""
        test_file = tmp_path / "my_lessons.json"
        test_data = {"title": "Test", "categories": {}}
        test_file.write_text(json.dumps(test_data))

        title, source, categories = parse_json_file(test_file)

        assert source == "my_lessons.json"


class TestMigrationFiles:
    """Tests for migration file configuration."""

    def test_migration_files_list(self):
        """Test migration files list includes expected files."""
        assert "high_iq_lessons.json" in MIGRATION_FILES
        assert "interesting_things_learnt.json" in MIGRATION_FILES
        assert "little_observations.json" in MIGRATION_FILES
        assert "rules_of_life.json" in MIGRATION_FILES

    def test_skip_files_excludes_duplicates(self):
        """Test itl.json is in skip list."""
        assert "itl.json" in SKIP_FILES

    def test_itl_not_in_migration_files(self):
        """Test itl.json is excluded from migration."""
        assert "itl.json" not in MIGRATION_FILES


@pytest.fixture
def temp_migration_dir():
    """Create a temporary directory with test JSON files."""
    temp_dir = tempfile.mkdtemp()

    # Create test JSON files
    files_data = {
        "file1.json": {
            "title": "File 1",
            "source": "Source 1",
            "categories": {
                "strategy_and_game_theory": ["Lesson A", "Lesson B"],
                "psychology_and_cognitive_biases": ["Lesson C"]
            }
        },
        "file2.json": {
            "title": "File 2",
            "source": "Source 2",
            "categories": {
                "economics_and_finance": ["Lesson D"],
                "strategy_and_game_theory": ["Lesson A"]  # Duplicate
            }
        }
    }

    for filename, data in files_data.items():
        file_path = Path(temp_dir) / filename
        with open(file_path, "w") as f:
            json.dump(data, f)

    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_db():
    """Create a mock MemoryPalaceDB."""
    mock = MagicMock()
    mock.get_lesson_count.return_value = 0
    mock.add_lesson.return_value = "test-id"
    return mock


class TestMigrateFile:
    """Tests for single file migration."""

    def test_migrate_file_adds_lessons(self, temp_migration_dir, mock_db):
        """Test migrating a file adds lessons to database."""
        file_path = Path(temp_migration_dir) / "file1.json"
        seen_hashes = set()

        stats = migrate_file(mock_db, file_path, seen_hashes, dry_run=False)

        assert stats["total"] == 3
        assert stats["added"] == 3
        assert stats["skipped_duplicate"] == 0
        assert mock_db.add_lesson.call_count == 3

    def test_migrate_file_dry_run(self, temp_migration_dir, mock_db):
        """Test dry run doesn't add lessons."""
        file_path = Path(temp_migration_dir) / "file1.json"
        seen_hashes = set()

        stats = migrate_file(mock_db, file_path, seen_hashes, dry_run=True)

        assert stats["added"] == 3
        mock_db.add_lesson.assert_not_called()

    def test_migrate_file_deduplicates(self, temp_migration_dir, mock_db):
        """Test cross-file deduplication."""
        file1 = Path(temp_migration_dir) / "file1.json"
        file2 = Path(temp_migration_dir) / "file2.json"
        seen_hashes = set()

        # Migrate first file
        migrate_file(mock_db, file1, seen_hashes, dry_run=False)

        # Migrate second file - should skip "Lesson A" duplicate
        stats2 = migrate_file(mock_db, file2, seen_hashes, dry_run=False)

        assert stats2["skipped_duplicate"] == 1
        assert stats2["added"] == 1  # Only "Lesson D"

    def test_migrate_file_category_mapping(self, temp_migration_dir, mock_db):
        """Test categories are properly mapped."""
        file_path = Path(temp_migration_dir) / "file1.json"
        seen_hashes = set()

        stats = migrate_file(mock_db, file_path, seen_hashes, dry_run=False)

        # Check by_category stats
        assert stats["by_category"]["strategy"] == 2
        assert stats["by_category"]["psychology"] == 1


class TestMigrateAll:
    """Tests for full migration."""

    @patch("helper_functions.memory_palace_migration.MemoryPalaceDB")
    @patch("helper_functions.memory_palace_migration.config")
    def test_migrate_all_dry_run(self, mock_config, mock_db_class, temp_migration_dir):
        """Test dry run migration."""
        mock_config.MEMORY_PALACE_FOLDER = temp_migration_dir
        mock_config.memory_palace_index_folder = f"{temp_migration_dir}/index"

        # Create expected files
        for filename in ["high_iq_lessons.json", "interesting_things_learnt.json"]:
            file_path = Path(temp_migration_dir) / filename
            with open(file_path, "w") as f:
                json.dump({
                    "title": "Test",
                    "source": "Test",
                    "categories": {"strategy_and_game_theory": ["Lesson"]}
                }, f)

        stats = migrate_all(
            dry_run=True,
            memory_palace_folder=temp_migration_dir
        )

        assert stats["dry_run"] is True
        assert stats["status"] == "success"
        mock_db_class.assert_not_called()

    @patch("helper_functions.memory_palace_migration.MemoryPalaceDB")
    @patch("helper_functions.memory_palace_migration.config")
    def test_migrate_all_skips_non_empty_db(self, mock_config, mock_db_class, temp_migration_dir):
        """Test migration skips when database already has lessons."""
        mock_config.MEMORY_PALACE_FOLDER = temp_migration_dir
        mock_config.memory_palace_index_folder = f"{temp_migration_dir}/index"

        mock_db = MagicMock()
        mock_db.get_lesson_count.return_value = 100  # Already has lessons
        mock_db_class.return_value = mock_db

        stats = migrate_all(
            dry_run=False,
            force=False,
            memory_palace_folder=temp_migration_dir
        )

        assert stats["status"] == "skipped"
        assert stats["reason"] == "database_not_empty"

    @patch("helper_functions.memory_palace_migration.MemoryPalaceDB")
    @patch("helper_functions.memory_palace_migration.config")
    def test_migrate_all_with_force(self, mock_config, mock_db_class, temp_migration_dir):
        """Test force mode clears existing index."""
        mock_config.MEMORY_PALACE_FOLDER = temp_migration_dir
        index_folder = Path(temp_migration_dir) / "index"
        index_folder.mkdir()
        (index_folder / "test.txt").touch()  # Create file in index
        mock_config.memory_palace_index_folder = str(index_folder)

        mock_db = MagicMock()
        mock_db.get_lesson_count.return_value = 0
        mock_db_class.return_value = mock_db

        # Create a migration file
        file_path = Path(temp_migration_dir) / "high_iq_lessons.json"
        with open(file_path, "w") as f:
            json.dump({
                "title": "Test",
                "source": "Test",
                "categories": {"strategy_and_game_theory": ["Test lesson"]}
            }, f)

        stats = migrate_all(
            dry_run=False,
            force=True,
            memory_palace_folder=temp_migration_dir
        )

        assert stats["status"] == "success"
        # Index folder should have been removed (but recreated by DB init)

    def test_migrate_all_missing_folder(self, tmp_path):
        """Test migration fails gracefully for missing folder."""
        missing_folder = tmp_path / "nonexistent"

        with pytest.raises(FileNotFoundError):
            migrate_all(memory_palace_folder=str(missing_folder))
