"""
Wiki configuration loader.
Extends the existing config.py pattern with wiki-specific settings.
"""
from config.config import config_yaml

# Wiki Configuration
_wiki = config_yaml.get("wiki", {})
wiki_enabled = _wiki.get("enabled", False)
wiki_vault_path = _wiki.get("vault_path", "./vault")

# LLM Council model names (as registered in LiteLLM proxy)
_council = _wiki.get("council", {})
wiki_entity_extractor_model = _council.get("entity_extractor_model", "gemini-3.1-pro")
wiki_prose_synthesizer_model = _council.get("prose_synthesizer_model", "gpt-5.4-mini")
wiki_cross_connector_model = _council.get("cross_connector_model", "glm-5")
wiki_contradiction_finder_model = _council.get("contradiction_finder_model", "kimi-2.5")
wiki_chairman_model = _council.get("chairman_model", "claude-opus-4-6")
wiki_migration_council_mode = _council.get("migration_council_mode", "migration")
wiki_lint_council_mode = _council.get("lint_council_mode", "full")

# Rate limits per model {model_name: {requests_per_minute, tokens_per_minute}}
wiki_rate_limits = _wiki.get("rate_limits", {})

# Auto-file threshold: min source citations for EDITH answers to auto-file
wiki_auto_file_citation_threshold = _wiki.get("auto_file_citation_threshold", 3)

# Migration settings
_migration = _wiki.get("migration", {})
wiki_migration_lessons_batch_size = _migration.get("lessons_batch_size", 10)
wiki_migration_archive_batch_size = _migration.get("archive_batch_size", 5)
wiki_migration_link_memory_batch_size = _migration.get("link_memory_batch_size", 5)

# Lint settings
_lint = _wiki.get("lint", {})
wiki_lint_enabled = _lint.get("enabled", True)
wiki_lint_run_with_newsletter_cron = _lint.get("run_with_newsletter_cron", True)
wiki_lint_orphan_page_threshold_days = _lint.get("orphan_page_threshold_days", 30)
wiki_lint_min_sources_high_confidence = _lint.get("min_sources_for_high_confidence", 5)
wiki_lint_min_sources_medium_confidence = _lint.get("min_sources_for_medium_confidence", 2)

# Backup settings
_backup = _wiki.get("backup", {})
wiki_backup_enabled = _backup.get("enabled", True)
wiki_backup_dir = _backup.get("backup_dir", "./memory_palace/backups/wiki")
wiki_backup_keep_last_n = _backup.get("keep_last_n", 30)

# Telegram notification settings
wiki_telegram_notify_on_auto_file = _wiki.get("telegram_notify_on_auto_file", True)
wiki_telegram_notify_on_lint_issues = _wiki.get("telegram_notify_on_lint_issues", True)
