import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = Path("newsletter_research_data")
CACHE_MAX_AGE_HOURS = 6
NEWS_CACHE_FILENAME = "news_cache.json"


def _cache_path(output_dir: Path = DEFAULT_OUTPUT_DIR) -> Path:
    return output_dir / NEWS_CACHE_FILENAME


def save_news_cache(news_data: Dict[str, str], output_dir: Path = DEFAULT_OUTPUT_DIR) -> bool:
    """
    Save news data to cache with timestamp.
    Returns True if successful.
    """
    cache_file = _cache_path(output_dir)
    cache_file.parent.mkdir(parents=True, exist_ok=True)

    cache_entry = {
        "timestamp": datetime.now().isoformat(),
        "date_iso": datetime.now().strftime("%Y-%m-%d"),
        "news": news_data,
    }

    try:
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(cache_entry, f, indent=2, ensure_ascii=False)
        logger.info("News cache saved to %s", cache_file)
        return True
    except Exception as e:
        logger.error("Failed to save news cache: %s", e)
        return False


def load_news_cache(
    output_dir: Path = DEFAULT_OUTPUT_DIR, max_age_hours: int = CACHE_MAX_AGE_HOURS
) -> Optional[Dict[str, str]]:
    """
    Load news data from cache if available and fresh.
    Returns None if cache is missing, expired, or invalid.
    """
    cache_file = _cache_path(output_dir)
    if not cache_file.exists():
        logger.info("No news cache found")
        return None

    try:
        with open(cache_file, "r", encoding="utf-8") as f:
            cache_entry = json.load(f)

        cache_time = datetime.fromisoformat(cache_entry["timestamp"])
        age_hours = (datetime.now() - cache_time).total_seconds() / 3600

        if age_hours > max_age_hours:
            logger.info(
                "News cache expired (%.1f hours old, max is %s)", age_hours, max_age_hours
            )
            return None

        if cache_entry.get("date_iso") != datetime.now().strftime("%Y-%m-%d"):
            logger.info("News cache is from a different day")
            return None

        logger.info("Loaded news cache (%.1f hours old)", age_hours)
        return cache_entry["news"]

    except Exception as e:
        logger.warning("Failed to load news cache: %s", e)
        return None


def is_cache_valid(
    output_dir: Path = DEFAULT_OUTPUT_DIR, max_age_hours: int = CACHE_MAX_AGE_HOURS
) -> bool:
    """Check if cache exists and is still valid."""
    return load_news_cache(output_dir, max_age_hours) is not None


def get_cache_info(
    output_dir: Path = DEFAULT_OUTPUT_DIR, max_age_hours: int = CACHE_MAX_AGE_HOURS
) -> Optional[Dict[str, Any]]:
    """
    Get information about the cache without loading full data.
    """
    cache_file = _cache_path(output_dir)
    if not cache_file.exists():
        return None

    try:
        with open(cache_file, "r", encoding="utf-8") as f:
            cache_entry = json.load(f)

        cache_time = datetime.fromisoformat(cache_entry["timestamp"])
        age_hours = (datetime.now() - cache_time).total_seconds() / 3600

        return {
            "exists": True,
            "timestamp": cache_entry["timestamp"],
            "date_iso": cache_entry.get("date_iso"),
            "age_hours": round(age_hours, 2),
            "has_tech": "technology" in cache_entry.get("news", {}),
            "has_financial": "financial" in cache_entry.get("news", {}),
            "has_india": "india" in cache_entry.get("news", {}),
            "cache_file": str(cache_file),
            "is_valid": age_hours < max_age_hours
            and cache_entry.get("date_iso") == datetime.now().strftime("%Y-%m-%d"),
        }
    except Exception as e:
        return {"exists": True, "error": str(e)}
