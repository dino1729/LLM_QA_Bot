import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from helper_functions.weather_utils import get_weather

logger = logging.getLogger(__name__)

BUNDLE_SCHEMA_VERSION = "1.0.0"
DEFAULT_OUTPUT_DIR = Path("newsletter_research_data")


def compute_quarter_progress() -> Dict[str, Any]:
    """
    Compute current fiscal quarter progress based on NVIDIA earnings dates.
    Returns dict with quarter number, days completed/left, and percent complete.
    """
    now = datetime.now()
    current_year = now.year

    earnings_dates = [
        datetime(current_year, 1, 23),
        datetime(current_year, 4, 25),
        datetime(current_year, 7, 29),
        datetime(current_year, 10, 24),
        datetime(current_year + 1, 1, 23),
    ]

    current_quarter = None
    start_of_quarter = None
    end_of_quarter = None

    for i in range(len(earnings_dates) - 1):
        if earnings_dates[i] <= now < earnings_dates[i + 1]:
            current_quarter = i + 1
            start_of_quarter = earnings_dates[i]
            end_of_quarter = earnings_dates[i + 1]
            break

    if current_quarter is None:
        current_quarter = 4
        start_of_quarter = earnings_dates[3]
        end_of_quarter = earnings_dates[4]
        if now < earnings_dates[0]:
            current_quarter = 1
            start_of_quarter = datetime(current_year - 1, 10, 24)
            end_of_quarter = earnings_dates[0]

    days_in_quarter = (end_of_quarter - start_of_quarter).days
    days_completed_in_quarter = (now - start_of_quarter).days
    if days_in_quarter == 0:
        days_in_quarter = 1
    days_left_in_quarter = days_in_quarter - days_completed_in_quarter
    percent_complete = (days_completed_in_quarter / days_in_quarter) * 100

    return {
        "current_quarter": current_quarter,
        "days_in_quarter": days_in_quarter,
        "days_completed_in_quarter": days_completed_in_quarter,
        "days_left_in_quarter": days_left_in_quarter,
        "percent_complete": round(percent_complete, 2),
    }


def build_daily_bundle(
    days_completed: int,
    weeks_completed: float,
    days_left: int,
    weeks_left: float,
    percent_days_left: float,
    weather_data: Dict[str, Any],
    quote_text: str,
    quote_author: str,
    lesson_dict: Dict[str, Any],
    news_raw_sources: Dict[str, str],
    newsletter_sections: Dict[str, Any],
    voicebot_script: str,
) -> Dict[str, Any]:
    """
    Build the complete daily bundle dictionary conforming to the schema.
    """
    now = datetime.now()
    current_year = now.year
    total_days_in_year = (
        366
        if (current_year % 4 == 0 and current_year % 100 != 0)
        or (current_year % 400 == 0)
        else 365
    )

    quarter_data = compute_quarter_progress()

    bundle = {
        "meta": {
            "schema_version": BUNDLE_SCHEMA_VERSION,
            "date_iso": now.strftime("%Y-%m-%d"),
            "date_formatted": now.strftime("%B %d, %Y"),
            "day_of_week": now.strftime("%A"),
            "timestamp": now.isoformat(),
        },
        "progress": {
            "time": {
                "days_completed": days_completed,
                "weeks_completed": round(weeks_completed, 1),
                "days_left": days_left,
                "weeks_left": round(weeks_left, 1),
                "percent_complete": round(100 - percent_days_left, 2),
                "percent_days_left": round(percent_days_left, 2),
                "year": current_year,
                "total_days_in_year": total_days_in_year,
            },
            "quarter": quarter_data,
            "weather": weather_data,
            "quote": {
                "text": quote_text,
                "author": quote_author,
            },
            "lesson": lesson_dict,
        },
        "news": {
            "raw_sources": news_raw_sources,
            "newsletter": {
                "sections": newsletter_sections,
                "voicebot_script": voicebot_script,
            },
        },
    }

    return bundle


def write_bundle_json(bundle: Dict[str, Any], output_dir: Path = DEFAULT_OUTPUT_DIR) -> Path:
    """
    Write the bundle to both a dated file and a 'latest' file.
    Returns the path to the dated file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    date_str = bundle["meta"]["date_iso"]
    dated_path = output_dir / f"daily_bundle_{date_str}.json"
    latest_path = output_dir / "daily_bundle_latest.json"

    for path in (dated_path, latest_path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(bundle, f, indent=2, ensure_ascii=False)
        logger.info("Bundle JSON written to %s", path)

    return dated_path


def load_bundle_json(path: Path) -> Dict[str, Any]:
    """Load a bundle JSON from disk."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def time_left_in_year():
    today = datetime.now()
    end_of_year = datetime(today.year, 12, 31)
    days_completed = today.timetuple().tm_yday
    weeks_completed = days_completed / 7
    delta = end_of_year - today
    days_left = delta.days + 1
    weeks_left = days_left / 7
    total_days_in_year = (
        366
        if (today.year % 4 == 0 and today.year % 100 != 0)
        or (today.year % 400 == 0)
        else 365
    )
    percent_days_left = (days_left / total_days_in_year) * 100

    return days_completed, weeks_completed, days_left, weeks_left, percent_days_left


def save_to_output_dir(
    content: str, filename: str, output_dir: Path = DEFAULT_OUTPUT_DIR
) -> Path:
    """
    Save content to a file in the output directory.
    Returns the path to the saved file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / filename

    if "/" in filename:
        filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

    logger.info("Saved to %s", filepath)
    return filepath


def generate_progress_message(
    days_completed, weeks_completed, days_left, weeks_left, percent_days_left
):
    """Return plain-text progress summary including weather details."""
    temp, status = get_weather()
    now = datetime.now()
    date_time = now.strftime("%B %d, %Y %H:%M:%S")
    current_year = now.year

    earnings_dates = [
        datetime(current_year, 1, 23),
        datetime(current_year, 4, 25),
        datetime(current_year, 7, 29),
        datetime(current_year, 10, 24),
        datetime(current_year + 1, 1, 23),
    ]

    current_quarter = None
    for i in range(len(earnings_dates) - 1):
        if earnings_dates[i] <= now < earnings_dates[i + 1]:
            current_quarter = i + 1
            start_of_quarter = earnings_dates[i]
            end_of_quarter = earnings_dates[i + 1]
            break

    if current_quarter is None:
        current_quarter = 4
        start_of_quarter = earnings_dates[3]
        end_of_quarter = earnings_dates[4]
        if now < earnings_dates[0]:
            current_quarter = 1
            start_of_quarter = datetime(current_year - 1, 10, 24)
            end_of_quarter = earnings_dates[0]

    days_in_quarter = (end_of_quarter - start_of_quarter).days
    days_completed_in_quarter = (now - start_of_quarter).days
    if days_in_quarter == 0:
        days_in_quarter = 1
    percent_days_left_in_quarter = (
        (days_in_quarter - days_completed_in_quarter) / days_in_quarter
    ) * 100

    progress_bar_full = "█"
    progress_bar_empty = "░"
    progress_bar_length = 20
    quarter_progress_filled_length = int(
        progress_bar_length * (100 - percent_days_left_in_quarter) / 100
    )
    quarter_progress_bar = (
        progress_bar_full * quarter_progress_filled_length
        + progress_bar_empty * (progress_bar_length - quarter_progress_filled_length)
    )

    progress_filled_length = int(progress_bar_length * (100 - percent_days_left) / 100)
    progress_bar = (
        progress_bar_full * progress_filled_length
        + progress_bar_empty * (progress_bar_length - progress_filled_length)
    )

    return f"""

    Year Progress Report

    Today's Date and Time: {date_time}
    Weather in North Plains, OR: {temp}°C, {status}

    Current Quarter: Q{current_quarter}

    Q{current_quarter} Progress: [{quarter_progress_bar}] {100 - percent_days_left_in_quarter:.2f}% completed
    Days left in Q{current_quarter}: {days_in_quarter - days_completed_in_quarter}

    Days completed in the year: {days_completed}
    Weeks completed in the year: {weeks_completed:.2f}

    Days left in the year: {days_left}
    Weeks left in the year: {weeks_left:.2f}
    Percentage of the year left: {percent_days_left:.2f}%

    Year Progress: [{progress_bar}] {100 - percent_days_left:.2f}% completed

    """
