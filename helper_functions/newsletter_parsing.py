from datetime import datetime
from typing import Any, Dict, List


def parse_newsletter_item(item_text: str) -> Dict[str, Any]:
    """
    Parse a single newsletter item from text format to structured dict.
    Expected format: "[Source] Headline | Date: MM/DD/YYYY | URL | Commentary: text"
    """
    result = {
        "source": "",
        "headline": "",
        "date_mmddyyyy": "",
        "url": "",
        "commentary": "",
    }

    if not item_text:
        return result

    item_text = item_text.strip()
    if item_text.startswith("- "):
        item_text = item_text[2:]

    if item_text.startswith("[") and "]" in item_text:
        end_bracket = item_text.index("]")
        result["source"] = item_text[1:end_bracket]
        item_text = item_text[end_bracket + 1 :].strip()

    if " | " in item_text:
        components = item_text.split(" | ")
        result["headline"] = components[0].strip()

        for component in components[1:]:
            component = component.strip()
            if component.startswith("Date:"):
                result["date_mmddyyyy"] = component.replace("Date:", "").strip()
            elif component.startswith("http"):
                result["url"] = component
            elif component.startswith("Commentary:"):
                result["commentary"] = component.replace("Commentary:", "").strip()
    else:
        result["headline"] = item_text

    return result


def parse_newsletter_text_to_sections(newsletter_text: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Parse the full newsletter text into structured sections dict.
    Returns dict with keys: tech, financial, india
    Each value is a list of item dicts.
    """
    sections = {"tech": [], "financial": [], "india": []}

    if not newsletter_text:
        return sections

    section_map = {
        "tech news update": "tech",
        "financial markets news update": "financial",
        "india news update": "india",
    }

    parts = newsletter_text.split("##")

    for part in parts:
        part = part.strip()
        if not part:
            continue

        section_key = None
        for header, key in section_map.items():
            if header in part.lower():
                section_key = key
                break

        if not section_key:
            continue

        lines = part.split("\n")
        for line in lines:
            line = line.strip()
            if line.startswith("- "):
                item = parse_newsletter_item(line)
                if item["headline"]:
                    sections[section_key].append(item)

    return sections


def generate_fallback_newsletter_sections() -> Dict[str, List[Dict[str, Any]]]:
    """Generate fallback newsletter sections when LLM generation fails."""
    today = datetime.now().strftime("%m/%d/%Y")

    fallback_item = {
        "source": "Update",
        "headline": "News update temporarily unavailable. Check back later for the latest updates.",
        "date_mmddyyyy": today,
        "url": "",
        "commentary": "We're working to bring you the latest news. Please check back soon.",
    }

    return {
        "tech": [fallback_item],
        "financial": [fallback_item],
        "india": [fallback_item],
    }
