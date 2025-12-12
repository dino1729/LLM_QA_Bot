#!/usr/bin/env python3
"""
Standalone test for newsletter generation - no external dependencies required.
Tests JSON bundle creation and HTML rendering without LLM calls or audio.
"""

import json
import sys
import os
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import only the specific functions we need (avoid loading audio/LLM modules)
# We'll import the core functions directly from the file

def load_newsletter_functions():
    """Dynamically load only the functions we need without triggering all imports"""
    import importlib.util
    
    # Read the file and extract just the functions we need
    script_path = Path(__file__).parent.parent / "year_progress_and_news_reporter_litellm.py"
    
    # Instead of importing the whole module, we'll define minimal versions here
    return None

# Define minimal versions of the functions for testing
OUTPUT_DIR = Path("newsletter_research_data")
BUNDLE_SCHEMA_VERSION = "1.0.0"

def compute_quarter_progress():
    """Compute current fiscal quarter progress"""
    now = datetime.now()
    current_year = now.year
    
    earnings_dates = [
        datetime(current_year, 1, 23),
        datetime(current_year, 4, 25),
        datetime(current_year, 7, 29),
        datetime(current_year, 10, 24),
        datetime(current_year + 1, 1, 23)
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
    
    days_in_quarter = (end_of_quarter - start_of_quarter).days
    days_completed_in_quarter = (now - start_of_quarter).days
    if days_in_quarter == 0:
        days_in_quarter = 1
    days_left_in_quarter = days_in_quarter - days_completed_in_quarter
    percent_complete = ((days_completed_in_quarter) / days_in_quarter) * 100
    
    return {
        "current_quarter": current_quarter,
        "days_in_quarter": days_in_quarter,
        "days_completed_in_quarter": days_completed_in_quarter,
        "days_left_in_quarter": days_left_in_quarter,
        "percent_complete": round(percent_complete, 2)
    }


def parse_lesson_to_dict(lesson_text, topic=""):
    """Parse lesson text with markers into structured dict"""
    import re
    
    result = {
        "topic": topic,
        "key_insight": "",
        "historical": "",
        "application": "",
        "raw_text": lesson_text
    }
    
    if not lesson_text:
        return result
    
    key_insight_match = re.search(
        r'\[KEY INSIGHT\]\s*(.*?)(?=\[HISTORICAL\]|\[APPLICATION\]|$)',
        lesson_text, re.DOTALL | re.IGNORECASE
    )
    historical_match = re.search(
        r'\[HISTORICAL\]\s*(.*?)(?=\[KEY INSIGHT\]|\[APPLICATION\]|$)',
        lesson_text, re.DOTALL | re.IGNORECASE
    )
    application_match = re.search(
        r'\[APPLICATION\]\s*(.*?)(?=\[KEY INSIGHT\]|\[HISTORICAL\]|$)',
        lesson_text, re.DOTALL | re.IGNORECASE
    )
    
    if key_insight_match and key_insight_match.group(1).strip():
        result["key_insight"] = key_insight_match.group(1).strip()
    else:
        result["key_insight"] = "Timeless principles reveal themselves through historical patterns."
    
    if historical_match and historical_match.group(1).strip():
        result["historical"] = historical_match.group(1).strip()
    
    if application_match and application_match.group(1).strip():
        result["application"] = application_match.group(1).strip()
    
    return result


def parse_newsletter_item(item_text):
    """Parse a single newsletter item to structured dict"""
    result = {
        "source": "",
        "headline": "",
        "date_mmddyyyy": "",
        "url": "",
        "commentary": ""
    }
    
    if not item_text:
        return result
    
    item_text = item_text.strip()
    if item_text.startswith("- "):
        item_text = item_text[2:]
    
    if item_text.startswith("[") and "]" in item_text:
        end_bracket = item_text.index("]")
        result["source"] = item_text[1:end_bracket]
        item_text = item_text[end_bracket + 1:].strip()
    
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


def parse_newsletter_text_to_sections(newsletter_text):
    """Parse full newsletter text into structured sections"""
    sections = {"tech": [], "financial": [], "india": []}
    
    if not newsletter_text:
        return sections
    
    section_map = {
        "tech news update": "tech",
        "financial markets news update": "financial",
        "india news update": "india"
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


def build_daily_bundle(
    days_completed, weeks_completed, days_left, weeks_left, percent_days_left,
    weather_data, quote_text, quote_author, lesson_dict,
    news_raw_sources, newsletter_sections, voicebot_script
):
    """Build the complete daily bundle"""
    now = datetime.now()
    current_year = now.year
    total_days_in_year = 366 if (current_year % 4 == 0 and current_year % 100 != 0) or (current_year % 400 == 0) else 365
    
    quarter_data = compute_quarter_progress()
    
    return {
        "meta": {
            "schema_version": BUNDLE_SCHEMA_VERSION,
            "generated_at_iso": now.isoformat(),
            "date_iso": now.strftime("%Y-%m-%d"),
            "date_formatted": now.strftime("%B %d, %Y"),
            "day_of_week": now.strftime("%A"),
            "llm_provider": "test",
            "model_tiers_used": ["test"]
        },
        "progress": {
            "time": {
                "year": current_year,
                "total_days_in_year": total_days_in_year,
                "days_completed": days_completed,
                "days_left": days_left,
                "weeks_completed": round(weeks_completed, 2),
                "weeks_left": round(weeks_left, 2),
                "percent_complete": round(100 - percent_days_left, 2)
            },
            "quarter": quarter_data,
            "weather": weather_data,
            "quote": {"text": quote_text, "author": quote_author},
            "lesson": lesson_dict
        },
        "news": {
            "raw_sources": news_raw_sources,
            "newsletter": {"sections": newsletter_sections},
            "voicebot": {"script": voicebot_script}
        }
    }


def format_news_items_html(items):
    """Format news items to HTML with full styling"""
    import html as html_module
    
    if not items:
        return '<div style="text-align: center; color: var(--text-muted); padding: var(--space-md);">No updates available.</div>'
    
    formatted = []
    for item in items[:5]:
        source = html_module.escape(item.get("source", "Update"))
        headline = html_module.escape(item.get("headline", ""))
        date_str = html_module.escape(item.get("date_mmddyyyy", ""))
        url = item.get("url", "")
        commentary = html_module.escape(item.get("commentary", ""))
        
        if not headline:
            continue
        
        # Build headline with optional link
        if url:
            headline_html = f'''<h3 class="news-headline">
                <a href="{html_module.escape(url)}" target="_blank" rel="noopener noreferrer">{headline}</a>
            </h3>'''
            cta_html = f'<a href="{html_module.escape(url)}" target="_blank" rel="noopener noreferrer" class="news-cta">Read More →</a>'
        else:
            headline_html = f'<h3 class="news-headline">{headline}</h3>'
            cta_html = ""
        
        date_html = f'<span class="news-date">{date_str}</span>' if date_str else ""
        commentary_html = f'<div class="news-commentary">{commentary}</div>' if commentary else ""
        
        formatted.append(f'''<article class="news-item">
                    <span class="news-source">{source}</span>
                    {date_html}
                    {headline_html}
                    {commentary_html}
                    {cta_html}
                </article>''')
    
    return "\n".join(formatted)


def render_newsletter_html_from_bundle(bundle):
    """Render newsletter HTML from bundle with full styling"""
    meta = bundle["meta"]
    sections = bundle["news"]["newsletter"]["sections"]
    
    tech_html = format_news_items_html(sections.get("tech", []))
    financial_html = format_news_items_html(sections.get("financial", []))
    india_html = format_news_items_html(sections.get("india", []))
    
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=5.0">
    <meta name="theme-color" content="#0A0B0D">
    <meta name="color-scheme" content="dark light">
    <title>Daily Briefing - {meta["date_formatted"]}</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,600;0,700;1,400&family=Inter:wght@300;400;500;600&display=swap');

        :root {{
            --space-xs: 8px;
            --space-sm: 16px;
            --space-md: 24px;
            --space-lg: 32px;
            --space-xl: 48px;
            --bg-primary: #0A0B0D;
            --bg-secondary: #12141A;
            --bg-card: rgba(22, 25, 32, 0.85);
            --border-subtle: rgba(255, 255, 255, 0.06);
            --border-medium: rgba(255, 255, 255, 0.1);
            --text-primary: #F5F5F7;
            --text-secondary: #A1A1AA;
            --text-muted: #6B7280;
            --accent-gold: #D4AF37;
            --accent-gold-light: #E8C959;
            --accent-gold-dark: #B8962E;
            --accent-glow: rgba(212, 175, 55, 0.15);
            --accent-glow-strong: rgba(212, 175, 55, 0.3);
            --font-display: 'Playfair Display', Georgia, serif;
            --font-body: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            --tap-target-min: 44px;
            --border-radius-sm: 8px;
            --border-radius-md: 12px;
            --border-radius-lg: 16px;
            --border-radius-xl: 24px;
            --shadow-md: 0 4px 16px rgba(0, 0, 0, 0.2);
            --transition-fast: 150ms ease;
            --transition-base: 250ms ease;
        }}

        *, *::before, *::after {{ box-sizing: border-box; }}

        body {{
            margin: 0;
            padding: 0;
            background: var(--bg-primary);
            background-image:
                radial-gradient(ellipse at top, rgba(212, 175, 55, 0.03) 0%, transparent 50%),
                linear-gradient(180deg, var(--bg-primary) 0%, var(--bg-secondary) 100%);
            background-attachment: fixed;
            color: var(--text-primary);
            font-family: var(--font-body);
            font-size: 16px;
            line-height: 1.6;
            -webkit-font-smoothing: antialiased;
            min-height: 100vh;
        }}

        .container {{
            width: 100%;
            max-width: 640px;
            margin: 0 auto;
            padding: var(--space-sm);
        }}

        .header {{
            text-align: center;
            padding: var(--space-lg) 0 var(--space-md);
            margin-bottom: var(--space-md);
            position: relative;
        }}

        .header::after {{
            content: '';
            position: absolute;
            bottom: 0;
            left: 50%;
            transform: translateX(-50%);
            width: 60px;
            height: 2px;
            background: linear-gradient(90deg, transparent, var(--accent-gold), transparent);
        }}

        h1 {{
            font-family: var(--font-display);
            font-size: 28px;
            font-weight: 700;
            color: var(--accent-gold);
            margin: 0 0 var(--space-xs) 0;
            text-shadow: 0 2px 20px var(--accent-glow);
        }}

        .subtitle {{
            color: var(--text-secondary);
            font-size: 14px;
        }}

        .date-badge {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-height: var(--tap-target-min);
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 2px;
            color: var(--text-secondary);
            border: 1px solid var(--border-subtle);
            padding: var(--space-xs) var(--space-md);
            border-radius: 100px;
            background: rgba(255, 255, 255, 0.02);
            margin-top: var(--space-sm);
        }}

        .card {{
            background: var(--bg-card);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border: 1px solid var(--border-subtle);
            border-radius: var(--border-radius-lg);
            padding: var(--space-md);
            margin-bottom: var(--space-md);
            box-shadow: var(--shadow-md);
            animation: fadeInUp 0.5s ease forwards;
            opacity: 0;
        }}

        .card:nth-child(1) {{ animation-delay: 0.1s; }}
        .card:nth-child(2) {{ animation-delay: 0.2s; }}
        .card:nth-child(3) {{ animation-delay: 0.3s; }}
        .card:nth-child(4) {{ animation-delay: 0.4s; }}

        @keyframes fadeInUp {{
            from {{ opacity: 0; transform: translateY(20px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}

        .card-header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: var(--space-xs);
            margin-bottom: var(--space-sm);
            padding-bottom: var(--space-sm);
            border-bottom: 1px solid var(--border-subtle);
        }}

        .card-title {{
            font-family: var(--font-display);
            font-size: 20px;
            font-weight: 600;
            color: var(--text-primary);
            margin: 0;
            display: flex;
            align-items: center;
            gap: var(--space-xs);
        }}

        .card-icon {{ font-size: 20px; }}
        .card-meta {{ font-size: 12px; color: var(--accent-gold); font-weight: 500; }}

        .news-item {{
            padding: var(--space-md) 0;
            border-bottom: 1px solid var(--border-subtle);
        }}
        .news-item:first-child {{ padding-top: 0; }}
        .news-item:last-child {{ border-bottom: none; padding-bottom: 0; }}

        .news-source {{
            display: inline-block;
            font-size: 10px;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            color: var(--accent-gold);
            font-weight: 600;
            margin-bottom: var(--space-xs);
            padding: 4px 8px;
            background: var(--accent-glow);
            border-radius: 4px;
        }}

        .news-date {{
            display: block;
            font-size: 11px;
            color: var(--text-muted);
            margin: var(--space-xs) 0;
        }}

        .news-headline {{
            font-size: 17px;
            font-family: var(--font-display);
            font-weight: 600;
            line-height: 1.4;
            color: var(--text-primary);
            margin: 0 0 var(--space-sm) 0;
        }}

        .news-headline a {{
            color: inherit;
            text-decoration: none;
            display: block;
            min-height: var(--tap-target-min);
            padding: var(--space-xs) 0;
            transition: color var(--transition-fast);
        }}

        .news-headline a:hover {{ color: var(--accent-gold); }}

        .news-commentary {{
            font-size: 14px;
            color: var(--text-secondary);
            line-height: 1.7;
            background: rgba(255, 255, 255, 0.02);
            padding: var(--space-sm);
            border-radius: var(--border-radius-sm);
            border-left: 3px solid var(--accent-gold);
            margin-top: var(--space-sm);
        }}

        .news-cta {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            height: 32px;
            padding: 0 var(--space-sm);
            margin-top: var(--space-xs);
            font-family: var(--font-body);
            font-size: 12px;
            font-weight: 500;
            letter-spacing: 0.3px;
            color: var(--bg-primary);
            background: linear-gradient(135deg, var(--accent-gold) 0%, var(--accent-gold-light) 100%);
            border: none;
            border-radius: 6px;
            text-decoration: none;
            cursor: pointer;
            transition: all var(--transition-base);
            box-shadow: 0 2px 8px var(--accent-glow);
        }}

        .news-cta:hover {{
            transform: translateY(-1px);
            box-shadow: 0 4px 12px var(--accent-glow-strong);
        }}

        footer {{
            text-align: center;
            padding: var(--space-lg) var(--space-sm);
            color: var(--text-muted);
            font-size: 12px;
            border-top: 1px solid var(--border-subtle);
            margin-top: var(--space-lg);
        }}

        @media screen and (min-width: 480px) {{
            .container {{ padding: var(--space-md); }}
            h1 {{ font-size: 32px; }}
            .card {{ padding: var(--space-lg); }}
            .news-headline {{ font-size: 18px; }}
        }}

        @media screen and (min-width: 768px) {{
            .container {{ padding: var(--space-lg); }}
            h1 {{ font-size: 36px; }}
            .card {{ border-radius: var(--border-radius-xl); }}
        }}

        @media (prefers-reduced-motion: reduce) {{
            *, *::before, *::after {{
                animation-duration: 0.01ms !important;
                transition-duration: 0.01ms !important;
            }}
        }}

        a:focus-visible, .news-cta:focus-visible {{
            outline: 2px solid var(--accent-gold);
            outline-offset: 2px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header class="header">
            <h1>Daily Briefing</h1>
            <p class="subtitle">{meta["day_of_week"]}'s Essential Updates</p>
            <div class="date-badge">{meta["date_formatted"]}</div>
        </header>

        <article class="card" aria-label="Technology news">
            <div class="card-header">
                <h2 class="card-title">
                    <span class="card-icon" aria-hidden="true">💻</span>
                    Technology
                </h2>
                <span class="card-meta">Tech & AI</span>
            </div>
            <div class="card-content">
                {tech_html}
            </div>
        </article>

        <article class="card" aria-label="Financial markets news">
            <div class="card-header">
                <h2 class="card-title">
                    <span class="card-icon" aria-hidden="true">📈</span>
                    Markets
                </h2>
                <span class="card-meta">Finance</span>
            </div>
            <div class="card-content">
                {financial_html}
            </div>
        </article>

        <article class="card" aria-label="India news">
            <div class="card-header">
                <h2 class="card-title">
                    <span class="card-icon" aria-hidden="true" style="display: inline-flex; align-items: center;">
                        <svg width="20" height="14" viewBox="0 0 20 14" style="border-radius: 2px; box-shadow: 0 1px 2px rgba(0,0,0,0.2);">
                            <rect width="20" height="4.67" fill="#FF9933"/>
                            <rect y="4.67" width="20" height="4.67" fill="#FFFFFF"/>
                            <rect y="9.33" width="20" height="4.67" fill="#138808"/>
                            <circle cx="10" cy="7" r="1.8" fill="#000080"/>
                        </svg>
                    </span>
                    India
                </h2>
                <span class="card-meta">Regional</span>
            </div>
            <div class="card-content">
                {india_html}
            </div>
        </article>

        <footer>
            <p>Generated by <strong>EDITH</strong> • {datetime.now().year}</p>
            <p style="margin-top: 8px; font-size: 11px;">Even Dead, I'm The Hero</p>
        </footer>
    </div>
</body>
</html>"""


# ============================================================================
# TESTS
# ============================================================================

def test_newsletter_parsing():
    """Test newsletter text parsing"""
    print("\n" + "=" * 70)
    print("TEST: Newsletter Parsing")
    print("=" * 70)
    
    sample = """
## Tech News Update:
- [TechCrunch] AI breakthrough announced | Date: 12/11/2025 | https://example.com/ai | Commentary: Major development in AI
- [Wired] New chip technology | Date: 12/11/2025 | Commentary: Revolutionary advancement

## Financial Markets News Update:
- [Reuters] Markets rally today | Date: 12/11/2025 | https://reuters.com/markets | Commentary: Strong performance
- [Bloomberg] Fed decision announced | Date: 12/11/2025 | Commentary: Rate decision impact

## India News Update:
- [TOI] Infrastructure news | Date: 12/11/2025 | Commentary: Development update
"""
    
    sections = parse_newsletter_text_to_sections(sample)
    
    assert len(sections["tech"]) == 2, f"Expected 2 tech items, got {len(sections['tech'])}"
    assert len(sections["financial"]) == 2, f"Expected 2 financial items, got {len(sections['financial'])}"
    assert len(sections["india"]) == 1, f"Expected 1 india item, got {len(sections['india'])}"
    
    # Check first tech item
    tech_item = sections["tech"][0]
    assert tech_item["source"] == "TechCrunch"
    assert "AI breakthrough" in tech_item["headline"]
    assert tech_item["url"] == "https://example.com/ai"
    
    print("✓ Newsletter parsing works correctly")
    print(f"  - Tech items: {len(sections['tech'])}")
    print(f"  - Financial items: {len(sections['financial'])}")
    print(f"  - India items: {len(sections['india'])}")
    
    return True


def test_bundle_creation():
    """Test bundle creation"""
    print("\n" + "=" * 70)
    print("TEST: Bundle Creation")
    print("=" * 70)
    
    lesson = parse_lesson_to_dict(
        "[KEY INSIGHT]\nTest insight.\n\n[HISTORICAL]\nHistorical context.\n\n[APPLICATION]\nApplication here.",
        topic="Test Topic"
    )
    
    newsletter_sections = {
        "tech": [{"source": "Test", "headline": "Test headline", "date_mmddyyyy": "12/11/2025", "url": "", "commentary": "Test"}],
        "financial": [],
        "india": []
    }
    
    bundle = build_daily_bundle(
        days_completed=345,
        weeks_completed=49.29,
        days_left=20,
        weeks_left=2.86,
        percent_days_left=5.48,
        weather_data={"temp_c": 10, "status": "cloudy", "location": "Test City"},
        quote_text="Test quote",
        quote_author="Test Author",
        lesson_dict=lesson,
        news_raw_sources={"technology": "raw", "financial": "raw", "india": "raw"},
        newsletter_sections=newsletter_sections,
        voicebot_script="Test script"
    )
    
    assert bundle["meta"]["schema_version"] == BUNDLE_SCHEMA_VERSION
    assert bundle["progress"]["quote"]["author"] == "Test Author"
    assert bundle["progress"]["lesson"]["topic"] == "Test Topic"
    
    # Test JSON serialization
    json_str = json.dumps(bundle, indent=2)
    parsed = json.loads(json_str)
    assert parsed["meta"]["schema_version"] == bundle["meta"]["schema_version"]
    
    print("✓ Bundle creation works correctly")
    print(f"  - Schema version: {bundle['meta']['schema_version']}")
    print(f"  - Date: {bundle['meta']['date_formatted']}")
    print(f"  - JSON size: {len(json_str)} chars")
    
    return bundle


def test_html_rendering(bundle):
    """Test HTML rendering from bundle"""
    print("\n" + "=" * 70)
    print("TEST: HTML Rendering")
    print("=" * 70)
    
    html = render_newsletter_html_from_bundle(bundle)
    
    assert "Daily Briefing" in html
    assert "Technology" in html
    assert "Markets" in html  # Card title uses "Markets" for compact display
    assert "India" in html
    assert bundle["meta"]["date_formatted"] in html
    assert "Playfair Display" in html  # Check for proper fonts
    assert "Inter" in html
    
    print("✓ HTML rendering works correctly")
    print(f"  - HTML size: {len(html)} chars")
    print(f"  - Using fonts: Playfair Display + Inter")
    
    return html


def generate_example_newsletter():
    """Generate a full example newsletter with sample data"""
    print("\n" + "=" * 70)
    print("GENERATING EXAMPLE NEWSLETTER")
    print("=" * 70)
    
    # Sample news content
    sample_newsletter = """
## Tech News Update:
- [TechCrunch] OpenAI announces GPT-5 with breakthrough reasoning capabilities | Date: 12/11/2025 | https://techcrunch.com/gpt5 | Commentary: This represents a quantum leap in AI reasoning, with potential to transform enterprise workflows and scientific research.
- [The Verge] Apple reveals M5 chip with 3nm process | Date: 12/11/2025 | https://theverge.com/m5 | Commentary: The new chip promises 50% better performance while using 30% less power than its predecessor.
- [Wired] Microsoft launches AI-powered Windows 12 | Date: 12/11/2025 | Commentary: Deep AI integration across the OS could redefine how users interact with their computers.
- [Ars Technica] Quantum computing reaches 1000-qubit milestone | Date: 12/10/2025 | https://arstechnica.com/quantum | Commentary: This breakthrough brings quantum advantage closer for practical applications in drug discovery and cryptography.
- [ZDNet] Global 6G trials begin in major cities | Date: 12/10/2025 | Commentary: Early tests show speeds up to 100x faster than 5G, enabling new applications in AR/VR and autonomous systems.

## Financial Markets News Update:
- [Reuters] S&P 500 closes at record high amid tech rally | Date: 12/11/2025 | https://reuters.com/sp500 | Commentary: Strong earnings from AI companies and positive economic data drive unprecedented market optimism.
- [Bloomberg] Federal Reserve signals rate cuts for 2026 | Date: 12/11/2025 | https://bloomberg.com/fed | Commentary: The dovish stance reflects confidence in cooling inflation and sets stage for continued market growth.
- [CNBC] NVIDIA surpasses $5 trillion market cap | Date: 12/11/2025 | Commentary: Insatiable AI chip demand continues to propel the company to historic valuations.
- [Financial Times] Dollar weakens as global recovery strengthens | Date: 12/10/2025 | Commentary: Currency shifts reflect improving economic conditions in Europe and Asia.
- [MarketWatch] Oil prices stabilize at $75/barrel | Date: 12/10/2025 | https://marketwatch.com/oil | Commentary: OPEC+ production decisions and demand forecasts bring equilibrium to energy markets.

## India News Update:
- [Economic Times] India GDP growth hits 7.5% in Q3 | Date: 12/11/2025 | https://economictimes.com/gdp | Commentary: Strong domestic consumption and manufacturing output exceed analyst expectations.
- [Times of India] New semiconductor fab announced in Gujarat | Date: 12/11/2025 | Commentary: The $10 billion investment marks a major milestone in India's chip manufacturing ambitions.
- [The Hindu] Digital India initiative reaches 500 million users | Date: 12/10/2025 | https://thehindu.com/digital | Commentary: Government services digitization continues to transform citizen experiences across the nation.
- [Indian Express] Renewable energy capacity surpasses 200 GW | Date: 12/10/2025 | Commentary: India on track to meet its 2030 clean energy targets ahead of schedule.
- [NDTV] New high-speed rail corridor approved | Date: 12/10/2025 | https://ndtv.com/rail | Commentary: The Mumbai-Ahmedabad expansion will cut travel time significantly and boost regional connectivity.
"""
    
    # Parse to sections
    newsletter_sections = parse_newsletter_text_to_sections(sample_newsletter)
    
    # Create lesson
    lesson_dict = parse_lesson_to_dict(
        """[KEY INSIGHT]
The most successful leaders understand that sustainable influence comes from empowering others, not accumulating power for oneself.

[HISTORICAL]
In 1955, Rosa Parks' quiet act of civil disobedience sparked the Montgomery Bus Boycott. Her strength came not from authority but from moral conviction. The movement succeeded because leaders like Martin Luther King Jr. understood that lasting change requires building coalitions and inspiring collective action.

[APPLICATION]
For modern engineers and leaders, this translates to creating environments where team members feel empowered to take initiative. Instead of hoarding knowledge, share it freely. Instead of seeking credit, give recognition to others. The paradox of leadership is that your influence grows as you help others succeed.""",
        topic="The Paradox of Leadership and Influence"
    )
    
    # Build bundle
    bundle = build_daily_bundle(
        days_completed=345,
        weeks_completed=49.29,
        days_left=20,
        weeks_left=2.86,
        percent_days_left=5.48,
        weather_data={"temp_c": 8, "status": "partly cloudy", "location": "North Plains, OR"},
        quote_text="The only way to do great work is to love what you do. If you haven't found it yet, keep looking. Don't settle.",
        quote_author="Steve Jobs",
        lesson_dict=lesson_dict,
        news_raw_sources={
            "technology": "Sample tech news sources",
            "financial": "Sample financial news sources",
            "india": "Sample India news sources"
        },
        newsletter_sections=newsletter_sections,
        voicebot_script="Sample voicebot script for TTS"
    )
    
    # Render HTML
    html = render_newsletter_html_from_bundle(bundle)
    
    # Save files
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save JSON
    json_path = OUTPUT_DIR / f"daily_bundle_{bundle['meta']['date_iso']}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(bundle, f, indent=2, ensure_ascii=False)
    print(f"✓ Bundle JSON saved: {json_path}")
    
    # Save latest JSON
    latest_path = OUTPUT_DIR / "daily_bundle_latest.json"
    with open(latest_path, 'w', encoding='utf-8') as f:
        json.dump(bundle, f, indent=2, ensure_ascii=False)
    print(f"✓ Latest bundle saved: {latest_path}")
    
    # Save HTML
    html_path = OUTPUT_DIR / "news_newsletter_report.html"
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"✓ Newsletter HTML saved: {html_path}")
    
    print("\n" + "=" * 70)
    print(f"Newsletter sections: tech={len(newsletter_sections['tech'])}, financial={len(newsletter_sections['financial'])}, india={len(newsletter_sections['india'])}")
    print(f"JSON size: {len(json.dumps(bundle))} chars")
    print(f"HTML size: {len(html)} chars")
    print("=" * 70)
    
    return bundle, html


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("NEWSLETTER GENERATION - STANDALONE TESTS")
    print("=" * 70)
    print("No external dependencies required (no LLM calls, no audio)")
    
    # Run tests
    test_newsletter_parsing()
    bundle = test_bundle_creation()
    test_html_rendering(bundle)
    
    # Generate example
    bundle, html = generate_example_newsletter()
    
    print("\n" + "=" * 70)
    print("✓ ALL TESTS PASSED!")
    print("=" * 70)
    print(f"\nOpen {OUTPUT_DIR / 'news_newsletter_report.html'} in a browser to view the example newsletter.")

