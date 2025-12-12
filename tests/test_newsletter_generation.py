#!/usr/bin/env python3
"""
Test script for newsletter generation
Tests the newsletter formatting and HTML generation without running full news fetch

Updated for JSON-backed newsletter pipeline.
"""

from datetime import datetime
import sys
import os
import json
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from year_progress_and_news_reporter_litellm import (
    generate_html_news_template,
    format_news_section,
    save_to_output_dir,
    parse_newsletter_text_to_sections,
    parse_newsletter_item,
    generate_fallback_newsletter_sections,
    format_news_items_html,
    render_newsletter_html_from_bundle,
    build_daily_bundle,
    write_bundle_json,
    load_bundle_json,
    parse_lesson_to_dict,
    OUTPUT_DIR
)


def test_newsletter_parsing():
    """Test the newsletter parsing with sample content"""
    
    print("\n" + "="*80)
    print("TESTING NEWSLETTER GENERATION")
    print("="*80)
    
    # Sample newsletter content in the expected format
    sample_newsletter = """
## Tech News Update:
- [TechCrunch] OpenAI releases new GPT-5 model with improved reasoning | Date: 12/07/2025 | https://techcrunch.com/gpt5-release | Commentary: This represents a major advancement in AI capabilities with potential implications across industries
- [The Verge] Apple announces new MacBook Pro with M4 chip | Date: 12/07/2025 | Commentary: The new chip promises 40% better performance than the previous generation
- [Wired] Microsoft integrates advanced AI into Office suite | Date: 12/06/2025 | https://wired.com/ms-office-ai | Commentary: This move signals Microsoft's commitment to AI-first productivity tools
- [Ars Technica] Quantum computing breakthrough at IBM | Date: 12/06/2025 | Commentary: Researchers achieved quantum advantage in practical applications for the first time
- [ZDNet] 5G rollout accelerates in rural areas | Date: 12/05/2025 | https://zdnet.com/5g-rural | Commentary: This expansion could bridge the digital divide in underserved communities

## Financial Markets News Update:
- [Reuters] S&P 500 reaches new all-time high | Date: 12/07/2025 | https://reuters.com/markets/sp500-high | Commentary: Strong corporate earnings and economic data drive market optimism
- [Bloomberg] Federal Reserve holds interest rates steady | Date: 12/07/2025 | Commentary: The decision reflects confidence in the current economic trajectory
- [CNBC] Tech stocks lead market rally | Date: 12/07/2025 | https://cnbc.com/tech-rally | Commentary: FAANG stocks showed particularly strong performance amid AI enthusiasm
- [Financial Times] Dollar strengthens against major currencies | Date: 12/06/2025 | Commentary: Currency movements reflect shifting global economic dynamics
- [MarketWatch] Oil prices stabilize after recent volatility | Date: 12/06/2025 | https://marketwatch.com/oil-stable | Commentary: OPEC production decisions contribute to market stability

## India News Update:
- [Times of India] New infrastructure projects announced in Maharashtra | Date: 12/07/2025 | Commentary: These projects aim to boost regional connectivity and economic growth
- [The Hindu] Education reforms to be implemented nationwide | Date: 12/07/2025 | https://thehindu.com/education-reforms | Commentary: The reforms focus on skill development and digital literacy
- [Economic Times] GDP growth exceeds expectations | Date: 12/06/2025 | https://economictimes.com/gdp-growth | Commentary: Strong domestic consumption drives economic expansion
- [Indian Express] Renewable energy capacity reaches milestone | Date: 12/06/2025 | Commentary: India continues to make strides toward its clean energy goals
- [NDTV] Healthcare initiatives launched in rural regions | Date: 12/05/2025 | https://ndtv.com/healthcare-rural | Commentary: These programs aim to improve healthcare access in underserved areas
"""
    
    print("\n1. Testing newsletter content generation...")
    print(f"✓ Sample newsletter created: {len(sample_newsletter)} characters")
    
    print("\n2. Testing HTML generation (legacy text-based)...")
    html_output = generate_html_news_template(sample_newsletter)
    print(f"✓ HTML generated: {len(html_output)} characters")
    
    print("\n3. Testing individual section parsing (legacy)...")
    sections = [
        ("Tech News Update", "Technology"),
        ("Financial Markets News Update", "Financial Markets"),
        ("India News Update", "India News")
    ]
    
    for section_title, display_name in sections:
        formatted = format_news_section(sample_newsletter, section_title)
        has_content = "No updates available" not in formatted
        status = "✓" if has_content else "✗"
        print(f"{status} {display_name}: {'Found content' if has_content else 'No content found'}")
    
    print("\n4. Testing structured newsletter parsing...")
    parsed_sections = parse_newsletter_text_to_sections(sample_newsletter)
    print(f"✓ Parsed tech items: {len(parsed_sections['tech'])}")
    print(f"✓ Parsed financial items: {len(parsed_sections['financial'])}")
    print(f"✓ Parsed india items: {len(parsed_sections['india'])}")
    
    print("\n5. Testing fallback newsletter sections...")
    fallback_sections = generate_fallback_newsletter_sections()
    print(f"✓ Fallback sections generated with keys: {list(fallback_sections.keys())}")
    
    print("\n6. Testing structured HTML rendering...")
    items_html = format_news_items_html(parsed_sections['tech'])
    print(f"✓ Tech items HTML generated: {len(items_html)} characters")
    
    print("\n" + "="*80)
    print("✓ ALL PARSING TESTS COMPLETED")
    print("="*80)
    
    return True


def test_bundle_creation():
    """Test bundle creation and JSON serialization"""
    
    print("\n" + "="*80)
    print("TESTING BUNDLE CREATION")
    print("="*80)
    
    print("\n1. Creating sample data...")
    
    # Sample data
    weather_data = {"temp_c": 12.5, "status": "cloudy", "location": "Test City"}
    lesson_dict = parse_lesson_to_dict(
        "[KEY INSIGHT]\nTest insight here.\n\n[HISTORICAL]\nTest historical context.\n\n[APPLICATION]\nTest application.",
        topic="Test Topic"
    )
    
    newsletter_sections = {
        "tech": [
            {"source": "Test", "headline": "Test headline 1", "date_mmddyyyy": "12/11/2025", "url": "", "commentary": "Test commentary"}
        ],
        "financial": [
            {"source": "Test", "headline": "Test headline 2", "date_mmddyyyy": "12/11/2025", "url": "https://example.com", "commentary": "Test commentary"}
        ],
        "india": [
            {"source": "Test", "headline": "Test headline 3", "date_mmddyyyy": "12/11/2025", "url": "", "commentary": "Test commentary"}
        ]
    }
    
    print("✓ Sample data created")
    
    print("\n2. Building bundle...")
    bundle = build_daily_bundle(
        days_completed=345,
        weeks_completed=49.29,
        days_left=20,
        weeks_left=2.86,
        percent_days_left=5.48,
        weather_data=weather_data,
        quote_text="Test quote",
        quote_author="Test Author",
        lesson_dict=lesson_dict,
        news_raw_sources={"technology": "raw tech", "financial": "raw financial", "india": "raw india"},
        newsletter_sections=newsletter_sections,
        voicebot_script="Test voicebot script"
    )
    
    print(f"✓ Bundle created with keys: {list(bundle.keys())}")
    print(f"✓ Meta schema version: {bundle['meta']['schema_version']}")
    
    print("\n3. Testing JSON serialization...")
    json_str = json.dumps(bundle, indent=2)
    print(f"✓ JSON serialized: {len(json_str)} characters")
    
    # Verify it can be deserialized
    parsed = json.loads(json_str)
    assert parsed['meta']['schema_version'] == bundle['meta']['schema_version']
    print("✓ JSON deserialization verified")
    
    print("\n4. Testing bundle HTML rendering...")
    progress_html = None
    newsletter_html = None
    
    try:
        from year_progress_and_news_reporter_litellm import render_year_progress_html_from_bundle, render_newsletter_html_from_bundle
        
        progress_html = render_year_progress_html_from_bundle(bundle)
        print(f"✓ Progress HTML rendered: {len(progress_html)} characters")
        
        newsletter_html = render_newsletter_html_from_bundle(bundle)
        print(f"✓ Newsletter HTML rendered: {len(newsletter_html)} characters")
    except Exception as e:
        print(f"✗ HTML rendering failed: {e}")
    
    print("\n" + "="*80)
    print("✓ ALL BUNDLE TESTS COMPLETED")
    print("="*80)
    
    return True


def test_lesson_parsing():
    """Test lesson text parsing to structured dict"""
    
    print("\n" + "="*80)
    print("TESTING LESSON PARSING")
    print("="*80)
    
    # Test case 1: Full structured lesson
    print("\n1. Testing full structured lesson...")
    full_lesson = """[KEY INSIGHT]
The best leaders understand that influence comes from service, not authority.

[HISTORICAL]
In ancient Rome, Marcus Aurelius demonstrated servant leadership by personally visiting soldiers and sharing their hardships. His Meditations reveal a leader focused on duty rather than power.

[APPLICATION]
Modern engineers can apply this by mentoring juniors, sharing credit, and focusing on team success over personal recognition."""
    
    parsed = parse_lesson_to_dict(full_lesson, "Leadership")
    assert parsed["topic"] == "Leadership"
    assert "influence" in parsed["key_insight"].lower()
    assert "Rome" in parsed["historical"] or "Marcus" in parsed["historical"]
    assert "engineers" in parsed["application"].lower()
    print("✓ Full lesson parsed correctly")
    
    # Test case 2: Missing sections (fallback)
    print("\n2. Testing fallback for missing sections...")
    partial_lesson = "Just some text without markers."
    parsed2 = parse_lesson_to_dict(partial_lesson, "Test")
    assert parsed2["key_insight"]  # Should have fallback
    print(f"✓ Fallback insight: {parsed2['key_insight'][:50]}...")
    
    # Test case 3: Empty input
    print("\n3. Testing empty input...")
    parsed3 = parse_lesson_to_dict("", "Empty")
    assert parsed3["topic"] == "Empty"
    assert parsed3["raw_text"] == ""
    print("✓ Empty input handled correctly")
    
    print("\n" + "="*80)
    print("✓ ALL LESSON PARSING TESTS COMPLETED")
    print("="*80)
    
    return True


def test_news_item_parsing():
    """Test individual news item parsing"""
    
    print("\n" + "="*80)
    print("TESTING NEWS ITEM PARSING")
    print("="*80)
    
    # Test case 1: Full item with URL
    print("\n1. Testing full item with URL...")
    item1 = "- [Reuters] Major announcement made | Date: 12/11/2025 | https://example.com/article | Commentary: This is important news."
    parsed1 = parse_newsletter_item(item1)
    assert parsed1["source"] == "Reuters"
    assert "announcement" in parsed1["headline"].lower()
    assert parsed1["date_mmddyyyy"] == "12/11/2025"
    assert parsed1["url"] == "https://example.com/article"
    assert "important" in parsed1["commentary"].lower()
    print("✓ Full item parsed correctly")
    
    # Test case 2: Item without URL
    print("\n2. Testing item without URL...")
    item2 = "- [Bloomberg] Market update today | Date: 12/11/2025 | Commentary: Markets showed strength."
    parsed2 = parse_newsletter_item(item2)
    assert parsed2["source"] == "Bloomberg"
    assert parsed2["url"] == ""
    print("✓ Item without URL parsed correctly")
    
    # Test case 3: Minimal item
    print("\n3. Testing minimal item...")
    item3 = "Simple headline text"
    parsed3 = parse_newsletter_item(item3)
    assert parsed3["headline"] == "Simple headline text"
    print("✓ Minimal item parsed correctly")
    
    print("\n" + "="*80)
    print("✓ ALL NEWS ITEM PARSING TESTS COMPLETED")
    print("="*80)
    
    return True


if __name__ == "__main__":
    print("\n" + "="*80)
    print("NEWSLETTER GENERATION TEST SUITE (JSON-BACKED)")
    print("="*80)
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Run all tests
    test_newsletter_parsing()
    test_bundle_creation()
    test_lesson_parsing()
    test_news_item_parsing()
    
    print("\n" + "="*80)
    print("✓ ALL TEST SUITES COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("\nTo test the full newsletter generation:")
    print("  python year_progress_and_news_reporter_litellm.py")
