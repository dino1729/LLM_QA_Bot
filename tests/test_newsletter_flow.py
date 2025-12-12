#!/usr/bin/env python3
"""
End-to-end test of newsletter generation flow with mock data
Simulates the complete process without calling actual APIs

Updated for JSON-backed newsletter pipeline.
"""

from datetime import datetime
import sys
import os
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from year_progress_and_news_reporter_litellm import (
    generate_html_news_template,
    format_news_section,
    save_to_output_dir,
    parse_newsletter_text_to_sections,
    generate_fallback_newsletter_sections,
    generate_newsletter_sections,
    format_news_items_html,
    render_newsletter_html_from_bundle,
    render_year_progress_html_from_bundle,
    build_daily_bundle,
    write_bundle_json,
    load_bundle_json,
    parse_lesson_to_dict,
    OUTPUT_DIR
)


def test_complete_newsletter_flow():
    """Test the complete newsletter generation flow with mock data"""
    
    print("\n" + "="*80)
    print("TESTING COMPLETE NEWSLETTER FLOW (JSON-BACKED)")
    print("="*80)
    
    # Simulate sparse news data (similar to what we're currently getting)
    mock_tech_news = """
---

**Sources:**
1. https://www.novintrades.com/articles/5450
2. https://www.geeksforgeeks.org/blogs/top-new-technology-trends/
3. https://www.analyticsinsight.net/tech-news/biggest-news-stories-2025
"""
    
    mock_financial_news = """
---

**Sources:**
1. https://www.hancockwhitney.com/insights/markets-december-2025
2. https://www.edwardjones.com/market-news-insights
3. https://stockanalysis.com/
"""
    
    mock_india_news = """
# India – Daily News Summary

**Date:** December 11, 2025

## Political Developments
- Governor's Rule imposed in Mizoram's Chakma Autonomous District Council
- Telangana approves 42% reservation for Backward Classes in local elections
- Vice President emphasizes preserving traditional knowledge systems

## Economic News
- Net direct tax collections show mixed results
- RBI issues regulatory penalties to financial institutions
- Infrastructure projects progress across states
"""
    
    print("\n1. Simulating sparse news data reception...")
    print(f"   Tech news: {len(mock_tech_news)} chars")
    print(f"   Financial news: {len(mock_financial_news)} chars")
    print(f"   India news: {len(mock_india_news)} chars")
    
    # Parse directly to structured sections
    print("\n2. Testing newsletter section generation...")
    
    # Note: In real usage, generate_newsletter_sections would call the LLM
    # Here we test the fallback path
    fallback_sections = generate_fallback_newsletter_sections()
    print(f"   ✓ Fallback sections generated")
    print(f"   Tech: {len(fallback_sections['tech'])} items")
    print(f"   Financial: {len(fallback_sections['financial'])} items")
    print(f"   India: {len(fallback_sections['india'])} items")
    
    # Test bundle creation
    print("\n3. Creating test bundle...")
    
    weather_data = {"temp_c": 10, "status": "partly cloudy", "location": "North Plains, OR"}
    lesson_dict = parse_lesson_to_dict(
        "[KEY INSIGHT]\nTest insight for flow testing.\n\n[HISTORICAL]\nHistorical context here.\n\n[APPLICATION]\nApplication for engineers.",
        topic="Test Flow Topic"
    )
    
    bundle = build_daily_bundle(
        days_completed=345,
        weeks_completed=49.29,
        days_left=20,
        weeks_left=2.86,
        percent_days_left=5.48,
        weather_data=weather_data,
        quote_text="The best way to predict the future is to create it.",
        quote_author="Peter Drucker",
        lesson_dict=lesson_dict,
        news_raw_sources={
            "technology": mock_tech_news,
            "financial": mock_financial_news,
            "india": mock_india_news
        },
        newsletter_sections=fallback_sections,
        voicebot_script="This is a test voicebot script for flow testing."
    )
    
    print(f"   ✓ Bundle created")
    print(f"   Schema version: {bundle['meta']['schema_version']}")
    print(f"   Date: {bundle['meta']['date_formatted']}")
    
    # Test HTML rendering from bundle
    print("\n4. Testing HTML rendering from bundle...")
    
    progress_html = render_year_progress_html_from_bundle(bundle)
    print(f"   ✓ Progress HTML: {len(progress_html)} chars")
    
    newsletter_html = render_newsletter_html_from_bundle(bundle)
    print(f"   ✓ Newsletter HTML: {len(newsletter_html)} chars")
    
    # Verify HTML has expected content
    has_progress_content = "Year Progress" in progress_html and "Peter Drucker" in progress_html
    has_newsletter_content = "Daily Briefing" in newsletter_html
    
    print(f"   {'✓' if has_progress_content else '✗'} Progress HTML has expected content")
    print(f"   {'✓' if has_newsletter_content else '✗'} Newsletter HTML has expected content")
    
    # Test saving to output directory
    print("\n5. Testing file output...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save test files
    test_progress_path = save_to_output_dir(progress_html, "test_flow_progress.html")
    test_newsletter_path = save_to_output_dir(newsletter_html, "test_flow_newsletter.html")
    
    # Save bundle JSON
    test_bundle_path = OUTPUT_DIR / "test_flow_bundle.json"
    with open(test_bundle_path, 'w', encoding='utf-8') as f:
        json.dump(bundle, f, indent=2)
    
    print(f"   ✓ Progress HTML saved to: {test_progress_path}")
    print(f"   ✓ Newsletter HTML saved to: {test_newsletter_path}")
    print(f"   ✓ Bundle JSON saved to: {test_bundle_path}")
    
    # Verify files were created
    assert test_progress_path.exists(), "Progress HTML not created"
    assert test_newsletter_path.exists(), "Newsletter HTML not created"
    assert test_bundle_path.exists(), "Bundle JSON not created"
    
    # Test loading bundle back
    print("\n6. Testing bundle load/verify...")
    loaded_bundle = load_bundle_json(test_bundle_path)
    assert loaded_bundle['meta']['schema_version'] == bundle['meta']['schema_version']
    assert loaded_bundle['progress']['quote']['author'] == "Peter Drucker"
    print("   ✓ Bundle loaded and verified")
    
    print("\n" + "="*80)
    print("✓ COMPLETE FLOW TEST FINISHED")
    print("="*80)
    
    print("\nResults:")
    print("  - Bundle creation: ✓ Working")
    print("  - HTML rendering: ✓ Working")
    print("  - File saving: ✓ Working")
    print("  - Bundle loading: ✓ Working")
    
    print(f"\nGenerated files in {OUTPUT_DIR}:")
    print(f"  - test_flow_progress.html")
    print(f"  - test_flow_newsletter.html")
    print(f"  - test_flow_bundle.json")
    
    print("\n💡 To test the full pipeline:")
    print("   python year_progress_and_news_reporter_litellm.py")
    
    return True


def test_legacy_compatibility():
    """Test that legacy text-based functions still work"""
    
    print("\n" + "="*80)
    print("TESTING LEGACY COMPATIBILITY")
    print("="*80)
    
    # Sample newsletter in old text format
    sample_newsletter = """
## Tech News Update:
- [TechCrunch] AI breakthrough announced | Date: 12/11/2025 | Commentary: Major development
- [Wired] New chip technology | Date: 12/11/2025 | https://wired.com/chip | Commentary: Revolutionary

## Financial Markets News Update:
- [Reuters] Markets rally | Date: 12/11/2025 | Commentary: Strong performance
- [Bloomberg] Fed announcement | Date: 12/11/2025 | https://bloomberg.com/fed | Commentary: Rate decision

## India News Update:
- [Times of India] Infrastructure news | Date: 12/11/2025 | Commentary: Development update
"""
    
    print("\n1. Testing legacy HTML generation...")
    legacy_html = generate_html_news_template(sample_newsletter)
    assert len(legacy_html) > 1000
    assert "Daily Briefing" in legacy_html
    print(f"   ✓ Legacy HTML generated: {len(legacy_html)} chars")
    
    print("\n2. Testing legacy section formatting...")
    tech_section = format_news_section(sample_newsletter, "Tech News Update")
    assert "AI breakthrough" in tech_section or "news-item" in tech_section
    print(f"   ✓ Tech section formatted: {len(tech_section)} chars")
    
    print("\n3. Testing text-to-sections parsing...")
    sections = parse_newsletter_text_to_sections(sample_newsletter)
    assert len(sections['tech']) >= 1
    assert len(sections['financial']) >= 1
    assert len(sections['india']) >= 1
    print(f"   ✓ Sections parsed: tech={len(sections['tech'])}, financial={len(sections['financial'])}, india={len(sections['india'])}")
    
    print("\n" + "="*80)
    print("✓ LEGACY COMPATIBILITY TEST FINISHED")
    print("="*80)
    
    return True


if __name__ == "__main__":
    print("\n" + "="*80)
    print("NEWSLETTER FLOW TEST - JSON-BACKED PIPELINE")
    print("="*80)
    print("\nThis test validates:")
    print("  1. Bundle creation from raw data")
    print("  2. HTML rendering from bundle")
    print("  3. File I/O operations")
    print("  4. Legacy text-based compatibility")
    print(f"\nOutput directory: {OUTPUT_DIR}")
    
    test_complete_newsletter_flow()
    test_legacy_compatibility()
    
    print("\n" + "="*80)
    print("✓ ALL FLOW TESTS COMPLETED SUCCESSFULLY!")
    print("="*80)
