#!/usr/bin/env python3
"""
Test script for newsletter generation
Tests the newsletter formatting and HTML generation without running full news fetch
"""

from datetime import datetime
import sys
import os

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from year_progress_and_news_reporter_litellm import (
    generate_gpt_response_newsletter,
    generate_html_news_template,
    format_news_section,
    save_message_to_file,
    generate_fallback_newsletter
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
    
    print("\n2. Testing HTML generation...")
    html_output = generate_html_news_template(sample_newsletter)
    print(f"✓ HTML generated: {len(html_output)} characters")
    
    print("\n3. Testing individual section parsing...")
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
    
    print("\n4. Saving test HTML output...")
    save_message_to_file(html_output, "test_newsletter.html")
    save_message_to_file(sample_newsletter, "test_newsletter.txt")
    print("✓ Test files saved to bing_data/")
    
    print("\n5. Testing fallback newsletter generation...")
    fallback = generate_fallback_newsletter("")
    print(f"✓ Fallback newsletter generated: {len(fallback)} characters")
    
    print("\n" + "="*80)
    print("✓ ALL TESTS COMPLETED")
    print("="*80)
    print("\nCheck the following files in bing_data/:")
    print("  - test_newsletter.html (open in browser to view)")
    print("  - test_newsletter.txt (view raw formatted content)")
    
    return True


def test_with_actual_files():
    """Test with actual news files if they exist"""
    
    print("\n" + "="*80)
    print("TESTING WITH ACTUAL NEWS FILES")
    print("="*80)
    
    news_files = [
        "bing_data/news_tech_report.txt",
        "bing_data/news_usa_report.txt",
        "bing_data/news_india_report.txt",
        "bing_data/news_newsletter_report.txt"
    ]
    
    print("\nChecking for existing news files...")
    for file_path in news_files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            status = "✓" if size > 100 else "⚠"
            print(f"{status} {os.path.basename(file_path)}: {size} bytes")
        else:
            print(f"✗ {os.path.basename(file_path)}: Not found")
    
    # Try to regenerate HTML from existing newsletter content
    newsletter_path = "bing_data/news_newsletter_report.txt"
    if os.path.exists(newsletter_path):
        with open(newsletter_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if content.strip():
            print("\n📄 Regenerating HTML from existing newsletter...")
            html = generate_html_news_template(content)
            save_message_to_file(html, "news_newsletter_report_regenerated.html")
            print("✓ Regenerated HTML saved")
        else:
            print("\n⚠ Newsletter file is empty")
    
    return True


if __name__ == "__main__":
    print("\n" + "="*80)
    print("NEWSLETTER GENERATION TEST SUITE")
    print("="*80)
    
    # Run parsing test with sample data
    test_newsletter_parsing()
    
    # Test with actual files if they exist
    test_with_actual_files()
    
    print("\n✓ Test suite completed!")
    print("\nTo test the full newsletter generation:")
    print("  python year_progress_and_news_reporter_litellm.py")

