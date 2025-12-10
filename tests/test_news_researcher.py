"""
Quick test script for news_researcher.py

Tests the multi-model news gathering system:
- Fast Model: Keyword extraction, source ranking
- Smart Model: News synthesis with deep reasoning
- Strategic Model: Final editorial enhancement

Aggregator Sources:
- TLDR Tech (https://tldr.tech/)
- AINews (https://news.smol.ai/issues/)
- Ben's Bites (https://www.bensbites.com/)
"""
import logging
from helper_functions.news_researcher import gather_daily_news, scrape_aggregator_headlines, extract_keywords_from_headlines
from config import config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

print("\n" + "="*80)
print("MULTI-MODEL NEWS RESEARCHER")
print("="*80)
print(f"Fast Model:       {config.litellm_fast_llm}")
print(f"Smart Model:      {config.litellm_smart_llm}")
print(f"Strategic Model:  {config.litellm_strategic_llm}")
print("="*80)

def test_aggregator_scraping():
    """Test aggregator headline scraping"""
    print("\n" + "="*80)
    print("TEST 1: Aggregator Headline Scraping")
    print("="*80)
    
    aggregators = ["tldr", "smol", "bensbites"]
    total_success = 0
    
    for aggregator in aggregators:
        print(f"\nTesting {aggregator}...")
        headlines = scrape_aggregator_headlines(aggregator)
        
        if headlines:
            print(f"  ✓ Successfully scraped {len(headlines)} headlines from {aggregator}")
            for i, h in enumerate(headlines, 1):
                print(f"    {i}. {h['title'][:70]}...")
            total_success += 1
        else:
            print(f"  ✗ Failed to scrape headlines from {aggregator}")
    
    print(f"\n  Overall: {total_success}/{len(aggregators)} aggregators working")
    return total_success >= 1  # At least one aggregator should work


def test_keyword_extraction():
    """Test keyword extraction from headlines"""
    print("\n" + "="*80)
    print("TEST 2: Keyword Extraction")
    print("="*80)
    
    headlines = [
        {'title': 'Claude AI launches new features for developers'},
        {'title': 'Nvidia announces next-generation GPU architecture'}
    ]
    
    keywords = extract_keywords_from_headlines(headlines, provider="litellm")
    
    if keywords:
        print(f"✓ Successfully extracted {len(keywords)} keywords:")
        for kw in keywords:
            print(f"  - {kw}")
    else:
        print("✗ Failed to extract keywords")
    
    return len(keywords) > 0


def test_tech_news_gathering():
    """Test full technology news gathering"""
    print("\n" + "="*80)
    print("TEST 3: Technology News Gathering (Full Integration)")
    print("="*80)
    
    try:
        news = gather_daily_news(
            category="technology",
            max_sources=3,  # Reduced for faster testing
            aggregator_limit=1,
            freshness_hours=24,
            provider="litellm"
        )
        
        if news and len(news) > 200:
            print(f"✓ Successfully gathered tech news ({len(news)} characters)")
            print("\nPreview (first 300 chars):")
            print(news[:300] + "...")
            
            # Check for sources
            if "Sources:" in news or "http" in news:
                print("✓ News includes source citations")
            
            return True
        else:
            print("✗ News gathering returned insufficient content")
            return False
            
    except Exception as e:
        print(f"✗ Error during news gathering: {e}")
        return False


if __name__ == "__main__":
    print("\n" + "="*80)
    print("NEWS RESEARCHER TEST SUITE")
    print("="*80)
    
    results = []
    
    # Run tests
    results.append(("Aggregator Scraping", test_aggregator_scraping()))
    results.append(("Keyword Extraction", test_keyword_extraction()))
    results.append(("Tech News Gathering", test_tech_news_gathering()))
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    total_passed = sum(1 for _, p in results if p)
    print(f"\nTotal: {total_passed}/{len(results)} tests passed")
    
    if total_passed == len(results):
        print("\n🎉 All tests passed! The news researcher is working correctly.")
    else:
        print("\n⚠️  Some tests failed. Check the logs above for details.")

