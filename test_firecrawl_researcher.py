"""
Simple test script for Firecrawl Researcher
Tests the new custom researcher without Tavily dependency
"""
from helper_functions.firecrawl_researcher import conduct_research_firecrawl

def test_simple_query():
    """Test with a simple, quick query"""
    print("=" * 80)
    print("Testing Firecrawl Researcher")
    print("=" * 80)
    print()
    
    # Use a simple query that should return quick results
    query = "What is Python programming language?"
    provider = "litellm"  # or "ollama"
    model_name = "gpt-oss-120b"  # One of your available models
    
    print(f"Query: {query}")
    print(f"Provider: {provider}")
    print(f"Model: {model_name}")
    print()
    print("-" * 80)
    print()
    
    try:
        report = conduct_research_firecrawl(
            query=query,
            provider=provider,
            model_name=model_name,
            max_sources=3  # Limit to 3 sources for quick test
        )
        
        print()
        print("=" * 80)
        print("RESEARCH REPORT")
        print("=" * 80)
        print()
        print(report)
        print()
        print("=" * 80)
        print("✓ Test Completed Successfully!")
        print("=" * 80)
        
    except Exception as e:
        print()
        print("=" * 80)
        print("✗ Test Failed")
        print("=" * 80)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print()
    print("╔════════════════════════════════════════════════════════════════════════════╗")
    print("║                    Firecrawl Researcher Test Suite                         ║")
    print("╚════════════════════════════════════════════════════════════════════════════╝")
    print()
    print("This will test the new Firecrawl-based researcher (no Tavily required!)")
    print()
    
    test_simple_query()
    
    print()
    print("Next Steps:")
    print("  1. If test passed, try it in the UI: python gradio_ui_full.py")
    print("  2. Go to 'AI Assistant' tab")
    print("  3. Select your provider and model")
    print("  4. Ask: 'What's the latest news about AI?'")
    print()

