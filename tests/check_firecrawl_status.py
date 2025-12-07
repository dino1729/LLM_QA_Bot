#!/usr/bin/env python3
"""
Check Firecrawl server status and test basic functionality
"""

import requests
from config import config

def check_firecrawl_server():
    """Check if Firecrawl server is accessible"""
    
    print("\n" + "="*80)
    print("FIRECRAWL SERVER STATUS CHECK")
    print("="*80)
    
    firecrawl_url = config.firecrawl_server_url
    print(f"\n📡 Checking Firecrawl server at: {firecrawl_url}")
    
    try:
        # Try to access the root endpoint
        response = requests.get(firecrawl_url, timeout=5)
        print(f"✓ Server is accessible (Status: {response.status_code})")
        
        # Try a test scrape
        print("\n🧪 Testing scrape functionality...")
        scrape_url = f"{firecrawl_url}/scrape"
        test_payload = {
            "url": "https://example.com",
            "formats": ["markdown"],
            "onlyMainContent": True
        }
        
        scrape_response = requests.post(scrape_url, json=test_payload, timeout=10)
        
        if scrape_response.status_code == 200:
            print("✓ Scrape endpoint is working")
            result = scrape_response.json()
            if "data" in result:
                print(f"✓ Data field present in response")
                if "markdown" in result["data"]:
                    content_len = len(result["data"]["markdown"])
                    print(f"✓ Successfully scraped test page ({content_len} characters)")
                else:
                    print("⚠ Markdown field not found in response")
            else:
                print("⚠ Data field not found in response")
        else:
            print(f"✗ Scrape endpoint failed (Status: {scrape_response.status_code})")
            print(f"   Response: {scrape_response.text[:200]}")
        
        print("\n" + "="*80)
        print("✓ FIRECRAWL CHECK COMPLETE")
        print("="*80)
        
        return True
        
    except requests.exceptions.ConnectionError:
        print("✗ Cannot connect to Firecrawl server")
        print("\n💡 Solutions:")
        print("   1. Start the Firecrawl server:")
        print("      docker run -p 3002:3002 firecrawl/firecrawl")
        print("   2. Or update config.yml to use a different retriever")
        print("   3. Check the server URL in config/config.yml")
        return False
        
    except requests.exceptions.Timeout:
        print("✗ Connection to Firecrawl server timed out")
        print("\n💡 Server may be slow or unresponsive")
        return False
        
    except Exception as e:
        print(f"✗ Error checking Firecrawl server: {e}")
        return False


def check_config():
    """Check relevant configuration settings"""
    
    print("\n" + "="*80)
    print("CONFIGURATION CHECK")
    print("="*80)
    
    print(f"\n📋 Retriever: {config.retriever}")
    print(f"📋 Firecrawl Server URL: {config.firecrawl_server_url}")
    print(f"📋 LiteLLM Base URL: {config.litellm_base_url}")
    print(f"📋 Ollama Base URL: {config.ollama_base_url}")
    
    # Check API keys (redacted)
    def check_key(name, value):
        if value and len(value) > 0:
            print(f"✓ {name}: Set ({len(value)} chars, redacted)")
        else:
            print(f"✗ {name}: Not set")
    
    check_key("LiteLLM API Key", config.litellm_api_key)
    check_key("OpenAI API Key", config.openai_api_key if hasattr(config, 'openai_api_key') else None)
    
    print("\n" + "="*80)
    

if __name__ == "__main__":
    check_config()
    check_firecrawl_server()
    
    print("\n💡 Tips:")
    print("   - If Firecrawl is not available, news fetching will be limited")
    print("   - The system will attempt to use fallback methods")
    print("   - Consider using newspaper3k or BeautifulSoup fallbacks")
    print()

