"""
Comprehensive Test Suite for Firecrawl Researcher
Tests connectivity, individual endpoints, and full research workflow.
"""
import requests
import json
import time
import sys
from helper_functions.firecrawl_researcher import conduct_research_firecrawl
from config import config

# Colors for terminal output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"

def print_header(title):
    print(f"\n{CYAN}{'=' * 80}{RESET}")
    print(f"{CYAN}{title.center(80)}{RESET}")
    print(f"{CYAN}{'=' * 80}{RESET}\n")

def print_status(step, status, message=""):
    symbol = "✓" if status else "✗"
    color = GREEN if status else RED
    print(f"{color}[{symbol}] {step}{RESET} {message}")

def check_server_health(url):
    """Check if Firecrawl server is reachable"""
    print_header("1. Connectivity Check")
    print(f"Checking Firecrawl server at: {url}")
    
    try:
        # Try health endpoint first
        try:
            response = requests.get(f"{url}/health", timeout=5)
            if response.status_code == 200:
                print_status("Server Reachable", True, "(Health endpoint active)")
                return True
        except:
            pass
            
        # Fallback to root
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            print_status("Server Reachable", True, "(Root endpoint active)")
            return True
        else:
            print_status("Server Reachable", True, f"(Status: {response.status_code})")
            return True
            
    except requests.exceptions.ConnectionError:
        print_status("Server Reachable", False, "Connection refused. Is the server running?")
        print(f"\n{YELLOW}Tip: Ensure Firecrawl is running on port 3002 (or your configured port).{RESET}")
        return False
    except Exception as e:
        print_status("Server Reachable", False, str(e))
        return False

def test_scrape_endpoint(url):
    """Test the /scrape endpoint directly"""
    print_header("2. Scrape Endpoint Test")
    
    target_url = "https://example.com"
    payload = {
        "url": target_url,
        "formats": ["markdown"],
    }
    
    try:
        print(f"Scraping: {target_url}...")
        response = requests.post(f"{url}/scrape", json=payload, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            if data.get("success") or "data" in data:
                markdown = data.get("data", {}).get("markdown", "")
                if "Example Domain" in markdown:
                    print_status("Scrape Success", True, "Content verified")
                    return True
                else:
                    print_status("Scrape Success", False, "Content mismatch")
                    print(f"Response snippet: {markdown[:100]}...")
            else:
                print_status("Scrape Success", False, "Invalid response format")
        else:
            print_status("Scrape Success", False, f"HTTP {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print_status("Scrape Endpoint", False, str(e))
    
    return False

def test_search_endpoint(url):
    """Test if /search endpoint is available (optional feature)"""
    print_header("3. Search Endpoint Test (Optional)")
    
    query = "test"
    payload = {
        "query": query,
        "limit": 1
    }
    
    try:
        print(f"Testing /search endpoint...")
        response = requests.post(f"{url}/search", json=payload, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if data.get("success") or "data" in data:
                print_status("Search Available", True, "Server supports search")
            else:
                print_status("Search Available", False, "Response format unexpected")
        elif response.status_code == 404:
            print_status("Search Available", False, "Endpoint not found (Normal for basic self-hosted)")
        else:
            print_status("Search Available", False, f"HTTP {response.status_code}")
            
    except Exception as e:
        print_status("Search Test", False, f"Skipped ({str(e)})")

def test_full_research_workflow():
    """Test the integrated researcher function - uses config for model names"""
    print_header("4. Full Research Workflow")
    
    query = "latest india-usa tariff deal news"
    # Use config defaults - model names should be configured in config.yml
    provider = config.firecrawl_default_provider if hasattr(config, 'firecrawl_default_provider') else "litellm"
    model_name = config.litellm_smart_llm if hasattr(config, 'litellm_smart_llm') and config.litellm_smart_llm else None
    
    print(f"Query: {query}")
    print(f"Provider: {provider}")
    print(f"Model: {model_name}")
    print("-" * 40)
    
    try:
        start_time = time.time()
        report = conduct_research_firecrawl(
            query=query,
            provider=provider,
            model_name=model_name,
            max_sources=5  # Use more sources for better testing
        )
        duration = time.time() - start_time
        
        if report and "Unable to" not in report and "failed" not in report.lower():
            print_status("Research Generation", True, f"Completed in {duration:.2f}s")
            print(f"\n{YELLOW}Report Preview:{RESET}")
            print("-" * 40)
            print(report[:500] + "..." if len(report) > 500 else report)
            print("-" * 40)
        else:
            print_status("Research Generation", False, "Report generation failed or returned error")
            print(f"Output: {report}")
            
    except Exception as e:
        print_status("Research Generation", False, str(e))
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    server_url = config.firecrawl_server_url if hasattr(config, 'firecrawl_server_url') else "http://localhost:3002"
    
    print_header("Firecrawl Researcher Diagnostic Suite")
    
    if check_server_health(server_url):
        test_scrape_endpoint(server_url)
        test_search_endpoint(server_url)
        test_full_research_workflow()
    else:
        print(f"\n{RED}Critical: Cannot connect to Firecrawl server. Skipping functionality tests.{RESET}")
        sys.exit(1)