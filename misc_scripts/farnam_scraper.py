import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time
import base64
from urllib.parse import urlparse
import sys

def save_as_pdf(url, output_filename=None):
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    
    # Set a common user agent
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36")
    
    driver = webdriver.Chrome(options=chrome_options)
    
    try:
        driver.get(url)
        time.sleep(3)  # Allow page to load
        
        # Print to PDF parameters
        print_params = {
            "printBackground": True,
            "paperWidth": 8.27,  # A4 size
            "paperHeight": 11.69
        }
        
        result = driver.execute_cdp_cmd("Page.printToPDF", print_params)
        
        if output_filename is None:
            parsed_url = urlparse(url)
            path = parsed_url.path.strip('/').replace('/', '_')
            output_filename = f"{path}.pdf" if path else f"{parsed_url.netloc}.pdf"
        
        with open(output_filename, "wb") as f:
            f.write(base64.b64decode(result['data']))
            
        print(f"Successfully saved {output_filename}")
        return True
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return False
    finally:
        driver.quit()

def save_links_from_file(file_path):
    with open(file_path, 'r') as file:
        links = file.readlines()
    
    for link in links:
        link = link.strip()
        if link:
            save_as_pdf(link)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage:")
        print("For single URL: python farnam_scraper.py <url>")
        print("For file with URLs: python farnam_scraper.py <links_file>")
        sys.exit(1)
    
    input_path = sys.argv[1]
    
    # Check if input is a URL or file
    if input_path.startswith(('http://', 'https://')):
        # Handle single URL
        save_as_pdf(input_path)
    else:
        # Handle file containing URLs
        save_links_from_file(input_path)