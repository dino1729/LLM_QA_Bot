import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, parse_qs

target_url = "https://blog.samaltman.com/"
newsletter_base = "https://blog.samaltman.com/"
output_filename = "extracted_article_list.txt"

def get_page_urls(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        base_url = response.url

        # Extract and filter URLs
        urls = set()
        for link in soup.find_all('a'):
            href = link.get('href')
            if href:
                href = href.strip()
                if href and not href.startswith(('javascript:', 'mailto:', 'tel:', '#')):
                    absolute_url = urljoin(base_url, href)
                    if absolute_url.startswith(newsletter_base) and 'page' not in absolute_url:
                        urls.add(absolute_url)
        return urls, soup
    except requests.exceptions.RequestException as e:
        print(f"Error fetching the page: {e}")
        return set(), None
    except Exception as e:
        print(f"An error occurred: {e}")
        return set(), None

def extract_all_urls(start_url):
    visited_urls = set()
    urls_to_visit = {start_url}
    all_urls = set()
    page_count = 0

    while urls_to_visit:
        current_url = urls_to_visit.pop()
        if current_url in visited_urls:
            continue
        visited_urls.add(current_url)
        page_count += 1

        print(f"Fetching page {page_count}: {current_url}")
        page_urls, soup = get_page_urls(current_url)
        all_urls.update(page_urls)
        print(f"Found {len(page_urls)} URLs on page {page_count}")

        # Find pagination links
        if soup:
            for link in soup.find_all('a'):
                href = link.get('href')
                if href and 'page' in href:
                    absolute_url = urljoin(current_url, href)
                    if absolute_url not in visited_urls:
                        urls_to_visit.add(absolute_url)

    print(f"Total pages visited: {page_count}")
    print(f"Total unique URLs found: {len(all_urls)}")
    return all_urls

# Extract all URLs starting from the target URL
all_newsletter_urls = extract_all_urls(target_url)

# Read existing URLs from the file
try:
    with open(output_filename, 'r') as f:
        existing_urls = set(f.read().splitlines())
except FileNotFoundError:
    existing_urls = set()

# Combine existing URLs with new URLs
new_urls = all_newsletter_urls - existing_urls
combined_urls = existing_urls.union(new_urls)

# Save to file
with open(output_filename, 'w') as f:
    f.write("\n".join(sorted(combined_urls)))

print(f"Successfully saved {len(combined_urls)} newsletter URLs to {output_filename}")
print(f"New URLs extracted: {len(new_urls)}")