import argparse
import os
from helper_functions.analyzers import analyze_article, analyze_ytvideo, analyze_media

def analyze_url(url, memorize=True):
    url = url.strip()
    if "youtube.com/watch" in url or "youtu.be/" in url:
        analysis = analyze_ytvideo(url, memorize)
        return analysis["summary"]
    elif any(x in url for x in [".m4a", ".mp3", ".wav", ".mp4", ".mkv"]):
        analysis = analyze_media(url, memorize)
        return analysis["summary"]
    elif "http" in url:
        analysis = analyze_article(url, memorize)
        return analysis["summary"]
    else:
        return "Invalid URL. Please enter a valid article or YouTube video URL."

parser = argparse.ArgumentParser(description="Process a file containing URLs to generate summaries.")
parser.add_argument("input_file", help="The path to the plaintext file containing URLs.")
args = parser.parse_args()

# Generate the output file name
input_file_name = os.path.splitext(args.input_file)[0]
output_file = f"{input_file_name}_summaries.txt"

# Ensure the output file exists
if not os.path.exists(output_file):
    open(output_file, 'w').close()

with open(args.input_file, 'r') as infile, open(output_file, 'a') as outfile:
    lines = infile.readlines()
    total_lines = len(lines)
    for i, line in enumerate(lines):
        print(f"Analyzing URL: {line.strip()}")  # Progress indicator
        summary = analyze_url(line)
        outfile.write(f"URL: {line.strip()}\n")
        outfile.write(f"Summary:\n{summary}\n")
        outfile.write("-" * 40 + "\n")  # Separator
        # Print progress bar
        progress = (i + 1) / total_lines * 100
        print(f"Progress: {progress:.2f}%")
