import argparse
from helper_functions.analyzers import analyze_article, analyze_ytvideo, analyze_media
 
parser = argparse.ArgumentParser(description="Process a URL to generate a summary.")
parser.add_argument("url", help="The URL of the article or YouTube video.")
args = parser.parse_args()

summary = ""
memorize = False

url = args.url.strip()
# Check if the url is a YouTube video url
if "youtube.com/watch" in url or "youtu.be/" in url:
    analysis = analyze_ytvideo(url, memorize)
    summary = analysis["summary"]
# Check if the url is a media file url. A media url will contain m4a, mp3, wav, mp4 or mkv in the url
elif any(x in url for x in [".m4a", ".mp3", ".wav", ".mp4", ".mkv"]):
    analysis = analyze_media(url, memorize)
    summary = analysis["summary"]
# Else, the url is an article url
elif "http" in url:
    analysis = analyze_article(url, memorize)
    summary = analysis["summary"]
else:
    summary = "Invalid URL. Please enter a valid article or YouTube video URL."

print(summary)
