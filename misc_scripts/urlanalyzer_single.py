import argparse
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from helper_functions.analyzers import analyze_article, analyze_ytvideo, analyze_media
from config import config

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

def send_email(subject, message):
    sender_email = config.yahoo_id
    receiver_email = "katam.dinesh@hotmail.com"
    password = config.yahoo_app_password

    email_message = MIMEMultipart()
    email_message["From"] = sender_email
    email_message["To"] = receiver_email
    email_message["Subject"] = subject

    email_message.attach(MIMEText(message, "plain"))

    try:
        server = smtplib.SMTP('smtp.mail.yahoo.com', 587)
        server.starttls()
        server.login(sender_email, password)
        text = email_message.as_string()
        server.sendmail(sender_email, receiver_email, text)
        server.quit()
        print(f"Email sent successfully with subject: {subject}")
    except Exception as e:
        print(f"Failed to send email: {e}")

parser = argparse.ArgumentParser(description="Process a file containing URLs to generate summaries.")
parser.add_argument("input_file", help="The path to the plaintext file containing URLs.")
args = parser.parse_args()

with open(args.input_file, 'r') as infile:
    lines = infile.readlines()

if lines:
    line = lines[0]
    print(f"Analyzing URL: {line.strip()}")  # Progress indicator
    summary = analyze_url(line)
    subject = "Daily Personal Content Analyzer"
    message = f"URL: {line.strip()}\nSummary:\n{summary}\n"
    send_email(subject, message)

    # Remove the processed line from the input file
    with open(args.input_file, 'w') as infile:
        infile.writelines(lines[1:])
else:
    print("No URLs to process.")
