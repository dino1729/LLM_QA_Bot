import json
import os
import requests
import openai
import PyPDF2
import requests
import re
import ast
import argparse
import logging
import dotenv
import supabase
import tiktoken

from datetime import datetime
from calendar import c
from hmac import new
from shutil import copyfileobj
from urllib.parse import parse_qs, urlparse
from IPython.display import Markdown, display
from langchain import OpenAI
from langchain.agents import initialize_agent
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import AzureOpenAI
from llama_index import (
    Document,
    GPTSimpleVectorIndex,
    GPTListIndex,
    LangchainEmbedding,
    LLMPredictor,
    PromptHelper,
    SimpleDirectoryReader,
    ServiceContext
)
from newspaper import Article
from bs4 import BeautifulSoup
from PIL import Image
from pytube import YouTube
from youtube_transcript_api import YouTubeTranscriptApi

# Get API key from environment variable
dotenv.load_dotenv()
os.environ["OPENAI_API_KEY"] = os.environ.get("AZUREOPENAIAPIKEY")
openai.api_type = os.environ.get("AZUREOPENAIAPITYPE")
openai.api_version = os.environ.get("AZUREOPENAIAPIVERSION")
openai.api_base = os.environ.get("AZUREOPENAIENDPOINT")
openai.api_key = os.environ.get("AZUREOPENAIAPIKEY")
LLM_DEPLOYMENT_NAME = "text-davinci-003"
#Supabase API key
SUPABASE_API_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
SUPABASE_URL = os.environ.get("PUBLIC_SUPABASE_URL")

# max LLM token input size
max_input_size = 4096
# set number of output tokens
num_output = 2048
# set maximum chunk overlap
max_chunk_overlap = 24
# set chunk size limit
chunk_size_limit = 256
# set prompt helper
prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)

#Update your deployment name accordingly
llm = AzureOpenAI(deployment_name=LLM_DEPLOYMENT_NAME, model_kwargs={
    "api_type": os.environ.get("AZUREOPENAIAPITYPE"),
    "api_version": os.environ.get("AZUREOPENAIAPIVERSION"),
})
llm_predictor = LLMPredictor(llm=llm)
embedding_llm = LangchainEmbedding(OpenAIEmbeddings(chunk_size=1))
service_context = ServiceContext.from_defaults(
    llm_predictor=llm_predictor,
    embed_model=embedding_llm,
    prompt_helper=prompt_helper,
    chunk_size_limit=chunk_size_limit
)

UPLOAD_FOLDER = os.path.join(".", "iosdata")  # set the upload folder path

# If the UPLOAD_FOLDER path does not exist, create it
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def build_index():
    documents = SimpleDirectoryReader(UPLOAD_FOLDER).load_data()
    index = GPTSimpleVectorIndex.from_documents(documents, service_context=service_context)
    #index = GPTListIndex.from_documents(documents, service_context=service_context)
    index.save_to_disk(os.path.join(UPLOAD_FOLDER, "index.json"))

def upload_data_to_supabase(index_data, title, url):
    
    # Insert the data for each document into the Supabase table
    supabase_client = supabase.Client(SUPABASE_URL, SUPABASE_API_KEY)
    for doc_id, doc_data in index_data["docstore"]["__data__"]["docs"].items():
        content_title = title
        content_url = url
        content_date = datetime.today().strftime('%B %d, %Y')
        content_text = doc_data['text']
        content_length = len(content_text)
        content_tokens = len(tiktoken.get_encoding("cl100k_base").encode(content_text))
        cleaned_content_text = re.sub(r'[^\w0-9./:^,&%@"!()?\\p{Sc}\'’“”]+|\s+', ' ', content_text, flags=re.UNICODE)
        embedding = index_data["vector_store"]["__data__"]["simple_vector_store_data_dict"]["embedding_dict"][doc_id]

        result = supabase_client.table('mp').insert({
            'content_title': content_title,
            'content_url': content_url,
            'content_date': content_date,
            'content': cleaned_content_text,
            'content_length': content_length,
            'content_tokens': content_tokens,
            'embedding': embedding
        }).execute()

def clearallfiles():
    # Ensure the UPLOAD_FOLDER is empty
    for file in os.listdir(UPLOAD_FOLDER):
        os.remove(os.path.join(UPLOAD_FOLDER, file))

def download_ytvideo(url, memorize):

    clearallfiles()
    if url:
        # Extract the video id from the url
        match = re.search(r"youtu\.be\/(.+)", url)
        if match:
            video_id = match.group(1)
        else:
            video_id = url.split("=")[1]
        try:
            # Use pytube to get the video title
            yt = YouTube(url)
            video_title = yt.title
        except Exception as e:
            print("Error occurred while getting video title:", str(e))
            video_title = video_id
        try:
            # Download the transcript using youtube_transcript_api
            #transcript_list = YouTubeTranscriptApi.get_transcripts([video_id])
            transcript_list = YouTubeTranscriptApi.get_transcripts([video_id],languages=['en-IN', 'en'])
        except Exception as e:
            # Handle the case where the video does not have transcripts
            print("Error occurred while downloading transcripts:", str(e))
            transcript_list = []
        # Check if the video has already generated transcripts
        if transcript_list:
            # Join all the transcript text into a single string
            transcript_text = " ".join([transcript["text"] for transcript in transcript_list[0][video_id]])
            # Save the transcript to a file in UPLOAD_FOLDER
            with open(os.path.join(UPLOAD_FOLDER, "transcript.txt"), "w") as f:
                f.write(transcript_text)
            # Build index
            build_index()
            # Upload data to Supabase if memorize is True
            if memorize:
                index_data = json.load(open(os.path.join(UPLOAD_FOLDER, "index.json")))
                upload_data_to_supabase(index_data, title=video_title, url=url)
            # Generate summary
            summary = summary_generator()
            # Generate example queries
            return summary
        # If the video does not have transcripts, download the video and post-process it locally
        else:
            return "Youtube video doesn't have transcripts"
    else:
        return "Please enter a valid Youtube URL"

def download_art(url, memorize):
    
    clearallfiles()
    if url:
        # Extract the article
        article = Article(url)
        try:
            article.download()
            article.parse()
            #Check if the article text has atleast 75 words
            if len(article.text.split()) < 75:
                raise Exception("Article is too short. Probably the article is behind a paywall.")
        except Exception as e:
            print("Failed to download and parse article from URL using newspaper package: %s. Error: %s", url, str(e))
            # Try an alternate method using requests and beautifulsoup
            try:
                req = requests.get(url)
                soup = BeautifulSoup(req.content, 'html.parser')
                article.text = soup.get_text()
            except Exception as e:
                print("Failed to download article using beautifulsoup method from URL: %s. Error: %s", url, str(e))
                return "Failed to download and parse article. Please check the URL and try again.", summary
        # Save the article to the UPLOAD_FOLDER
        with open(os.path.join(UPLOAD_FOLDER, "article.txt"), 'w') as f:
            f.write(article.text)
        # Build index
        build_index()
        # Upload data to Supabase if memorize is True
        if memorize:
            index_data = json.load(open(os.path.join(UPLOAD_FOLDER, "index.json")))
            upload_data_to_supabase(index_data, title=article.title, url=url)
        # Generate summary
        summary = summary_generator()

        return summary
    else:
        return "Please enter a valid URL"

def ask_fromfullcontext(question):
    
    index = GPTSimpleVectorIndex.load_from_disk(os.path.join(UPLOAD_FOLDER, "index.json"), service_context=service_context)
    #index = GPTListIndex.load_from_disk(UPLOAD_FOLDER + "/index.json", service_context=service_context)
    response = index.query(question, response_mode="tree_summarize")
    answer = response.response
    
    return answer

def summary_generator():
    try:
        summary = ask_fromfullcontext("Summarize the input context while preserving the main points and information, using no less than 10 sentences. Use your own words and avoid copying word-for-word from the provided context. Do not include any irrelevant information such as discounts, promotions, sponsorships or advertisements in your summary and stick to the core message of the content.").lstrip('\n')
    except Exception as e:
        print("Error occurred while generating summary:", str(e))
        summary = "Summary not available"
    return summary

if __name__ == "__main__":
    logger = logging.getLogger()
    logger.level = logging.WARN
    parser = argparse.ArgumentParser(description="Process a URL to generate a summary.")
    parser.add_argument("url", help="The URL of the article or YouTube video.")
    args = parser.parse_args()

    summary = ""
    memorize = False
    url = args.url.strip()
    if "youtube.com/watch" in url or "youtu.be/" in url:
        summary = download_ytvideo(url, memorize)
    elif "http" in url:
        summary = download_art(url, memorize)
    else:
        summary = "Invalid URL. Please enter a valid article or YouTube video URL.", "Summary not available"

    print(summary)
