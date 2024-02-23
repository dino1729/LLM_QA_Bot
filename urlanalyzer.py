import config
import json
import os
import requests
import re
import dotenv
import supabase
import tiktoken
import argparse
import wget
import whisper
import logging
import sys
from openai import AzureOpenAI as OpenAIAzure
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_audio
from datetime import datetime
from newspaper import Article
from bs4 import BeautifulSoup
from pytube import YouTube
from youtube_transcript_api import YouTubeTranscriptApi
from llama_index.embeddings import AzureOpenAIEmbedding
from llama_index.llms import AzureOpenAI
from llama_index import (
    VectorStoreIndex,
    SummaryIndex,
    PromptHelper,
    SimpleDirectoryReader,
    ServiceContext,
    StorageContext,
    load_index_from_storage,
    get_response_synthesizer,
    set_global_service_context,
)
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.node_parser import SemanticSplitterNodeParser
from llama_index.prompts import PromptTemplate

def build_index():

    documents = SimpleDirectoryReader(UPLOAD_FOLDER).load_data()
    questionindex = VectorStoreIndex.from_documents(documents)
    questionindex.set_index_id("vector_index")
    questionindex.storage_context.persist(persist_dir=VECTOR_FOLDER)
    
    summaryindex = SummaryIndex.from_documents(documents)
    summaryindex.set_index_id("summary_index")
    summaryindex.storage_context.persist(persist_dir=SUMMARY_FOLDER)

def upload_data_to_supabase(metadata_index, embedding_index, title, url):
    
    # Insert the data for each document into the Supabase table
    supabase_client = supabase.Client(public_supabase_url, supabase_service_role_key)
    for doc_id, doc_data in metadata_index["docstore/data"].items():
        content_title = title
        content_url = url
        content_date = datetime.today().strftime('%B %d, %Y')
        content_text = doc_data["__data__"]["text"]
        content_length = len(content_text)
        content_tokens = len(tiktoken.get_encoding("cl100k_base").encode(content_text))
        cleaned_content_text = re.sub(r'[^\w0-9./:^,&%@"!()?\\p{Sc}\'’“”]+|\s+', ' ', content_text, flags=re.UNICODE)
        embedding = embedding_index["embedding_dict"][doc_id]

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
    for root, dirs, files in os.walk(UPLOAD_FOLDER):
        for file in files:
            file_path = os.path.join(root, file)
            os.remove(file_path)

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
                metadata_index = json.load(open(os.path.join(VECTOR_FOLDER, "docstore.json")))
                embedding_index = json.load(open(os.path.join(VECTOR_FOLDER, "vector_store.json")))
                upload_data_to_supabase(metadata_index, embedding_index, title=video_title, url=url)
            # Generate summary
            summary = summary_generator()
            return summary
        # If the video does not have transcripts, download the video and post-process it locally
        else:
            yt = YouTube(url)
            yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first().download(UPLOAD_FOLDER)
            build_index()
            # Upload data to Supabase if memorize is True
            if memorize:
                metadata_index = json.load(open(os.path.join(VECTOR_FOLDER, "docstore.json")))
                embedding_index = json.load(open(os.path.join(VECTOR_FOLDER, "vector_store.json")))
                upload_data_to_supabase(metadata_index, embedding_index, title=video_title, url=url)
            # Generate summary
            summary = summary_generator()
            return summary
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
                pass
        # Save the article to the UPLOAD_FOLDER
        with open(os.path.join(UPLOAD_FOLDER, "article.txt"), 'w') as f:
            f.write(article.text)
        # Build index
        build_index()
        # Upload data to Supabase if memorize is True
        if memorize:
            metadata_index = json.load(open(os.path.join(VECTOR_FOLDER, "docstore.json")))
            embedding_index = json.load(open(os.path.join(VECTOR_FOLDER, "vector_store.json")))
            upload_data_to_supabase(metadata_index, embedding_index, title=article.title, url=url)
        # Generate summary
        summary = summary_generator()
        return summary
    else:
        return "Please enter a valid URL"

def download_media(url, memorize):

    clearallfiles()
    if url:
        try:
            # Check if the url contains overcast.fm and extract the media url
            if "overcast.fm" in url:
                req = requests.get(url)
                soup = BeautifulSoup(req.content, 'html.parser')
                audio_tag = soup.find('audio')
                # look for mp3, m4a, wav, mkv, mp4 in the html content
                media_url = audio_tag.source['src']
                url = media_url
                
            media = wget.download(url, out=UPLOAD_FOLDER)
            # Extract the file name
            filename_with_path = media
            file_name = os.path.basename(filename_with_path)
            # Get extention of file name
            ext = file_name.split(".")[-1].lower()
            # Check if the file is an audio file
            if ext in ["m4a", "mp3", "wav"]:
                # Rename the file to audio.mp3
                os.rename(os.path.join(UPLOAD_FOLDER, file_name), os.path.join(UPLOAD_FOLDER, "audio.mp3"))
            # Check if the file is a video file
            elif ext in ["mp4", "mkv"]:
                # Rename the file to video.mp4
                os.rename(os.path.join(UPLOAD_FOLDER, file_name), os.path.join(UPLOAD_FOLDER, "video.mp4"))
                # Extract the audio from the video and save it as audio.mp3
                ffmpeg_extract_audio(os.path.join(UPLOAD_FOLDER, "video.mp4"), os.path.join(UPLOAD_FOLDER, "audio.mp3"))
                # Delete the video file
                os.remove(os.path.join(UPLOAD_FOLDER, "video.mp4"))
            else:
                raise Exception("Invalid media file format")
        except Exception as e:
            pass
        # Use whisper to extract the transcript from the audio
        model = whisper.load_model("base")
        media_text = model.transcribe(os.path.join(UPLOAD_FOLDER, "audio.mp3"))
        # Save the transcript to a file in UPLOAD_FOLDER
        with open(os.path.join(UPLOAD_FOLDER, "media_transcript.txt"), "w") as f:
            f.write(media_text["text"])
        # Delete the audio file
        os.remove(os.path.join(UPLOAD_FOLDER, "audio.mp3"))
        # Build index
        build_index()
        # Upload data to Supabase if the memorize checkbox is checked
        if memorize:
            metadata_index = json.load(open(os.path.join(VECTOR_FOLDER, "docstore.json")))
            embedding_index = json.load(open(os.path.join(VECTOR_FOLDER, "vector_store.json")))
            upload_data_to_supabase(metadata_index, embedding_index, title=file_name, url=url)
        # Generate summary
        summary = summary_generator()
        return summary
    else:
        return "Please enter a valid URL"

def ask_fromfullcontext(question, fullcontext_template):
    
    storage_context = StorageContext.from_defaults(persist_dir=SUMMARY_FOLDER)
    summary_index = load_index_from_storage(storage_context, index_id="summary_index")
    # SummaryIndexRetriever
    retriever = summary_index.as_retriever(
        retriever_mode="default",
    )
    # configure response synthesizer
    response_synthesizer = get_response_synthesizer(
        response_mode="tree_summarize",
        summary_template=fullcontext_template,
    )
    # assemble query engine
    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
    )
    response = query_engine.query(question)

    answer = response.response
    
    return answer

def summary_generator():
    
    global summary
    try:
        summary = ask_fromfullcontext("Generate a summary of the input context. Be as verbose as possible.", summary_template).lstrip('\n')
    except Exception as e:
        print("Error occurred while generating summary:", str(e))
        summary = "Summary not available"
    return summary

if __name__ == "__main__":
    
    logging.basicConfig(stream=sys.stdout, level=logging.CRITICAL)
    logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

    azure_api_key = config.azure_api_key
    azure_api_base = config.azure_api_base
    azure_embeddingapi_version = config.azure_embeddingapi_version
    azure_chatapi_version = config.azure_chatapi_version
    azure_gpt4_deploymentid = config.azure_gpt4_deploymentid
    azure_gpt35_deploymentid = config.azure_gpt35_deploymentid
    azure_embedding_deploymentid = config.azure_embedding_deploymentid

    bing_api_key = config.bing_api_key
    bing_endpoint = config.bing_endpoint
    bing_news_endpoint = config.bing_news_endpoint
    openweather_api_key = config.openweather_api_key

    llama2_api_key = config.llama2_api_key
    llama2_api_base = config.llama2_api_base
    supabase_service_role_key = config.supabase_service_role_key
    public_supabase_url = config.public_supabase_url
    pinecone_api_key = config.pinecone_api_key
    pinecone_environment = config.pinecone_environment

    client = OpenAIAzure(
        api_key=azure_api_key,
        azure_endpoint=azure_api_base,
        api_version=azure_chatapi_version,
    )

    # max LLM token input size
    num_output = 1024
    max_chunk_overlap_ratio = 0.1
    max_input_size = 128000
    context_window = 128000
    prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap_ratio)
    # Set a flag for lite mode: Choose lite mode if you dont want to analyze videos without transcripts
    lite_mode = False

    llm = AzureOpenAI(
        deployment_name=azure_gpt4_deploymentid, 
        model="gpt-4-0125-preview",
        api_key=azure_api_key,
        azure_endpoint=azure_api_base,
        api_version=azure_chatapi_version,
        temperature=0.25,
        max_tokens=num_output,
    )
    embedding_llm =AzureOpenAIEmbedding(
        deployment_name=azure_embedding_deploymentid,
        model="text-embedding-3-large",
        api_key=azure_api_key,
        azure_endpoint=azure_api_base,
        api_version=azure_embeddingapi_version,
        max_retries=3,
        embed_batch_size=1,
    )

    splitter = SemanticSplitterNodeParser(buffer_size=1, breakpoint_percentile_threshold=95, embed_model=embedding_llm)

    service_context = ServiceContext.from_defaults(
        llm=llm,
        embed_model=embedding_llm,
        prompt_helper=prompt_helper,
        context_window=context_window,
        node_parser=splitter,
    )
    set_global_service_context(service_context)

    #UPLOAD_FOLDER = './data'  # set the upload folder path
    UPLOAD_FOLDER = os.path.join(".", "iosdata")
    SUMMARY_FOLDER = os.path.join(UPLOAD_FOLDER, "summary_index")
    VECTOR_FOLDER = os.path.join(UPLOAD_FOLDER, "vector_index")

    sum_template = (
        "You are a world-class text summarizer. We have provided context information below. \n"
        "---------------------\n"
        "{context_str}"
        "\n---------------------\n"
        "Based on the context provided, your task is to summarize the input context while effectively conveying the main points and relevant information. The summary should be presented in a numbered list of at least 10 key points and takeaways, with a catchy headline at the top. It is important to refrain from directly copying word-for-word from the original context. Additionally, please ensure that the summary excludes any extraneous details such as discounts, promotions, sponsorships, or advertisements, and remains focused on the core message of the content.\n"
        "---------------------\n"
        "Using both the context information and also using your own knowledge, "
        "answer the question: {query_str}\n"
    )
    summary_template = PromptTemplate(sum_template)

    # If the UPLOAD_FOLDER path does not exist, create it
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    if not os.path.exists(SUMMARY_FOLDER):
        os.makedirs(SUMMARY_FOLDER)
    # if not os.path.exists(VECTOR_FOLDER ):
    #     os.makedirs(VECTOR_FOLDER)
    
    parser = argparse.ArgumentParser(description="Process a URL to generate a summary.")
    parser.add_argument("url", help="The URL of the article or YouTube video.")
    args = parser.parse_args()

    summary = ""
    memorize = False

    url = args.url.strip()
    # Check if the url is a YouTube video url
    if "youtube.com/watch" in url or "youtu.be/" in url:
        summary = download_ytvideo(url, memorize)
    # Check if the url is a media file url. A media url will contain m4a, mp3, wav, mp4 or mkv in the url
    elif any(x in url for x in [".m4a", ".mp3", ".wav", ".mp4", ".mkv"]):
        summary = download_media(url, memorize)
    # Else, the url is an article url
    elif "http" in url:
        summary = download_art(url, memorize)
    else:
        summary = "Invalid URL. Please enter a valid article or YouTube video URL.", "Summary not available"

    print(summary)
