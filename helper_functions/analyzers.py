from config import config
import os
import shutil
import json
import supabase
import re
from datetime import datetime
import tiktoken
import ast
import whisper
import wget
import requests
import logging
import sys
from newspaper import Article
from bs4 import BeautifulSoup
from pytube import YouTube
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_audio
from youtube_transcript_api import YouTubeTranscriptApi
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from openai import AzureOpenAI as OpenAIAzure
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.core import VectorStoreIndex, PromptHelper, SimpleDirectoryReader, StorageContext, load_index_from_storage, get_response_synthesizer
from llama_index.core.indices import SummaryIndex
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import PromptTemplate
from llama_index.core import Settings

logging.basicConfig(stream=sys.stdout, level=logging.CRITICAL)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

azure_api_key = config.azure_api_key
azure_api_base = config.azure_api_base
azure_embeddingapi_version = config.azure_embeddingapi_version
azure_chatapi_version = config.azure_chatapi_version
azure_gpt4_deploymentid = config.azure_gpt4_deploymentid
openai_gpt4_modelname = config.openai_gpt4_modelname
azure_gpt4omini_deploymentid = config.azure_gpt4omini_deploymentid
openai_gpt4omini_modelname = config.openai_gpt4omini_modelname
azure_embedding_deploymentid = config.azure_embedding_deploymentid
openai_embedding_modelname = config.openai_embedding_modelname

supabase_service_role_key = config.supabase_service_role_key
public_supabase_url = config.public_supabase_url

sum_template = config.sum_template
eg_template = config.eg_template
ques_template = config.ques_template

temperature = config.temperature
max_tokens = config.max_tokens
model_name = config.model_name
num_output = config.num_output
max_chunk_overlap_ratio = config.max_chunk_overlap_ratio
max_input_size = config.max_input_size
context_window = config.context_window
keywords = config.keywords

# Set a flag for lite mode: Choose lite mode if you dont want to analyze videos without transcripts
lite_mode = False

Settings.client = OpenAIAzure(
    api_key=azure_api_key,
    azure_endpoint=azure_api_base,
    api_version=azure_chatapi_version,
)
Settings.llm = AzureOpenAI(
    azure_deployment=azure_gpt4_deploymentid,
    api_key=azure_api_key,
    azure_endpoint=azure_api_base,
    api_version=azure_chatapi_version,
)
Settings.embed_model = AzureOpenAIEmbedding(
    azure_deployment=azure_embedding_deploymentid,
    api_key=azure_api_key,
    azure_endpoint=azure_api_base,
    api_version=azure_embeddingapi_version,
    max_retries=3,
    embed_batch_size=1,
)
text_splitter = SentenceSplitter()
Settings.text_splitter = text_splitter
Settings.prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap_ratio)

example_qs = []
summary = "No Summary available yet"
example_queries = config.example_queries
summary_template = PromptTemplate(sum_template)
example_template = PromptTemplate(eg_template)
qa_template = PromptTemplate(ques_template)

UPLOAD_FOLDER = config.UPLOAD_FOLDER
SUMMARY_FOLDER = config.SUMMARY_FOLDER
VECTOR_FOLDER = config.VECTOR_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(SUMMARY_FOLDER):
    os.makedirs(SUMMARY_FOLDER)
if not os.path.exists(VECTOR_FOLDER):
    os.makedirs(VECTOR_FOLDER)

def clearallfiles():
    # Ensure the UPLOAD_FOLDER is empty
    for root, dir, files in os.walk(UPLOAD_FOLDER):
        for file in files:
            file_path = os.path.join(root, file)
            os.remove(file_path)

def fileformatvaliditycheck(files):
    # Function to check validity of file formats
    for file in files:
        file_name = file.name
        # Get extention of file name
        ext = file_name.split(".")[-1].lower()
        if ext not in ["pdf", "txt", "docx", "png", "jpg", "jpeg", "mp3"]:
            return False
    return True

def build_index():

    documents = SimpleDirectoryReader(UPLOAD_FOLDER).load_data()
    questionindex = VectorStoreIndex.from_documents(documents)
    questionindex.set_index_id("vector_index")
    questionindex.storage_context.persist(persist_dir=VECTOR_FOLDER)
    
    summaryindex = SummaryIndex.from_documents(documents)
    summaryindex.set_index_id("summary_index")
    summaryindex.storage_context.persist(persist_dir=SUMMARY_FOLDER)

def upload_data_to_supabase(title, url):
    try:
        # Load metadata and embedding index
        metadata_index = json.load(open(os.path.join(VECTOR_FOLDER, "docstore.json")))
        embedding_index = json.load(open(os.path.join(VECTOR_FOLDER, "default__vector_store.json")))
        
        # Check if the metadata_index and embedding_index are not empty
        if not metadata_index["docstore/data"] or not embedding_index["embedding_dict"]:
            return "The folder is empty, no data to upload."
        
        # Initialize the Supabase client
        supabase_client = supabase.Client(public_supabase_url, supabase_service_role_key)
        
        # Insert the data for each document into the Supabase table
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
        
        return "All documents have been uploaded successfully."

    except ValueError as ve:
        return str(ve)
    except Exception as e:
        return f"An unexpected error occurred: {e}"

def summary_generator():
    
    try:
        summary = ask_fromfullcontext("Generate a summary of the input context. Be as verbose as possible.", summary_template).lstrip('\n')
    except Exception as e:
        print("Error occurred while generating summary:", str(e))
        summary = "Summary not available"
    return summary

def example_generator():
    
    try:
        llmresponse = ask_fromfullcontext("Generate upto 8 questions. Output must be a valid python literal formatted as a list of lists, where each inner list contains only one question in string format enclosed in double qoutes and square braces. It must be compatible for postprocessing with ast.literal_eval", example_template).lstrip('\n')
        # Debugging
        print("LLM Response:", llmresponse)

        # Check if the response is wrapped in a code block
        if llmresponse.startswith("```python") and llmresponse.endswith("```"):
            # Extract the content inside the code block
            llmresponse = llmresponse.strip("```python").strip()

        #MDebugging
        print("LLM Response after stripping:", llmresponse)

        example_qs = ast.literal_eval(llmresponse.rstrip())
        # Debugging
        print(example_qs)
    except Exception as e:
        print("Error occurred while generating examples:", str(e))
        example_qs = example_queries
    return example_qs

def ask_fromfullcontext(question, fullcontext_template):
    
    storage_context = StorageContext.from_defaults(persist_dir=SUMMARY_FOLDER)
    summary_index = load_index_from_storage(storage_context, index_id="summary_index")
    retriever = summary_index.as_retriever(
        retriever_mode="default",
    )
    response_synthesizer = get_response_synthesizer(
        response_mode="tree_summarize",
        summary_template=fullcontext_template,
    )
    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
    )
    response = query_engine.query(question)
    answer = response.response
    return answer

def extract_video_id(url):
    # Extract the video id from the url
    match = re.search(r"youtu\.be\/(.+)", url)
    if match:
        return match.group(1)
    return url.split("=")[1]

def get_video_title(url, video_id):
    try:
        yt = YouTube(url)
        return yt.title
    except Exception as e:
        print(f"Error occurred while getting video title: {str(e)}")
        return video_id

def transcript_extractor(video_id):
    try:
        transcript_list = YouTubeTranscriptApi.get_transcripts([video_id], languages=['en-IN', 'en'])
        transcript_text = " ".join([transcript["text"] for transcript in transcript_list[0][video_id]])
        with open(os.path.join(UPLOAD_FOLDER, "transcript.txt"), "w") as f:
            f.write(transcript_text)
        return True
    except Exception as e:
        print(f"Error occurred while downloading transcripts: {str(e)}")
        return False

def video_downloader(url):
    yt = YouTube(url)
    yt.streams.filter(progressive=True, file_extension="mp4").order_by("resolution").desc().first().download(UPLOAD_FOLDER, filename="video.mp4")

def process_video(url, memorize, lite_mode):
    video_id = extract_video_id(url)
    video_title = get_video_title(url, video_id)
    video_memoryupload_status = "No data uploaded to Supabase."
    if transcript_extractor(video_id):
        build_index()
        if memorize:
            video_memoryupload_status = upload_data_to_supabase(title=video_title, url=url)
        summary = summary_generator()
        example_queries = example_generator()
        return "Youtube transcript downloaded and Index built successfully!", summary, example_queries, video_title, video_memoryupload_status
    else:
        if not lite_mode:
            video_downloader(url)
            build_index()
            if memorize:
                video_memoryupload_status = upload_data_to_supabase(title=video_title, url=url)
            summary = summary_generator()
            example_queries = example_generator()
            return "Youtube video downloaded and Index built successfully!", summary, example_queries, video_title, video_memoryupload_status
        else:
            return "Youtube transcripts do not exist for this video!", None, None

def analyze_ytvideo(url, memorize, lite_mode=False):
    clearallfiles()
    if not url or "youtube.com" not in url and "youtu.be" not in url:
        return {
            "message": "Please enter a valid Youtube URL",
            "summary": None,
            "example_queries": None,
            "video_title": None,
            "video_memoryupload_status": None
        }
    message, summary, example_queries, video_title, video_memoryupload_status = process_video(url, memorize, lite_mode)
    # Create a dictionary with the results
    results = {
        "message": message,
        "summary": summary,
        "example_queries": example_queries,
        "video_title": video_title,
        "video_memoryupload_status": video_memoryupload_status
    }
    return results

def fileformatvaliditycheck(files):
    # Function to check validity of file formats
    for file in files:
        file_name = file.name
        # Get extention of file name
        ext = file_name.split(".")[-1].lower()
        if ext not in ["pdf", "txt", "docx", "png", "jpg", "jpeg", "mp3"]:
            return False
    return True

def savetodisk(files):
    
    filenames = []
    for file in files:
        filename_with_path = file.name
        file_name = os.path.basename(filename_with_path)
        filenames.append(file_name)
        with open(filename_with_path, 'rb') as f:
            with open(os.path.join(UPLOAD_FOLDER, file_name), 'wb') as f1:
                shutil.copyfileobj(f, f1)
    return filenames

def process_files(files, memorize):
    file_format_validity = fileformatvaliditycheck(files)
    if not file_format_validity:
        return "Please upload documents in pdf/txt/docx/png/jpg/jpeg format only.", None, None
    uploaded_filenames = savetodisk(files)
    file_memoryupload_status = "No data uploaded to Supabase."
    for filename in uploaded_filenames:
        file_title = filename
        build_index()
        if memorize:
            file_memoryupload_status = upload_data_to_supabase(title=file_title, url="Local")
    summary = summary_generator()
    example_queries = example_generator()
    
    return "Files uploaded and Index built successfully!", summary, example_queries, file_title, file_memoryupload_status

def analyze_file(files, memorize):
    clearallfiles()
    if not files:
        return {
            "message": "Please upload a file before proceeding",
            "summary": None,
            "example_queries": None,
            "file_title": None,
            "file_memoryupload_status": None
        }
    message, summary, example_queries, file_title, file_memoryupload_status = process_files(files, memorize)
    # Create a dictionary with the results
    results = {
        "message": message,
        "summary": summary,
        "example_queries": example_queries,
        "file_title": file_title,
        "file_memoryupload_status": file_memoryupload_status
    }
    
    return results

def download_and_parse_article(url):
    article = Article(url)
    try:
        article.download()
        article.parse()
        if len(article.text.split()) < 75:
            raise Exception("Article is too short. Probably the article is behind a paywall.")
        return article
    except Exception as e:
        print(f"Failed to download and parse article from URL using newspaper package: {url}. Error: {str(e)}")
        return None

def alternative_article_download(url):
    try:
        req = requests.get(url)
        soup = BeautifulSoup(req.content, 'html.parser')
        return soup.get_text()
    except Exception as e:
        print(f"Failed to download article using beautifulsoup method from URL: {url}. Error: {str(e)}")
        return None

def save_article_text(text):
    with open(os.path.join(UPLOAD_FOLDER, "article.txt"), 'w') as f:
        f.write(text)

def process_article(url, memorize):
    article = download_and_parse_article(url)
    if not article:
        article_text = alternative_article_download(url)
        if not article_text:
            return "Failed to download and parse article. Please check the URL and try again.", None, None
        else:
            save_article_text(article_text)
            article_title = "Unknown"  # Since we don't have the title from the alternative method
    else:
        save_article_text(article.text)
        article_title = article.title
    article_memoryupload_status = "No data uploaded to Supabase."
    build_index()
    if memorize:
        article_memoryupload_status = upload_data_to_supabase(title=article_title, url=url)
    summary = summary_generator()
    example_queries = example_generator()

    return "Article downloaded and Index built successfully!", summary, example_queries, article_title, article_memoryupload_status

def analyze_article(url, memorize):
    clearallfiles()
    if not url:
        return {
            "message": "Please enter a valid URL",
            "summary": None,
            "example_queries": None,
            "article_title": None,
            "article_memoryupload_status": None
        }
    message, summary, example_queries, article_title, article_memoryupload_status = process_article(url, memorize)
    # Create a dictionary with the results
    results = {
        "message": message,
        "summary": summary,
        "example_queries": example_queries,
        "article_title": article_title,
        "article_memoryupload_status": article_memoryupload_status
    }
    return results

def extract_media_url_from_overcast(url):
    req = requests.get(url)
    soup = BeautifulSoup(req.content, 'html.parser')
    audio_tag = soup.find('audio')
    return audio_tag.source['src']

def download_media_file(url):
    return wget.download(url, UPLOAD_FOLDER)

def rename_and_extract_audio(file_name):
    ext = file_name.split(".")[-1].lower()
    if ext in ["m4a", "mp3", "wav"]:
        os.rename(os.path.join(UPLOAD_FOLDER, file_name), os.path.join(UPLOAD_FOLDER, "audio.mp3"))
    elif ext in ["mp4", "mkv"]:
        os.rename(os.path.join(UPLOAD_FOLDER, file_name), os.path.join(UPLOAD_FOLDER, "video.mp4"))
        ffmpeg_extract_audio(os.path.join(UPLOAD_FOLDER, "video.mp4"), os.path.join(UPLOAD_FOLDER, "audio.mp3"))
        os.remove(os.path.join(UPLOAD_FOLDER, "video.mp4"))
    else:
        raise Exception("Invalid media file format")

def transcribe_audio_with_whisper():
    model = whisper.load_model("base")
    return model.transcribe(os.path.join(UPLOAD_FOLDER, "audio.mp3"))

def save_transcript(text):
    with open(os.path.join(UPLOAD_FOLDER, "media_transcript.txt"), "w") as f:
        f.write(text)

def process_media(url, memorize):
    try:
        if "overcast.fm" in url:
            url = extract_media_url_from_overcast(url)
        file_name_with_path = download_media_file(url)
        media_title = os.path.basename(file_name_with_path)
        rename_and_extract_audio(media_title)
    except Exception as e:
        print(f"Failed to download media using wget from URL: {url}. Error: {str(e)}")
        return "Failed to download media. Please check the URL and try again.", None, None

    media_text = transcribe_audio_with_whisper()
    save_transcript(media_text["text"])
    os.remove(os.path.join(UPLOAD_FOLDER, "audio.mp3"))
    media_memoryupload_status = "No data uploaded to Supabase."

    build_index()
    if memorize:
        media_memoryupload_status = upload_data_to_supabase(title=media_title, url=url)
    summary = summary_generator()
    example_queries = example_generator()

    return "Media downloaded and Index built successfully!", summary, example_queries, media_title, media_memoryupload_status

def analyze_media(url, memorize):
    clearallfiles()
    if not url:
        return {
            "message": "Please enter a valid media URL",
            "summary": None,
            "example_queries": None,
            "media_title": None,
            "media_memoryupload_status": None
        }

    message, summary, example_queries, media_title, media_memoryupload_status = process_media(url, memorize)
    
    # Create a dictionary with the results
    results = {
        "message": message,
        "summary": summary,
        "example_queries": example_queries,
        "media_title": media_title,
        "media_memoryupload_status": media_memoryupload_status
    }
    
    return results
