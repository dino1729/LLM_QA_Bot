"""
Content Analyzers for Files, Videos, Articles, and Media
Uses unified LLM client, removed Supabase dependency
"""
from config import config
from helper_functions.llm_client import get_client
import os
import shutil
import re
import tiktoken
import ast
import whisper
import wget
import requests
import logging
import sys
from newspaper import Article
from bs4 import BeautifulSoup
import yt_dlp
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_audio
from youtube_transcript_api import YouTubeTranscriptApi
from llama_index.core import VectorStoreIndex, PromptHelper, SimpleDirectoryReader, StorageContext, load_index_from_storage, get_response_synthesizer
from llama_index.core.indices import SummaryIndex
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import PromptTemplate
from llama_index.core import Settings
from helper_functions.memory_palace_local import save_memory

logging.basicConfig(stream=sys.stdout, level=logging.CRITICAL)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# Configuration
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

# Set a flag for lite mode: Choose lite mode if you don't want to analyze videos without transcripts
lite_mode = False

# Initialize unified LLM client (using fast model for summary/example generation)
default_client = get_client(provider="litellm", model_tier="fast")

Settings.llm = default_client.get_llamaindex_llm()
Settings.embed_model = default_client.get_llamaindex_embedding()
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
    """Clear all files in the upload folder"""
    for root, dirs, files in os.walk(UPLOAD_FOLDER):
        for file in files:
            file_path = os.path.join(root, file)
            os.remove(file_path)


def fileformatvaliditycheck(files):
    """Check if file formats are valid"""
    for file in files:
        file_name = file.name
        ext = file_name.split(".")[-1].lower()
        if ext not in ["pdf", "txt", "docx", "png", "jpg", "jpeg", "mp3"]:
            return False
    return True


def build_index():
    """Build vector and summary indexes from uploaded documents"""
    documents = SimpleDirectoryReader(UPLOAD_FOLDER).load_data()
    questionindex = VectorStoreIndex.from_documents(documents)
    questionindex.set_index_id("vector_index")
    questionindex.storage_context.persist(persist_dir=VECTOR_FOLDER)

    summaryindex = SummaryIndex.from_documents(documents)
    summaryindex.set_index_id("summary_index")
    summaryindex.storage_context.persist(persist_dir=SUMMARY_FOLDER)


def summary_generator():
    """Generate summary from indexed documents"""
    try:
        summary = ask_fromfullcontext(
            "Generate a concise, high-level executive summary of the input context.",
            summary_template
        ).lstrip('\n')
        print(f"Summary generated successfully, length: {len(summary) if summary else 0}")
    except Exception as e:
        print(f"Error occurred while generating summary: {str(e)}")
        summary = "Summary not available"
    return summary


def example_generator():
    """Generate example questions from indexed documents"""
    try:
        llmresponse = ask_fromfullcontext(
            "Generate upto 8 questions. Output must be a valid python literal formatted as a list of lists, where each inner list contains only one question in string format enclosed in double quotes and square braces. It must be compatible for postprocessing with ast.literal_eval",
            example_template
        ).lstrip('\n')

        # Check if the response is wrapped in a code block
        if llmresponse.startswith("```python") and llmresponse.endswith("```"):
            llmresponse = llmresponse.strip("```python").strip()

        example_qs = ast.literal_eval(llmresponse.rstrip())
    except Exception as e:
        print(f"Error occurred while generating examples: {str(e)}")
        example_qs = example_queries
    return example_qs


def ask_fromfullcontext(question, fullcontext_template):
    """Query the summary index with a question"""
    storage_context = StorageContext.from_defaults(persist_dir=SUMMARY_FOLDER)
    summary_index = load_index_from_storage(storage_context, index_id="summary_index")
    retriever = summary_index.as_retriever(retriever_mode="default")
    response_synthesizer = get_response_synthesizer(
        llm=Settings.llm,  # Explicitly pass the LLM from Settings
        response_mode="tree_summarize",
        summary_template=fullcontext_template,
    )
    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
    )
    response = query_engine.query(question)
    answer = response.response
    if not answer:
        logging.warning("Empty response from query engine")
    return answer


# ========== YouTube Video Analysis ==========

def extract_video_id(url):
    """Extract video ID from YouTube URL"""
    match = re.search(r"youtu\.be\/(.+)", url)
    if match:
        return match.group(1)
    if "youtube.com" in url:
        return url.split("v=")[1].split("&")[0]
    return url


def get_video_title(url, video_id):
    """Get YouTube video title"""
    try:
        with yt_dlp.YoutubeDL() as ydl:
            info = ydl.extract_info(url, download=False)
            return info.get('title', video_id)
    except Exception as e:
        print(f"Error occurred while getting video title: {str(e)}")
        return video_id


def transcript_extractor(video_id):
    """Extract transcript from YouTube video"""
    try:
        api = YouTubeTranscriptApi()
        transcript = api.fetch(video_id, languages=['en-IN', 'en'])
        transcript_text = " ".join([snippet.text for snippet in transcript.snippets])
        with open(os.path.join(UPLOAD_FOLDER, "transcript.txt"), "w") as f:
            f.write(transcript_text)
        return True
    except Exception as e:
        print(f"Error occurred while downloading transcripts: {str(e)}")
        return False


def video_downloader(url):
    """Download YouTube video"""
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': os.path.join(UPLOAD_FOLDER, 'video.mp4')
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])


def process_video(url, memorize, lite_mode):
    """Process YouTube video - extract transcript or download video"""
    video_id = extract_video_id(url)
    video_title = get_video_title(url, video_id)

    if transcript_extractor(video_id):
        build_index()
        summary = summary_generator()
        example_queries = example_generator()
        return "Youtube transcript downloaded and Index built successfully!", summary, example_queries, video_title
    else:
        if not lite_mode:
            try:
                video_downloader(url)
                build_index()
                summary = summary_generator()
                example_queries = example_generator()
                return "Youtube video downloaded and Index built successfully!", summary, example_queries, video_title
            except Exception as e:
                return f"Failed to download video: {str(e)}", None, None, video_title
        else:
            return "Youtube transcripts do not exist for this video!", None, None, None


def analyze_ytvideo(url, memorize, lite_mode=False, model_name="LITELLM"):
    """
    Analyze YouTube video
    """
    clearallfiles()
    if not url or ("youtube.com" not in url and "youtu.be" not in url):
        return {
            "message": "Please enter a valid Youtube URL",
            "summary": None,
            "example_queries": None,
            "video_title": None,
            "video_memoryupload_status": "Memory upload skipped"
        }
    message, summary, example_queries, video_title = process_video(url, memorize, lite_mode)
    
    memory_status = "Memory upload skipped"
    if memorize and summary:
        try:
            memory_status = save_memory(
                title=video_title,
                content=summary,
                source_type="video",
                source_ref=url,
                model_name=model_name
            )
        except Exception as e:
            print(f"Memory Palace save failed: {e}")
            memory_status = f"Memory upload failed: {e}"

    results = {
        "message": message,
        "summary": summary,
        "example_queries": example_queries,
        "video_title": video_title,
        "video_memoryupload_status": memory_status
    }
    return results


# ========== File Analysis ==========

def savetodisk(files):
    """Save uploaded files to disk"""
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
    """Process uploaded files"""
    file_format_validity = fileformatvaliditycheck(files)
    if not file_format_validity:
        return "Please upload documents in pdf/txt/docx/png/jpg/jpeg format only.", None, None, None

    uploaded_filenames = savetodisk(files)
    file_title = uploaded_filenames[-1] if uploaded_filenames else "Unknown"

    build_index()
    summary = summary_generator()
    example_queries = example_generator()

    return "Files uploaded and Index built successfully!", summary, example_queries, file_title


def analyze_file(files, memorize, model_name="LITELLM"):
    """
    Analyze uploaded files
    """
    clearallfiles()
    if not files:
        return {
            "message": "Please upload a file before proceeding",
            "summary": None,
            "example_queries": None,
            "file_title": None,
            "file_memoryupload_status": "Memory upload skipped"
        }
    message, summary, example_queries, file_title = process_files(files, memorize)
    
    memory_status = "Memory upload skipped"
    if memorize and summary:
        try:
            # For files, source_ref is the filename of the last processed file
            memory_status = save_memory(
                title=file_title,
                content=summary,
                source_type="file",
                source_ref=file_title,
                model_name=model_name
            )
        except Exception as e:
            print(f"Memory Palace save failed: {e}")
            memory_status = f"Memory upload failed: {e}"

    results = {
        "message": message,
        "summary": summary,
        "example_queries": example_queries,
        "file_title": file_title,
        "file_memoryupload_status": memory_status
    }
    return results


# ========== Article Analysis ==========

def download_and_parse_article(url):
    """Download and parse article using newspaper3k"""
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
    """Alternative article download using BeautifulSoup"""
    try:
        req = requests.get(url)
        soup = BeautifulSoup(req.content, 'html.parser')
        return soup.get_text()
    except Exception as e:
        print(f"Failed to download article using beautifulsoup method from URL: {url}. Error: {str(e)}")
        return None


def save_article_text(text):
    """Save article text to file"""
    with open(os.path.join(UPLOAD_FOLDER, "article.txt"), 'w') as f:
        f.write(text)


def process_article(url, memorize):
    """Process article from URL"""
    article = download_and_parse_article(url)
    if not article:
        article_text = alternative_article_download(url)
        if not article_text:
            return "Failed to download and parse article. Please check the URL and try again.", None, None, None
        else:
            save_article_text(article_text)
            article_title = "Unknown"
    else:
        save_article_text(article.text)
        article_title = article.title

    build_index()
    summary = summary_generator()
    example_queries = example_generator()

    return "Article downloaded and Index built successfully!", summary, example_queries, article_title


def analyze_article(url, memorize, model_name="LITELLM"):
    """
    Analyze article from URL
    """
    clearallfiles()
    if not url:
        return {
            "message": "Please enter a valid URL",
            "summary": None,
            "example_queries": None,
            "article_title": None,
            "article_memoryupload_status": "Memory upload skipped"
        }
    message, summary, example_queries, article_title = process_article(url, memorize)
    
    memory_status = "Memory upload skipped"
    if memorize and summary:
        try:
            memory_status = save_memory(
                title=article_title,
                content=summary,
                source_type="article",
                source_ref=url,
                model_name=model_name
            )
        except Exception as e:
            print(f"Memory Palace save failed: {e}")
            memory_status = f"Memory upload failed: {e}"

    results = {
        "message": message,
        "summary": summary,
        "example_queries": example_queries,
        "article_title": article_title,
        "article_memoryupload_status": memory_status
    }
    return results


# ========== Media Analysis ==========

def extract_media_url_from_overcast(url):
    """Extract media URL from Overcast podcast player"""
    req = requests.get(url)
    soup = BeautifulSoup(req.content, 'html.parser')
    audio_tag = soup.find('audio')
    # Try to get src from audio tag itself, or from source child element
    if 'src' in audio_tag.attrs:
        return audio_tag['src']
    else:
        source_tag = audio_tag.find('source')
        return source_tag['src'] if source_tag else None


def download_media_file(url):
    """Download media file"""
    return wget.download(url, UPLOAD_FOLDER)


def rename_and_extract_audio(file_name):
    """Rename and extract audio from media file"""
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
    """Transcribe audio using Whisper"""
    model = whisper.load_model("base")
    return model.transcribe(os.path.join(UPLOAD_FOLDER, "audio.mp3"))


def save_transcript(text):
    """Save transcript to file"""
    with open(os.path.join(UPLOAD_FOLDER, "media_transcript.txt"), "w") as f:
        f.write(text)


def process_media(url, memorize):
    """Process media file (audio/video) from URL"""
    try:
        if "overcast.fm" in url:
            url = extract_media_url_from_overcast(url)
        file_name_with_path = download_media_file(url)
        media_title = os.path.basename(file_name_with_path)
        rename_and_extract_audio(media_title)
    except Exception as e:
        print(f"Failed to download media using wget from URL: {url}. Error: {str(e)}")
        return "Failed to download media. Please check the URL and try again.", None, None, None

    media_text = transcribe_audio_with_whisper()
    save_transcript(media_text["text"])
    os.remove(os.path.join(UPLOAD_FOLDER, "audio.mp3"))

    build_index()
    summary = summary_generator()
    example_queries = example_generator()

    return "Media downloaded and Index built successfully!", summary, example_queries, media_title


def analyze_media(url, memorize, model_name="LITELLM"):
    """
    Analyze media from URL
    """
    clearallfiles()
    if not url:
        return {
            "message": "Please enter a valid media URL",
            "summary": None,
            "example_queries": None,
            "media_title": None,
            "media_memoryupload_status": "Memory upload skipped"
        }

    message, summary, example_queries, media_title = process_media(url, memorize)

    memory_status = "Memory upload skipped"
    if memorize and summary:
        try:
            memory_status = save_memory(
                title=media_title,
                content=summary,
                source_type="media",
                source_ref=url,
                model_name=model_name
            )
        except Exception as e:
            print(f"Memory Palace save failed: {e}")
            memory_status = f"Memory upload failed: {e}"

    results = {
        "message": message,
        "summary": summary,
        "example_queries": example_queries,
        "media_title": media_title,
        "media_memoryupload_status": memory_status
    }
    
    return results


# Note: upload_data_to_supabase function has been removed as Supabase is no longer used
