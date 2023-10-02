import json
import os
import requests
import openai
import requests
import re
import dotenv
import supabase
import tiktoken
import argparse
from datetime import datetime
from newspaper import Article
from bs4 import BeautifulSoup
from pytube import YouTube
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.embeddings import OpenAIEmbeddings
from llama_index.llms import AzureOpenAI
from llama_index import (
    VectorStoreIndex,
    SummaryIndex,
    LangchainEmbedding,
    PromptHelper,
    SimpleDirectoryReader,
    ServiceContext,
    StorageContext,
    load_index_from_storage,
    get_response_synthesizer,
    set_global_service_context,
)
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.text_splitter import SentenceSplitter
from llama_index.node_parser import SimpleNodeParser
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
    supabase_client = supabase.Client(SUPABASE_URL, SUPABASE_API_KEY)
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
                metadata_index = json.load(open(os.path.join(VECTOR_FOLDER, "docstore.json")))
                embedding_index = json.load(open(os.path.join(VECTOR_FOLDER, "vector_store.json")))
                upload_data_to_supabase(metadata_index, embedding_index, title=video_title, url=url)
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
                return summary
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

def ask_fromfullcontext(question, fullcontext_template):
    
    # Reset OpenAI API type and base
    openai.api_type = azure_api_type
    openai.api_base = azure_api_base
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
    
    # logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    # logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

    # Get API key from environment variable
    dotenv.load_dotenv()
    azure_api_key = os.environ["AZURE_API_KEY"]
    azure_api_type = "azure"
    azure_api_base = os.environ.get("AZURE_API_BASE")
    azure_embeddingapi_version = os.environ.get("AZURE_EMBEDDINGAPI_VERSION")
    azure_chatapi_version = os.environ.get("AZURE_CHATAPI_VERSION")
    EMBEDDINGS_DEPLOYMENT_NAME = "text-embedding-ada-002"
    #Supabase API key
    SUPABASE_API_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    SUPABASE_URL = os.environ.get("PUBLIC_SUPABASE_URL")

    os.environ["OPENAI_API_KEY"] = azure_api_key
    openai.api_type = azure_api_type
    openai.api_base = azure_api_base
    openai.api_key = azure_api_key

   # Check if user set the davinci model flag
    gpt4_flag = False
    if gpt4_flag:
        LLM_DEPLOYMENT_NAME = "gpt-4-32k"
        LLM_MODEL_NAME = "gpt-4-32k"
        openai.api_version = azure_chatapi_version
        max_input_size = 96000
        context_window = 32000
    else:
        LLM_DEPLOYMENT_NAME = "gpt-35-turbo-16k"
        LLM_MODEL_NAME = "gpt-35-turbo-16k"
        openai.api_version = azure_chatapi_version
        max_input_size = 48000
        context_window = 16000

    # max LLM token input size
    num_output = 1024
    max_chunk_overlap_ratio = 0.1
    chunk_size = 256
    prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap_ratio)
    text_splitter = SentenceSplitter(
        separator=" ",
        chunk_size=chunk_size,
        chunk_overlap=20,
        paragraph_separator="\n\n\n",
        secondary_chunking_regex="[^,.;。]+[,.;。]?",
        tokenizer=tiktoken.encoding_for_model("gpt-35-turbo").encode
    )
    node_parser = SimpleNodeParser(text_splitter=text_splitter)
    # Set a flag for lite mode: Choose lite mode if you dont want to analyze videos without transcripts
    lite_mode = True

    llm = AzureOpenAI(
        engine=LLM_DEPLOYMENT_NAME, 
        model=LLM_MODEL_NAME,
        api_key=azure_api_key,
        api_base=azure_api_base,
        api_type=azure_api_type,
        api_version=azure_chatapi_version,
        temperature=0.5,
        max_tokens=num_output,
    )
    embedding_llm = LangchainEmbedding(
        OpenAIEmbeddings(
            model=EMBEDDINGS_DEPLOYMENT_NAME,
            deployment=EMBEDDINGS_DEPLOYMENT_NAME,
            openai_api_key=azure_api_key,
            openai_api_base=azure_api_base,
            openai_api_type=azure_api_type,
            openai_api_version=azure_embeddingapi_version,
            chunk_size=16,
            max_retries=3,
        ),
        embed_batch_size=1,
    )
    service_context = ServiceContext.from_defaults(
        llm=llm,
        embed_model=embedding_llm,
        prompt_helper=prompt_helper,
        chunk_size=chunk_size,
        context_window=context_window,
        node_parser=node_parser,
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
    if "youtube.com/watch" in url or "youtu.be/" in url:
        summary = download_ytvideo(url, memorize)
    elif "http" in url:
        summary = download_art(url, memorize)
    else:
        summary = "Invalid URL. Please enter a valid article or YouTube video URL.", "Summary not available"

    print(summary)
