import json
import os
import requests
import gradio as gr
import openai
import PyPDF2
import requests
import re
import ast
import dotenv
import logging
import shutil
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

logger = logging.getLogger()
logger.level = logging.WARN

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
num_output = 1024
# set maximum chunk overlap
max_chunk_overlap = 24
# set chunk size limit
chunk_size_limit = 256
# set prompt helper
prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)

# Set a flag for lite mode. Choose lite mode if you dont want to analyze videos without transcripts
lite_mode = True

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

#UPLOAD_FOLDER = './data'  # set the upload folder path
UPLOAD_FOLDER = os.path.join(".", "data")
example_queries = [["Generate key 5 point summary"], ["What are 5 main ideas of this article?"], ["What are the key lessons learned and insights in this video?"], ["List key insights and lessons learned from the paper"], ["What are the key takeaways from this article?"]]
example_qs = []
summary = "No Summary available yet"

# If the UPLOAD_FOLDER path does not exist, create it
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Function to generate the trip plan
def generate_trip_plan(city, days):
    #Check if the days input is a number and throw an error if it is not
    try:
        days = int(days)
        prompt = f"List the popular tourist attractions in {city} including top rated restaurants that can be visited in {days} days. Be sure to arrage the places optimized for distance and time and output must contain a numbered list with a short, succinct description of each place."
        completions = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=1024,
            n=1,
            stop=None,
            temperature=0.5,
        )
        message = completions.choices[0].text
        return f"Here is your trip plan for {city} for {days} day(s): {message}"
    except:
        return "Please enter a number for days."

def craving_satisfier(city, food_craving):
    # If the food craving is input as "idk", generate a random food craving
    if food_craving in ["idk","I don't know","I don't know what I want","I don't know what I want to eat","I don't know what I want to eat.","Idk"]:
        # Generate a random food craving
        food_craving = openai.Completion.create(
            engine="text-davinci-003",
            prompt="Generate a random food craving",
            max_tokens=64,
            n=1,
            stop=None,
            temperature=0.8,
        )
        food_craving = food_craving.choices[0].text
        # Remove 2 new line characters from the beginning of the string
        food_craving = food_craving[2:]
        print(f"Don't worry, yo! I think you are craving for {food_craving}!")
    else:
        print(f"That's a great choice! My mouth is watering just thinking about {food_craving}!")

    prompt = f"I'm looking for 3 restaurants in {city} that serves {food_craving}. Just give me a list of 3 restaurants with short address."
    completions = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=128,
        n=1,
        stop=None,
        temperature=0.5,
        top_p=1.0,
    )
    message = completions.choices[0].text
    # Remove new line characters from the beginning of the string
    message = message[1:]
    return f'Here are 3 restaurants in {city} that serve {food_craving}! Bon Appetit!! {message}'

def fileformatvaliditycheck(files):
    # Function to check validity of file formats
    for file in files:
        file_name = file.name
        # Get extention of file name
        ext = file_name.split(".")[-1].lower()
        if ext not in ["pdf", "txt", "docx", "png", "jpg", "jpeg"]:
            return False
    return True

def savetodisk(files):
    
    filenames = []
    # Save the files to the UPLOAD_FOLDER
    for file in files:
        # Extract the file name
        filename_with_path = file.name
        file_name = os.path.basename(filename_with_path)
        filenames.append(file_name)
        # Open the file in read-binary mode
        with open(filename_with_path, 'rb') as f:
            # Save the file to the UPLOAD_FOLDER
            with open(os.path.join(UPLOAD_FOLDER, file_name), 'wb') as f1:
                shutil.copyfileobj(f, f1)
    return filenames

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

def upload_file(files, memorize):

    clearallfiles()
    global example_queries, summary
    # Basic checks
    if not files:
        return "Please upload a file before proceeding", gr.Dataset.update(samples=example_queries), summary

    fileformatvalidity = fileformatvaliditycheck(files)
    # Check if all the files are in the correct format
    if not fileformatvalidity:
        return "Please upload documents in pdf/txt/docx/png/jpg/jpeg format only.", gr.Dataset.update(samples=example_queries), summary

    # Save files to UPLOAD_FOLDER
    uploaded_filenames = savetodisk(files)
    # Build index
    build_index()
    # Upload data to Supabase if the memorize checkbox is checked
    if memorize:
        index_data = json.load(open(os.path.join(UPLOAD_FOLDER, "index.json")))
        upload_data_to_supabase(index_data, title=uploaded_filenames[0], url="Local")
    # Generate summary
    summary = summary_generator()
    # Generate example queries
    example_queries = example_generator()

    return "Files uploaded and Index built successfully!", gr.Dataset.update(samples=example_queries), summary

def download_ytvideo(url, memorize):

    clearallfiles()
    global example_queries, summary
    if url:
        # Check if the URL belongs to YouTube
        if "youtube.com" in url or "youtu.be" in url:
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
                # Upload data to Supabase if the memorize checkbox is checked
                if memorize:
                    index_data = json.load(open(os.path.join(UPLOAD_FOLDER, "index.json")))
                    upload_data_to_supabase(index_data, title=video_title, url=url)
                # Generate summary
                summary = summary_generator()
                # Generate example queries
                example_queries = example_generator()
                return "Youtube transcript downloaded and Index built successfully!", gr.Dataset.update(samples=example_queries), summary
            # If the video does not have transcripts, download the video and post-process it locally
            else:
                if not lite_mode:
                    # Download the video and post-process it if there are no captions
                    yt = YouTube(url)
                    yt.streams.filter(progressive=True, file_extension="mp4").order_by("resolution").desc().first().download(UPLOAD_FOLDER, filename="video.mp4")
                    # Clear files from UPLOAD_FOLDER
                    #clearnonvideos()
                    # Build index
                    build_index()
                    # Upload data to Supabase if the memorize checkbox is checked
                    if memorize:
                        index_data = json.load(open(os.path.join(UPLOAD_FOLDER, "index.json")))
                        upload_data_to_supabase(index_data, title=video_title, url=url)
                    # Generate summary
                    summary = summary_generator()
                    # Generate example queries
                    example_queries = example_generator()

                    return "Youtube video downloaded and Index built successfully!", gr.Dataset.update(samples=example_queries), summary
                elif lite_mode:
                    return "Youtube transcripts do not exist for this video!", gr.Dataset.update(samples=example_queries), summary
        else:
            return "Please enter a valid Youtube URL", gr.Dataset.update(samples=example_queries), summary
    else:
        return "Please enter a valid Youtube URL", gr.Dataset.update(samples=example_queries), summary

def download_art(url, memorize):

    clearallfiles()
    global example_queries, summary
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
                return "Failed to download and parse article. Please check the URL and try again.", gr.Dataset.update(samples=example_queries), summary
        # Save the article to the UPLOAD_FOLDER
        with open(os.path.join(UPLOAD_FOLDER, "article.txt"), 'w') as f:
            f.write(article.text)
        # Build index
        build_index()
        # Upload data to Supabase if the memorize checkbox is checked
        if memorize:
            index_data = json.load(open(os.path.join(UPLOAD_FOLDER, "index.json")))
            upload_data_to_supabase(index_data, title=article.title, url=url)
        # Generate summary
        summary = summary_generator()
        # Generate example queries
        example_queries = example_generator()

        return "Article downloaded and Index built successfully!", gr.Dataset.update(samples=example_queries), summary
    else:
        return "Please enter a valid URL", gr.Dataset.update(samples=example_queries), summary

def ask(question, history):
    
    history = history or []
    s = list(sum(history, ()))
    s.append(question)
    inp = ' '.join(s)

    index = GPTSimpleVectorIndex.load_from_disk(os.path.join(UPLOAD_FOLDER, "index.json"), service_context=service_context)
    #index = GPTListIndex.load_from_disk(UPLOAD_FOLDER + "/index.json", service_context=service_context)
    response = index.query(question, similarity_top_k=3, mode="embedding")
    answer = response.response

    history.append((question, answer))

    return history, history

def ask_query(question):

    index = GPTSimpleVectorIndex.load_from_disk(os.path.join(UPLOAD_FOLDER, "index.json"), service_context=service_context)
    #index = GPTListIndex.load_from_disk(UPLOAD_FOLDER + "/index.json", service_context=service_context)
    response = index.query(question, similarity_top_k=3, mode="embedding")
    answer = response.response

    return answer

def ask_fromfullcontext(question):
    
    index = GPTSimpleVectorIndex.load_from_disk(os.path.join(UPLOAD_FOLDER, "index.json"), service_context=service_context)
    #index = GPTListIndex.load_from_disk(UPLOAD_FOLDER + "/index.json", service_context=service_context)
    response = index.query(question, response_mode="tree_summarize")
    answer = response.response
    
    return answer

def example_generator():
    
    global example_queries, example_qs
    try:
        llmresponse = ask_fromfullcontext("You are a helpful assistant that is helping the user to gain more knowledge about the input context. Generate atleast 5 relevant questions that would enable the user to get key ideas from the input context. Disregard any irrelevant information such as discounts, promotions, sponsorships or advertisements from the context. Output must be must in the form of python list of 5 strings, 1 string for each question enclosed in double quotes.")
        example_qs = [[str(item)] for item in ast.literal_eval(llmresponse.rstrip())]
    except Exception as e:
        print("Error occurred while generating examples:", str(e))
        example_qs = example_queries
    return example_qs

def summary_generator():
    
    global summary
    try:
        summary = ask_fromfullcontext("Summarize the input context while preserving the main points and information into a numbered list of at least 10 key points and takeaways, with a catchy headline at the top. Try to use your own words, and avoid copying word-for-word from the provided context. Do not include any irrelevant information such as discounts, promotions, sponsorships or advertisements in your summary and stick to the core message of the content.").lstrip('\n')
    except Exception as e:
        print("Error occurred while generating summary:", str(e))
        summary = "Summary not available"
    return summary

def update_examples():
    
    global example_queries
    example_queries = example_generator()
    return gr.Dataset.update(samples=example_queries)

def load_example(example_id):
    
    global example_queries
    return example_queries[example_id][0]

def cleartext(query, output):
    # Function to clear text
    return ["", ""]

def clearhistory(field1, field2, field3):
    # Function to clear history
    return ["", "", ""]

with gr.Blocks(css="#chatbot .overflow-y-auto{height:500px}") as llmapp:
    gr.Markdown(
        """
        <h1><center><b>LLM Bot</center></h1>
        """
    )
    gr.Markdown(
        """
        This app uses the Transformer magic to answer all your questions! Check the "Memorize" box if you want to add the information to your memory palace!
        """
    )
    with gr.Row():
        memorize = gr.Checkbox(label="I want this information stored in my memory palace!")
    with gr.Row():
        with gr.Column(scale=1, min_width=250):
            with gr.Box():
                files = gr.File(label="Upload the files to be analyzed", file_count="multiple")
                with gr.Row():
                    upload_button = gr.Button(value="Upload", scale=0)
                    upload_output = gr.Textbox(label="Upload Status")
            with gr.Tab(label="Video Analyzer"):
                yturl = gr.Textbox(placeholder="Input must be a Youtube URL", label="Enter Youtube URL")
                with gr.Row():
                    download_button = gr.Button(value="Download", scale=0)
                    download_output = gr.Textbox(label="Video download Status")
            with gr.Tab(label="Article Analyzer"):
                arturl = gr.Textbox(placeholder="Input must be a URL", label="Enter Article URL")
                with gr.Row():
                    adownload_button = gr.Button(value="Download", scale=0)
                    adownload_output = gr.Textbox(label="Article download Status")
        with gr.Column(scale=2, min_width=650):
            with gr.Box():
                summary_output = gr.Textbox(placeholder="Summary will be generated here", label="Key takeaways")
                chatbot = gr.Chatbot(elem_id="chatbot", label="LLM Bot")
                state = gr.State([])
                with gr.Row():
                    query = gr.Textbox(show_label=False, placeholder="Enter text and press enter")
                    submit_button = gr.Button(value="Ask", scale=0)
                    clearquery_button = gr.Button(value="Clear", scale=0)
                examples = gr.Dataset(samples=example_queries, components=[query], type="index")
                submit_button.click(ask, inputs=[query, state], outputs=[chatbot, state])
                query.submit(ask, inputs=[query, state], outputs=[chatbot, state])
            clearchat_button = gr.Button(value="Clear Chat", scale=0)
    with gr.Row():
        with gr.Tab(label="Trip Generator"):
            with gr.Row():
                city_name = gr.Textbox(placeholder="Enter the name of the city", label="City Name")
                number_of_days = gr.Textbox(placeholder="Enter the number of days", label="Number of Days")
                city_button = gr.Button(value="Plan", scale=0)
            with gr.Row():
                city_output = gr.Textbox(label="Trip Plan")
                clear_trip_button = gr.Button(value="Clear", scale=0)
        with gr.Tab(label="Cravings Generator"):
            with gr.Row():
                craving_city_name = gr.Textbox(placeholder="Enter the name of the city", label="City Name")
                craving_cuisine = gr.Textbox(placeholder="What kind of food are you craving for? Enter idk if you don't know what you want to eat", label="Food")
                craving_button = gr.Button(value="Cook", scale=0)
            with gr.Row():
                craving_output = gr.Textbox(label="Food Places")
                clear_craving_button = gr.Button(value="Clear", scale=0)

    # Upload button for uploading files
    upload_button.click(upload_file, inputs=[files, memorize], outputs=[upload_output, examples, summary_output], show_progress=True)
    # Download button for downloading youtube video
    download_button.click(download_ytvideo, inputs=[yturl, memorize], outputs=[download_output, examples, summary_output], show_progress=True)
    # Download button for downloading article
    adownload_button.click(download_art, inputs=[arturl, memorize], outputs=[adownload_output, examples, summary_output], show_progress=True)
    # City Planner button
    city_button.click(generate_trip_plan, inputs=[city_name, number_of_days], outputs=[city_output], show_progress=True)
    # Cravings button
    craving_button.click(craving_satisfier, inputs=[craving_city_name, craving_cuisine], outputs=[craving_output], show_progress=True)

    # Load example queries
    examples.click(load_example, inputs=[examples], outputs=[query])

    clearquery_button.click(cleartext, inputs=[query, query], outputs=[query, query])
    clearchat_button.click(cleartext, inputs=[query, chatbot], outputs=[query,chatbot])
    clear_trip_button.click(clearhistory, inputs=[city_name, number_of_days, city_output], outputs=[city_name, number_of_days, city_output])
    clear_craving_button.click(clearhistory, inputs=[craving_city_name, craving_cuisine, craving_output], outputs=[craving_city_name, craving_cuisine, craving_output])
    # live = True

if __name__ == '__main__':
    llmapp.launch(server_name='0.0.0.0', server_port=7860)
