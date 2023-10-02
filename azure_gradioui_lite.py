import json
import os
import pinecone
import requests
import gradio as gr
import openai
import requests
import re
import ast
import dotenv
import logging
import shutil
import supabase
import tiktoken
import sys
import cohere
import google.generativeai as palm
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
from llama_index.retrievers import VectorIndexRetriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.indices.postprocessor import SimilarityPostprocessor
from llama_index.text_splitter import SentenceSplitter
from llama_index.node_parser import SimpleNodeParser
from llama_index.prompts import PromptTemplate
from llama_index.agent import OpenAIAgent
from llama_hub.tools.weather.base import OpenWeatherMapToolSpec

def generate_trip_plan(city, days):
    #Check if the days input is a number and throw an error if it is not
    try:
        days = int(days)

        tripsystem_prompt = [{
            "role": "system",
            "content": "You are a world class trip planner who is knowledgeable about all the tourist attractions in the world. You will serve the user by planning a trip for them and respecting all their preferences."
        }]
        conversation = tripsystem_prompt.copy()
        user_message = f"Craft a thorough and detailed travel itinerary for {city}. This itinerary should encompass the city's most frequented tourist attractions, as well as its top-rated restaurants, all of which should be visitable within a timeframe of {days} days. The itinerary should be strategically organized to take into account the distance between each location and the time required to travel there, maximizing efficiency. Moreover, please include specific time windows for each location, arranged in ascending order, to facilitate effective planning. The final output should be a numbered list, where each item corresponds to a specific location. Accompany each location with a brief yet informative description to provide context and insight."
        conversation.append({"role": "user", "content": str(user_message)})

        openai.api_type = azure_api_type
        openai.api_base = azure_api_base
        response = openai.ChatCompletion.create(
            engine="gpt-35-turbo",
            messages=conversation,
            max_tokens=2048,
            temperature=0.3,
        )
        message = response['choices'][0]['message']['content']
        conversation.append({"role": "assistant", "content": str(message)})
        return f"Here is your trip plan for {city} for {days} day(s): {message}"
    except:
        return "Please enter a number for days."

def craving_satisfier(city, food_craving):
    # If the food craving is input as "idk", generate a random food craving
    if food_craving in ["idk","I don't know","I don't know what I want","I don't know what I want to eat","I don't know what I want to eat.","Idk"]:
        # Generate a random food craving
        foodsystem_prompt = [{
            "role": "system",
            "content": "You are a world class food recommender who is knowledgeable about all the food items in the world. The user will ask you to generate a food craving and you must respond in one-word answer."
        }]
        conversation1 = foodsystem_prompt.copy()
        user_message1 = f"I don't know what to eat and I want you to generate a random cuisine. Be as creative as possible"
        conversation1.append({"role": "user", "content": str(user_message1)})
        openai.api_type = azure_api_type
        openai.api_base = azure_api_base
        response1 = openai.ChatCompletion.create(
            engine="gpt-35-turbo",
            messages=conversation1,
            max_tokens=32,
            temperature=0.5,
        )
        food_craving = response1['choices'][0]['message']['content']
        conversation1.append({"role": "assistant", "content": str(food_craving)})
        print(f"Don't worry, yo! I think you are craving for {food_craving}!")
    else:
        print(f"That's a great choice! My mouth is watering just thinking about {food_craving}!")

    restaurantsystem_prompt = [{
        "role": "system",
        "content": "You are a world class restaurant recommender who is knowledgeable about all the restaurants in the world. You will serve the user by recommending restaurants."
    }]
    conversation2 = restaurantsystem_prompt.copy()
    user_message2 = f"I'm looking for 8 restaurants in {city} that serves {food_craving}. Provide me with a list of six restaurants, including their brief addresses. Also, mention one dish from each that particularly stands out, ensuring it contains neither beef nor pork."
    conversation2.append({"role": "user", "content": str(user_message2)})

    openai.api_type = azure_api_type
    openai.api_base = azure_api_base
    response2 = openai.ChatCompletion.create(
        engine="gpt-35-turbo",
        messages=conversation2,
        max_tokens=2048,
        temperature=0.4,
    )
    message = response2['choices'][0]['message']['content']
    conversation2.append({"role": "assistant", "content": str(message)})

    return f'Here are 8 restaurants in {city} that serve {food_craving}! Bon Appetit!! \n {message}'

def extract_context_frompinecone(query):
    
    holybook = "gita"
    pinecone.init(api_key=pinecone_api_key, environment=pinecone_environment)
    index = pinecone.Index(holybook)
    # Get embeddings for the query
    try:
        response = openai.Embedding.create(
            input=[query], 
            engine=EMBEDDINGS_DEPLOYMENT_NAME,
            )
        embedding = response["data"][0]["embedding"]
        # Find contex in pinecone
        with open(f"./holybook/{holybook}.json", "r") as f:
            data = json.loads(f.read())
        res = index.query(vector=(embedding), top_k=8)
        ids = [i["id"] for i in res["matches"]]
        context = ""
        for id in ids:
            context = context + str(id) + ": " + data[str(id)] + "\n\n"
    except Exception as e:
        print("Error occurred while generating context:", str(e))
        context = "Error occurred while extracting context from Pinecone. Answer the question without context."

    return context

def gita_answer(query, history, model_name, max_tokens, temperature):

    systemprompt = [{
        "role": "system",
        "content": "You are not an AI Language model. You will be a Bhagwad Gita assistant to the user. Restrict yourself to the context of the question."
    }]
    try:
        # Set the initial conversation to the default system prompt
        conversation = systemprompt.copy()
        for human, assistant in history:
            conversation.append({"role": "user", "content": human})
            conversation.append({"role": "assistant", "content": assistant})
        context = extract_context_frompinecone(query)
        userprompt = f"Here are some verses that could help answer my question:\n\n{context}\n\nMy question: {query}\n\nYour answer:\n\n"
        conversation.append({"role": "user", "content": userprompt})
        try:
            # Generate a response using the selected model
            assistant_reply = generate_chat(model_name, conversation, temperature, max_tokens)
        except Exception as e:
            print("Model error:", str(e))
            print("Resetting conversation...")
            conversation = systemprompt.copy()
    except Exception as e:
        print("Error occurred while generating response:", str(e))
        conversation = systemprompt.copy()

    return assistant_reply

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

def clearallfiles_bing():
    # Ensure the UPLOAD_FOLDER is empty
    for root, dirs, files in os.walk(BING_FOLDER):
        for file in files:
            file_path = os.path.join(root, file)
            os.remove(file_path)

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
        metadata_index = json.load(open(os.path.join(VECTOR_FOLDER, "docstore.json")))
        embedding_index = json.load(open(os.path.join(VECTOR_FOLDER, "vector_store.json")))
        upload_data_to_supabase(metadata_index, embedding_index, title=uploaded_filenames[0], url="Local")
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
                    metadata_index = json.load(open(os.path.join(VECTOR_FOLDER, "docstore.json")))
                    embedding_index = json.load(open(os.path.join(VECTOR_FOLDER, "vector_store.json")))
                    upload_data_to_supabase(metadata_index, embedding_index, title=video_title, url=url)
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
                        metadata_index = json.load(open(os.path.join(VECTOR_FOLDER, "docstore.json")))
                        embedding_index = json.load(open(os.path.join(VECTOR_FOLDER, "vector_store.json")))
                        upload_data_to_supabase(metadata_index, embedding_index, title=video_title, url=url)
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
            metadata_index = json.load(open(os.path.join(VECTOR_FOLDER, "docstore.json")))
            embedding_index = json.load(open(os.path.join(VECTOR_FOLDER, "vector_store.json")))
            upload_data_to_supabase(metadata_index, embedding_index, title=article.title, url=url)
        # Generate summary
        summary = summary_generator()
        # Generate example queries
        example_queries = example_generator()

        return "Article downloaded and Index built successfully!", gr.Dataset.update(samples=example_queries), summary
    else:
        return "Please enter a valid URL", gr.Dataset.update(samples=example_queries), summary

def generate_chat(model_name, conversation, temperature, max_tokens):
    if model_name == "COHERE":
        co = cohere.Client(cohere_api_key)
        response = co.generate(
            model='command-nightly',
            prompt=str(conversation).replace("'", '"'),
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.generations[0].text
    elif model_name == "PALM":
        palm.configure(api_key=google_palm_api_key)
        response = palm.chat(
            model="models/chat-bison-001",
            messages=str(conversation).replace("'", '"'),
            temperature=temperature,
        )
        return response.last
    elif model_name == "GPT4":
        openai.api_type = azure_api_type
        openai.api_base = azure_api_base
        openai.api_version = azure_chatapi_version
        openai.api_key = azure_api_key
        response = openai.ChatCompletion.create(
            engine="gpt-4",
            messages=conversation,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=0.9,
            frequency_penalty=0.6,
            presence_penalty=0.1
        )
        return response['choices'][0]['message']['content']
    elif model_name == "GPT35TURBO":
        openai.api_type = azure_api_type
        openai.api_base = azure_api_base
        openai.api_version = azure_chatapi_version
        openai.api_key = azure_api_key
        response = openai.ChatCompletion.create(
            engine="gpt-35-turbo",
            messages=conversation,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0.6,
            presence_penalty=0.1
        )
    elif model_name == "WIZARDVICUNA7B":
        openai.api_type = llama2_api_type
        openai.api_key = llama2_api_key
        openai.api_base = llama2_api_base
        response = openai.ChatCompletion.create(
            model="wizardvicuna7b-uncensored-hf",
            messages=conversation,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response['choices'][0]['message']['content']
    else:
        return "Invalid model name"

def text_extractor(url):

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
        return article.text
    else:
        return None

def saveextractedtext_to_file(text, filename):

    # Save the output to the article.txt file
    file_path = os.path.join(BING_FOLDER, filename)
    with open(file_path, 'w') as file:
        file.write(text)

    return f"Text saved to {file_path}"

def get_bing_results(query, num=10):

    clearallfiles_bing()
    # Construct a request
    mkt = 'en-US'
    params = { 'q': query, 'mkt': mkt, 'count': num, 'responseFilter': ['Webpages','News'] }
    headers = { 'Ocp-Apim-Subscription-Key': bing_api_key }
    response = requests.get(bing_endpoint, headers=headers, params=params)
    response_data = response.json()  # Parse the JSON response

    # Extract snippets and append them into a single text variable
    all_snippets = [result['snippet'] for result in response_data['webPages']['value']]
    combined_snippets = '\n'.join(all_snippets)
    
    # Format the results as a string
    output = f"Here is the context from Bing for the query: '{query}':\n"
    output += combined_snippets

    # Save the output to a file
    saveextractedtext_to_file(output, "bing_results.txt")
    # Query the results using llama-index
    answer = str(simple_query(BING_FOLDER, query)).strip()

    return answer

def get_bing_news_results(query, num=5):

    clearallfiles_bing()
    # Construct a request
    mkt = 'en-US'
    params = { 'q': query, 'mkt': mkt, 'freshness': 'Day', 'count': num }
    headers = { 'Ocp-Apim-Subscription-Key': bing_api_key }
    response = requests.get(bing_news_endpoint, headers=headers, params=params)
    response_data = response.json()  # Parse the JSON response
    #pprint(response_data)

    # Extract text from the urls and append them into a single text variable
    all_urls = [result['url'] for result in response_data['value']]
    all_snippets = [text_extractor(url) for url in all_urls]

    # Combine snippets with titles and article names
    combined_output = ""
    for i, (snippet, result) in enumerate(zip(all_snippets, response_data['value'])):
        title = f"Article {i + 1}: {result['name']}"
        if len(snippet.split()) >= 75:  # Check if article has at least 75 words
            combined_output += f"\n{title}\n{snippet}\n"

    # Format the results as a string
    output = f"Here's scraped text from top {num} articles for: '{query}':\n"
    output += combined_output

    # Save the output to a file
    saveextractedtext_to_file(output, "bing_results.txt")
    # Summarize the bing search response
    bingsummary = str(summarize(BING_FOLDER)).strip()

    return bingsummary

def get_weather_data(query):
    
    # Initialize OpenWeatherMapToolSpec
    weather_tool = OpenWeatherMapToolSpec(
        key=openweather_api_key,
    )

    agent = OpenAIAgent.from_tools(
        weather_tool.to_tool_list(),
        llm=llm,
        verbose=False,
    )

    return str(agent.chat(query))

def summarize(data_folder):
    
    # Reset OpenAI API type and base
    openai.api_type = azure_api_type
    openai.api_base = azure_api_base
    # Initialize a document
    documents = SimpleDirectoryReader(data_folder).load_data()
    #index = VectorStoreIndex.from_documents(documents)
    summary_index = SummaryIndex.from_documents(documents)
    # SummaryIndexRetriever
    retriever = summary_index.as_retriever(
        retriever_mode='default',
    )
    # configure response synthesizer
    response_synthesizer = get_response_synthesizer(
        response_mode="tree_summarize",
        summary_template=summary_template,
    )
    # assemble query engine
    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
    )
    response = query_engine.query("Generate a summary of the input context. Be as verbose as possible, while keeping the summary concise and to the point.")

    return response

def simple_query(data_folder, query):
    
    # Reset OpenAI API type and base
    openai.api_type = azure_api_type
    openai.api_base = azure_api_base
    # Initialize a document
    documents = SimpleDirectoryReader(data_folder).load_data()
    #index = VectorStoreIndex.from_documents(documents)
    vector_index = VectorStoreIndex.from_documents(documents)
    # configure retriever
    retriever = VectorIndexRetriever(
        index=vector_index,
        similarity_top_k=6,
    )
    # # configure response synthesizer
    response_synthesizer = get_response_synthesizer(
        text_qa_template=qa_template,
    )
    # # assemble query engine
    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
        node_postprocessors=[
            SimilarityPostprocessor(similarity_cutoff=0.7)
        ],
    )
    response = query_engine.query(query)

    return response

def internet_connected_chatbot(query, history, model_name, max_tokens, temperature):
    
    try:
        # Set the initial conversation to the default system prompt
        conversation = system_prompt.copy()
        for human, assistant in history:
            conversation.append({"role": "user", "content": human})
            conversation.append({"role": "assistant", "content": assistant})
        conversation.append({"role": "user", "content": query})

        try:
            # If the query contains any of the keywords, perform a Bing search
            if any(keyword in query.lower() for keyword in keywords):
                # If the query contains news
                if "news" in query.lower():
                    assistant_reply = get_bing_news_results(query)
                # If the query contains weather
                elif "weather" in query.lower():
                    assistant_reply = get_weather_data(query)
                else:
                    assistant_reply = get_bing_results(query)
            else:
                # Generate a response using the selected model
                assistant_reply = generate_chat(model_name, conversation, temperature, max_tokens)
        except Exception as e:
            print("Model error:", str(e))
            print("Resetting conversation...")
            conversation = system_prompt.copy()

    except Exception as e:
        print("Error occurred while generating response:", str(e))
        conversation = system_prompt.copy()

    return assistant_reply

def ask(question, history):
    
    history = history or []
    answer = ask_query(question)

    return answer

def ask_query(question):

    # Reset OpenAI API type and base
    openai.api_type = azure_api_type
    openai.api_base = azure_api_base
    storage_context = StorageContext.from_defaults(persist_dir=VECTOR_FOLDER)
    vector_index = load_index_from_storage(storage_context, index_id="vector_index")
    # configure retriever
    retriever = VectorIndexRetriever(
        index=vector_index,
        similarity_top_k=10,
    )
    # # configure response synthesizer
    response_synthesizer = get_response_synthesizer(
        text_qa_template=qa_template,
    )
    # # assemble query engine
    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
        node_postprocessors=[
            SimilarityPostprocessor(similarity_cutoff=0.7)
        ],
    )
    response = query_engine.query(question)
    answer = response.response

    return answer

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

def example_generator():
    
    global example_queries, example_qs
    try:
        llmresponse = ask_fromfullcontext("Generate 5 questions exactly in the format mentioned", example_template).lstrip('\n')
        example_qs = [[str(item)] for item in ast.literal_eval(llmresponse.rstrip())]
    except Exception as e:
        print("Error occurred while generating examples:", str(e))
        example_qs = example_queries
    return example_qs

def summary_generator():
    
    global summary
    try:
        summary = ask_fromfullcontext("Generate a summary of the input context. Be as verbose as possible.", summary_template).lstrip('\n')
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

def clearfield(field):
    # Function to clear text
    return [""]

def clearhistory(field1, field2, field3):
    # Function to clear history
    return ["", "", ""]

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# Get API key from environment variable
dotenv.load_dotenv()
cohere_api_key = os.environ["COHERE_API_KEY"]
google_palm_api_key = os.environ["GOOGLE_PALM_API_KEY"]
azure_api_key = os.environ["AZURE_API_KEY"]
azure_api_type = "azure"
azure_api_base = os.environ.get("AZURE_API_BASE")
azure_api_version = os.environ.get("AZURE_API_VERSION")
azure_chatapi_version = os.environ.get("AZURE_CHATAPI_VERSION")

EMBEDDINGS_DEPLOYMENT_NAME = "text-embedding-ada-002"
# LocalAL API Base
llama2_api_type = "open_ai"
llama2_api_key = os.environ.get("LLAMA2_API_KEY")
llama2_api_base = os.environ.get("LLAMA2_API_BASE")
#Supabase API key
SUPABASE_API_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
SUPABASE_URL = os.environ.get("PUBLIC_SUPABASE_URL")
#Pinecone API key
pinecone_api_key = os.environ.get("PINECONE_API_KEY")
pinecone_environment = os.environ.get("PINECONE_ENVIRONMENT")

bing_api_key = os.getenv("BING_API_KEY")
bing_endpoint = os.getenv("BING_ENDPOINT") + "/v7.0/search"
bing_news_endpoint = os.getenv("BING_ENDPOINT") + "/v7.0/news/search"

openweather_api_key = os.environ.get("OPENWEATHER_API_KEY")

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
    print("Using gpt4-32k model.")
else:
    LLM_DEPLOYMENT_NAME = "gpt-35-turbo-16k"
    LLM_MODEL_NAME = "gpt-35-turbo-16k"
    openai.api_version = azure_chatapi_version
    max_input_size = 48000
    context_window = 16000
    print("Using gpt-3p5-turbo-16k model.")

system_prompt = [{
    "role": "system",
    "content": "You are a helpful and super-intelligent voice assistant, that accurately answers user queries. Be accurate, helpful, concise, and clear."
}]
conversation = system_prompt.copy()
temperature = 0.5
max_tokens = 420
model_name = "PALM"
# Define a list of keywords that trigger Bing search
keywords = ["latest", "current", "recent", "update", "best", "top", "news", "weather", "summary", "previous"]
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
    tokenizer=tiktoken.encoding_for_model("gpt-4").encode
)
node_parser = SimpleNodeParser(text_splitter=text_splitter)

# Set a flag for lite mode: Choose lite mode if you dont want to analyze videos without transcripts
lite_mode = True

llm = AzureOpenAI(
    engine=LLM_DEPLOYMENT_NAME, 
    model=LLM_MODEL_NAME,
    openai_api_key=azure_api_key,
    openai_api_base=azure_api_base,
    openai_api_type=azure_api_type,
    temperature=0.5,
    max_tokens=1024,
)
embedding_llm = LangchainEmbedding(
    OpenAIEmbeddings(
        model=EMBEDDINGS_DEPLOYMENT_NAME,
        deployment=EMBEDDINGS_DEPLOYMENT_NAME,
        openai_api_key=azure_api_key,
        openai_api_base=azure_api_base,
        openai_api_type=azure_api_type,
        openai_api_version=azure_api_version,
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
UPLOAD_FOLDER = os.path.join(".", "data")
BING_FOLDER = os.path.join(".", "bing_data")
SUMMARY_FOLDER = os.path.join(UPLOAD_FOLDER, "summary_index")
VECTOR_FOLDER = os.path.join(UPLOAD_FOLDER, "vector_index")

example_queries = [["Generate key 5 point summary"], ["What are 5 main ideas of this article?"], ["What are the key lessons learned and insights in this video?"], ["List key insights and lessons learned from the paper"], ["What are the key takeaways from this article?"]]
example_qs = []
summary = "No Summary available yet"

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
eg_template = (
    "You are a world-class question generator. We have provided context information below. Here is the context:\n"
    "---------------------\n"
    "{context_str}"
    "\n---------------------\n"
    "Based on the context provided, your task is to generate 5 relevant questions that would enable the user to get key ideas from the input context. Disregard any irrelevant information such as discounts, promotions, sponsorships or advertisements from the context. Output must be must in the form of python list of 5 strings, 1 string for each question enclosed in double quotes\n"
    "---------------------\n"
    "{query_str}\n"
)
example_template = PromptTemplate(eg_template)
ques_template = (
    "You are a world-class personal assistant. You will be provided snippets of information from the main context based on user's query. Here is the context:\n"
    "---------------------\n"
    "{context_str}\n"
    "\n---------------------\n"
    "Based on the context provided, your task is to answer the user's question to the best of your ability. Try to long answers to certain questions in the form of a bulleted list. It is important to refrain from directly copying word-for-word from the original context. Additionally, please ensure that the summary excludes any extraneous details such as discounts, promotions, sponsorships, or advertisements, and remains focused on the core message of the content.\n"
    "---------------------\n"
    "Using both the context information and also using your own knowledge, "
    "answer the question: {query_str}\n"
)
qa_template = PromptTemplate(ques_template)

# If the UPLOAD_FOLDER path does not exist, create it
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(BING_FOLDER):
    os.makedirs(BING_FOLDER)
if not os.path.exists(SUMMARY_FOLDER ):
    os.makedirs(SUMMARY_FOLDER)
if not os.path.exists(VECTOR_FOLDER ):
    os.makedirs(VECTOR_FOLDER)

# theme = gr.Theme.from_hub("gstaff/xkcd")
theme = gr.themes.Soft()

with gr.Blocks(theme=theme) as llmapp:
    gr.Markdown(
        """
        <h1><center><b>LLM Bot</center></h1>
        """
    )
    gr.Markdown(
        """
        <center>
        <br>
        This app uses the Transformer magic to answer all your questions! <br>
        Check the "Memorize" box if you want to add the information to your memory palace! <br>
        Using the default gpt-35-turbo-16k model! <br>
        </center>
        """
    )
    with gr.Tab(label="LLM APP"):
        with gr.Box():
            with gr.Row():
                memorize = gr.Checkbox(label="I want this information stored in my memory palace!")
            with gr.Row():
                with gr.Tab(label="Video Analyzer"):
                    yturl = gr.Textbox(placeholder="Input must be a Youtube URL", label="Enter Youtube URL")
                    with gr.Row():
                        download_output = gr.Textbox(label="Video download Status")
                        download_button = gr.Button(value="Download", scale=0)
                with gr.Tab(label="Article Analyzer"):
                    arturl = gr.Textbox(placeholder="Input must be a URL", label="Enter Article URL")
                    with gr.Row():
                        adownload_output = gr.Textbox(label="Article download Status")
                        adownload_button = gr.Button(value="Download", scale=0)
                with gr.Tab(label="File Analyzer"):
                    files = gr.Files(label="Upload the files to be analyzed")
                    with gr.Row():
                        upload_output = gr.Textbox(label="Upload Status")
                        upload_button = gr.Button(value="Upload", scale=0)
            with gr.Row():
                with gr.Box():
                    summary_output = gr.Textbox(placeholder="Summary will be generated here", label="Key takeaways")
                    chatui = gr.ChatInterface(
                        ask,
                        submit_btn="Ask",
                        retry_btn=None,
                        undo_btn=None,
                    )
                    query = chatui.textbox
                    examples = gr.Dataset(label="Questions", samples=example_queries, components=[query], type="index")
    with gr.Tab(label="AI Assistant"):
        gr.ChatInterface(
            internet_connected_chatbot,
            additional_inputs=[
                gr.Radio(label="Model", choices=["COHERE", "PALM", "GPT4", "GPT35TURBO", "WIZARDVICUNA7B"], value="PALM"),
                gr.Slider(10, 1680, value=840, label = "Max Output Tokens"),
                gr.Slider(0.1, 0.9, value=0.5, label = "Temperature"),
            ],
            examples=[["Latest news summary"], ["Explain special theory of relativity"], ["Latest Chelsea FC news"], ["Latest news from India"],["What is the latest GDP per capita of India?"]],
            submit_btn="Ask",
            retry_btn=None,
            undo_btn=None,
        )
    with gr.Tab(label="Fun"):
        with gr.Tab(label="City Planner"):
            with gr.Row():
                city_name = gr.Textbox(placeholder="Enter the name of the city", label="City Name")
                number_of_days = gr.Textbox(placeholder="Enter the number of days", label="Number of Days")
                city_button = gr.Button(value="Plan", scale=0)
            with gr.Row():
                city_output = gr.Textbox(label="Trip Plan", show_copy_button=True)
                clear_trip_button = gr.Button(value="Clear", scale=0)
        with gr.Tab(label="Bhagawad Gita"):
            gitachat = gr.ChatInterface(
                gita_answer,
                additional_inputs=[
                    gr.Radio(label="Model", choices=["COHERE", "PALM", "GPT4", "GPT35TURBO", "WIZARDVICUNA7B"], value="GPT35TURBO"),
                    gr.Slider(10, 1680, value=840, label = "Max Output Tokens"),
                    gr.Slider(0.1, 0.9, value=0.5, label = "Temperature"),
                ],
                examples=[["What is the meaning of life?"], ["What is the purpose of life?"], ["What is the meaning of death?"], ["What is the purpose of death?"], ["What is the meaning of existence?"], ["What is the purpose of existence?"], ["What is the meaning of the universe?"], ["What is the purpose of the universe?"], ["What is the meaning of the world?"], ["What is the purpose of the world?"]],
                submit_btn="Ask",
                retry_btn=None,
                undo_btn=None,
            )
            gita_question = gitachat.textbox
            # with gr.Row():
            #     gitareferences = gr.Textbox(placeholder= "References will be generated here", label="References")
            #     with gr.Column():
            #         gitaviewreferences = gr.Button(value="View References", scale=0)
            #         gitaclear_button = gr.Button(value="Clear", scale=0)
        with gr.Tab(label="Cravings Generator"):
            with gr.Row():
                craving_city_name = gr.Textbox(placeholder="Enter the name of the city", label="City Name")
                craving_cuisine = gr.Textbox(placeholder="What are you craving for? Enter idk if you don't know what you want to eat", label="Food")
                craving_button = gr.Button(value="Cook", scale=0)
            with gr.Row():
                craving_output = gr.Textbox(label="Food Places", show_copy_button=True)
                clear_craving_button = gr.Button(value="Clear", scale=0)


    upload_button.click(upload_file, inputs=[files, memorize], outputs=[upload_output, examples, summary_output], show_progress=True)
    download_button.click(download_ytvideo, inputs=[yturl, memorize], outputs=[download_output, examples, summary_output], show_progress=True)
    adownload_button.click(download_art, inputs=[arturl, memorize], outputs=[adownload_output, examples, summary_output], show_progress=True)
    city_button.click(generate_trip_plan, inputs=[city_name, number_of_days], outputs=[city_output], show_progress=True)
    craving_button.click(craving_satisfier, inputs=[craving_city_name, craving_cuisine], outputs=[craving_output], show_progress=True)
    examples.click(load_example, inputs=[examples], outputs=[query])
    clear_trip_button.click(clearhistory, inputs=[city_name, number_of_days, city_output], outputs=[city_name, number_of_days, city_output])
    clear_craving_button.click(clearhistory, inputs=[craving_city_name, craving_cuisine, craving_output], outputs=[craving_city_name, craving_cuisine, craving_output])
    #gitaviewreferences.click(extract_context_frompinecone, inputs=[gita_question], outputs=[gitareferences], show_progress=True)
    #gitaclear_button.click(clearfield, inputs=[gitareferences], outputs=[gitareferences])

if __name__ == '__main__':
    llmapp.launch(server_name='0.0.0.0', server_port=7860)
