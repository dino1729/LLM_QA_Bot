import dotenv
import os
import cohere
import google.generativeai as palm
import requests
from bs4 import BeautifulSoup
from newspaper import Article
import openai
from langchain.embeddings import OpenAIEmbeddings
from llama_index.llms import AzureOpenAI
from llama_index import (
    VectorStoreIndex,
    SummaryIndex,
    LangchainEmbedding,
    PromptHelper,
    SimpleDirectoryReader,
    ServiceContext,
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
from llama_hub.tools.bing_search.base import BingSearchToolSpec
import tiktoken
import argparse
import logging
import sys
import random

def clearallfiles():
    # Ensure the UPLOAD_FOLDER is empty
    for root, dirs, files in os.walk(UPLOAD_FOLDER):
        for file in files:
            file_path = os.path.join(root, file)
            os.remove(file_path)

def saveextractedtext_to_file(text, filename):

    # Save the output to the article.txt file
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    with open(file_path, 'w') as file:
        file.write(text)

    return f"Text saved to {file_path}"

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
        except:
            # Try an alternate method using requests and beautifulsoup
            try:
                req = requests.get(url)
                soup = BeautifulSoup(req.content, 'html.parser')
                article.text = soup.get_text()
            except:
                pass
        return article.text
    else:
        return None

def get_bing_results(query, num=10):

    clearallfiles()
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
    answer = str(simple_query(UPLOAD_FOLDER, query)).strip()

    return answer

def get_bing_news_results(query, num=5):

    clearallfiles()
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
    bingsummary = str(summarize(UPLOAD_FOLDER)).strip()

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

def get_bing_agent(query):
    
        bing_tool = BingSearchToolSpec(
            api_key=bing_api_key,
        )
    
        agent = OpenAIAgent.from_tools(
            bing_tool.to_tool_list(),
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
            top_p=0.9,
            frequency_penalty=0.6,
            presence_penalty=0.1
        )
        return response['choices'][0]['message']['content']
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

if __name__ == "__main__":

    logging.basicConfig(stream=sys.stdout, level=logging.CRITICAL)
    logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

    # Get API keys from environment variables
    dotenv.load_dotenv()
    cohere_api_key = os.environ["COHERE_API_KEY"]
    google_palm_api_key = os.environ["GOOGLE_PALM_API_KEY"]
    azure_api_key = os.environ["AZURE_API_KEY"]
    azure_api_type = "azure"
    azure_api_base = os.environ.get("AZURE_API_BASE")
    azure_api_version = os.environ.get("AZURE_API_VERSION")
    azure_chatapi_version = os.environ.get("AZURE_CHATAPI_VERSION")
    EMBEDDINGS_DEPLOYMENT_NAME = "text-embedding-ada-002"

    llama2_api_type = "open_ai"
    llama2_api_key = os.environ.get("LLAMA2_API_KEY")
    llama2_api_base = os.environ.get("LLAMA2_API_BASE")

    bing_api_key = os.getenv("BING_API_KEY")
    bing_endpoint = os.getenv("BING_ENDPOINT") + "/v7.0/search"
    bing_news_endpoint = os.getenv("BING_ENDPOINT") + "/v7.0/news/search"

    openweather_api_key = os.environ.get("OPENWEATHER_API_KEY")

    # max LLM token input size
    max_input_size = 96000
    num_output = 1024
    max_chunk_overlap_ratio = 0.1
    chunk_size = 256
    context_window = 32000
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

    os.environ["OPENAI_API_KEY"] = azure_api_key
    openai.api_type = azure_api_type
    openai.api_base = azure_api_base
    openai.api_key = azure_api_key

    # Check if user set the gpt4 model flag
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

    llm = AzureOpenAI(
        engine=LLM_DEPLOYMENT_NAME, 
        model=LLM_MODEL_NAME,
        api_key=azure_api_key,
        api_base=azure_api_base,
        api_type=azure_api_type,
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
    sum_template = (
        "You are a world-class text summarizer connected to the internet. We have provided context information below from the internet below. \n"
        "---------------------\n"
        "{context_str}"
        "\n---------------------\n"
        "Based on the context provided, your task is to summarize the input context while effectively conveying the main points and relevant information. The summary should be presented in the format of news headlines. It is important to refrain from directly copying word-for-word from the original context. Additionally, please ensure that the summary excludes any extraneous details such as discounts, promotions, sponsorships, or advertisements, and remains focused on the core message of the content.\n"
        "---------------------\n"
        "Using both the latest context information and also using your own knowledge, "
        "answer the question: {query_str}\n"
    )
    summary_template = PromptTemplate(sum_template)

    ques_template = (
        "You are a world-class personal assistant connected to the internet. You will be provided snippets of information from the internet based on user's query. Here is the context:\n"
        "---------------------\n"
        "{context_str}\n"
        "\n---------------------\n"
        "Based on the context provided, your task is to answer the user's question to the best of your ability. It is important to refrain from directly copying word-for-word from the original context. Additionally, please ensure that the summary excludes any extraneous details such as discounts, promotions, sponsorships, or advertisements, and remains focused on the core message of the content.\n"
        "---------------------\n"
        "Using both the latest context information and also using your own knowledge, "
        "answer the question: {query_str}\n"
    )
    qa_template = PromptTemplate(ques_template)

    system_prompt = [{
        "role": "system",
        "content": "You are a helpful and super-intelligent voice assistant, that accurately answers user queries. Be accurate, helpful, concise, and clear."
    }]
    model_names = ["WIZARDVICUNA7B", "PALM", "GPT4", "GPT35TURBO" "COHERE"]
    model_index = 0
    model_name = model_names[model_index]
    temperature = 0.3
    max_tokens = 1024

    # Initialize the conversation
    conversation = system_prompt.copy() 

    UPLOAD_FOLDER = os.path.join(".", "data")
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)

   # Define a list of keywords that trigger Bing search
    keywords = ["latest", "current", "recent", "update", "best", "top", "news", "weather", "summary", "previous"]

    # Get user query from args
    parser = argparse.ArgumentParser(description="Bing+ChatGPT tool.")
    parser.add_argument("query", help="User query")
    args = parser.parse_args()
    userquery = args.query

    # User Query formatted into array
    user_message = {"role": "user", "content": userquery}
    conversation.append(user_message)


    # Check if the query contains any of the keywords
    if any(keyword in userquery.lower() for keyword in keywords):
        # Check if the query contains the word news
        if "news" in userquery.lower():
            bing_newsresults = get_bing_news_results(userquery)
            print(bing_newsresults)
        # Check if the query contains the word weather
        elif "weather" in userquery.lower():
            weatherdata = get_weather_data(userquery)
            print(weatherdata)
        else:
            bing_searchresults = get_bing_results(userquery)
            print(bing_searchresults)
    else:
        # Query the results using random model
        random_model = random.choice(model_names)
        #assistant_reply = generate_chat(random_model, conversation, temperature, max_tokens)
        assistant_reply = get_bing_agent(userquery)
        new_assistant_message = {"role": "assistant", "content": assistant_reply}
        conversation.append(new_assistant_message)
        #print(f"{random_model} says: {assistant_reply}")
        print(assistant_reply)



