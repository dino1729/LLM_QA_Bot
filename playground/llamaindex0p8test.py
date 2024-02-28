import json
import os
from datetime import datetime
from urllib import response

import tiktoken
import re
import logging

import sys
from llama_index.core import VectorStoreIndex, ListIndex, PromptHelper, SimpleDirectoryReader, ServiceContext, StorageContext, load_index_from_storage, get_response_synthesizer, Prompt
from llama_index.embeddings.langchain import LangchainEmbedding
from langchain.embeddings import OpenAIEmbeddings
import dotenv

import openai
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.core import set_global_service_context
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core import StorageContext
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.indices.list import ListIndexLLMRetriever
from llama_index.core.response_synthesizers import TreeSummarize

# Set OpenAI API key
dotenv.load_dotenv()
os.environ["OPENAI_API_KEY"] = os.environ.get("AZUREOPENAIAPIKEY")
openai.api_type = os.environ.get("AZUREOPENAIAPITYPE")
openai.api_base = os.environ.get("AZUREOPENAIENDPOINT")
openai.api_key = os.environ.get("AZUREOPENAIAPIKEY")

EMBEDDINGS_DEPLOYMENT_NAME = "text-embedding-ada-002"

# Check if user set the davinci model flag
davincimodel_flag = False
if davincimodel_flag:
    LLM_DEPLOYMENT_NAME = "text-davinci-003"
    LLM_MODEL_NAME = "text-davinci-003"
    openai.api_version = os.environ.get("AZUREOPENAIAPIVERSION")
    print("Using text-davinci-003 model.")
else:
    LLM_DEPLOYMENT_NAME = "gpt-3p5-turbo-16k"
    LLM_MODEL_NAME = "gpt-35-turbo-16k"
    openai.api_version = os.environ.get("AZURECHATAPIVERSION")
    print("Using gpt-3p5-turbo-16k model.")

max_input_size = 4096
num_output = 1024
max_chunk_overlap_ratio = 0.12
chunk_size = 512
context_window = 4096

UPLOAD_FOLDER = os.path.join(".", "data")
LIST_FOLDER = os.path.join(UPLOAD_FOLDER, "list_index")
VECTOR_FOLDER = os.path.join(UPLOAD_FOLDER, "vector_index")

if not os.path.exists(LIST_FOLDER ):
    os.makedirs(LIST_FOLDER)
if not os.path.exists(VECTOR_FOLDER ):
    os.makedirs(VECTOR_FOLDER)

text_splitter = SentenceSplitter(
  separator=" ",
  chunk_size=chunk_size,
  chunk_overlap=20,
  paragraph_separator="\n\n\n"
)

node_parser = SimpleNodeParser(text_splitter=text_splitter)
prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap_ratio)

llm = AzureOpenAI(
    engine=LLM_DEPLOYMENT_NAME, 
    model=LLM_MODEL_NAME,
    openai_api_key=openai.api_key,
    openai_api_base=openai.api_base,
    openai_api_type=openai.api_type,
    openai_api_version=openai.api_version,
    temperature=0.5,
    max_tokens=1024,
)

embedding_llm = LangchainEmbedding(
    OpenAIEmbeddings(
        model=EMBEDDINGS_DEPLOYMENT_NAME,
        deployment=EMBEDDINGS_DEPLOYMENT_NAME,
        openai_api_key=openai.api_key,
        openai_api_base=openai.api_base,
        openai_api_type=openai.api_type,
        openai_api_version=openai.api_version,
        chunk_size=32,
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

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)  # logging.DEBUG for more verbose output
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

sum_template = (
    "You are a world-class text summarizer. We have provided context information below. \n"
    "---------------------\n"
    "{context_str}\n"
    "\n---------------------\n"
    "Based on the information provided, your task is to summarize the input context while effectively conveying the main points and relevant information. The summary should be presented in a numbered list of at least 10 key points and takeaways, with a catchy headline at the top. It is important to refrain from directly copying word-for-word from the original context. Additionally, please ensure that the summary excludes any extraneous details such as discounts, promotions, sponsorships, or advertisements, and remains focused on the core message of the content.\n"
    "---------------------\n"
    "Using both the context information and also using your own knowledge, "
    "answer the question: {query_str}\n"
    "If the context isn't helpful, you can also answer the question on your own.\n"
)
summary_template = Prompt(sum_template)
eg_template = (
    "You are a helpful assistant that is helping the user to gain more knowledge about the input context. You will be provided snippets of information from the main context based on user's query. Here is the context:\n"
    "---------------------\n"
    "{context_str}\n"
    "\n---------------------\n"
    "Based on the information provided, your task is to generate atleast 5 relevant questions that would enable the user to get key ideas from the input context. Disregard any irrelevant information such as discounts, promotions, sponsorships or advertisements from the context. Output must be must in the form of python list of 5 strings, 1 string for each question enclosed in double quotes. Be sure to double check your answer to see if it is in the format requested\n"
    "---------------------\n"
    "Using both the context information and also using your own knowledge, "
    "answer the question: {query_str}\n"
    "If the context isn't helpful, you can also answer the question on your own.\n"
)
example_template = Prompt(eg_template)
ques_template = (
    "You are a world-class personal assistant. You will be provided snippets of information from the main context based on user's query. Here is the context:\n"
    "---------------------\n"
    "{context_str}\n"
    "\n---------------------\n"
    "Based on the information provided, your task is to answer the user's question to the best of your ability. It is important to refrain from directly copying word-for-word from the original context. Additionally, please ensure that the summary excludes any extraneous details such as discounts, promotions, sponsorships, or advertisements, and remains focused on the core message of the content.\n"
    "---------------------\n"
    "Using both the context information and also using your own knowledge, "
    "answer the question: {query_str}\n"
    "If the context isn't helpful, you can also answer the question on your own.\n"
)
qa_template = Prompt(ques_template)
refine_template_str = (
    "The original question is as follows: {query_str}\n"
    "We have provided an existing answer: {existing_answer}\n"
    "We have the opportunity to refine the existing answer "
    "(only if needed) with some more context below.\n"
    "------------\n"
    "{context_msg}\n"
    "------------\n"
    "Using both the new context and your own knowledege, update or repeat the existing answer.\n"
)
refine_template = Prompt(refine_template_str)

documents = SimpleDirectoryReader(UPLOAD_FOLDER).load_data()

listindex = ListIndex.from_documents(documents)
listindex.set_index_id("list_index")
listindex.storage_context.persist(persist_dir=LIST_FOLDER)

vectorindex = VectorStoreIndex.from_documents(documents)
vectorindex.set_index_id("vector_index")
vectorindex.storage_context.persist(persist_dir=VECTOR_FOLDER)

# Rebuild the list index from storage
storage_context = StorageContext.from_defaults(persist_dir=LIST_FOLDER)
list_index = load_index_from_storage(storage_context, index_id="list_index")

# # ListIndexRetriever
# retriever = ListIndexLLMRetriever(
#     index=list_index,
#     choice_batch_size=10,
# )
# # configure response synthesizer
# response_synthesizer = get_response_synthesizer(
#     response_mode="tree_summarize",
#     text_qa_template=summary_template,
# )
# # assemble query engine
# query_engine1 = RetrieverQueryEngine(
#     retriever=retriever,
#     response_synthesizer=response_synthesizer,
# )
# response1 = query_engine1.query("Generate a summary of the input context. Be as verbose as possible")

# # ListIndexRetriever
# retriever = list_index.as_retriever(retriever_mode='default')
# response_synthesizer = get_response_synthesizer(
#     response_mode="tree_summarize",
#     text_qa_template=summary_template,
# )
# # tree summarize
# query_engine1 = RetrieverQueryEngine(
#     retriever=retriever,
#     response_synthesizer=response_synthesizer,
# )
# response1 = query_engine1.query("Generate a summary of the input context. Be as verbose as possible")

query_engine1 = list_index.as_query_engine(
    response_mode="tree_summarize",
)
response1 = query_engine1.query("Generate a summary of the input context. Be as verbose as possible")

print("*****************LIST INDEX SUMMARY TEST*******************")
print(response1)

# Rebuild the vector index from storage
storage_context = StorageContext.from_defaults(persist_dir=VECTOR_FOLDER)
vector_index = load_index_from_storage(storage_context, index_id="vector_index")
# configure retriever
retriever = VectorIndexRetriever(
    index=vector_index,
    similarity_top_k=6,
)
# # configure response synthesizer
response_synthesizer = get_response_synthesizer(
    response_mode="refine",
    text_qa_template=qa_template,
    refine_template=refine_template
)
# # assemble query engine
query_engine = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=response_synthesizer,
    node_postprocessors=[
        SimilarityPostprocessor(similarity_cutoff=0.7)
    ],
)
response = query_engine.query("Tell me something interesting about the input context")
print("*****************VECTOR INDEX QA TEST*******************")
print(response)

# Test metadata extraction
def upload_data_to_supabase(metadata_index, embedding_index, title, url):
    
    # Insert the data for each document into the Supabase table
    # supabase_client = supabase.Client(SUPABASE_URL, SUPABASE_API_KEY)
    for doc_id, doc_data in metadata_index["docstore/data"].items():
        content_title = title
        content_url = url
        content_date = datetime.today().strftime('%B %d, %Y')
        content_text = doc_data["__data__"]["text"]
        content_length = len(content_text)
        content_tokens = len(tiktoken.get_encoding("cl100k_base").encode(content_text))
        cleaned_content_text = re.sub(r'[^\w0-9./:^,&%@"!()?\\p{Sc}\'’“”]+|\s+', ' ', content_text, flags=re.UNICODE)
        embedding = embedding_index["embedding_dict"][doc_id]
        # result = supabase_client.table('mp').insert({
        #     'content_title': content_title,
        #     'content_url': content_url,
        #     'content_date': content_date,
        #     'content': cleaned_content_text,
        #     'content_length': content_length,
        #     'content_tokens': content_tokens,
        #     'embedding': embedding
        # }).execute()
        print("\nContent Title: ", content_title)
        print("\nContent URL: ", content_url)
        print("\nContent Date: ", content_date)
        print("\nContent Text: ", cleaned_content_text)
        print("\nContent Length: ", content_length)
        print("\nContent Tokens: ", content_tokens)
        #print("\nEmbedding: ", embedding)

    return content_title, content_url, content_date, content_length, content_tokens, cleaned_content_text, embedding

# Test upload data to Supabase
# metadata_index = json.load(open(os.path.join(VECTOR_FOLDER, "docstore.json")))
# embedding_index = json.load(open(os.path.join(VECTOR_FOLDER, "vector_store.json")))
# #print(index_data)
# upload_data_to_supabase(metadata_index, embedding_index, "Test Title", "Test URL")
