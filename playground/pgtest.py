import json
import os
import datetime

import tiktoken
import re
import logging
import sys
from llama_index import (
    VectorStoreIndex,
    ListIndex,
    LangchainEmbedding,
    PromptHelper,
    SimpleDirectoryReader,
    ServiceContext,
    StorageContext,
    load_index_from_storage,
    get_response_synthesizer,
    Prompt,
)
from langchain.embeddings import OpenAIEmbeddings
import dotenv

import openai
from llama_index.llms import AzureOpenAI
from llama_index import set_global_service_context

from llama_index.text_splitter import SentenceSplitter
from llama_index.node_parser import SimpleNodeParser

from llama_index.storage import StorageContext
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.indices.postprocessor import SimilarityPostprocessor
from llama_index.retrievers import VectorIndexRetriever

# Set OpenAI API key
dotenv.load_dotenv()
os.environ["OPENAI_API_KEY"] = os.environ.get("AZUREOPENAIAPIKEY")
openai.api_type = os.environ.get("AZUREOPENAIAPITYPE")
openai.api_version = os.environ.get("AZUREOPENAIAPIVERSION")
openai.api_base = os.environ.get("AZUREOPENAIENDPOINT")
openai.api_key = os.environ.get("AZUREOPENAIAPIKEY")
LLM_DEPLOYMENT_NAME = "text-davinci-003"
EMBEDDINGS_DEPLOYMENT_NAME = "text-embedding-ada-002"

max_input_size = 4096
num_output = 1024
max_chunk_overlap_ratio = 0.12
chunk_size = 512
context_window = 4096

UPLOAD_FOLDER = os.path.join('./data')
LIST_FOLDER = os.path.join('./data/list_index')
VECTOR_FOLDER = os.path.join('./data/vector_index')

if not os.path.exists(LIST_FOLDER ):
    os.makedirs(LIST_FOLDER)
if not os.path.exists(VECTOR_FOLDER ):
    os.makedirs(VECTOR_FOLDER)

text_splitter = SentenceSplitter(
  separator=" ",
  chunk_size=512,
  chunk_overlap=20,
  backup_separators=["\n"],
  paragraph_separator="\n\n\n"
)

node_parser = SimpleNodeParser(text_splitter=text_splitter)
prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap_ratio)

llm = AzureOpenAI(
    engine=LLM_DEPLOYMENT_NAME, 
    model=LLM_DEPLOYMENT_NAME,
    openai_api_key=openai.api_key,
    openai_api_base=openai.api_base,
    openai_api_type=openai.api_type,
    openai_api_version=openai.api_version,
    temperature=0.5
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

# storage_context = StorageContext.from_defaults(
#     docstore=SimpleDocumentStore.from_persist_dir(persist_dir=UPLOAD_FOLDER),
#     vector_store=SimpleVectorStore.from_persist_dir(persist_dir=UPLOAD_FOLDER),
#     index_store=SimpleIndexStore.from_persist_dir(persist_dir=UPLOAD_FOLDER),
# )

set_global_service_context(service_context)

logging.basicConfig(stream=sys.stdout, level=logging.INFO)  # logging.DEBUG for more verbose output
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


sum_template = (
    "You are a world-class text summarizer. We have provided context information below. \n"
    "---------------------\n"
    "{context_str}"
    "\n---------------------\n"
    "Based on the information provided, your task is to summarize the input context while effectively conveying the main points and relevant information. The summary should be presented in the style of a news reader, using your own words to accurately capture the essence of the content. It is important to refrain from directly copying word-for-word from the original context. Additionally, please ensure that the summary excludes any extraneous details such as discounts, promotions, sponsorships, or advertisements, and remains focused on the core message of the content.\n"
    "---------------------\n"
    "{query_str}"
)
summary_template = Prompt(sum_template)

ques_template = (
    "You are a world-class personal assistant connected to the internet. You will be provided snippets of information from the internet based on user's query. Here is the context:\n"
    "---------------------\n"
    "{context_str}"
    "\n---------------------\n"
    "Based on the information provided, your task is to answer the user's question to the best of your ability. You can use your own knowledge base to answer the question and only use the relavant information from the internet incase you don't have knowledge of the latest information to correctly answer user's question\n"
    "---------------------\n"
    "{query_str}"
)
qa_template = Prompt(ques_template)

documents = SimpleDirectoryReader(UPLOAD_FOLDER).load_data()

listindex = ListIndex.from_documents(documents)
listindex.set_index_id("list_index")
listindex.storage_context.persist(persist_dir=LIST_FOLDER)

vectorindex = VectorStoreIndex.from_documents(documents)
vectorindex.set_index_id("vector_index")
vectorindex.storage_context.persist(persist_dir=VECTOR_FOLDER)


# Rebuild the list index from storage
storage_context = StorageContext.from_defaults(persist_dir=LIST_FOLDER)
index = load_index_from_storage(storage_context, index_id="list_index")
retriever = index.as_retriever(retriever_mode='default')
query_engine1 = RetrieverQueryEngine.from_args(retriever, response_mode='tree_summarize', text_qa_template=summary_template)
response = query_engine1.query("Generate a summary of the input context. Be as verbose as possible")
print(response)

# Rebuild the vector index from storage
storage_context = StorageContext.from_defaults(persist_dir=VECTOR_FOLDER)
index = load_index_from_storage(storage_context, index_id="vector_index")
# configure retriever
retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=5,
)
# # configure response synthesizer
response_synthesizer = get_response_synthesizer(text_qa_template=qa_template)
# # assemble query engine
query_engine = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=response_synthesizer,
    node_postprocessors=[
        SimilarityPostprocessor(similarity_cutoff=0.7)
    ],
    )
response = query_engine.query("Tell me something interesting about the input context")
print(response)


# Test metadata extraction
def upload_data_to_supabase(index_data, title, url):
    
    # Insert the data for each document into the Supabase table
    # supabase_client = supabase.Client(SUPABASE_URL, SUPABASE_API_KEY)
    for doc_id, doc_data in index_data["docstore"]["__data__"]["docs"].items():
        content_title = title
        content_url = url
        content_date = datetime.today().strftime('%B %d, %Y')
        content_text = doc_data['text']
        content_length = len(content_text)
        content_tokens = len(tiktoken.get_encoding("cl100k_base").encode(content_text))
        cleaned_content_text = re.sub(r'[^\w0-9./:^,&%@"!()?\\p{Sc}\'’“”]+|\s+', ' ', content_text, flags=re.UNICODE)
        embedding = index_data["vector_store"]["__data__"]["simple_vector_store_data_dict"]["embedding_dict"][doc_id]

        # result = supabase_client.table('mp').insert({
        #     'content_title': content_title,
        #     'content_url': content_url,
        #     'content_date': content_date,
        #     'content': cleaned_content_text,
        #     'content_length': content_length,
        #     'content_tokens': content_tokens,
        #     'embedding': embedding
        # }).execute()

    return content_title, content_url, content_date, content_text, content_length, content_tokens, cleaned_content_text, embedding

# Test upload data to Supabase
index_data = json.load(open(os.path.join(VECTOR_FOLDER, "docstore.json")))
upload_data_to_supabase(index_data, "Test Title", "Test URL")
