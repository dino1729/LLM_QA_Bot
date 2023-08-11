import os
import logging
from re import U
import sys
from llama_index import (
    VectorStoreIndex,
    ListIndex,
    LangchainEmbedding,
    LLMPredictor,
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
from matplotlib.sankey import UP
import openai
from llama_index.llms import AzureOpenAI
from llama_index import set_global_service_context
from llama_index.text_splitter import TokenTextSplitter
from llama_index.text_splitter import SentenceSplitter
from llama_index.node_parser import SimpleNodeParser
from llama_index.storage.docstore import SimpleDocumentStore
from llama_index.storage.index_store import SimpleIndexStore
from llama_index.vector_stores import SimpleVectorStore
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
chunk_size_limit = 256
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
    chunk_size_limit=chunk_size_limit,
    context_window=context_window,
    node_parser=node_parser,
)

# storage_context = StorageContext.from_defaults(
#     docstore=SimpleDocumentStore.from_persist_dir(persist_dir=UPLOAD_FOLDER),
#     vector_store=SimpleVectorStore.from_persist_dir(persist_dir=UPLOAD_FOLDER),
#     index_store=SimpleIndexStore.from_persist_dir(persist_dir=UPLOAD_FOLDER),
# )

set_global_service_context(service_context)

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)  # logging.DEBUG for more verbose output
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
response = query_engine.query("Tell me something about the Telugu Desam Party")
print(response)

