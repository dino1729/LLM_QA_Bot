import os
import logging
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
)
from langchain.embeddings import OpenAIEmbeddings
import dotenv
import openai
from llama_index.llms import AzureOpenAI

# Set OpenAI API key
dotenv.load_dotenv()
os.environ["OPENAI_API_KEY"] = os.environ.get("AZUREOPENAIAPIKEY")
openai.api_type = os.environ.get("AZUREOPENAIAPITYPE")
openai.api_version = os.environ.get("AZUREOPENAIAPIVERSION")
openai.api_base = os.environ.get("AZUREOPENAIENDPOINT")
openai.api_key = os.environ.get("AZUREOPENAIAPIKEY")
LLM_DEPLOYMENT_NAME = "text-davinci-003"
EMBEDDINGS_DEPLOYMENT_NAME = "text-embedding-ada-002"
# max LLM token input size
max_input_size = 4096
# set number of output tokens
num_output = 1024
# set maximum chunk overlap
max_chunk_overlap_ratio = 0.12
# set chunk size limit
chunk_size_limit = 256
# set context window
context_window = 4096
# set prompt helper
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
    ),
    embed_batch_size=1,
)
service_context = ServiceContext.from_defaults(
    llm=llm,
    embed_model=embedding_llm,
    prompt_helper=prompt_helper,
    chunk_size_limit=chunk_size_limit,
    context_window=context_window,
)

# Set up logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# Create a list index
UPLOAD_FOLDER = os.path.join(".", "data")

documents = SimpleDirectoryReader(UPLOAD_FOLDER).load_data()
index = ListIndex.from_documents(documents, service_context=service_context)
# index.set_index_id("list_index")
# index.storage_context.persist(persist_dir=UPLOAD_FOLDER)

# storage_context = StorageContext.from_defaults(persist_dir=UPLOAD_FOLDER)
# index = load_index_from_storage(storage_context, index_id="list_index")

query_engine = index.as_query_engine(service_context=service_context)
response = query_engine.query("What did the author do after his time at Y Combinator?")
print(response)

# # Load documents from a directory
# documents = SimpleDirectoryReader("data").load_data()

# # Create a tree index from the documents
# new_index = TreeIndex.from_documents(documents, service_context=service_context)
# # Query the index
# query_engine = new_index.as_query_engine()
# response = query_engine.query("What did the author do growing up?")
# print(response)

# # Query the index with a different question
# response = query_engine.query("What did the author do after his time at Y Combinator?")
# print(response)

# # Create a tree index with a different child_branch_factor
# query_engine = new_index.as_query_engine(child_branch_factor=2)
# response = query_engine.query("What did the author do growing up?")
# print(response)

# # Build the tree index during query-time
# documents = SimpleDirectoryReader("data").load_data()
# index_light = TreeIndex.from_documents(documents, build_tree=False, service_context=service_context)
# query_engine = index_light.as_query_engine(retriever_mode="all_leaf", response_mode="tree_summarize")
# response = query_engine.query("What did the author do after his time at Y Combinator?")
# print(response)

# # Build the tree index with a custom summary prompt
# documents = SimpleDirectoryReader("data").load_data()
# query_str = "What did the author do growing up?"
# SUMMARY_PROMPT_TMPL = (
#     "Context information is below. \n"
#     "---------------------\n"
#     "{context_str}"
#     "\n---------------------\n"
#     "Given the context information and not prior knowledge, "
#     f"answer the question: {query_str}\n"
# )
# SUMMARY_PROMPT = Prompt(SUMMARY_PROMPT_TMPL)
# index_with_query = TreeIndex.from_documents(documents, summary_template=SUMMARY_PROMPT, service_context=service_context)
# query_engine = index_with_query.as_query_engine(retriever_mode="root")
# response = index_with_query.query(query_str)
# print(response)

# # Create a keyword table index
# documents = SimpleDirectoryReader("data").load_data()
# index = KeywordTableIndex.from_documents(documents, service_context=service_context)
# query_engine = index.as_query_engine()
# response = query_engine.query("What did the author do after his time at Y Combinator?")
# print(response)


