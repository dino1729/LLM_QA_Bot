import os
import logging
import sys
import dotenv
import openai
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import AzureOpenAI
from llama_index import (
    VectorStoreIndex,
    TreeIndex,
    LangchainEmbedding,
    LLMPredictor,
    PromptHelper,
    SimpleDirectoryReader,
    ServiceContext,
    StorageContext,
    load_index_from_storage,
    get_response_synthesizer,
)
# Get API key from environment variable
dotenv.load_dotenv()
os.environ["OPENAI_API_KEY"] = os.environ.get("AZUREOPENAIAPIKEY")
openai.api_type = os.environ.get("AZUREOPENAIAPITYPE")
openai.api_version = os.environ.get("AZUREOPENAIAPIVERSION")
openai.api_base = os.environ.get("AZUREOPENAIENDPOINT")
openai.api_key = os.environ.get("AZUREOPENAIAPIKEY")
LLM_DEPLOYMENT_NAME = "text-davinci-003"
# max LLM token input size
max_input_size = 4096
# set number of output tokens
num_output = 1024
# set maximum chunk overlap
max_chunk_overlap_ratio = 0.12
# set chunk size limit
chunk_size_limit = 256
# set prompt helper
prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap_ratio)

#Update your deployment name accordingly
llm = AzureOpenAI(deployment_name="text-davinci-003", model="text-davinci-003")
llm_predictor = LLMPredictor(llm=llm)
embedding_llm = LangchainEmbedding(
    OpenAIEmbeddings(
        model="text-embedding-ada-002",
        deployment="text-embedding-ada-002",
        openai_api_key=openai.api_key,
        openai_api_base=openai.api_base,
        openai_api_type=openai.api_type,
        openai_api_version=openai.api_version,
    ),
    embed_batch_size=1,
)
service_context = ServiceContext.from_defaults(
    llm_predictor=llm_predictor,
    embed_model=embedding_llm,
    prompt_helper=prompt_helper,
    chunk_size_limit=chunk_size_limit
)

# Set up logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# Load documents from a directory
documents = SimpleDirectoryReader("data").load_data()

# Create a tree index from the documents
new_index = TreeIndex.from_documents(documents, service_context=service_context)

# Query the index
query_engine = new_index.as_query_engine(service_context=service_context)
response = query_engine.query("What did the author do growing up?")
print(response)

# Query the index with a different question
response = query_engine.query("What did the author do after his time at Y Combinator?")
print(response)
