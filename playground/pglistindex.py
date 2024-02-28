from llama_index.core import ServiceContext, PromptHelper, SimpleDirectoryReader, ListIndex
from llama_index.embeddings.langchain import LangchainEmbedding

import os
import openai
import dotenv
from langchain.embeddings import OpenAIEmbeddings
from llama_index.llms.azure_openai import AzureOpenAI
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

# Get API key from environment variable
dotenv.load_dotenv()
os.environ["OPENAI_API_KEY"] = os.environ.get("AZUREOPENAIAPIKEY")
openai.api_type = os.environ.get("AZUREOPENAIAPITYPE")
openai.api_version = os.environ.get("AZUREOPENAIAPIVERSION_CHAT")
openai.api_base = os.environ.get("AZUREOPENAIENDPOINT")
openai.api_key = os.environ.get("AZUREOPENAIAPIKEY")
LLM_DEPLOYMENT_NAME = "text-davinci-003"

#Update your deployment name accordingly
# llm = AzureOpenAI(deployment_name=LLM_DEPLOYMENT_NAME, model_kwargs={
#     "api_type": os.environ.get("AZUREOPENAIAPITYPE"),
#     "api_version": os.environ.get("AZUREOPENAIAPIVERSION"),
# })
llm = AzureOpenAI(engine="gpt-3p5-turbo-old", model="gpt-35-turbo", temperature=0.0)

embedding_llm = LangchainEmbedding(OpenAIEmbeddings(chunk_size=1, max_retries=3))
service_context = ServiceContext.from_defaults(
    llm=llm,
    embed_model=embedding_llm,
    prompt_helper=prompt_helper,
    chunk_size_limit=chunk_size_limit
)
UPLOAD_FOLDER = os.path.join(".", "data")

documents = SimpleDirectoryReader(UPLOAD_FOLDER).load_data()
index = ListIndex.from_documents(documents, service_context=service_context)
query_engine = index.as_query_engine(service_context=service_context)
response = query_engine.query("What did the author do after his time at Y Combinator?")
print(response)
