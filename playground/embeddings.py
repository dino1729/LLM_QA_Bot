
import openai
import dotenv
import os
from langchain.embeddings.openai import OpenAIEmbeddings

# Get API key from environment variable
dotenv.load_dotenv()
os.environ["OPENAI_API_KEY"] = os.environ.get("AZUREOPENAIAPIKEY")
openai.api_type = os.environ.get("AZUREOPENAIAPITYPE")
openai.api_version = os.environ.get("AZUREOPENAIAPIVERSION")
openai.api_base = os.environ.get("AZUREOPENAIENDPOINT")
openai.api_key = os.environ.get("AZUREOPENAIAPIKEY")

#openai.api_version = "2023-05-15"

# response = openai.Embedding.create(
#     input="Dinesh is the next big thing in the future",
#     engine="text-embedding-ada-002",
# )
# print(response)
# embeddings = response['data'][0]['embedding']
# print(embeddings)

embeddings = OpenAIEmbeddings(
    model="text-embedding-ada-002",
    deployment="text-embedding-ada-002",
    openai_api_key=openai.api_key,
    openai_api_base=openai.api_base,
    openai_api_type=openai.api_type,
    openai_api_version=openai.api_version,
    chunk_size=256,
)
text1 = "Dinesh is the next big thing in the future"
query_result1 = embeddings.embed_query(text1)
print(query_result1)

text2 = "Dinesh was the best thing in the past"
query_result2 = embeddings.embed_query(text2)
print(query_result2)
