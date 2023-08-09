import openai
import dotenv
import os

# Get API key from environment variable
dotenv.load_dotenv()
os.environ["OPENAI_API_KEY"] = os.environ.get("AZUREOPENAIAPIKEY")
openai.api_type = os.environ.get("AZUREOPENAIAPITYPE")
openai.api_version = os.environ.get("AZUREOPENAIAPIVERSION")
openai.api_base = os.environ.get("AZUREOPENAIENDPOINT")
openai.api_key = os.environ.get("AZUREOPENAIAPIKEY")

#openai.api_version = "2023-05-15"

response = openai.Embedding.create(
    input="Dinesh is the next big thing in the future",
    engine="text-embedding-ada-002",
)
embeddings = response['data'][0]['embedding']
print(embeddings)
