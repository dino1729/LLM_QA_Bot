import os
import sys
from openai import AzureOpenAI, OpenAI
import random
import dotenv

# Add the parent directory to sys.path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
print(f"Debug: Added parent directory to path: {parent_dir}")

from helper_functions.chat_generation import generate_chat
dotenv.load_dotenv()

if __name__ == '__main__':
    llama2_api_type = "open_ai"
    llama2_api_key = os.environ.get("LLAMA2_API_KEY")
    llama2_api_base = os.environ.get("LLAMA2_API_BASE")
    azure_api_key = os.environ["AZURE_API_KEY"]
    azure_api_type = "azure"
    azure_api_base = os.environ.get("AZURE_API_BASE")
    azure_embeddingapi_version = os.environ.get("AZURE_EMBEDDINGAPI_VERSION")
    azure_chatapi_version = os.environ.get("AZURE_CHATAPI_VERSION")

    system_prompt = [{
        "role": "system",
        "content": "You are a helpful and super-intelligent voice assistant, that accurately answers user queries. Be accurate, helpful, concise, and clear."
    }]
    conversation = system_prompt.copy()
    temperature = 0.1
    max_tokens = 150

    while True:
        user_query = input("Enter your query: ")
        conversation.append(({"role": "user", "content": user_query}))

        model_name = random.choice(["GROQ"])

        assistant_reply = generate_chat(model_name, conversation, temperature, max_tokens)
        print("Bot: ", assistant_reply)

        conversation.append(({"role": "assistant", "content": assistant_reply}))
