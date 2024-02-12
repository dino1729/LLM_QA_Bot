import os
from openai import AzureOpenAI, OpenAI
import random
import dotenv
dotenv.load_dotenv()

def generate_chat(model_name, conversation, temperature, max_tokens):

    if model_name == "GPT4":
        client = AzureOpenAI(
            api_key=azure_api_key,
            azure_endpoint=azure_api_base,
            api_version=azure_chatapi_version,
        )
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=conversation,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=0.9,
            frequency_penalty=0.6,
            presence_penalty=0.1
        )
        return response.choices[0].message.content
    
    elif model_name == "GPT35TURBO":
        client = AzureOpenAI(
            api_key=azure_api_key,
            azure_endpoint=azure_api_base,
            api_version=azure_chatapi_version,
        )
        response = client.chat.completions.create(
            model="gpt-35-turbo",
            messages=conversation,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=0.9,
            frequency_penalty=0.6,
            presence_penalty=0.1
        )
        return response.choices[0].message.content
    
    elif model_name == "WIZARDVICUNA7B":
        local_client = OpenAI(
            api_key=llama2_api_key,
            base_url=llama2_api_base,
        )
        response = local_client.chat.completions.create(
            model="wizardvicuna7b-uncensored-hf",
            messages=conversation,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content
    else:
        return "Invalid model name"

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

        model_name = random.choice(["GPT4", "GPT35TURBO"])

        assistant_reply = generate_chat(model_name, conversation, temperature, max_tokens)
        print("Bot: ", assistant_reply)

        conversation.append(({"role": "assistant", "content": assistant_reply}))
