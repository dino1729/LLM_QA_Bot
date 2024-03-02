# Import necessary libraries
import cohere
import google.generativeai as palm
import google.generativeai as genai
from groq import Groq
from openai import AzureOpenAI as OpenAIAzure
from openai import OpenAI
from config import config

cohere_api_key = config.cohere_api_key
google_api_key = config.google_api_key
gemini_model_name = config.gemini_model_name
groq_api_key = config.groq_api_key
azure_api_key = config.azure_api_key
azure_api_base = config.azure_api_base
azure_chatapi_version = config.azure_chatapi_version
azure_chatapi_version = config.azure_chatapi_version
azure_gpt4_deploymentid = config.azure_gpt4_deploymentid
azure_gpt35_deploymentid = config.azure_gpt35_deploymentid
llama2_api_key = config.llama2_api_key
llama2_api_base = config.llama2_api_base

def generate_chat(model_name, conversation, temperature, max_tokens):

    client = OpenAIAzure(
        api_key=azure_api_key,
        azure_endpoint=azure_api_base,
        api_version=azure_chatapi_version,
    )
    if model_name == "COHERE":

        co = cohere.Client(cohere_api_key)
        response = co.generate(
            model='command-nightly',
            prompt=str(conversation).replace("'", '"'),
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.generations[0].text
    
    elif model_name == "PALM":

        palm.configure(api_key=google_api_key)
        response = palm.chat(
            model="models/chat-bison-001",
            messages=str(conversation).replace("'", '"'),
            temperature=temperature,
        )
        return response.last
    
    elif model_name == "GEMINI":
    
        genai.configure(api_key=google_api_key)
        generation_config = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
            "top_p": 0.9,
            "top_k": 1,
        }
        gemini = genai.GenerativeModel(model_name= gemini_model_name,generation_config=generation_config)
        response = gemini.generate_content(str(conversation).replace("'", '"'))
        return response.text
    
    elif model_name == "GPT4":

        response = client.chat.completions.create(
            model=azure_gpt4_deploymentid,
            messages=conversation,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=0.9,
            frequency_penalty=0.6,
            presence_penalty=0.1
        )
        return response.choices[0].message.content
    
    elif model_name == "GPT35TURBO":

        response = client.chat.completions.create(
            model=azure_gpt35_deploymentid,
            messages=conversation,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=0.9,
            frequency_penalty=0.6,
            presence_penalty=0.1
        )
        return response.choices[0].message.content
    
    elif model_name == "MIXTRAL8x7B":

        local_client = OpenAI(
            api_key=llama2_api_key,
            base_url=llama2_api_base,
        )
        response = local_client.chat.completions.create(
            model="mixtral8x7b",
            messages=conversation,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content
    
    elif model_name == "GROQ_LLAMA":

        groq_client = Groq(
            api_key=groq_api_key,
        )
        response = groq_client.chat.completions.create(
            model="llama2-70b-4096",
            messages=conversation,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=0.9,
            frequency_penalty=0.6,
            presence_penalty=0.1
        )
        return response.choices[0].message.content
    
    elif model_name == "GROQ_MIXTRAL":

        groq_client = Groq(
            api_key=groq_api_key,
        )
        response = groq_client.chat.completions.create(
            model="mixtral-8x7b-32768",
            messages=conversation,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=0.9,
            frequency_penalty=0.6,
            presence_penalty=0.1
        )
        return response.choices[0].message.content

    else:
        return "Invalid model name"
