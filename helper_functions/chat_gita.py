from helper_functions.chat_generation import generate_chat
from openai import AzureOpenAI as OpenAIAzure
from config import config
import pinecone
import json

azure_api_key = config.azure_api_key
azure_api_base = config.azure_api_base
azure_chatapi_version = config.azure_chatapi_version
azure_chatapi_version = config.azure_chatapi_version
azure_gpt4_deploymentid = config.azure_gpt4_deploymentid
azure_gpt4omini_deploymentid = config.azure_gpt4omini_deploymentid
azure_embeddingapi_version = config.azure_embeddingapi_version
azure_embedding_deploymentid = config.azure_embedding_deploymentid

pinecone_api_key = config.pinecone_api_key
pinecone_environment = config.pinecone_environment

def extract_context_frompinecone(query):
    
    embed_client = OpenAIAzure(
        api_key = azure_api_key,  
        api_version = azure_embeddingapi_version,
        azure_endpoint =azure_api_base,
    )
    holybook = "gita"
    pinecone.init(api_key=pinecone_api_key, environment=pinecone_environment)
    index = pinecone.Index(holybook)
    # Get embeddings for the query
    try:
        response = embed_client.embeddings.create(
            input=[query], 
            model=azure_embedding_deploymentid,
            )
        embedding = response.data[0].embedding
        # Find contex in pinecone
        with open(f"./holybook/{holybook}.json", "r") as f:
            data = json.loads(f.read())
        res = index.query(vector=(embedding), top_k=8)
        ids = [i["id"] for i in res["matches"]]
        context = ""
        for id in ids:
            context = context + str(id) + ": " + data[str(id)] + "\n\n"
    except Exception as e:
        print("Error occurred while generating context:", str(e))
        context = "Error occurred while extracting context from Pinecone. Answer the question without context."

    return context

def gita_answer(query, history, model_name, max_tokens, temperature):

    systemprompt = [{
        "role": "system",
        "content": "You are not an AI Language model. You will be a Bhagwad Gita assistant to the user. Restrict yourself to the context of the question."
    }]
    try:
        # Set the initial conversation to the default system prompt
        conversation = systemprompt.copy()
        for human, assistant in history:
            conversation.append({"role": "user", "content": human})
            conversation.append({"role": "assistant", "content": assistant})
        context = extract_context_frompinecone(query)
        userprompt = f"Here are some verses that could help answer my question:\n\n{context}\n\nMy question: {query}\n\nYour answer:\n\n"
        conversation.append({"role": "user", "content": userprompt})
        try:
            # Generate a response using the selected model
            assistant_reply = generate_chat(model_name, conversation, temperature, max_tokens)
        except Exception as e:
            print("Model error:", str(e))
            print("Resetting conversation...")
            conversation = systemprompt.copy()
    except Exception as e:
        print("Error occurred while generating response:", str(e))
        conversation = systemprompt.copy()

    return assistant_reply
