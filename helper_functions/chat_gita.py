"""
Bhagawad Gita Chatbot
Uses unified LLM client and Pinecone for embeddings
"""
from helper_functions.chat_generation import generate_chat
from helper_functions.llm_client import get_client
from config import config
import json

# Configuration
pinecone_api_key = config.pinecone_api_key if hasattr(config, 'pinecone_api_key') else ""
pinecone_environment = config.pinecone_environment if hasattr(config, 'pinecone_environment') else ""

# Initialize unified client for embeddings
embedding_client = get_client(provider="litellm", model_tier="smart")


def extract_context_frompinecone(query):
    """
    Extract context from Pinecone vector database

    Args:
        query: User query

    Returns:
        Context from relevant Gita verses
    """
    if not pinecone_api_key:
        return "Pinecone is not configured. Please add your Pinecone API key to config.yml"

    try:
        import pinecone

        holybook = "gita"
        pinecone.init(api_key=pinecone_api_key, environment=pinecone_environment)
        index = pinecone.Index(holybook)

        # Get embeddings for the query using unified client
        embedding = embedding_client.get_embedding(query)

        # Find context in pinecone
        with open(f"./holybook/{holybook}.json", "r") as f:
            data = json.loads(f.read())

        res = index.query(vector=embedding, top_k=8)
        ids = [i["id"] for i in res["matches"]]

        context = ""
        for id in ids:
            context = context + str(id) + ": " + data[str(id)] + "\n\n"

        return context

    except Exception as e:
        print(f"Error occurred while generating context: {str(e)}")
        return "Error occurred while extracting context from Pinecone. Answer the question without context."


def gita_answer(query, history, model_name, max_tokens, temperature):
    """
    Generate answer based on Bhagawad Gita context

    Args:
        query: User question
        history: Chat history
        model_name: Model to use
        max_tokens: Max tokens for response
        temperature: Sampling temperature

    Returns:
        Assistant's response with Gita context
    """
    systemprompt = [{
        "role": "system",
        "content": "You are not an AI Language model. You will be a Bhagwad Gita assistant to the user. Restrict yourself to the context of the question."
    }]

    try:
        # Build conversation from history
        conversation = systemprompt.copy()
        for human, assistant in history:
            conversation.append({"role": "user", "content": human})
            conversation.append({"role": "assistant", "content": assistant})

        # Get context from Pinecone
        context = extract_context_frompinecone(query)

        # Build user prompt with context
        userprompt = f"Here are some verses that could help answer my question:\n\n{context}\n\nMy question: {query}\n\nYour answer:\n\n"
        conversation.append({"role": "user", "content": userprompt})

        try:
            # Generate response using the selected model
            assistant_reply = generate_chat(model_name, conversation, temperature, max_tokens)
        except Exception as e:
            print(f"Model error: {str(e)}")
            print("Resetting conversation...")
            conversation = systemprompt.copy()
            assistant_reply = "Sorry, I encountered an error. Please try again."

    except Exception as e:
        print(f"Error occurred while generating response: {str(e)}")
        conversation = systemprompt.copy()
        assistant_reply = "Sorry, I encountered an error. Please try again."

    return assistant_reply
