"""
Bhagawad Gita Chatbot
Uses unified LLM client and Pinecone for embeddings
"""
import json

from helper_functions.chat_generation import generate_chat
from helper_functions.llm_client import get_client
from config import config

try:
    from pinecone import Pinecone
except ImportError:
    Pinecone = None

# Configuration
pinecone_apikey = config.pinecone_api_key if hasattr(config, 'pinecone_api_key') else ""
pinecone_environment = config.pinecone_environment if hasattr(config, 'pinecone_environment') else ""

# Initialize unified client for embeddings
# Uses configured default_embeddings_tier from config.yml
embedding_client = get_client(provider="litellm", model_tier=config.default_embeddings_tier)


def extract_context_frompinecone(query):
    """
    Extract context from Pinecone vector database

    Args:
        query: User query

    Returns:
        Context from relevant Gita verses
    """
    if Pinecone is None or pinecone_apikey is None:
        return ""

    try:
        holybook = "gita"
        pinecone_client = Pinecone(api_key=pinecone_apikey)
        index = pinecone_client.Index(holybook)

        # Get embeddings for the query using unified client
        embedding = embedding_client.get_embedding(query)

        res = index.query(vector=embedding, top_k=8)
        matches = getattr(res, "matches", None)
        if matches is None and isinstance(res, dict):
            matches = res.get("matches", [])

        context_chunks = []
        for match in matches or []:
            metadata = getattr(match, "metadata", None)
            if metadata is None and isinstance(match, dict):
                metadata = match.get("metadata", {})

            verse_text = ""
            if isinstance(metadata, dict):
                verse_text = metadata.get("text", "")

            if not verse_text and isinstance(match, dict):
                match_id = match.get("id")
                if match_id:
                    with open(f"./holybook/{holybook}.json", "r") as f:
                        data = json.loads(f.read())
                    verse_text = data.get(str(match_id), "")

            if verse_text:
                context_chunks.append(verse_text)

        return "\n\n".join(context_chunks)

    except Exception as e:
        print(f"Error occurred while generating context: {str(e)}")
        return ""


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
    try:
        context = extract_context_frompinecone(query)
        systemprompt = [{
            "role": "system",
            "content": (
                "You are not an AI Language model. You will be a Bhagavad Gita assistant to the user. "
                "Restrict yourself to the context of the question.\n\n"
                f"Relevant Bhagavad Gita context:\n{context}"
            )
        }]

        # Build conversation from history
        conversation = systemprompt.copy()
        for entry in history:
            if isinstance(entry, dict):
                conversation.append({
                    "role": entry.get("role", "user"),
                    "content": entry.get("content", ""),
                })
            else:
                human, assistant = entry
                conversation.append({"role": "user", "content": human})
                conversation.append({"role": "assistant", "content": assistant})

        conversation.append({"role": "user", "content": query})

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
