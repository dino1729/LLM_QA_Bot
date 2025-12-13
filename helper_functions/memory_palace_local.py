import os
import shutil
import re
import logging
import asyncio
from typing import List, Optional, Dict, AsyncIterator
from datetime import datetime

from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    Document,
    Settings
)
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import get_response_synthesizer
from llama_index.core.prompts import PromptTemplate

from config import config
from helper_functions.llm_client import get_client, UnifiedLLMClient

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MEMORY_PALACE_ROOT = config.MEMORY_PALACE_FOLDER

# Create root dir if not exists
if not os.path.exists(MEMORY_PALACE_ROOT):
    os.makedirs(MEMORY_PALACE_ROOT)

def _sanitize_filename(name: str) -> str:
    """Sanitize string for use in directory names"""
    return re.sub(r'[^a-zA-Z0-9_-]', '_', name)

def get_memory_palace_index_dir(provider: str, embedding_model: str) -> str:
    """
    Get the specific index directory for a provider and embedding model combination.
    This ensures we don't mix embeddings from different models.
    """
    sanitized_provider = _sanitize_filename(provider)
    # Embedding model name might contain slashes or colons, sanitize it
    sanitized_model = _sanitize_filename(embedding_model)
    
    dir_name = f"{sanitized_provider}__{sanitized_model}"
    return os.path.join(MEMORY_PALACE_ROOT, dir_name)

def load_or_create_index(persist_dir: str) -> VectorStoreIndex:
    """
    Load existing index from storage or create a new empty one if it doesn't exist.
    """
    if os.path.exists(persist_dir) and os.listdir(persist_dir):
        try:
            storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
            index = load_index_from_storage(storage_context)
            logger.info(f"Loaded existing index from {persist_dir}")
            return index
        except Exception as e:
            logger.error(f"Failed to load index from {persist_dir}: {e}. Creating new one.")
            # If load fails, fall back to creating new
            pass
    
    if not os.path.exists(persist_dir):
        os.makedirs(persist_dir)
        
    # Create empty index
    index = VectorStoreIndex.from_documents([])
    index.storage_context.persist(persist_dir=persist_dir)
    logger.info(f"Created new index at {persist_dir}")
    return index

def save_memory(
    title: str,
    content: str,
    source_type: str,
    source_ref: str,
    model_name: str
) -> str:
    """
    Save a new memory to the persistent store.
    
    Args:
        title: Title of the memory
        content: The main content (summary, takeaways, etc.)
        source_type: Type of source (video, article, etc.)
        source_ref: URL or filename
        model_name: The model name string (e.g. "LITELLM:model-name" or "OLLAMA:model-name") to derive embedding config
    """
    # 1. Parse model to determine embedding configuration
    # We rely on helper_functions.llm_client.get_client to configure Settings.embed_model
    # But here we just need to know WHICH dir to use.
    # We'll instantiate a client briefly to get the embedding model name accurately.
    
    # Simple parsing to avoid circular deps if possible, or use the one from gradio_ui_full which parses strings.
    # Let's assume we can parse it similarly to how the app does it.
    provider = "litellm"
    if model_name.lower().startswith("ollama"):
        provider = "ollama"
    # For others like "GEMINI", "GROQ", we often fall back to LiteLLM for embeddings in this app structure,
    # unless specific embedding models are configured. 
    # Let's check config.
    
    if provider == "litellm":
        embedding_model = config.litellm_embedding
    else:
        embedding_model = config.ollama_embedding
        
    # 2. Get persistence directory
    persist_dir = get_memory_palace_index_dir(provider, embedding_model)
    
    # 3. Create Document
    full_text = f"Title: {title}\n\nKey Takeaways:\n{content}\n\nSource: {source_ref}"
    
    doc = Document(
        text=full_text,
        metadata={
            "source_type": source_type,
            "source_title": title,
            "source_ref": source_ref,
            "created_at": datetime.now().isoformat(),
            "type": "memory"
        }
    )
    
    # 4. Load/Create Index and Insert
    # Note: We assume Settings.embed_model is already set correctly by the caller (via api_lock + set_model_context)
    # However, to be safe, we should probably ensure the index uses the correct embed model.
    # LlamaIndex uses the global Settings.embed_model when inserting.
    
    index = load_or_create_index(persist_dir)
    index.insert(doc)
    index.storage_context.persist(persist_dir=persist_dir)
    
    return "Saved to Memory Palace"

def search_memories(
    query: str,
    model_name: str,
    top_k: int = 5
) -> List[Dict]:
    """
    Search for memories similar to the query.
    """
    # Determine provider/embedding model
    provider = "litellm"
    if model_name.lower().startswith("ollama"):
        provider = "ollama"
    
    if provider == "litellm":
        embedding_model = config.litellm_embedding
    else:
        embedding_model = config.ollama_embedding
        
    persist_dir = get_memory_palace_index_dir(provider, embedding_model)
    
    if not os.path.exists(persist_dir) or not os.listdir(persist_dir):
        return []
        
    storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
    index = load_index_from_storage(storage_context)
    
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=top_k
    )
    
    nodes = retriever.retrieve(query)
    
    results = []
    for node in nodes:
        results.append({
            "content": node.get_content(),
            "score": node.score,
            "metadata": node.metadata
        })
        
    return results

def prepare_memory_stream(
    message: str,
    history: List[List[str]],
    model_name: str,
    top_k: int = 5
):
    """
    Prepare a streaming response using retrieved memories.
    Returns a response object (StreamingResponse) from LlamaIndex query engine.
    """
    # Determine provider/embedding model
    provider = "litellm"
    if model_name.lower().startswith("ollama"):
        provider = "ollama"
    
    if provider == "litellm":
        embedding_model = config.litellm_embedding
    else:
        embedding_model = config.ollama_embedding
        
    persist_dir = get_memory_palace_index_dir(provider, embedding_model)
    
    if not os.path.exists(persist_dir) or not os.listdir(persist_dir):
        raise ValueError("Memory Palace is empty. Please save some memories first.")
        
    storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
    index = load_index_from_storage(storage_context)
    
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=top_k
    )
    
    # Custom prompt to include chat history context if needed, 
    # though LlamaIndex ChatEngine is usually better for history.
    # But for now we stick to RetrieverQueryEngine as per plan and existing docqa patterns.
    # We can format history into the query or prompt.
    
    # Let's use a custom QA template that emphasizes using the memory
    mp_qa_template_str = (
        "You are a helpful AI assistant with access to a Memory Palace.\n"
        "Use the following retrieved memories to answer the user's question.\n"
        "If the answer is not in the memories, you can say so or use your general knowledge but mention it's not from memory.\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "User Question: {query_str}\n"
        "Answer: "
    )
    mp_qa_template = PromptTemplate(mp_qa_template_str)
    
    response_synthesizer = get_response_synthesizer(
        text_qa_template=mp_qa_template,
        streaming=True,
        llm=Settings.llm 
    )
    
    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer
    )
    
    # We could prepend history to the message, or rely on the fact that 
    # the user question usually contains the intent.
    # For simplicity and matching current docqa, we just query with the message.
    
    response = query_engine.query(message)
    return response

def reset_memory_palace(model_name: str) -> str:
    """
    Reset (delete) the memory palace index for the specific embedding model.
    """
    # Determine provider/embedding model
    provider = "litellm"
    if model_name.lower().startswith("ollama"):
        provider = "ollama"
    
    if provider == "litellm":
        embedding_model = config.litellm_embedding
    else:
        embedding_model = config.ollama_embedding
        
    persist_dir = get_memory_palace_index_dir(provider, embedding_model)
    
    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)
        return f"Memory Palace reset for {provider} (embeddings: {embedding_model})"
    else:
        return "Memory Palace was already empty."

