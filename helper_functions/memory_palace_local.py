"""
Memory Palace Local - SimpleVectorStore-based persistent memory storage.
No LlamaIndex dependency.
"""
import os
import shutil
import re
import logging
from typing import List, Dict
from datetime import datetime

from helper_functions.vector_store import SimpleVectorStore, Document
from config import config
from helper_functions.llm_client import get_client

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
    sanitized_model = _sanitize_filename(embedding_model)
    dir_name = f"{sanitized_provider}__{sanitized_model}"
    return os.path.join(MEMORY_PALACE_ROOT, dir_name)


def _get_client_for_model(model_name: str):
    """Get a UnifiedLLMClient based on model_name string."""
    provider = "litellm"
    if model_name.lower().startswith("ollama"):
        provider = "ollama"
    return get_client(provider=provider, model_tier="smart")


def _get_embedding_model(provider: str) -> str:
    """Get the embedding model name for a provider."""
    if provider == "litellm":
        return config.litellm_embedding
    else:
        return config.ollama_embedding


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
        model_name: The model name string to derive embedding config
    """
    provider = "litellm"
    if model_name.lower().startswith("ollama"):
        provider = "ollama"

    embedding_model = _get_embedding_model(provider)
    persist_dir = get_memory_palace_index_dir(provider, embedding_model)

    # Create Document
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

    # Get embedding function
    client = _get_client_for_model(model_name)
    store = SimpleVectorStore(persist_dir=persist_dir, embed_fn=client.get_embedding)
    store.insert(doc)

    return "Saved to Memory Palace"


def search_memories(
    query: str,
    model_name: str,
    top_k: int = 5
) -> List[Dict]:
    """Search for memories similar to the query."""
    provider = "litellm"
    if model_name.lower().startswith("ollama"):
        provider = "ollama"

    embedding_model = _get_embedding_model(provider)
    persist_dir = get_memory_palace_index_dir(provider, embedding_model)

    if not os.path.exists(persist_dir) or not os.listdir(persist_dir):
        return []

    client = _get_client_for_model(model_name)
    store = SimpleVectorStore(persist_dir=persist_dir, embed_fn=client.get_embedding)
    results = store.search(query, top_k=top_k)

    return [
        {
            "content": r.text,
            "score": r.score,
            "metadata": r.metadata
        }
        for r in results
    ]


class StreamingResult:
    """Wrapper to provide .response_gen for backwards compatibility."""

    def __init__(self, generator):
        self.response_gen = generator


def prepare_memory_stream(
    message: str,
    history: List[List[str]],
    model_name: str,
    top_k: int = 5
):
    """
    Prepare a streaming response using retrieved memories.
    Returns a StreamingResult with .response_gen generator.
    """
    provider = "litellm"
    if model_name.lower().startswith("ollama"):
        provider = "ollama"

    embedding_model = _get_embedding_model(provider)
    persist_dir = get_memory_palace_index_dir(provider, embedding_model)

    if not os.path.exists(persist_dir) or not os.listdir(persist_dir):
        raise ValueError("Memory Palace is empty. Please save some memories first.")

    client = _get_client_for_model(model_name)
    store = SimpleVectorStore(persist_dir=persist_dir, embed_fn=client.get_embedding)
    results = store.search(message, top_k=top_k)
    context = "\n\n".join([r.text for r in results])

    mp_qa_prompt = (
        "You are a helpful AI assistant with access to a Memory Palace.\n"
        "Use the following retrieved memories to answer the user's question.\n"
        "If the answer is not in the memories, you can say so or use your general knowledge but mention it's not from memory.\n"
        "---------------------\n"
        f"{context}\n"
        "---------------------\n"
        f"User Question: {message}\n"
        "Answer: "
    )

    gen = client.stream_chat_completion(
        messages=[{"role": "user", "content": mp_qa_prompt}],
        temperature=config.temperature,
        max_tokens=config.max_tokens
    )

    return StreamingResult(gen)


def reset_memory_palace(model_name: str) -> str:
    """Reset (delete) the memory palace index for the specific embedding model."""
    provider = "litellm"
    if model_name.lower().startswith("ollama"):
        provider = "ollama"

    embedding_model = _get_embedding_model(provider)
    persist_dir = get_memory_palace_index_dir(provider, embedding_model)

    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)
        return f"Memory Palace reset for {provider} (embeddings: {embedding_model})"
    else:
        return "Memory Palace was already empty."
