"""
Chat streaming using direct RAG (retrieve + stream LLM response).
No LlamaIndex dependency - uses SimpleVectorStore and UnifiedLLMClient.
"""
from helper_functions.vector_store import SimpleVectorStore, StreamingResult
from helper_functions.llm_client import get_client
from config import config
import logging
import os

logger = logging.getLogger(__name__)

# Module-level cache: persist_dir -> SimpleVectorStore instance
_store_cache: dict = {}


def prepare_chat_stream(question, model_name, vector_folder, qa_template, parse_model_name_func):
    """
    Prepare the chat stream by retrieving relevant chunks and streaming an LLM response.
    Returns a StreamingResult with .response_gen generator.
    """
    provider, tier, actual_model = parse_model_name_func(model_name)
    client = get_client(provider=provider, model_tier=tier, model_name=actual_model)

    if not os.path.exists(vector_folder) or not os.listdir(vector_folder):
        raise Exception("Index not found. Please upload documents first.")

    # Reuse cached store or create and cache a new one
    store = _store_cache.get(vector_folder)
    if store is None:
        store = SimpleVectorStore(persist_dir=vector_folder, embed_fn=client.get_embedding)
        _store_cache[vector_folder] = store
    else:
        # Update embed_fn in case the client changed
        store.embed_fn = client.get_embedding

    results = store.search(question, top_k=10)

    if not results:
        logger.warning("No relevant document chunks found for query: %s", question)
        raise Exception(
            "No relevant document chunks found for query; "
            "try uploading more documents or broadening the query."
        )

    context = "\n\n".join([r.text for r in results])

    # Format prompt - qa_template is a string with {context_str} and {query_str} placeholders
    template_str = qa_template if isinstance(qa_template, str) else str(qa_template)
    prompt = template_str.format(context_str=context, query_str=question)

    # Stream LLM response
    gen = client.stream_chat_completion(
        messages=[{"role": "user", "content": prompt}],
        temperature=config.temperature,
        max_tokens=config.max_tokens
    )

    return StreamingResult(gen)
