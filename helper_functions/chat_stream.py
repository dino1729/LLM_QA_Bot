import asyncio
from llama_index.core import StorageContext, load_index_from_storage, get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from helper_functions.llm_client import get_client
from llama_index.core import Settings as LlamaSettings
import logging
import os

logger = logging.getLogger(__name__)

def prepare_chat_stream(question, model_name, vector_folder, qa_template, parse_model_name_func):
    """
    Prepare the chat stream by setting up the engine and executing the query.
    Returns the response object which contains .response_gen
    """
    provider, tier, actual_model = parse_model_name_func(model_name)
    
    original_llm = LlamaSettings.llm
    original_embed = LlamaSettings.embed_model
    
    client = None
    if provider in ["litellm", "ollama"]:
        client = get_client(provider=provider, model_tier=tier, model_name=actual_model)
        # We set globals because load_index_from_storage/retriever might implicit use them
        LlamaSettings.llm = client.get_llamaindex_llm()
        LlamaSettings.embed_model = client.get_llamaindex_embedding()
    
    # Use the local client for synthesizer to ensure it sticks even if globals change
    llm_instance = LlamaSettings.llm 

    try:
        if not os.path.exists(vector_folder) or not os.listdir(vector_folder):
             raise Exception("Index not found. Please upload documents first.")

        storage_context = StorageContext.from_defaults(persist_dir=vector_folder)
        vector_index = load_index_from_storage(storage_context, index_id="vector_index")
        
        retriever = VectorIndexRetriever(
            index=vector_index,
            similarity_top_k=10,
        )
        
        # Explicitly pass LLM to synthesizer
        response_synthesizer = get_response_synthesizer(
            text_qa_template=qa_template,
            streaming=True,
            llm=llm_instance
        )
        
        query_engine = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
        )
        
        # Run query - this performs retrieval and returns a StreamingResponse object
        response = query_engine.query(question)
        return response
            
    except Exception as e:
        logger.error(f"Error preparing chat stream: {e}")
        raise e
    finally:
        # Restore settings immediately after query setup is done
        # The retrieval is done. The generator in response will use the captured llm_instance.
        if provider in ["litellm", "ollama"]:
            LlamaSettings.llm = original_llm
            LlamaSettings.embed_model = original_embed
