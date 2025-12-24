import os
import sys
import logging
import traceback
import asyncio
import tempfile
import shutil
import uuid
import re
from typing import List, Optional, Any
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, File, UploadFile, Form, Body, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from helper_functions.chat_stream import prepare_chat_stream
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv

# Import helpers
from helper_functions.chat_generation_with_internet import internet_connected_chatbot
from helper_functions.trip_planner import generate_trip_plan
from helper_functions.food_planner import craving_satisfier
from helper_functions.analyzers import analyze_article, analyze_ytvideo, analyze_media, analyze_file, clearallfiles
from helper_functions.chat_generation import generate_chat
from helper_functions import gptimage_tool as tool
from helper_functions.llm_client import get_client, list_available_models
from helper_functions.memory_palace_local import (
    save_memory,
    search_memories,
    prepare_memory_stream,
    reset_memory_palace
)
from llama_index.core import Settings as LlamaSettings
from llama_index.core import StorageContext, load_index_from_storage, get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import PromptTemplate
from config import config

# Load env vars
load_dotenv()

# Configure logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
VECTOR_FOLDER = config.VECTOR_FOLDER
SUMMARY_FOLDER = config.SUMMARY_FOLDER
qa_template = PromptTemplate(config.ques_template)

# Global Lock for LlamaIndex Settings
api_lock = asyncio.Lock()

app = FastAPI(title="LLM QA Bot API")


def configure_cors(app: FastAPI) -> None:
    """
    Configure CORS middleware with environment-driven, secure origins.
    
    Reads configuration from environment variables (priority) or config.yml:
    - ALLOWED_ORIGINS / cors.allowed_origins: Comma-separated list of allowed origins
    - CORS_ALLOW_ORIGIN_REGEX / cors.allow_origin_regex: Regex pattern for dynamic subdomains
    - ENVIRONMENT / cors.environment: "development" enables permissive "*", default is "production"
    
    In production:
    - Rejects wildcard "*" and requires explicit origins
    - Logs error and fails fast if no valid origins are configured
    
    In development:
    - Allows wildcard "*" for convenience (logs warning)
    """
    import re
    
    environment = getattr(config, "cors_environment", "production").lower().strip()
    allowed_origins_raw = getattr(config, "cors_allowed_origins", "")
    allow_origin_regex = getattr(config, "cors_allow_origin_regex", "")
    
    is_development = environment == "development"
    
    # Parse and validate allowed origins (comma-separated)
    allowed_origins: List[str] = []
    has_explicit_wildcard = False
    
    if allowed_origins_raw:
        for origin in allowed_origins_raw.split(","):
            origin = origin.strip()
            if origin:
                # Validate origin format (must be a valid URL or "*")
                if origin == "*":
                    has_explicit_wildcard = True
                    if is_development:
                        logger.warning(
                            "CORS: Wildcard '*' is enabled in DEVELOPMENT mode. "
                            "Do NOT use this setting in production!"
                        )
                        allowed_origins = ["*"]
                        break
                    else:
                        logger.error(
                            "CORS: Wildcard '*' origin is NOT allowed in production. "
                            "Please configure explicit origins via ALLOWED_ORIGINS or "
                            "config.yml cors.allowed_origins. Skipping '*'."
                        )
                        continue
                elif origin.startswith("http://") or origin.startswith("https://"):
                    allowed_origins.append(origin)
                else:
                    logger.warning(f"CORS: Invalid origin '{origin}' skipped. Origins must start with http:// or https://")
    
    # Validate regex pattern if provided
    validated_regex: Optional[str] = None
    if allow_origin_regex:
        try:
            re.compile(allow_origin_regex)
            validated_regex = allow_origin_regex
            logger.info(f"CORS: Using origin regex pattern: {allow_origin_regex}")
        except re.error as e:
            logger.error(f"CORS: Invalid regex pattern '{allow_origin_regex}': {e}. Ignoring regex.")
    
    # In development mode with no explicit origins, default to permissive wildcard "*"
    if is_development and not allowed_origins and not validated_regex:
        logger.warning(
            "CORS: No origins configured in DEVELOPMENT mode. "
            "Defaulting to wildcard '*'. Do NOT use this setting in production!"
        )
        allowed_origins = ["*"]
    
    # Fail fast in production if no valid origins are configured
    if not is_development and not allowed_origins and not validated_regex:
        error_msg = (
            "CORS: No valid origins configured for production! "
            "Set ALLOWED_ORIGINS environment variable (comma-separated URLs) or "
            "configure cors.allowed_origins in config.yml. "
            "For dynamic subdomains, set CORS_ALLOW_ORIGIN_REGEX or cors.allow_origin_regex. "
            "Example: ALLOWED_ORIGINS='https://app.example.com,https://admin.example.com' "
            "Falling back to localhost origins for safety."
        )
        logger.error(error_msg)
        # Fallback to common localhost origins for local development/testing
        allowed_origins = [
            "http://localhost:3000",
            "http://localhost:5173",
            "http://localhost:7860",
            "http://127.0.0.1:3000",
            "http://127.0.0.1:5173",
            "http://127.0.0.1:7860",
        ]
        logger.warning(f"CORS: Using fallback localhost origins: {allowed_origins}")
    
    # Log the final configuration
    if allowed_origins == ["*"]:
        logger.info("CORS: Configured with permissive wildcard '*' (DEVELOPMENT ONLY)")
    else:
        logger.info(f"CORS: Configured with {len(allowed_origins)} explicit origin(s)")
        for origin in allowed_origins[:5]:  # Log first 5
            logger.info(f"  - {origin}")
        if len(allowed_origins) > 5:
            logger.info(f"  ... and {len(allowed_origins) - 5} more")
    
    # Configure CORS middleware
    # Note: allow_origin_regex and allow_origins are mutually exclusive in Starlette
    # If regex is set, it takes precedence for matching, but we still set origins for non-regex cases
    cors_kwargs = {
        "allow_credentials": True,
        "allow_methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
        "allow_headers": ["*"],  # Allow all headers (common for APIs)
    }
    
    if validated_regex and not (allowed_origins == ["*"]):
        # Use regex for dynamic subdomain matching
        cors_kwargs["allow_origin_regex"] = validated_regex
        # Also include explicit origins as fallback
        if allowed_origins:
            cors_kwargs["allow_origins"] = allowed_origins
    else:
        cors_kwargs["allow_origins"] = allowed_origins
    
    app.add_middleware(CORSMiddleware, **cors_kwargs)


# Configure CORS with secure, environment-driven settings
configure_cors(app)

# --- Helper Classes & Functions ---

class FileWrapper:
    """Wrapper to mimic Gradio file object for analyzers"""
    def __init__(self, path):
        self.name = path

def parse_model_name(model_name):
    """
    Parse model name to extract provider and tier/model
    
    Supports formats:
    - "LITELLM:model_name" -> ("litellm", "smart", "model_name")
    - "OLLAMA:model_name" -> ("ollama", "smart", "model_name")
    - "LITELLM_FAST" -> ("litellm", "fast", None)
    - Legacy formats for backward compatibility
    
    Returns:
        Tuple of (provider, tier, model_name)
    """
    # New dynamic format: PROVIDER:model_name
    if ":" in model_name:
        parts = model_name.split(":", 1)
        provider = parts[0].lower()
        actual_model = parts[1]
        return provider, "smart", actual_model
    
    # Legacy tier-based format
    if model_name.startswith("LITELLM_"):
        tier = model_name.replace("LITELLM_", "").lower()
        return "litellm", tier, None
    elif model_name == "LITELLM":
        return "litellm", config.default_parse_fallback_tier, None
    elif model_name.startswith("OLLAMA_"):
        tier = model_name.replace("OLLAMA_", "").lower()
        return "ollama", tier, None
    elif model_name == "OLLAMA":
        return "ollama", config.default_parse_fallback_tier, None
    else:
        # Use configured fallback provider and tier
        return config.default_parse_fallback_provider, config.default_parse_fallback_tier, None

async def set_model_context(model_name: str):
    """Set the LLM model for the current session (Thread/Async safe wrapper)"""
    provider, tier, actual_model = parse_model_name(model_name)
    if provider in ["litellm", "ollama"]:
        client = get_client(provider=provider, model_tier=tier, model_name=actual_model)
        LlamaSettings.llm = client.get_llamaindex_llm()
        LlamaSettings.embed_model = client.get_llamaindex_embedding()
        logger.info(f"Set model context: provider={provider}, tier={tier}, model={actual_model}")

def ask_query(question, model_name=None):
    """Query the vector index using the specified model"""
    # Use config default if no model specified
    if model_name is None:
        model_name = config.default_chat_model_name
    # Note: This function runs synchronously as per original logic.
    # In FastAPI we wrap it or just run it. LlamaIndex is mostly sync.
    provider, tier, actual_model = parse_model_name(model_name)

    if provider in ["litellm", "ollama"]:
        client = get_client(provider=provider, model_tier=tier, model_name=actual_model)
        original_llm = LlamaSettings.llm
        original_embed = LlamaSettings.embed_model
        LlamaSettings.llm = client.get_llamaindex_llm()
        LlamaSettings.embed_model = client.get_llamaindex_embedding()

    try:
        if not os.path.exists(VECTOR_FOLDER) or not os.listdir(VECTOR_FOLDER):
             return "Index not found. Please upload documents first."

        storage_context = StorageContext.from_defaults(persist_dir=VECTOR_FOLDER)
        vector_index = load_index_from_storage(storage_context, index_id="vector_index")
        retriever = VectorIndexRetriever(
            index=vector_index,
            similarity_top_k=10,
        )
        response_synthesizer = get_response_synthesizer(
            text_qa_template=qa_template,
        )
        query_engine = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
        )
        response = query_engine.query(question)
        answer = str(response) # response.response
    except Exception as e:
        logger.error(f"Error querying index: {e}")
        return f"Error: {str(e)}"
    finally:
        if provider in ["litellm", "ollama"]:
            LlamaSettings.llm = original_llm
            LlamaSettings.embed_model = original_embed

    return answer

# --- Pydantic Models ---
# Default values are loaded from config.yml via config module

class AnalyzeUrlRequest(BaseModel):
    url: str
    memorize: bool = False
    model_name: str = config.default_analyze_model_name

class ChatRequest(BaseModel):
    message: str
    history: List[List[str]] = []
    model_name: str = config.default_chat_model_name

class InternetChatRequest(BaseModel):
    message: str
    history: List[List[str]]
    model_name: str = config.default_internet_chat_model_name
    max_tokens: int = 4096
    temperature: float = 0.5

class TripRequest(BaseModel):
    city: str
    days: str
    model_name: str = config.default_trip_model_name

class CravingRequest(BaseModel):
    city: str
    cuisine: str
    model_name: str = config.default_cravings_model_name

class ImageGenRequest(BaseModel):
    prompt: str
    enhanced_prompt: str = ""
    size: str = "1024x1024"
    provider: str = config.default_image_provider

class ImageEditRequest(BaseModel):
    img_path: str
    prompt: str
    enhanced_prompt: str = ""
    size: str = "1024x1024"
    provider: str = config.default_image_provider

class PromptEnhanceRequest(BaseModel):
    prompt: str
    provider: str = config.default_image_provider

class SurpriseRequest(BaseModel):
    provider: str = config.default_image_provider

class MemorySaveRequest(BaseModel):
    title: str
    content: str
    source_type: str
    source_ref: str
    model_name: str = config.default_memory_model_name

class MemorySearchRequest(BaseModel):
    query: str
    model_name: str = config.default_memory_model_name
    top_k: int = 5

class MemoryChatRequest(BaseModel):
    message: str
    history: List[List[str]] = []
    model_name: str = config.default_memory_model_name
    top_k: int = 5

class MemoryResetRequest(BaseModel):
    model_name: str = config.default_memory_model_name


# --- API Endpoints ---

@app.get("/api/health")
async def health_check():
    return {"status": "ok"}

@app.get("/api/models/{provider}")
async def get_models(provider: str):
    """
    Get list of available models for a provider (litellm or ollama)
    
    Args:
        provider: Either "litellm" or "ollama"
    
    Returns:
        List of model names
    """
    if provider.lower() not in ["litellm", "ollama"]:
        raise HTTPException(status_code=400, detail="Provider must be 'litellm' or 'ollama'")
    
    try:
        models = list_available_models(provider.lower())
        return {"models": models}
    except Exception as e:
        logger.error(f"Error fetching models for {provider}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Analyzers

@app.post("/api/analyze/file")
async def endpoint_analyze_file(
    files: List[UploadFile] = File(...),
    memorize: bool = Form(False),
    model_name: str = Form(config.default_analyze_model_name)
):
    async with api_lock:
        await set_model_context(model_name)
        
        tmp_dir = tempfile.mkdtemp()
        file_objs = []
        try:
            for file in files:
                # Sanitize filename to prevent path traversal attacks:
                # 1. Extract basename to strip any directory components
                # 2. Keep only safe characters (letters, numbers, dash, underscore, dot)
                # 3. Add UUID prefix to avoid filename collisions
                raw_name = os.path.basename(file.filename) if file.filename else "upload"
                # Remove any character not in the safe whitelist
                safe_name = re.sub(r'[^a-zA-Z0-9._-]', '_', raw_name)
                # Ensure we have a valid filename (not empty, not just dots)
                if not safe_name or safe_name.strip('.') == '':
                    safe_name = "upload"
                # Add UUID prefix to ensure uniqueness
                unique_filename = f"{uuid.uuid4().hex[:8]}_{safe_name}"
                file_path = os.path.join(tmp_dir, unique_filename)
                with open(file_path, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)
                file_objs.append(FileWrapper(file_path))
            
            result = analyze_file(file_objs, memorize)
            return result
        except Exception as e:
            logger.error(f"Error analyzing file: {e}")
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            shutil.rmtree(tmp_dir)

@app.post("/api/analyze/youtube")
async def endpoint_analyze_youtube(req: AnalyzeUrlRequest):
    async with api_lock:
        await set_model_context(req.model_name)
        return analyze_ytvideo(req.url, req.memorize)

@app.post("/api/analyze/article")
async def endpoint_analyze_article(req: AnalyzeUrlRequest):
    async with api_lock:
        await set_model_context(req.model_name)
        return analyze_article(req.url, req.memorize)

@app.post("/api/analyze/media")
async def endpoint_analyze_media(req: AnalyzeUrlRequest):
    async with api_lock:
        await set_model_context(req.model_name)
        return analyze_media(req.url, req.memorize)

@app.post("/api/docqa/reset")
async def endpoint_reset():
    clearallfiles()
    return {"status": "Database reset"}

@app.post("/api/docqa/ask")
async def endpoint_ask(req: ChatRequest):
    async with api_lock:
        # ask_query uses global settings, so we need lock
        answer = ask_query(req.message, req.model_name)
        return {"answer": answer}

@app.post("/api/docqa/ask_stream")
async def endpoint_ask_stream(req: ChatRequest):
    async with api_lock:
        # Prepare the stream inside the lock to handle retrieval and settings safely
        # prepare_chat_stream runs the retrieval and returns a response object with .response_gen
        response = prepare_chat_stream(
            question=req.message,
            model_name=req.model_name,
            vector_folder=VECTOR_FOLDER,
            qa_template=qa_template,
            parse_model_name_func=parse_model_name
        )
        
    # The generator response.response_gen can be iterated outside the lock 
    # because it uses the specific LLM instance captured during setup
    return StreamingResponse(response.response_gen, media_type="text/event-stream")

# Internet Chat
@app.post("/api/chat/internet")
async def endpoint_chat_internet(req: InternetChatRequest):
    # Internet connected chatbot handles its own context/history
    # But it might use global LlamaIndex settings?
    # helper_functions/chat_generation_with_internet.py uses get_client and Settings.
    # It seems to set Settings.llm inside the module globally on import, 
    # but the function `internet_connected_chatbot` doesn't explicitly set them?
    # Wait, internet_connected_chatbot calls `generate_chat` or `get_web_results`.
    # `get_web_results` calls `firecrawl_researcher` which calls `conduct_research_firecrawl`.
    # Let's assume it's safe or we need the lock if it modifies global Settings.
    # Given the complexity, locking is safer.
    async with api_lock:
        response = internet_connected_chatbot(
            query=req.message,
            history=req.history,
            model_name=req.model_name,
            max_tokens=req.max_tokens,
            temperature=req.temperature
        )
        return {"response": response}

# Memory Palace Endpoints
@app.post("/api/memory_palace/save")
async def endpoint_memory_save(req: MemorySaveRequest):
    async with api_lock:
        await set_model_context(req.model_name)
        try:
            result = save_memory(
                title=req.title,
                content=req.content,
                source_type=req.source_type,
                source_ref=req.source_ref,
                model_name=req.model_name
            )
            return {"status": result}
        except Exception as e:
            logger.error(f"Error saving to memory palace: {e}")
            raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/memory_palace/search")
async def endpoint_memory_search(req: MemorySearchRequest):
    # Search doesn't necessarily need the LLM context if it just loads the index and uses it,
    # but VectorIndexRetriever might trigger embedding generation for the query.
    # So we should lock and set context.
    async with api_lock:
        await set_model_context(req.model_name)
        try:
            results = search_memories(
                query=req.query,
                model_name=req.model_name,
                top_k=req.top_k
            )
            return {"results": results}
        except Exception as e:
            logger.error(f"Error searching memory palace: {e}")
            raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/memory_palace/ask_stream")
async def endpoint_memory_ask_stream(req: MemoryChatRequest):
    async with api_lock:
        await set_model_context(req.model_name)
        try:
            response = prepare_memory_stream(
                message=req.message,
                history=req.history,
                model_name=req.model_name,
                top_k=req.top_k
            )
        except Exception as e:
            logger.error(f"Error preparing memory stream: {e}")
            raise HTTPException(status_code=500, detail=str(e))
            
    # Stream response outside lock
    return StreamingResponse(response.response_gen, media_type="text/event-stream")

@app.post("/api/memory_palace/reset")
async def endpoint_memory_reset(req: MemoryResetRequest):
    # Reset deletes files, doesn't need LLM context usually, but we need to know the folder path which depends on model name
    try:
        result = reset_memory_palace(req.model_name)
        return {"status": result}
    except Exception as e:
        logger.error(f"Error resetting memory palace: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Fun Tools
@app.post("/api/fun/trip")
async def endpoint_trip(req: TripRequest):
    # generate_trip_plan takes (city, days, model_name)
    result = generate_trip_plan(req.city, req.days, req.model_name)
    # result is a list/tuple? In Gradio it was returning single value for output
    # check gradio_ui_full.py: outputs=[city_output]
    # generate_trip_plan likely returns a string.
    return {"plan": result}

@app.post("/api/fun/cravings")
async def endpoint_cravings(req: CravingRequest):
    result = craving_satisfier(req.city, req.cuisine, req.model_name)
    return {"recommendation": result}

# Image Studio
@app.post("/api/image/generate")
async def endpoint_image_generate(req: ImageGenRequest):
    final_prompt = req.enhanced_prompt if req.enhanced_prompt and req.enhanced_prompt.strip() else req.prompt
    img_path = tool.run_generate_unified(final_prompt, size=req.size, provider=req.provider)
    # img_path is a local path. We need to serve it.
    # It likely saves to helper_functions/data/ or similar.
    # We should ensure it's accessible. 
    # Let's return the URL path relative to our static mount if possible, or serve it via a generic /api/files endpoint.
    # For now, return the path.
    return {"image_path": img_path}

@app.post("/api/image/edit")
async def endpoint_image_edit(req: ImageEditRequest):
    # Image edit requires a source image path.
    # The frontend should upload the image first?
    # Or if it's already generated, we pass the path.
    # If it's a new upload, we need an upload endpoint for image studio.
    
    # For simplicity, let's assume the user uploads an image to edit via a separate endpoint first,
    # or we handle upload in this request (multipart).
    # But the request model above assumes string path.
    
    # Let's check how Gradio handled it: `edit_img` (Image component) -> `ui_edit_wrapper` -> `tool.run_edit_unified`.
    # Gradio passes the file path of the uploaded image.
    
    # We need an endpoint to upload image for editing.
    if not os.path.exists(req.img_path):
        raise HTTPException(status_code=404, detail="Image not found")
        
    final_prompt = req.enhanced_prompt if req.enhanced_prompt and req.enhanced_prompt.strip() else req.prompt
    try:
        edited_path = tool.run_edit_unified(req.img_path, final_prompt, size=req.size, provider=req.provider)
        return {"image_path": edited_path}
    except Exception as e:
        logger.error(f"Image edit failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/image/upload")
async def endpoint_image_upload(file: UploadFile = File(...)):
    # Save uploaded image for editing
    if not os.path.exists("data/uploads"):
        os.makedirs("data/uploads")
    
    file_path = f"data/uploads/{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    return {"file_path": file_path}

@app.post("/api/image/enhance")
async def endpoint_enhance(req: PromptEnhanceRequest):
    res = tool.prompt_enhancer_unified(req.prompt, req.provider)
    return {"enhanced_prompt": res}

@app.post("/api/image/surprise")
async def endpoint_surprise(req: SurpriseRequest):
    res = tool.generate_surprise_prompt_unified(req.provider)
    return {"prompt": res}

@app.get("/api/files/{file_path:path}")
async def serve_file(file_path: str):
    """
    Serve files from allowed directories only.
    Security: Uses path resolution and safe containment checks to prevent path traversal.
    """
    from pathlib import Path
    
    # Whitelist of allowed file extensions (case-insensitive)
    ALLOWED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.gif', '.webp'}
    
    # Allowed directories - resolved to absolute paths to handle symlinks and normalize
    allowed_dirs = [
        Path("helper_functions/data").resolve(),
        Path("data/uploads").resolve(),
        Path("playground").resolve()
    ]
    
    # Resolve the requested path to absolute, handling symlinks and .. traversal
    try:
        resolved_path = Path(file_path).resolve()
    except (ValueError, OSError):
        return {"error": "Invalid file path"}
    
    # Check file extension against whitelist
    if resolved_path.suffix.lower() not in ALLOWED_EXTENSIONS:
        return {"error": "File type not allowed"}
    
    # Safe containment check using os.path.commonpath:
    # This properly handles path traversal attempts like /allowed/../../../etc/passwd
    is_allowed = False
    for allowed_dir in allowed_dirs:
        try:
            # commonpath returns the longest common sub-path; if it equals allowed_dir,
            # then resolved_path is contained within allowed_dir
            common = Path(os.path.commonpath([allowed_dir, resolved_path]))
            if common == allowed_dir:
                is_allowed = True
                break
        except ValueError:
            # commonpath raises ValueError if paths are on different drives (Windows)
            continue
    
    if not is_allowed:
        return {"error": "Access denied"}
    
    # Final check: file must exist and be a regular file (not directory/symlink to dir)
    if not resolved_path.exists() or not resolved_path.is_file():
        return {"error": "File not found"}
    
    return FileResponse(str(resolved_path))


# Serve Frontend
@app.get("/{full_path:path}")
async def serve_app(full_path: str):
    if full_path.startswith("api/"):
        return {"error": "API route not found"}
        
    file_path = os.path.join("frontend/dist", full_path)
    if os.path.exists(file_path) and os.path.isfile(file_path):
        return FileResponse(file_path)
    
    index_path = "frontend/dist/index.html"
    if os.path.exists(index_path):
        return FileResponse(index_path)
    
    return {"error": "Frontend not built. Run 'npm run build' in frontend directory."}

if __name__ == "__main__":
    print("Starting LLM QA Bot on http://0.0.0.0:7860")
    uvicorn.run(app, host="0.0.0.0", port=7860)
