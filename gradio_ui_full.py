"""
LLM QA Bot - Full Featured Gradio UI
Supports LiteLLM, Ollama, Gemini, Cohere, and Groq with Firecrawl integration
"""

import os
import gradio as gr
import logging
import sys
import traceback
import requests
from openai import OpenAI
from llama_index.core import StorageContext, load_index_from_storage, get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import PromptTemplate
from helper_functions.chat_generation_with_internet import internet_connected_chatbot
from helper_functions.trip_planner import generate_trip_plan
from helper_functions.food_planner import craving_satisfier
from helper_functions.analyzers import analyze_article, analyze_ytvideo, analyze_media, analyze_file, clearallfiles
from helper_functions.chat_generation import generate_chat
from config import config
# Import image tool functions
from helper_functions import gptimage_tool as tool

# --- OpenAI Client Initialization ---
# Create one client instance to reuse for image generation (requires OpenAI API)
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE")
)
# --- End OpenAI Client Initialization ---

# Model choices for Document Q&A (simplified)
DOCQA_MODEL_CHOICES = [
    "LITELLM",
    "OLLAMA"
]

# Model choices for other tabs (full list)
MODEL_CHOICES = [
    "LITELLM_FAST",
    "LITELLM_SMART",
    "LITELLM_STRATEGIC",
    "OLLAMA_FAST",
    "OLLAMA_SMART",
    "OLLAMA_STRATEGIC",
    "GEMINI",
    "GEMINI_THINKING",
    "COHERE",
    "GROQ",
    "GROQ_LLAMA",
    "GROQ_MIXTRAL"
]

# Configuration
VECTOR_FOLDER = config.VECTOR_FOLDER
SUMMARY_FOLDER = config.SUMMARY_FOLDER
example_queries = config.example_queries
example_memorypalacequeries = config.example_memorypalacequeries if hasattr(config, 'example_memorypalacequeries') else []
example_internetqueries = config.example_internetqueries if hasattr(config, 'example_internetqueries') else []
sum_template = config.sum_template
eg_template = config.eg_template
ques_template = config.ques_template
summary_template = PromptTemplate(sum_template)
example_template = PromptTemplate(eg_template)
qa_template = PromptTemplate(ques_template)

# Global variables
summary = "No Summary available yet"


# --- Helper Functions ---
def parse_model_name(model_name):
    """Parse model name to extract provider and tier"""
    if model_name.startswith("LITELLM_"):
        tier = model_name.replace("LITELLM_", "").lower()
        return "litellm", tier
    elif model_name == "LITELLM":
        # Simplified option: use fast model for summary/example generation
        return "litellm", "fast"
    elif model_name.startswith("OLLAMA_"):
        tier = model_name.replace("OLLAMA_", "").lower()
        return "ollama", tier
    elif model_name == "OLLAMA":
        # Simplified option: use fast model for summary/example generation
        return "ollama", "fast"
    else:
        # For other models (GEMINI, COHERE, GROQ), use the default
        return "litellm", "fast"


# --- Content Analysis Functions ---
def set_model_for_session(model_name):
    """Set the LLM model for the current session"""
    from helper_functions.llm_client import get_client
    from llama_index.core import Settings as LlamaSettings

    provider, tier = parse_model_name(model_name)

    if provider in ["litellm", "ollama"]:
        client = get_client(provider=provider, model_tier=tier)
        LlamaSettings.llm = client.get_llamaindex_llm()
        LlamaSettings.embed_model = client.get_llamaindex_embedding()
        print(f"Set model for session: provider={provider}, tier={tier}")


def upload_file(files, memorize, model_name):
    """Upload and process files with the selected model"""
    global example_queries, summary, query_component
    
    # Set the model for this session
    set_model_for_session(model_name)

    analysis = analyze_file(files, memorize)
    message = analysis["message"]
    summary = analysis["summary"]
    example_queries = analysis["example_queries"]
    file_title = analysis["file_title"]
    file_memoryupload_status = analysis["file_memoryupload_status"]

    # Return results + locked model selector + model state
    return (message, gr.Dataset(components=[query_component], samples=example_queries), summary, 
            file_title, file_memoryupload_status, gr.Dropdown(interactive=False), model_name)


def download_ytvideo(url, memorize, model_name):
    """Download and process YouTube video with the selected model"""
    global example_queries, summary, video_title, query_component
    
    # Set the model for this session
    set_model_for_session(model_name)

    analysis = analyze_ytvideo(url, memorize)
    message = analysis["message"]
    summary = analysis["summary"]
    example_queries = analysis["example_queries"]
    video_title = analysis["video_title"]
    video_memoryupload_status = analysis["video_memoryupload_status"]

    # Return results + locked model selector + model state
    return (message, gr.Dataset(components=[query_component], samples=example_queries), summary, 
            video_title, video_memoryupload_status, gr.Dropdown(interactive=False), model_name)


def download_art(url, memorize, model_name):
    """Download and process article with the selected model"""
    global example_queries, summary, article_title, query_component
    
    # Set the model for this session
    set_model_for_session(model_name)

    analysis = analyze_article(url, memorize)
    message = analysis["message"]
    summary = analysis["summary"]
    example_queries = analysis["example_queries"]
    article_title = analysis["article_title"]
    article_memoryupload_status = analysis["article_memoryupload_status"]

    # Return results + locked model selector + model state
    return (message, gr.Dataset(components=[query_component], samples=example_queries), summary, 
            article_title, article_memoryupload_status, gr.Dropdown(interactive=False), model_name)


def download_media(url, memorize, model_name):
    """Download and process media with the selected model"""
    global example_queries, summary, media_title, query_component
    
    # Set the model for this session
    set_model_for_session(model_name)

    analysis = analyze_media(url, memorize)
    message = analysis["message"]
    summary = analysis["summary"]
    example_queries = analysis["example_queries"]
    media_title = analysis["media_title"]
    media_memoryupload_status = analysis["media_memoryupload_status"]

    # Return results + locked model selector + model state
    return (message, gr.Dataset(components=[query_component], samples=example_queries), summary, 
            media_title, media_memoryupload_status, gr.Dropdown(interactive=False), model_name)


# --- Q&A Functions ---
def ask(question, history, model_name):
    """Ask a question using the selected model"""
    history = history or []
    answer = ask_query(question, model_name)
    return answer


def ask_query(question, model_name="LITELLM_SMART"):
    """Query the vector index using the specified model"""
    from helper_functions.llm_client import get_client
    from llama_index.core import Settings as LlamaSettings

    # Parse model name to get provider and tier
    provider, tier = parse_model_name(model_name)

    # Create a client for the selected model
    if provider in ["litellm", "ollama"]:
        client = get_client(provider=provider, model_tier=tier)
        # Temporarily set the LLM and embedding model for this query
        original_llm = LlamaSettings.llm
        original_embed = LlamaSettings.embed_model
        LlamaSettings.llm = client.get_llamaindex_llm()
        LlamaSettings.embed_model = client.get_llamaindex_embedding()

    try:
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
        answer = response.response
    finally:
        # Restore original LLM and embedding model if we changed them
        if provider in ["litellm", "ollama"]:
            LlamaSettings.llm = original_llm
            LlamaSettings.embed_model = original_embed

    return answer


def load_example(example_id):
    global example_queries
    return example_queries[example_id][0]


def clearfield(field):
    return [""]


def clearhistory(field1, field2, field3):
    return ["", "", ""]


def clear_trip_plan(field1, field2):
    return "", "", "Your trip plan will appear here..."


def clear_craving_plan(field1, field2):
    return "", "", "Your food recommendations will appear here..."


def toggle_model_local(is_local):
    """Toggle between local and remote models"""
    if is_local:
        return "OLLAMA_FAST"
    else:
        return "LITELLM_FAST"


def fetch_litellm_models():
    """
    Fetch available models from LiteLLM API
    
    Returns:
        List of model names
    """
    try:
        base_url = config.litellm_base_url
        api_key = config.litellm_api_key
        
        # LiteLLM uses the OpenAI-compatible /v1/models endpoint
        models_url = f"{base_url}/models"
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        response = requests.get(models_url, headers=headers, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        # Extract model IDs from the response
        if "data" in data:
            models = [model["id"] for model in data["data"]]
            print(f"Fetched {len(models)} models from LiteLLM")
            return models
        else:
            print("Unexpected response format from LiteLLM")
            return []
            
    except Exception as e:
        print(f"Error fetching LiteLLM models: {e}")
        return []


def fetch_ollama_models():
    """
    Fetch available models from Ollama API
    
    Returns:
        List of model names
    """
    try:
        base_url = config.ollama_base_url
        
        # Ollama has a specific endpoint for listing models
        # Remove the /v1 suffix if present, as Ollama uses different endpoints
        base_url = base_url.replace("/v1", "")
        models_url = f"{base_url}/api/tags"
        
        response = requests.get(models_url, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        # Extract model names from the response
        if "models" in data:
            models = [model["name"] for model in data["models"]]
            print(f"Fetched {len(models)} models from Ollama")
            return models
        else:
            print("Unexpected response format from Ollama")
            return []
            
    except Exception as e:
        print(f"Error fetching Ollama models: {e}")
        return []


def update_model_dropdown(provider):
    """
    Update the model dropdown based on selected provider
    
    Args:
        provider: Either "LiteLLM" or "Ollama"
    
    Returns:
        Updated dropdown with new choices and default value
    """
    if provider == "LiteLLM":
        models = fetch_litellm_models()
        if not models:
            # Fallback to default choices if API call fails
            models = MODEL_CHOICES
        else:
            # Prefix models with provider identifier for routing
            models = [f"LITELLM:{model}" for model in models]
        default_model = models[0] if models else f"LITELLM:{config.litellm_default_model}"
    else:  # Ollama
        models = fetch_ollama_models()
        if not models:
            # Fallback to default choices if API call fails
            models = MODEL_CHOICES
        else:
            # Prefix models with provider identifier for routing
            models = [f"OLLAMA:{model}" for model in models]
        default_model = models[0] if models else f"OLLAMA:{config.ollama_default_model}"
    
    return gr.Dropdown(choices=models, value=default_model)


# --- Image Tool UI Helper Functions ---
def clear_generate():
    return "", "", None, "1024x1024"


def clear_edit():
    return None, "Turn the subject into a cyberpunk character with neon lights", "", None, "1024x1024"


def ui_generate_wrapper(prompt: str, enhanced_prompt: str, size: str, provider: str = "openai"):
    final_prompt = enhanced_prompt if enhanced_prompt and enhanced_prompt.strip() else prompt
    img_path = tool.run_generate_unified(final_prompt, size=size, provider=provider)
    return img_path


def ui_edit_wrapper(img_path: str, prompt: str, enhanced_prompt: str, size: str, provider: str = "openai"):
    if not img_path:
        raise gr.Error("Please upload an image to edit.")
    edited_img_path = img_path
    final_prompt = enhanced_prompt if enhanced_prompt and enhanced_prompt.strip() else prompt
    try:
        edited_img_path = tool.run_edit_unified(img_path, final_prompt, size=size, provider=provider)
    except Exception as e:
        print(f"Error during image edit: {e}")
        traceback.print_exc()
        gr.Warning(f"Image edit failed: {e}. Returning original image.")
        return img_path
    return edited_img_path


# --- Main UI ---
logging.basicConfig(stream=sys.stdout, level=logging.CRITICAL)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

with gr.Blocks(theme='davehornik/Tealy', fill_height=True) as llmapp:
    gr.Markdown(
        """
        <h1><center><b>LLM Bot: Your AI-Powered Knowledge Companion</center></h1>
        """
    )
    gr.Markdown(
        """
        <center>
        <br>
        Dive into the world of AI with LLM Bot, your go-to assistant for exploring, learning, and storing knowledge. <br>
        Powered by LiteLLM, Ollama, Gemini, Cohere, and Groq with Firecrawl web scraping. <br>
        </center>
        """
    )

    with gr.Tab(label="Document Q&A"):
        with gr.Row():
            with gr.Column():
                docqa_model_selector = gr.Dropdown(
                    label="Model Selection",
                    choices=DOCQA_MODEL_CHOICES,
                    value="LITELLM",
                    interactive=True,
                    info="Select model before uploading documents. This will be used for analysis and Q&A."
                )
                memorize = gr.Checkbox(label="Store in vector index", value=False, visible=False)
                reset_button = gr.Button(value="Reset Database. Deletes all local data")
                docqa_model_state = gr.State(value="LITELLM")  # Store the locked-in model
        with gr.Row():
            with gr.Column():
                with gr.Tab(label="Video Analyzer"):
                    with gr.Row():
                        with gr.Column(scale=0):
                            yturl = gr.Textbox(placeholder="Input must be a Youtube URL", label="Enter Youtube URL")
                            video_output = gr.Textbox(label="Video Processing Status")
                            video_button = gr.Button(value="Process")
                        with gr.Column(scale=0):
                            video_title = gr.Textbox(placeholder="Video title to be generated here", label="Video Title")
                            video_memoryupload_status = gr.Textbox(label="Status", visible=False)
                    with gr.Accordion(label="Key takeaways", open=False):
                        vsummary_output = gr.Markdown(value="Summary will be generated here")
                with gr.Tab(label="Article Analyzer"):
                    with gr.Row():
                        with gr.Column(scale=0):
                            arturl = gr.Textbox(placeholder="Input must be a URL", label="Enter Article URL")
                            article_output = gr.Textbox(label="Article Processing Status")
                            article_button = gr.Button(value="Process")
                        with gr.Column(scale=0):
                            article_title = gr.Textbox(placeholder="Article title to be generated here", label="Article Title")
                            article_memoryupload_status = gr.Textbox(label="Status", visible=False)
                    with gr.Accordion(label="Key takeaways", open=False):
                        asummary_output = gr.Markdown(value="Summary will be generated here")
                with gr.Tab(label="File Analyzer"):
                    with gr.Row():
                        with gr.Column(scale=0):
                            files = gr.Files(label="Supported types: pdf, txt, docx, png, jpg, jpeg, mp3")
                            file_output = gr.Textbox(label="File Processing Status")
                            file_button = gr.Button(value="Process")
                        with gr.Column(scale=0):
                            file_title = gr.Textbox(placeholder="File title to be generated here", label="File Title")
                            file_memoryupload_status = gr.Textbox(label="Status", visible=False)
                    with gr.Accordion(label="Key takeaways", open=False):
                        fsummary_output = gr.Markdown(value="Summary will be generated here")
                with gr.Tab(label="Media URL Analyzer"):
                    with gr.Row():
                        with gr.Column(scale=0):
                            mediaurl = gr.Textbox(placeholder="Input must be a URL", label="Enter Media URL")
                            media_output = gr.Textbox(label="Media Processing Status")
                            media_button = gr.Button(value="Process")
                        with gr.Column(scale=0):
                            media_title = gr.Textbox(placeholder="Media title to be generated here", label="Media Title")
                            media_memoryupload_status = gr.Textbox(label="Status", visible=False)
                    with gr.Accordion(label="Key takeaways", open=False):
                        msummary_output = gr.Markdown(value="Summary will be generated here")
        with gr.Row():
            with gr.Column(scale=8):
                chatui = gr.ChatInterface(
                    ask,
                    additional_inputs=[docqa_model_state],
                    submit_btn="Ask",
                    fill_height=True,
                    type="messages"
                )
                query_component = chatui.textbox
            with gr.Column(scale=2):
                examples = gr.Dataset(label="Questions", samples=example_queries, components=[query_component], type="index")

    with gr.Tab(label="AI Assistant"):
        gr.Markdown("### Internet-Connected AI with Firecrawl and GPT Researcher")
        gr.Markdown("_Uses Firecrawl for web scraping and GPT Researcher for deep research._")
        
        with gr.Row():
            ai_provider_selector = gr.Radio(
                label="LLM Provider",
                choices=["LiteLLM", "Ollama"],
                value="LiteLLM",
                info="Select your LLM provider to fetch available models"
            )
        
        ai_chatinterface = gr.ChatInterface(
            internet_connected_chatbot,
            additional_inputs=[
                gr.Dropdown(label="Model", choices=MODEL_CHOICES, value="LITELLM_SMART"),
                gr.Slider(10, 8680, value=4840, label="Max Output Tokens"),
                gr.Slider(0.1, 0.9, value=0.5, label="Temperature"),
            ],
            examples=example_internetqueries if example_internetqueries else [
                ["What's the latest news about AI?"],
                ["What's the weather in New York?"],
                ["Search for information about quantum computing"]
            ],
            submit_btn="Ask",
            fill_height=True,
            type="messages"
        )
        
        # Get reference to the model dropdown for updating
        ai_model_dropdown = ai_chatinterface.additional_inputs[0]

    with gr.Tab(label="Fun"):
        with gr.Tab(label="City Planner"):
            with gr.Row():
                city_name = gr.Textbox(placeholder="Enter the name of the city", label="City Name")
                number_of_days = gr.Textbox(placeholder="Enter the number of days", label="Number of Days")
                trip_local_checkbox = gr.Checkbox(label="LOCAL", value=False, scale=0, info="Use local Ollama model")
            with gr.Row():
                city_button = gr.Button(value="Plan", scale=0)
                clear_trip_button = gr.Button(value="Clear", scale=0)
            with gr.Accordion(label="Trip Plan", open=True):
                city_output = gr.Markdown(value="Your trip plan will appear here...")
            trip_model_state = gr.State(value="LITELLM_FAST")
        with gr.Tab(label="Cravings Generator"):
            with gr.Row():
                craving_city_name = gr.Textbox(placeholder="Enter the name of the city", label="City Name")
                craving_cuisine = gr.Textbox(placeholder="What are you craving for? Enter idk if you don't know what you want to eat", label="Food")
                craving_local_checkbox = gr.Checkbox(label="LOCAL", value=False, scale=0, info="Use local Ollama model")
            with gr.Row():
                craving_button = gr.Button(value="Cook", scale=0)
                clear_craving_button = gr.Button(value="Clear", scale=0)
            with gr.Accordion(label="Food Places", open=True):
                craving_output = gr.Markdown(value="Your food recommendations will appear here...")
            craving_model_state = gr.State(value="LITELLM_FAST")

    # --- Image Studio Tab ---
    with gr.Tab("Image Studio"):
        gr.Markdown("## Generate and Edit Images using AI Models")
        with gr.Row():
            image_provider = gr.Dropdown(
                label="AI Provider",
                choices=["nvidia"],
                value="nvidia",
                info="NVIDIA: Stable Diffusion 3"
            )
        with gr.Tab("Generate"):
            clear_gen_btn = gr.Button("Clear")
            with gr.Row():
                gen_prompt = gr.Textbox(
                    label="Prompt",
                    value="A futuristic cityscape at sunset, synthwave style",
                    scale=4
                )
                gen_size = gr.Dropdown(
                    label="Size",
                    choices=["1024x1024", "1024x1536", "1536x1024"],
                    value="1024x1024",
                    scale=1
                )
            with gr.Row():
                enhance_btn = gr.Button("Enhance Prompt")
                surprise_gen_btn = gr.Button("🎁 Surprise Me!")
            gen_enhanced_prompt = gr.Textbox(label="Enhanced Prompt (Editable)", interactive=True)
            with gr.Row():
                gen_ex1_btn = gr.Button("🌆 Synthwave City")
                gen_ex2_btn = gr.Button("🐶 Dog Astronaut")
                gen_ex3_btn = gr.Button("🍔 Giant Burger")
                gen_ex4_btn = gr.Button("🎨 Watercolor Forest")
            gen_btn = gr.Button("Generate Image", variant="primary")
            gen_out = gr.Image(label="Generated Image", show_download_button=True)

        with gr.Tab("Edit"):
            clear_edit_btn = gr.Button("Clear")
            edit_img = gr.Image(type="filepath", label="Image to Edit", sources=["upload"], height=400)
            with gr.Row():
                edit_prompt = gr.Textbox(
                    label="Edit Prompt",
                    value="Turn the subject into a cyberpunk character with neon lights",
                    scale=4
                )
                edit_size = gr.Dropdown(
                    label="Size",
                    choices=["1024x1024", "1024x1536", "1536x1024"],
                    value="1024x1024",
                    scale=1
                )
            with gr.Row():
                edit_enhance_btn = gr.Button("Enhance Prompt")
                surprise_edit_btn = gr.Button("🎁 Surprise Me!")
            edit_enhanced_prompt = gr.Textbox(label="Enhanced Prompt (Editable)", interactive=True)
            with gr.Row():
                ghibli_btn = gr.Button("🎨 Ghibli Style")
                simp_btn = gr.Button("📺 Simpsons")
                sp_btn = gr.Button("☃️ South Park")
                comic_btn = gr.Button("💥 Comic Style")
            edit_btn = gr.Button("Edit Image", variant="primary")
            edit_out = gr.Image(label="Edited Image", show_download_button=True)
    # --- End Image Studio Tab ---

    # --- Event Handlers ---
    # Document Q&A Event Handlers
    video_button.click(download_ytvideo, inputs=[yturl, memorize, docqa_model_selector], 
                       outputs=[video_output, examples, vsummary_output, video_title, video_memoryupload_status, 
                               docqa_model_selector, docqa_model_state], show_progress=True)
    article_button.click(download_art, inputs=[arturl, memorize, docqa_model_selector], 
                        outputs=[article_output, examples, asummary_output, article_title, article_memoryupload_status, 
                                docqa_model_selector, docqa_model_state], show_progress=True)
    file_button.click(upload_file, inputs=[files, memorize, docqa_model_selector], 
                     outputs=[file_output, examples, fsummary_output, file_title, file_memoryupload_status, 
                             docqa_model_selector, docqa_model_state], show_progress=True)
    media_button.click(download_media, inputs=[mediaurl, memorize, docqa_model_selector], 
                      outputs=[media_output, examples, msummary_output, media_title, media_memoryupload_status, 
                              docqa_model_selector, docqa_model_state], show_progress=True)

    # AI Assistant Event Handlers
    ai_provider_selector.change(
        update_model_dropdown,
        inputs=[ai_provider_selector],
        outputs=[ai_model_dropdown]
    )
    
    # Fun Features Event Handlers
    city_button.click(generate_trip_plan, inputs=[city_name, number_of_days, trip_model_state], outputs=[city_output], show_progress=True)
    craving_button.click(craving_satisfier, inputs=[craving_city_name, craving_cuisine, craving_model_state], outputs=[craving_output], show_progress=True)
    clear_trip_button.click(clear_trip_plan, inputs=[city_name, number_of_days], outputs=[city_name, number_of_days, city_output])
    clear_craving_button.click(clear_craving_plan, inputs=[craving_city_name, craving_cuisine], outputs=[craving_city_name, craving_cuisine, craving_output])
    
    # Local checkbox handlers - update the hidden state
    trip_local_checkbox.change(toggle_model_local, inputs=[trip_local_checkbox], outputs=[trip_model_state])
    craving_local_checkbox.change(toggle_model_local, inputs=[craving_local_checkbox], outputs=[craving_model_state])

    examples.click(load_example, inputs=[examples], outputs=[query_component])

    reset_button.click(clearallfiles)
    reset_button.click(lambda: ["", "", "", "", "", "", "", "", "", "", "", "", "", "", "", ""], 
                      inputs=[], 
                      outputs=[video_output, vsummary_output, video_title, video_memoryupload_status, 
                              article_output, asummary_output, article_title, article_memoryupload_status, 
                              file_output, fsummary_output, file_title, file_memoryupload_status, 
                              media_output, msummary_output, media_title, media_memoryupload_status])
    # Unlock model selector on reset
    reset_button.click(lambda: gr.Dropdown(interactive=True), inputs=[], outputs=[docqa_model_selector])

    # Image Studio Event Handlers
    # Generate Tab
    enhance_btn.click(
        lambda p, prov: tool.prompt_enhancer_unified(p, prov),
        inputs=[gen_prompt, image_provider],
        outputs=[gen_enhanced_prompt],
        show_progress="Generating enhanced prompt..."
    )
    surprise_gen_btn.click(
        lambda prov: tool.generate_surprise_prompt_unified(prov),
        inputs=[image_provider],
        outputs=[gen_prompt],
        show_progress="Generating surprise prompt..."
    )
    clear_gen_btn.click(
        clear_generate,
        inputs=None,
        outputs=[gen_prompt, gen_enhanced_prompt, gen_out, gen_size]
    )
    gen_ex1_btn.click(lambda: "A futuristic cityscape at sunset, synthwave style", None, gen_prompt)
    gen_ex2_btn.click(lambda: "A golden retriever wearing a space helmet, digital art", None, gen_prompt)
    gen_ex3_btn.click(lambda: "A giant cheeseburger resting on a mountaintop", None, gen_prompt)
    gen_ex4_btn.click(lambda: "A dense forest painted in watercolor style", None, gen_prompt)
    gen_btn.click(
        ui_generate_wrapper,
        inputs=[gen_prompt, gen_enhanced_prompt, gen_size, image_provider],
        outputs=[gen_out],
        show_progress="Generating image..."
    )

    # Edit Tab
    edit_enhance_btn.click(
        lambda p, prov: tool.prompt_enhancer_unified(p, prov),
        inputs=[edit_prompt, image_provider],
        outputs=[edit_enhanced_prompt],
        show_progress="Generating enhanced prompt..."
    )
    surprise_edit_btn.click(
        lambda prov: tool.generate_surprise_prompt_unified(prov),
        inputs=[image_provider],
        outputs=[edit_prompt],
        show_progress="Generating surprise prompt..."
    )
    clear_edit_btn.click(
        clear_edit,
        inputs=None,
        outputs=[edit_img, edit_prompt, edit_enhanced_prompt, edit_out, edit_size]
    )
    ghibli_btn.click(lambda: "Convert the picture into Ghibli style animation", None, edit_prompt)
    simp_btn.click(lambda: "Turn the subjects into a Simpsons character", None, edit_prompt)
    sp_btn.click(lambda: "Turn the subjects into a South Park character", None, edit_prompt)
    comic_btn.click(lambda: "Convert the picture into a comic book style drawing with compelling futuristic story and cool dialogues", None, edit_prompt)
    edit_btn.click(
        ui_edit_wrapper,
        inputs=[edit_img, edit_prompt, edit_enhanced_prompt, edit_size, image_provider],
        outputs=[edit_out],
        show_progress="Editing image..."
    )
    # --- End Event Handlers ---


if __name__ == '__main__':
    llmapp.launch(server_name='0.0.0.0', server_port=7860)
