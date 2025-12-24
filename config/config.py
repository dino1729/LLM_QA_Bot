"""
Configuration loader for LLM_QA_Bot.
All model identifiers and settings are loaded from config.yml.
No hardcoded model defaults - everything must be specified in config.yml.
"""
import yaml
import dotenv
import os

# Resolve config directory relative to this module's location
# This ensures imports work regardless of the current working directory
config_dir = os.path.dirname(os.path.abspath(__file__))

# load yaml config
with open(os.path.join(config_dir, "config.yml"), "r") as f:
    config_yaml = yaml.safe_load(f)

# load .env config (if exists)
env_path = os.path.join(config_dir, ".env")
if os.path.exists(env_path):
    config_env = dotenv.dotenv_values(env_path)
else:
    config_env = {}

# API Keys - Gemini
google_api_key = config_yaml.get("google_api_key")
gemini_model_name = config_yaml.get("gemini_model_name")
gemini_thinkingmodel_name = config_yaml.get("gemini_thinkingmodel_name")

# Cohere
cohere_api_key = config_yaml.get("cohere_api_key")
cohere_model_name = config_yaml.get("cohere_model_name")

# Groq - all model names must be specified in config.yml (no hardcoded defaults)
groq_api_key = config_yaml.get("groq_api_key")
groq_model_name = config_yaml.get("groq_model_name")
groq_llama_model_name = config_yaml.get("groq_llama_model_name")
groq_mixtral_model_name = config_yaml.get("groq_mixtral_model_name")

# LiteLLM Configuration - all model names must be specified in config.yml
litellm_base_url = config_yaml.get("litellm_base_url")
litellm_api_key = config_yaml.get("litellm_api_key")
litellm_fast_llm = config_yaml.get("litellm_fast_llm")
litellm_smart_llm = config_yaml.get("litellm_smart_llm")
litellm_strategic_llm = config_yaml.get("litellm_strategic_llm")
litellm_embedding = config_yaml.get("litellm_embedding")
litellm_default_model = config_yaml.get("litellm_default_model")

# Ollama Configuration - all model names must be specified in config.yml
ollama_base_url = config_yaml.get("ollama_base_url")
ollama_fast_llm = config_yaml.get("ollama_fast_llm")
ollama_smart_llm = config_yaml.get("ollama_smart_llm")
ollama_strategic_llm = config_yaml.get("ollama_strategic_llm")
ollama_embedding = config_yaml.get("ollama_embedding")
ollama_default_model = config_yaml.get("ollama_default_model")

# Retriever Configuration
retriever = config_yaml.get("retriever")
firecrawl_server_url = config_yaml.get("firecrawl_server_url")
tavily_api_key = config_yaml.get("tavily_api_key")
firecrawl_default_provider = config_yaml.get("firecrawl_default_provider")
firecrawl_default_model_name = config_yaml.get("firecrawl_default_model_name")

# NVIDIA NIM Configuration (for Image Studio)
nvidia_api_key = config_yaml.get("nvidia_api_key")
nvidia_base_url = config_yaml.get("nvidia_base_url")
nvidia_image_gen_url = config_yaml.get("nvidia_image_gen_url")
nvidia_vision_model = config_yaml.get("nvidia_vision_model")
nvidia_text_model = config_yaml.get("nvidia_text_model")

# OpenAI Image Generation Configuration - no hardcoded defaults
openai_image_model = config_yaml.get("openai_image_model")
openai_image_enhancement_model = config_yaml.get("openai_image_enhancement_model")

# Whisper Configuration (for audio transcription)
whisper_model_name = config_yaml.get("whisper_model_name")

# Riva TTS Configuration (for text-to-speech)
# Empty string means use service default voice
riva_tts_voice_name = config_yaml.get("riva_tts_voice_name", "")

# Chatterbox TTS Configuration (GPU-accelerated on-device TTS)
chatterbox_tts_model_type = config_yaml.get("chatterbox_tts_model_type")
chatterbox_tts_cfg_weight = config_yaml.get("chatterbox_tts_cfg_weight", 0.5)
chatterbox_tts_exaggeration = config_yaml.get("chatterbox_tts_exaggeration", 0.5)
chatterbox_tts_audio_prompt_path = config_yaml.get("chatterbox_tts_audio_prompt_path")

# Newsletter TTS Voice Configuration
newsletter_progress_voice = config_yaml.get("newsletter_progress_voice")
newsletter_news_voice = config_yaml.get("newsletter_news_voice")

# Newsletter LLM Tier Configuration
# Options: "fast", "smart", "strategic"
newsletter_progress_llm_tier = config_yaml.get("newsletter_progress_llm_tier", "smart")
newsletter_news_llm_tier = config_yaml.get("newsletter_news_llm_tier", "smart")

# Podcast Configuration
podcast_enabled = config_yaml.get("podcast_enabled", False)
podcast_voice_a = config_yaml.get("podcast_voice_a")
podcast_voice_a_provider = config_yaml.get("podcast_voice_a_provider")
podcast_voice_a_model_name = config_yaml.get("podcast_voice_a_model_name")
podcast_voice_b = config_yaml.get("podcast_voice_b")
podcast_voice_b_provider = config_yaml.get("podcast_voice_b_provider")
podcast_voice_b_model_name = config_yaml.get("podcast_voice_b_model_name")
podcast_target_duration_seconds = config_yaml.get("podcast_target_duration_seconds", 600)
podcast_max_turns = config_yaml.get("podcast_max_turns", 20)
podcast_context_window_turns = config_yaml.get("podcast_context_window_turns", 6)
podcast_max_sentences_per_turn = config_yaml.get("podcast_max_sentences_per_turn", 4)
podcast_overlap_ms = config_yaml.get("podcast_overlap_ms", 100)
podcast_normalization_lufs = config_yaml.get("podcast_normalization_lufs", -16.0)
podcast_background_music_path = config_yaml.get("podcast_background_music_path")
podcast_intro_music_path = config_yaml.get("podcast_intro_music_path")
podcast_outro_music_path = config_yaml.get("podcast_outro_music_path")
podcast_ducking_db = config_yaml.get("podcast_ducking_db", -30)

# Azure Configuration
azure_api_key = config_yaml.get("azure_api_key")
azure_api_base = config_yaml.get("azure_api_base")
azure_chatapi_version = config_yaml.get("azure_chatapi_version")
azure_gpt4_deploymentid = config_yaml.get("azure_gpt4_deploymentid")
azure_embeddingapi_version = config_yaml.get("azure_embeddingapi_version")
azure_embedding_deploymentid = config_yaml.get("azure_embedding_deploymentid")

# Supabase Configuration
supabase_service_role_key = config_yaml.get("supabase_service_role_key")
public_supabase_url = config_yaml.get("public_supabase_url")

# CORS Configuration
# Priority: Environment variables > config.yml > defaults
# Environment variables: ALLOWED_ORIGINS, CORS_ALLOW_ORIGIN_REGEX, ENVIRONMENT
cors_config = config_yaml.get("cors", {})
cors_allowed_origins = os.environ.get("ALLOWED_ORIGINS", cors_config.get("allowed_origins", ""))
cors_allow_origin_regex = os.environ.get("CORS_ALLOW_ORIGIN_REGEX", cors_config.get("allow_origin_regex", ""))
cors_environment = os.environ.get("ENVIRONMENT", cors_config.get("environment", "production"))

# Weather and other services
openweather_api_key = config_yaml.get("openweather_api_key")
pyowm_api_key = config_yaml.get("pyowm_api_key")

# Email configuration
yahoo_id = config_yaml.get("yahoo_id")
yahoo_app_password = config_yaml.get("yahoo_app_password")

# Default model selectors for API endpoints
# These are loaded from the 'defaults' section in config.yml
defaults_config = config_yaml.get("defaults", {})
default_analyze_model_name = defaults_config.get("analyze_model_name")
default_chat_model_name = defaults_config.get("chat_model_name")
default_internet_chat_model_name = defaults_config.get("internet_chat_model_name")
default_trip_model_name = defaults_config.get("trip_model_name")
default_cravings_model_name = defaults_config.get("cravings_model_name")
default_memory_model_name = defaults_config.get("memory_model_name")
default_image_provider = defaults_config.get("image_provider")

# Research agent configuration
research_agent_tool_calling_hint = config_yaml.get("research_agent_tool_calling_hint")

# Paths
UPLOAD_FOLDER = config_yaml['paths']['UPLOAD_FOLDER']
WEB_SEARCH_FOLDER = config_yaml['paths']['WEB_SEARCH_FOLDER']
SUMMARY_FOLDER = config_yaml['paths']['SUMMARY_FOLDER']
VECTOR_FOLDER = config_yaml['paths']['VECTOR_FOLDER']
MEMORY_PALACE_FOLDER = config_yaml['paths'].get('MEMORY_PALACE_FOLDER')

# Settings
temperature = config_yaml['settings']['temperature']
max_tokens = config_yaml['settings']['max_tokens']
model_name = config_yaml['settings']['model_name']
num_output = config_yaml['settings']['num_output']
max_chunk_overlap_ratio = config_yaml['settings']['max_chunk_overlap_ratio']
max_input_size = config_yaml['settings']['max_input_size']
context_window = config_yaml['settings']['context_window']
default_chatbot_model = config_yaml['settings'].get('default_chatbot_model')

# Load prompts.yml config
prompts_file_path = os.path.join(config_dir, "prompts.yml")
with open(prompts_file_path, "r") as f:
    prompts_config = yaml.safe_load(f)

# Accessing the templates
sum_template = prompts_config["sum_template"]
eg_template = prompts_config["eg_template"]
ques_template = prompts_config["ques_template"]

system_prompt_content = prompts_config["system_prompt_content"]
system_prompt = [{
    "role": "system",
    "content": system_prompt_content
}]

example_queries = prompts_config['example_queries']
example_memorypalacequeries = prompts_config['example_memorypalacequeries']
example_internetqueries = prompts_config['example_internetqueries']
example_bhagawatgeetaqueries = prompts_config['example_bhagawatgeetaqueries']
keywords = prompts_config['keywords']
