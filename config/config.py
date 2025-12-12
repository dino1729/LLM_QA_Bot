import yaml
import dotenv
import os

config_dir = os.path.join(".", "config")

# load yaml config
with open(os.path.join(config_dir, "config.yml"), "r") as f:
    config_yaml = yaml.safe_load(f)

# load .env config (if exists)
env_path = os.path.join(config_dir, ".env")
if os.path.exists(env_path):
    config_env = dotenv.dotenv_values(env_path)
else:
    config_env = {}

# API Keys
google_api_key = config_yaml.get("google_api_key", "")
gemini_model_name = config_yaml.get("gemini_model_name", "")
gemini_thinkingmodel_name = config_yaml.get("gemini_thinkingmodel_name", "")
groq_api_key = config_yaml.get("groq_api_key", "")
groq_model_name = config_yaml.get("groq_model_name", "deepseek-r1-distill-llama-70b")
groq_llama_model_name = config_yaml.get("groq_llama_model_name", "llama3-70b-8192")
groq_mixtral_model_name = config_yaml.get("groq_mixtral_model_name", "mixtral-8x7b-32768")

# LiteLLM Configuration
litellm_base_url = config_yaml.get("litellm_base_url", "")
litellm_api_key = config_yaml.get("litellm_api_key", "")
litellm_fast_llm = config_yaml.get("litellm_fast_llm", "")
litellm_smart_llm = config_yaml.get("litellm_smart_llm", "")
litellm_strategic_llm = config_yaml.get("litellm_strategic_llm", "")
litellm_embedding = config_yaml.get("litellm_embedding", "")
litellm_default_model = config_yaml.get("litellm_default_model", "gpt-4")

# Ollama Configuration
ollama_base_url = config_yaml.get("ollama_base_url", "")
ollama_fast_llm = config_yaml.get("ollama_fast_llm", "")
ollama_smart_llm = config_yaml.get("ollama_smart_llm", "")
ollama_strategic_llm = config_yaml.get("ollama_strategic_llm", "")
ollama_embedding = config_yaml.get("ollama_embedding", "")
ollama_default_model = config_yaml.get("ollama_default_model", "llama2")

# Retriever Configuration
retriever = config_yaml.get("retriever", "")
firecrawl_server_url = config_yaml.get("firecrawl_server_url", "http://localhost:3002")
tavily_api_key = config_yaml.get("tavily_api_key", "")

# NVIDIA NIM Configuration (for Image Studio)
nvidia_api_key = config_yaml.get("nvidia_api_key", "")
nvidia_base_url = config_yaml.get("nvidia_base_url", "")
nvidia_image_gen_url = config_yaml.get("nvidia_image_gen_url", "")
nvidia_vision_model = config_yaml.get("nvidia_vision_model", "")
nvidia_text_model = config_yaml.get("nvidia_text_model", "")

# OpenAI Image Generation Configuration
openai_image_model = config_yaml.get("openai_image_model", "gpt-image-1")
openai_image_enhancement_model = config_yaml.get("openai_image_enhancement_model", "gpt-4o")

# CORS Configuration
# Priority: Environment variables > config.yml > defaults
# Environment variables: ALLOWED_ORIGINS, CORS_ALLOW_ORIGIN_REGEX, ENVIRONMENT
cors_config = config_yaml.get("cors", {})
cors_allowed_origins = os.environ.get("ALLOWED_ORIGINS", cors_config.get("allowed_origins", ""))
cors_allow_origin_regex = os.environ.get("CORS_ALLOW_ORIGIN_REGEX", cors_config.get("allow_origin_regex", ""))
cors_environment = os.environ.get("ENVIRONMENT", cors_config.get("environment", "production"))

# Weather and other services
openweather_api_key = config_yaml.get("openweather_api_key", "")
pyowm_api_key = config_yaml.get("pyowm_api_key", "")

# Email configuration
yahoo_id = config_yaml.get("yahoo_id", "")
yahoo_app_password = config_yaml.get("yahoo_app_password", "")

UPLOAD_FOLDER = config_yaml['paths']['UPLOAD_FOLDER']
WEB_SEARCH_FOLDER = config_yaml['paths']['WEB_SEARCH_FOLDER']
SUMMARY_FOLDER = config_yaml['paths']['SUMMARY_FOLDER']
VECTOR_FOLDER = config_yaml['paths']['VECTOR_FOLDER']

temperature = config_yaml['settings']['temperature']
max_tokens = config_yaml['settings']['max_tokens']
model_name = config_yaml['settings']['model_name']
num_output = config_yaml['settings']['num_output']
max_chunk_overlap_ratio = config_yaml['settings']['max_chunk_overlap_ratio']
max_input_size = config_yaml['settings']['max_input_size']
context_window = config_yaml['settings']['context_window']
default_chatbot_model = config_yaml['settings'].get('default_chatbot_model', 'GEMINI')

# Assuming config_dir is already defined as shown in your existing config.py
prompts_file_path = os.path.join(config_dir, "prompts.yml")

# Load prompts.yml config
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
