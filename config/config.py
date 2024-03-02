import yaml
import dotenv
import os

config_dir = os.path.join(".", "config")

# load yaml config
with open(os.path.join(config_dir, "config.yml"), "r") as f:
    config_yaml = yaml.safe_load(f)

# load .env config
config_env = dotenv.dotenv_values(os.path.join(config_dir, ".env"))

# config parameters
azure_api_base = config_yaml["azure_api_base"]
azure_api_key = config_yaml["azure_api_key"]
azure_chatapi_version = config_yaml["azure_chatapi_version"]
azure_embeddingapi_version = config_yaml["azure_embeddingapi_version"]

azure_gpt4_deploymentid = config_yaml["azure_gpt4_deploymentid"]
openai_gpt4_modelname = config_yaml["openai_gpt4_modelname"]
azure_gpt35_deploymentid = config_yaml["azure_gpt35_deploymentid"]
openai_gpt35_modelname = config_yaml["openai_gpt35_modelname"]
azure_embedding_deploymentid = config_yaml["azure_embedding_deploymentid"]
openai_embedding_modelname = config_yaml["openai_embedding_modelname"]

llama2_api_key = config_yaml["llama2_api_key"]
llama2_api_base = config_yaml["llama2_api_base"]

rvctts_api_base = config_yaml["rvctts_api_base"]

public_supabase_url = config_yaml["public_supabase_url"]
supabase_service_role_key = config_yaml["supabase_service_role_key"]

pinecone_api_key = config_yaml["pinecone_api_key"]
pinecone_environment = config_yaml["pinecone_environment"]

cohere_api_key = config_yaml["cohere_api_key"]
google_api_key = config_yaml["google_api_key"]
gemini_model_name = config_yaml["gemini_model_name"]
groq_api_key = config_yaml["groq_api_key"]

bing_api_key = config_yaml["bing_api_key"]
bing_endpoint = config_yaml["bing_endpoint"] + "/v7.0/search"
bing_news_endpoint = config_yaml["bing_endpoint"] + "/v7.0/news/search"

azurespeechkey = config_yaml["azurespeechkey"]
azurespeechregion = config_yaml["azurespeechregion"]
azuretexttranslatorkey = config_yaml["azuretexttranslatorkey"]

openweather_api_key = config_yaml["openweather_api_key"]

UPLOAD_FOLDER = config_yaml['paths']['UPLOAD_FOLDER']
BING_FOLDER = config_yaml['paths']['BING_FOLDER']
SUMMARY_FOLDER = config_yaml['paths']['SUMMARY_FOLDER']
VECTOR_FOLDER = config_yaml['paths']['VECTOR_FOLDER']

temperature = config_yaml['settings']['temperature']
max_tokens = config_yaml['settings']['max_tokens']
model_name = config_yaml['settings']['model_name']
num_output = config_yaml['settings']['num_output']
max_chunk_overlap_ratio = config_yaml['settings']['max_chunk_overlap_ratio']
max_input_size = config_yaml['settings']['max_input_size']
context_window = config_yaml['settings']['context_window']

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
keywords = prompts_config['keywords']
