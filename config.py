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
azure_gpt35_deploymentid = config_yaml["azure_gpt35_deploymentid"]
azure_embedding_deploymentid = config_yaml["azure_embedding_deploymentid"]

llama2_api_key = config_yaml["llama2_api_key"]
llama2_api_base = config_yaml["llama2_api_base"]

rvctts_api_base = config_yaml["rvctts_api_base"]

public_supabase_url = config_yaml["public_supabase_url"]
supabase_service_role_key = config_yaml["supabase_service_role_key"]

pinecone_api_key = config_yaml["pinecone_api_key"]
pinecone_environment = config_yaml["pinecone_environment"]

cohere_api_key = config_yaml["cohere_api_key"]
google_api_key = config_yaml["google_api_key"]

bing_api_key = config_yaml["bing_api_key"]
bing_endpoint = config_yaml["bing_endpoint"] + "/v7.0/search"
bing_news_endpoint = config_yaml["bing_endpoint"] + "/v7.0/news/search"

azurespeechkey = config_yaml["azurespeechkey"]
azurespeechregion = config_yaml["azurespeechregion"]
azuretexttranslatorkey = config_yaml["azuretexttranslatorkey"]

openweather_api_key = config_yaml["openweather_api_key"]
