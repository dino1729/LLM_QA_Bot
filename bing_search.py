import argparse
from config import config
from helper_functions.chat_generation_with_internet import internet_connected_chatbot

# Get user query from args
parser = argparse.ArgumentParser(description="Bing+ChatGPT tool.")
parser.add_argument("query", help="User query")
args = parser.parse_args()
userquery = args.query

system_prompt = config.system_prompt
temperature = config.temperature
max_tokens = config.max_tokens
model_name = config.model_name

assistant_reply = internet_connected_chatbot(userquery, [], model_name, max_tokens, temperature)
print(assistant_reply)
