from calendar import c
from numpy import dot
from openai import OpenAI
from openai import AzureOpenAI as OpenAIAzure
import os
import tiktoken

import dotenv
dotenv.load_dotenv()

azure_api_key = os.getenv("AZURE_API_KEY")
azure_api_base = os.getenv("AZURE_API_BASE")
azure_api_type = "azure"
azure_chatapi_version = os.getenv("AZURE_CHATAPI_VERSION")

client = OpenAIAzure(
    api_key=azure_api_key,
    azure_endpoint=azure_api_base,
    api_version=azure_chatapi_version,
)

system_prompt = [{
    "role": "system",
    "content": "You are a helpful and super-intelligent voice assistant, that accurately answers user queries. Be accurate, helpful, concise, and clear."
}]
conversation = system_prompt.copy()

def count_tokens_from_messages(messages, answer, model="gpt-3.5-turbo"):
    
    encoding = tiktoken.encoding_for_model(model)

    if model == "gpt-3.5-turbo-16k":
        tokens_per_message = 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif model == "gpt-3.5-turbo":
        tokens_per_message = 4
        tokens_per_name = -1
    elif model == "gpt-4":
        tokens_per_message = 3
        tokens_per_name = 1
    else:
        raise ValueError(f"Unknown model: {model}")

    # input
    n_input_tokens = 0
    for message in messages:
        n_input_tokens += tokens_per_message
        for key, value in message.items():
            n_input_tokens += len(encoding.encode(value))
            if key == "name":
                n_input_tokens += tokens_per_name

    n_input_tokens += 2

    # output
    n_output_tokens = 1 + len(encoding.encode(answer))

    return n_input_tokens, n_output_tokens

# r_gen = client.chat.completions.create(
#     model="gpt-4",
#     messages=system_prompt,
#     stream=True,
#     temperature=0.7,
#     max_tokens=150,
#     top_p=0.9,
#     frequency_penalty=0.6,
#     presence_penalty=0.1
# )
# answer = ""

# for r_item in r_gen:
#     if r_item.choices:
#         delta = r_item.choices[0].delta
#         if "content" in delta:
#             answer += delta["content"]
#             n_input_tokens, n_output_tokens = count_tokens_from_messages(system_prompt, answer, model="gpt-4")
#             print(f"Input tokens: {n_input_tokens}, output tokens: {n_output_tokens}")
#             print(answer)

while True:

    user_query = input("You: ")
    conversation.append({"role": "user", "content": user_query})
    
    r_gen = client.chat.completions.create(
        model="gpt-4",
        messages=conversation,
        stream=True,
        temperature=0.7,
        max_tokens=150,
        top_p=0.9,
        frequency_penalty=0.6,
        presence_penalty=0.1
    )
    answer = ""

    # # Loop over each item in the streaming response
    # for r_item in r_gen:
    #     # Check if there are choices and a delta with content
    #     if r_item.choices and r_item.choices[0].delta and r_item.choices[0].delta.content:
    #         # Extract the content from the delta
    #         delta_content = r_item.choices[0].delta.content
            
    #         # Print the delta content
    #         print(delta_content, end='', flush=True)
            
    #         # Append the content to the complete response
    #         answer += delta_content

    # # Check if the loop exited because the finish_reason was 'stop'
    # if r_item.choices and r_item.choices[0].finish_reason == 'stop':
    #     print("\n\nThe API indicated that the response is complete.")
    #     print("Bot:", answer)

    for r_item in r_gen:
        if r_item.choices:
            delta = r_item.choices[0].delta
            if delta.content:
                answer += delta.content
                n_input_tokens, n_output_tokens = count_tokens_from_messages(system_prompt, answer, model="gpt-4")
                print(f"Input tokens: {n_input_tokens}, output tokens: {n_output_tokens}")
                print(answer)


    conversation.append({"role": "assistant", "content": answer})
