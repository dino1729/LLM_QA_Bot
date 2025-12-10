import random
from helper_functions.chat_generation_with_internet import internet_connected_chatbot
from config import config

if __name__ == '__main__':

    system_prompt = [{
        "role": "system",
        "content": "You are a helpful and super-intelligent voice assistant, that accurately answers user queries. Be accurate, helpful, concise, and clear."
    }]
    conversation = system_prompt.copy()
    temperature = 0.5
    max_tokens = 4840

    while True:
        user_query = input("Enter your query: ")

        # model_name = random.choice(["GROQ", "GEMINI"])
        model_name = config.default_chatbot_model
        print("Model: ", model_name)

        assistant_reply = internet_connected_chatbot(user_query, conversation, model_name, max_tokens, temperature)
        print("Bot: ", assistant_reply)

        conversation.append(({"role": "assistant", "content": assistant_reply}))
