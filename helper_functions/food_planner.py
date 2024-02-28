from openai import AzureOpenAI as OpenAIAzure
from config import config

azure_api_key = config.azure_api_key
azure_api_base = config.azure_api_base
azure_chatapi_version = config.azure_chatapi_version
azure_chatapi_version = config.azure_chatapi_version
azure_gpt35_deploymentid = config.azure_gpt35_deploymentid

def craving_satisfier(city, food_craving):

    client = OpenAIAzure(
        api_key=azure_api_key,
        azure_endpoint=azure_api_base,
        api_version=azure_chatapi_version,
    )
    # If the food craving is input as "idk", generate a random food craving
    if food_craving in ["idk","I don't know","I don't know what I want","I don't know what I want to eat","I don't know what I want to eat.","Idk"]:
        # Generate a random food craving
        foodsystem_prompt = [{
            "role": "system",
            "content": "You are a world class food recommender who is knowledgeable about all the food items in the world. You must respond in one-word answer."
        }]
        conversation1 = foodsystem_prompt.copy()
        user_message1 = f"I don't know what to eat and I want you to generate a random cuisine. Be as creative as possible"
        conversation1.append({"role": "user", "content": str(user_message1)})

        response1 = client.chat.completions.create(
            model=azure_gpt35_deploymentid,
            messages=conversation1,
            max_tokens=32,
            temperature=0.5,
        )
        food_craving = response1.choices[0].message.content
        conversation1.append({"role": "assistant", "content": str(food_craving)})
        print(f"Don't worry, yo! I think you are craving for {food_craving}!")
    else:
        print(f"That's a great choice! My mouth is watering just thinking about {food_craving}!")

    restaurantsystem_prompt = [{
        "role": "system",
        "content": "You are a world class restaurant recommender who is knowledgeable about all the restaurants in the world. You will serve the user by recommending restaurants."
    }]
    conversation2 = restaurantsystem_prompt.copy()
    user_message2 = f"I'm looking for 8 restaurants in {city} that serves {food_craving}. Provide me with a list of eight restaurants, including their brief addresses. Also, mention one dish from each that particularly stands out, ensuring it contains neither beef nor pork."
    conversation2.append({"role": "user", "content": str(user_message2)})
    response2 = client.chat.completions.create(
        model=azure_gpt35_deploymentid,
        messages=conversation2,
        max_tokens=2048,
        temperature=0.4,
    )
    message = response2.choices[0].message.content
    conversation2.append({"role": "assistant", "content": str(message)})

    return f'Here are 8 restaurants in {city} that serve {food_craving}! Bon Appetit!! \n {message}'
