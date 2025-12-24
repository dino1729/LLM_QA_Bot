"""
Food/Craving Satisfier
Uses unified LLM client via chat_generation
"""
from config import config
from helper_functions.chat_generation import generate_chat


def craving_satisfier(city, food_craving, model_name=None):
    """
    Find restaurants based on food cravings

    Args:
        city: City name
        food_craving: Type of food or "idk" for random
        model_name: Model to use (default: from config.default_cravings_model_name)

    Returns:
        Restaurant recommendations
    """
    # Use config default if no model specified
    if model_name is None:
        model_name = config.default_cravings_model_name
    try:
        # If the food craving is "idk", generate a random food craving
        if food_craving.lower() in ["idk", "i don't know", "i don't know what i want", "i don't know what i want to eat", "i don't know what i want to eat."]:
            # Generate a random food craving
            foodsystem_prompt = [{
                "role": "system",
                "content": "You are a world class food recommender who is knowledgeable about all the food items in the world. You must respond in one-word answer."
            }]
            conversation1 = foodsystem_prompt.copy()
            user_message1 = "I don't know what to eat and I want you to generate a random cuisine. Be as creative as possible"
            conversation1.append({"role": "user", "content": str(user_message1)})

            food_craving = generate_chat(
                model_name=model_name,
                conversation=conversation1,
                temperature=0.5,
                max_tokens=32
            )

            conversation1.append({"role": "assistant", "content": str(food_craving)})
            print(f"Don't worry, yo! I think you are craving for {food_craving}!")
        else:
            print(f"That's a great choice! My mouth is watering just thinking about {food_craving}!")

        # Find restaurants
        restaurantsystem_prompt = [{
            "role": "system",
            "content": "You are a world class restaurant recommender who is knowledgeable about all the restaurants in the world. You will serve the user by recommending restaurants."
        }]
        conversation2 = restaurantsystem_prompt.copy()
        user_message2 = f"I'm looking for 8 restaurants in {city} that serves {food_craving}. Provide me with a list of eight restaurants, including their brief addresses. Also, mention one dish from each that particularly stands out, ensuring it contains neither beef nor pork."
        conversation2.append({"role": "user", "content": str(user_message2)})

        response = generate_chat(
            model_name=model_name,
            conversation=conversation2,
            temperature=0.4,
            max_tokens=2048
        )

        conversation2.append({"role": "assistant", "content": str(response)})

        return f'Here are 8 restaurants in {city} that serve {food_craving}! Bon Appetit!! \n{response}'

    except Exception as e:
        return f"Error generating restaurant recommendations: {str(e)}"
