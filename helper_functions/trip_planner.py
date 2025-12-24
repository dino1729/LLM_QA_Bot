"""
Trip Planner
Uses unified LLM client via chat_generation
"""
from config import config
from helper_functions.chat_generation import generate_chat


def generate_trip_plan(city, days, model_name=None):
    """
    Generate a trip plan for a city

    Args:
        city: City name
        days: Number of days
        model_name: Model to use (default: from config.default_trip_model_name)

    Returns:
        Trip plan string
    """
    # Use config default if no model specified
    if model_name is None:
        model_name = config.default_trip_model_name
    # Check if days input is a number
    try:
        days = int(days)

        tripsystem_prompt = [{
            "role": "system",
            "content": "You are a world class trip planner who is knowledgeable about all the tourist attractions in the world. You will serve the user by planning a trip for them and respecting all their preferences."
        }]
        conversation = tripsystem_prompt.copy()

        user_message = f"Craft a thorough and detailed travel itinerary for {city}. This itinerary should encompass the city's most frequented tourist attractions, as well as its top-rated restaurants, all of which should be visitable within a timeframe of {days} days including the best budget-friendly hotels/resorts to stay for {days-1} nights. The itinerary should be strategically organized to take into account the distance between each location and the time required to travel there, maximizing efficiency. Moreover, please include specific time windows for each location, arranged in ascending order, to facilitate effective planning. The final output should be a numbered list, where each item corresponds to a specific location. Accompany each location with a brief yet informative description to provide context and insight."

        conversation.append({"role": "user", "content": str(user_message)})

        # Generate response using selected model
        response = generate_chat(
            model_name=model_name,
            conversation=conversation,
            temperature=0.3,
            max_tokens=2048
        )

        conversation.append({"role": "assistant", "content": str(response)})

        return f"Here is your trip plan for {city} for {days} day(s): {response}"

    except ValueError:
        return "Please enter a number for days."
    except Exception as e:
        return f"Error generating trip plan: {str(e)}"
