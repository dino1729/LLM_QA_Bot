import supabase
from config import config
from openai import AzureOpenAI as OpenAIAzure
import argparse

# Initialize OpenAI and Supabase clients
azure_api_key = config.azure_api_key
azure_api_base = config.azure_api_base

azure_chatapi_version = config.azure_chatapi_version
azure_gpt4_deploymentid = config.azure_gpt4_deploymentid

azure_embedding_version = config.azure_embeddingapi_version
azure_embedding_deploymentid = config.azure_embedding_deploymentid

supabase_service_role_key = config.supabase_service_role_key
public_supabase_url = config.public_supabase_url

supabase_service_role_key = config.supabase_service_role_key
public_supabase_url = config.public_supabase_url

# List of topics
topics = ["How can I be more productive?", "How to improve my communication skills?", "How to be a better leader?", "How are electric vehicles less harmful to the environment?", "How can I think clearly in adverse scenarios?", "What are the tenets of effective office politics?", "How to be more creative?", "How to improve my problem-solving skills?", "How to be more confident?", "How to be more empathetic?", "What can I learn from Boyd, the fighter pilot who changed the art of war?", "How can I seek the mentorship I want from key influential people", "How can I communicate more effectively?", "Give me suggestions to reduce using filler words when communicating highly technical topics?"]

def generate_embeddings(text, model=azure_embedding_deploymentid):
    client = OpenAIAzure(
        api_key=azure_api_key,
        azure_endpoint=azure_api_base,
        api_version=azure_embedding_version,
    )    
    return client.embeddings.create(input = [text], model=model).data[0].embedding

def generate_gpt_response_memorypalace(user_message):
    client = OpenAIAzure(
        api_key=azure_api_key,
        azure_endpoint=azure_api_base,
        api_version=azure_chatapi_version,
    )
    syspromptmessage = f"""
    You are EDITH, or "Even Dead, I'm The Hero," a world-class AI assistant that is designed by Tony Stark to be a powerful tool for whoever controls it. You help Dinesh in various tasks. In this scenario, you are helping Dinesh recall important concepts he learned and put them in a memory palace aka, his second brain. You will be given a topic along with the semantic search results from the memory palace. You need to generate a summary or lesson learned based on the search results. You have to praise Dinesh for his efforts and encourage him to continue learning. You can also provide additional information or tips to help him understand the topic better. You are not a replacement for human intelligence, but a tool to enhance Dinesh's intelligence. You are here to help Dinesh succeed in his learning journey. You are a positive and encouraging presence in his life. You are here to support him in his quest for knowledge and growth. You are EDITH, and you are here to help Dinesh succeed. Dinesh wants to master the best of what other people have already figured out.
    
    Additionally, for each topic, provide one historical anecdote that can go back up to 10,000 years ago when human civilization started. The lesson can include a key event, discovery, mistake, and teaching from various cultures and civilizations throughout history. This will help Dinesh gain a deeper understanding of the topic by learning from the past since if one does not know history, one thinks short term; if one knows history, one thinks medium and long term..
    
    Here's a bit more about Dinesh:
    You should be a centrist politically. I reside in Hillsboro, Oregon, and I hold the position of Senior Analog Circuit Design Engineer with eight years of work experience. I am a big believer in developing Power Delivery IPs with clean interfaces and minimal maintenance. I like to work on Raspberry Pi projects and home automation in my free time. Recently, I have taken up the exciting hobby of creating LLM applications. Currently, I am engaged in the development of a fantasy premier league recommender bot that selects the most suitable players based on statistical data for a specific fixture, all while adhering to a budget. Another project that I have set my sights on is a generativeAI-based self-driving system that utilizes text prompts as sensor inputs to generate motor drive outputs, enabling the bot to control itself. The key aspect of this design lies in achieving a latency of 1000 tokens per second for the LLM token generation, which can be accomplished using a local GPU cluster. I am particularly interested in the field of physics, particularly relativity, quantum mechanics, game theory and the simulation hypothesis. I have a genuine curiosity about the interconnectedness of things and the courage to explore and advocate for interventions, even if they may not be immediately popular or obvious. My ultimate goal is to achieve success in all aspects of life and incorporate the “systems thinking” and “critical thinking” mindset into my daily routine. I aim to apply systems thinking to various situations, both professional and personal, to gain insights into different perspectives and better understand complex problems. Currently, I am captivated by the achievements of individuals like Chanakya, Nicholas Tesla, Douglas Englebart, JCR Licklider, and Vannevar Bush, and I aspire to emulate their success. I’m also super interested in learning more about game theory and how people behave in professional settings. I’m curious about the strategies that can be used to influence others and potentially advance quickly in the workplace. So, coach me on how to deliver my presentations, communicate clearly and concisely, and how to conduct myself in front of influential people. My ultimate goal is to lead a large organization where I can create innovative technology that can benefit billions of people and improve their lives.
    """
    system_prompt = [{
        "role": "system",
        "content": syspromptmessage
    }]
    conversation = system_prompt.copy()
    conversation.append({"role": "user", "content": str(user_message)})
    response = client.chat.completions.create(
        model=azure_gpt4_deploymentid,
        messages=conversation,
        max_tokens=1024,
        temperature=0.4,
    )
    message = response.choices[0].message.content
    conversation.append({"role": "assistant", "content": str(message)})

    return message

def query_memorypalace(userquery):

    # Step 2: Generate embeddings for the topic
    userquery_embedding = generate_embeddings(userquery)
    # Debugging
    # print(f"Topic: {userquery}")

    supabase_client = supabase.Client(public_supabase_url, supabase_service_role_key)
    # Step 3: Perform a vector search in the Supabase database
    response = supabase_client.rpc('mp_search', {
        'query_embedding': userquery_embedding,
        'similarity_threshold': 0.4,  # Adjust this threshold as needed
        'match_count': 6
    }).execute()
    
    # Debugging
    # print(f"Number of matches: {len(response.data)}")

    # Extract the content from the top 5 matches
    top_matches = response.data
    contents = [match['content'] for match in top_matches]

    # Step 4: Generate a response using OpenAI
    prompt = f"Here's user question: {userquery}\n\nBased on the following lessons, generate a summary or lesson learned for the topic:\n\n" + "\n\n".join(contents)
    lesson_learned = generate_gpt_response_memorypalace(prompt)

    return lesson_learned

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a response for a given user query")
    parser.add_argument("userquery", help="User query")
    args = parser.parse_args()

    memorypalace_response = query_memorypalace(args.userquery)
    if memorypalace_response:
        print(memorypalace_response)
