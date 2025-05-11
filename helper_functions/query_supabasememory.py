import supabase
from config import config
from openai import AzureOpenAI as OpenAIAzure
import argparse
from typing import AsyncIterator, List, Tuple, Union, Dict, Optional
import asyncio
import random

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

syspromptmessage = f"""
    You are EDITH, or "Even Dead, I'm The Hero," a world-class AI assistant that is designed by Tony Stark to be a powerful tool for whoever controls it. You help Dinesh in various tasks. In this scenario, you are helping Dinesh recall important concepts he learned and put them in a memory palace aka, his second brain. You will be given a topic along with the semantic search results from the memory palace. You need to generate a summary or lesson learned based on the search results. You have to praise Dinesh for his efforts and encourage him to continue learning. You can also provide additional information or tips to help him understand the topic better. You are not a replacement for human intelligence, but a tool to enhance Dinesh's intelligence. You are here to help Dinesh succeed in his learning journey. You are a positive and encouraging presence in his life. You are here to support him in his quest for knowledge and growth. You are EDITH, and you are here to help Dinesh succeed. Dinesh wants to master the best of what other people have already figured out.
    
    Additionally, for each topic, provide one historical anecdote that can go back up to 10,000 years ago when human civilization started. The lesson can include a key event, discovery, mistake, and teaching from various cultures and civilizations throughout history. This will help Dinesh gain a deeper understanding of the topic by learning from the past since if one does not know history, one thinks short term; if one knows history, one thinks medium and long term..
    
    Here's a bit more about Dinesh:
    You should be a centrist politically. I reside in Hillsboro, Oregon, and I hold the position of Senior Analog Circuit Design Engineer with eight years of work experience. I am a big believer in developing Power Delivery IPs with clean interfaces and minimal maintenance. I like to work on Raspberry Pi projects and home automation in my free time. Recently, I have taken up the exciting hobby of creating LLM applications. Currently, I am engaged in the development of a fantasy premier league recommender bot that selects the most suitable players based on statistical data for a specific fixture, all while adhering to a budget. Another project that I have set my sights on is a generativeAI-based self-driving system that utilizes text prompts as sensor inputs to generate motor drive outputs, enabling the bot to control itself. The key aspect of this design lies in achieving a latency of 1000 tokens per second for the LLM token generation, which can be accomplished using a local GPU cluster. I am particularly interested in the field of physics, particularly relativity, quantum mechanics, game theory and the simulation hypothesis. I have a genuine curiosity about the interconnectedness of things and the courage to explore and advocate for interventions, even if they may not be immediately popular or obvious. My ultimate goal is to achieve success in all aspects of life and incorporate the “systems thinking” and “critical thinking” mindset into my daily routine. I aim to apply systems thinking to various situations, both professional and personal, to gain insights into different perspectives and better understand complex problems. Currently, I am captivated by the achievements of individuals like Chanakya, Nicholas Tesla, Douglas Englebart, JCR Licklider, and Vannevar Bush, and I aspire to emulate their success. I’m also super interested in learning more about game theory and how people behave in professional settings. I’m curious about the strategies that can be used to influence others and potentially advance quickly in the workplace. So, coach me on how to deliver my presentations, communicate clearly and concisely, and how to conduct myself in front of influential people. My ultimate goal is to lead a large organization where I can create innovative technology that can benefit billions of people and improve their lives.

    ---
    
    INSTRUCTIONS FOR OUTPUT (always follow this structure):
    1. Carefully read and synthesize the most important, actionable, and unique insights from the provided knowledge chunks.
    2. Weigh the lessons by their similarity index, prioritizing those with higher values.
    3. Present your output as a concise list of key lessons learned, suitable for quick review.
    4. After the list, provide a separate section titled 'Historical Anecdote' with a relevant story, event, or teaching from any culture or civilization (up to 10,000 years ago) that illustrates or deepens understanding of the topic.
    5. End with a brief, positive encouragement for Dinesh to continue learning.
    6. Do not repeat the input verbatim; always synthesize and rephrase for clarity and impact.
    ---
"""

def generate_embeddings(text, model=azure_embedding_deploymentid):
    client = OpenAIAzure(
        api_key=azure_api_key,
        azure_endpoint=azure_api_base,
        api_version=azure_embedding_version,
    )    
    return client.embeddings.create(input = [text], model=model).data[0].embedding

def enhance_query_with_llm(userquery: str) -> str:
    """Use Azure OpenAI to expand and clarify the user query for better semantic search."""
    client = OpenAIAzure(
        api_key=azure_api_key,
        azure_endpoint=azure_api_base,
        api_version=azure_chatapi_version,
    )
    system_message = (
        "You are an expert at rephrasing and expanding user queries to maximize semantic search recall. "
        "Given a short or ambiguous user query, rewrite it as a detailed, explicit, and verbose description, "
        "including synonyms, related concepts, and clarifying context, but do not answer the question. "
        "Output only the improved query, nothing else."
    )
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": userquery}
    ]
    response = client.chat.completions.create(
        model=azure_gpt4_deploymentid,
        messages=messages,
        max_tokens=128,
        temperature=0.3
    )
    improved_query = response.choices[0].message.content.strip()
    return improved_query

# Add a debug flag to the function signature and use it to control debug printing
async def query_memorypalace_stream(
    userquery: str, 
    chat_history: Optional[List[Union[Tuple[str, str], Dict[str, str]]]] = None,
    debug: bool = False
) -> AsyncIterator[str]:

    try:
        # Enhance user query for better semantic search
        improved_userquery = enhance_query_with_llm(userquery)
        if debug:
            print(f"\n--- DEBUG: Improved user query for embedding ---\n{improved_userquery}\n--- END DEBUG ---\n")
        # Generate embeddings using improved query
        userquery_embedding = generate_embeddings(improved_userquery)

        # Initialize Supabase client
        supabase_client = supabase.Client(public_supabase_url, supabase_service_role_key)

        # Recursive/hierarchical similarity search
        similarity_thresholds = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
        max_matches = 10
        seen_ids = set()
        results = []
        for threshold in similarity_thresholds:
            if len(results) >= max_matches:
                break
            remaining = max_matches - len(results)
            response = supabase_client.rpc(
                'mp_search',
                {
                    'query_embedding': userquery_embedding,
                    'similarity_threshold': threshold,
                    'match_count': remaining
                }
            ).execute()
            # Add new results only
            for match in response.data:
                match_id = match.get('id') or match.get('pk') or id(match)
                if match_id not in seen_ids:
                    results.append(match)
                    seen_ids.add(match_id)
            # If enough results, stop
            if len(results) >= max_matches:
                break

        # Extract content and similarity
        contents = [
            {
                'content': match['content'],
                'similarity': match.get('similarity', 'N/A'),
                'metadata': {k: v for k, v in match.items() if k not in ['content', 'similarity']}
            }
            for match in results
        ]

        # Process chat history
        history_text = ""
        if chat_history:
            try:
                if chat_history and isinstance(chat_history[0], dict):
                    history_text = "\n\n".join(
                        f"User: {entry['user']}\nAssistant: {entry['assistant']}"
                        for entry in chat_history
                    )
                elif chat_history:
                    history_text = "\n\n".join(
                        f"User: {user}\nAssistant: {assistant}"
                        for user, assistant in chat_history
                    )
            except (IndexError, KeyError, AttributeError) as e:
                history_text = ""

        # Format retrieved chunks for LLM with similarity index
        formatted_chunks = "\n\n".join(
            f"[Similarity: {c['similarity']}]\n{c['content']}" for c in contents
        )

        # Construct prompt
        prompt = (
            f"Here's the chat history:\n\n{history_text}\n\n"
            f"Here's user question: {userquery}\n\n"
            f"Based on the following lessons (each with a similarity index), generate a summary or lesson learned for the topic:\n\n"
            f"{formatted_chunks}"
        )

        # Debug print: show what is being sent to the LLM only if debug is True
        if debug:
            print("\n--- DEBUG: Prompt sent to LLM ---\n")
            print(prompt)
            print("\n--- END DEBUG ---\n")

        # Initialize Azure OpenAI client
        client = OpenAIAzure(
            api_key=azure_api_key,
            azure_endpoint=azure_api_base,
            api_version=azure_chatapi_version,
        )
        # Set up the messages
        messages = [
            {
                "role": "system",
                "content": syspromptmessage
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        # Create the completion with streaming
        gpt_response = client.chat.completions.create(
            model=azure_gpt4_deploymentid,
            messages=messages,
            max_tokens=1024,
            temperature=0.4,
            stream=True
        )
        accumulated_response = ""
        for chunk in gpt_response:
            if chunk.choices:
                delta = chunk.choices[0].delta
                if delta.content:
                    accumulated_response += delta.content
                    yield accumulated_response
    except Exception as e:
        print(f"Encountered an error: {e}")
        yield f"I apologize, but I encountered an error while processing your request. Please try again. Error: {str(e)}"

def main():
    parser = argparse.ArgumentParser(description="Test query_memorypalace_stream standalone.")
    parser.add_argument('--query', type=str, default=None, help="User query to test.")
    parser.add_argument('--debug', action='store_true', help="Show debug info for LLM prompt.")
    args = parser.parse_args()

    # If no query is provided, pick a random topic
    if args.query is None:
        args.query = random.choice(topics)
        print(f"No query provided. Using random topic: {args.query}\n")

    async def run_query():
        print(f"Querying memory palace for: {args.query}\n")
        accumulated_output = None
        async for chunk in query_memorypalace_stream(args.query, debug=args.debug):
            accumulated_output = chunk
        if accumulated_output is not None:
            print(accumulated_output)
        else:
            print("No output received from LLM.")

    asyncio.run(run_query())

if __name__ == "__main__":
    main()
