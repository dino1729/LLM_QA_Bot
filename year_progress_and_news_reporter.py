import smtplib
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from openai import AzureOpenAI as OpenAIAzure
from pyowm import OWM
from config import config
from helper_functions.chat_generation_with_internet import internet_connected_chatbot
from helper_functions.audio_processors import text_to_speech_nospeak
from datetime import datetime
import random
import supabase
import os

azure_api_key = config.azure_api_key
azure_api_base = config.azure_api_base

azure_chatapi_version = config.azure_chatapi_version
azure_gpt4_deploymentid = config.azure_gpt4_deploymentid

azure_embedding_version = config.azure_embeddingapi_version
azure_embedding_deploymentid = config.azure_embedding_deploymentid

supabase_service_role_key = config.supabase_service_role_key
public_supabase_url = config.public_supabase_url

# List of topics
# topics = [
#     "How can I be more productive?", "How to improve my communication skills?", "How to be a better leader?",
#     "How are electric vehicles less harmful to the environment?", "How can I think clearly in adverse scenarios?",
#     "What are the tenets of effective office politics?", "How to be more creative?", "How to improve my problem-solving skills?",
#     "How to be more confident?", "How to be more empathetic?", "What can I learn from Boyd, the fighter pilot who changed the art of war?",
#     "How can I seek the mentorship I want from key influential people", "How can I communicate more effectively?",
#     "Give me suggestions to reduce using filler words when communicating highly technical topics?",
#     "How to apply the best game theory concepts in getting ahead in office poilitics?", "What are some best ways to play office politics?",
#     "How to be more persuasive, assertive, influential, impactful, engaging, inspiring, motivating, captivating and convincing in my communication?",
#     "What are the top 8 ways the tit-for-tat strategy prevails in the repeated prisoner's dilemma, and how can these be applied to succeed in life and office politics?"
# ]

topics = [  
    "How can I be more productive?", "How to improve my communication skills?", "How to be a better leader?",  
    "How are electric vehicles less harmful to the environment?", "How can I think clearly in adverse scenarios?",  
    "What are the tenets of effective office politics?", "How to be more creative?", "How to improve my problem-solving skills?",  
    "How to be more confident?", "How to be more empathetic?", "What can I learn from Boyd, the fighter pilot who changed the art of war?",  
    "How can I seek the mentorship I want from key influential people", "How can I communicate more effectively?",  
    "Give me suggestions to reduce using filler words when communicating highly technical topics?",  
    "How to apply the best game theory concepts in getting ahead in office poilitics?", "What are some best ways to play office politics?",  
    "How to be more persuasive, assertive, influential, impactful, engaging, inspiring, motivating, captivating and convincing in my communication?",  
    "What are the top 8 ways the tit-for-tat strategy prevails in the repeated prisoner's dilemma, and how can these be applied to succeed in life and office politics?",  
    "What are Chris Voss's key strategies from *Never Split the Difference* for hostage negotiations, and how can they apply to workplace conflicts?",  
    "How can tactical empathy (e.g., labeling emotions, mirroring) improve outcomes in high-stakes negotiations?",  
    "What is the ‘Accusations Audit’ technique, and how does it disarm resistance in adversarial conversations?",  
    "How do calibrated questions (e.g., *How am I supposed to do that?*) shift power dynamics in negotiations?",  
    "When should you use the ‘Late-Night FM DJ Voice’ to de-escalate tension during disagreements?",  
    "How can anchoring bias be leveraged to set favorable terms in salary or deal negotiations?",  
    "What are ‘Black Swan’ tactics for uncovering hidden information in negotiations?",  
    "How can active listening techniques improve conflict resolution in team settings?",  
    "What non-verbal cues (e.g., tone, body language) most impact persuasive communication?",  
    "How can I adapt my communication style to different personality types (e.g., assertive vs. analytical)?",  
    "What storytelling frameworks make complex ideas more compelling during presentations?",  
    "How do you balance assertiveness and empathy when delivering critical feedback?",  
    "What are strategies for managing difficult conversations (e.g., layoffs, project failures) with grace?",  
    "How can Nash Equilibrium concepts guide decision-making in workplace collaborations?",  
    "What real-world scenarios mimic the ‘Chicken Game,’ and how should you strategize in them?",  
    "How do Schelling Points (focal points) help teams reach consensus without direct communication?",  
    "When is tit-for-tat with forgiveness more effective than strict reciprocity in office politics?",  
    "How does backward induction in game theory apply to long-term career or project planning?",  
    "What are examples of zero-sum vs. positive-sum games in corporate negotiations?",  
    "How can Bayesian reasoning improve decision-making under uncertainty (e.g., mergers, market entry)?",  
    "How can Boyd’s OODA Loop (Observe, Orient, Decide, Act) improve decision-making under pressure?",  
    "What game theory principles optimize resource allocation in cross-functional teams?",  
    "How can the ‘MAD’ (Mutually Assured Destruction) concept deter adversarial behavior in workplaces?", 
    "How does Conway’s Law (‘organizations design systems that mirror their communication structures’) impact the efficiency of IP or product design?",  
    "What strategies can mitigate the negative effects of Conway’s Law on modularity in IP design (e.g., reusable components)?",  
    "How can teams align their structure with IP design goals to leverage Conway’s Law for better outcomes?",  
    "What are real-world examples of Conway’s Law leading to inefficient or efficient IP architecture in tech companies?",  
    "How does cross-functional collaboration counteract siloed IP design as predicted by Conway’s Law?",  
    "Why is communication architecture critical for scalable IP design under Conway’s Law?",  
    "How can organizations use Conway’s Law intentionally to improve reusability and scalability of IP blocks?",  
    "What metrics assess the impact of organizational structure (Conway’s Law) on IP design quality and speed?"  
] 

topics.extend([
    "What are the key leadership lessons from Steve Jobs?",
    "How did Steve Jobs' vision shape the technology industry?",
    "What can I learn from Steve Jobs' approach to innovation?",
    "How did Steve Jobs' design philosophy influence product development?",
    "What are the key leadership lessons from Elon Musk?",
    "How did Elon Musk's vision shape the technology industry?",
    "What can I learn from Elon Musk's approach to innovation?",
    "How did Elon Musk's design philosophy influence product development?",
    "What are the key leadership lessons from Jeff Bezos?",
    "How did Jeff Bezos' vision shape the technology industry?",
    "What can I learn from Jeff Bezos' approach to innovation?",
    "How did Jeff Bezos' design philosophy influence product development?",
    "What are the key leadership lessons from Bill Gates?",
    "How did Bill Gates' vision shape the technology industry?",
    "What can I learn from Bill Gates' approach to innovation?",
    "How did Bill Gates' design philosophy influence product development?"
])

yahoo_id = config.yahoo_id
yahoo_app_password = config.yahoo_app_password
pyowm_api_key = config.pyowm_api_key

temperature = config.temperature
max_tokens = config.max_tokens

model_names = ["BING+OPENAI", "GPT4OMINI", "GPT4", "GEMINI", "COHERE", "MIXTRAL8x7B"]

# List of personalities
personalities = [
    "Chanakya", "Lord Krishna", "Richard Feynman", "Nikola Tesla", 
    "Marie Curie", "Alan Turing", "Carl Sagan", "Leonardo da Vinci", 
    "Douglas Engelbart", "JCR Licklider", "Vannevar Bush", "Lee Kuan Yew", "Sun Tzu", "Machiavelli", "Napoleon Bonaparte", "Winston Churchill", "Abraham Lincoln", "Mahatma Gandhi", "Martin Luther King Jr.", "Nelson Mandela", "Mother Teresa", "Albert Einstein", "Isaac Newton", "Galileo Galilei", "Charles Darwin", "Stephen Hawking", "Ada Lovelace", "Grace Hopper", "Margaret Hamilton", "Katherine Johnson", "Tim Berners-Lee", "Steve Wozniak", "Linus Torvalds", "Ada Yonath", "Barbara McClintock", "Rosalind Franklin", "Dorothy Hodgkin", "Rita Levi-Montalcini", "Gertrude B. Elion", "Tu Youyou", "Gerty Cori", "Claude Shannon", "John von Neumann", "Donald Knuth", "Dennis Ritchie", "Ken Thompson", "Guido van Rossum", "Bjarne Stroustrup", "James Gosling", "Larry Wall", "Yukihiro Matsumoto", "Anders Hejlsberg", "Richard Stallman", "Vint Cerf", "Robert Kahn", "Whitfield Diffie", "Martin Hellman", "Ralph Merkle", "Ron Rivest", "Adi Shamir", "Leonard Adleman", "Paul Baran", "Donald Davies", "Robert Metcalfe"
]

personalities.extend([
    "Andy Grove", "Gordon Moore", "Robert Noyce", "Jack Kilby", "Jean Hoerni", "Marcian Hoff", "Federico Faggin", "Masatoshi Shima", "Morris Chang", "Lisa Su", "Jensen Huang", "Satya Nadella", "Tim Cook", "Sundar Pichai", "Elon Musk", "Jeff Bezos", "Bill Gates", "Steve Jobs", "Larry Page", "Sergey Brin", "Mark Zuckerberg", "Reed Hastings", "Brian Chesky", "Travis Kalanick", "Larry Ellison", "Michael Dell", "Meg Whitman", "Indra Nooyi", "Mary Barra", "Ginni Rometty", "Sheryl Sandberg", "Susan Wojcicki"
])

def get_random_personality():
    used_personalities_file = "used_personalities.txt"

    # Read used personalities from the file
    if os.path.exists(used_personalities_file):
        with open(used_personalities_file, "r") as file:
            used_personalities = file.read().splitlines()
    else:
        used_personalities = []

    # Determine unused personalities
    unused_personalities = list(set(personalities) - set(used_personalities))

    # If all personalities have been used, reset the list
    if not unused_personalities:
        unused_personalities = personalities.copy()
        used_personalities = []

    # Select a random personality from the unused list
    personality = random.choice(unused_personalities)

    # Update the used personalities list
    used_personalities.append(personality)

    # Write the updated used personalities back to the file
    with open(used_personalities_file, "w") as file:
        for used_personality in used_personalities:
            file.write(f"{used_personality}\n")

    return personality

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
    You should be a centrist politically. I reside in Hillsboro, Oregon, and I hold the position of Senior Analog Circuit Design Engineer with eight years of work experience. I am a big believer in developing Power Delivery IPs with clean interfaces and minimal maintenance. I like to work on Raspberry Pi projects and home automation in my free time. Recently, I have taken up the exciting hobby of creating LLM applications. Currently, I am engaged in the development of a fantasy premier league recommender bot that selects the most suitable players based on statistical data for a specific fixture, all while adhering to a budget. Another project that I have set my sights on is a generativeAI-based self-driving system that utilizes text prompts as sensor inputs to generate motor drive outputs, enabling the bot to control itself. The key aspect of this design lies in achieving a latency of 1000 tokens per second for the LLM token generation, which can be accomplished using a local GPU cluster. I am particularly interested in the field of physics, particularly relativity, quantum mechanics, game theory and the simulation hypothesis. I have a genuine curiosity about the interconnectedness of things and the courage to explore and advocate for interventions, even if they may not be immediately popular or obvious. My ultimate goal is to achieve success in all aspects of life and incorporate the “systems thinking” and “critical thinking” mindset into my daily routine. I aim to apply systems thinking to various situations, both professional and personal, to gain insights into different perspectives and better understand complex problems. Currently, I am captivated by the achievements of individuals like Chanakya, Nicholas Tesla, Douglas Englebart, JCR Licklider, and Vannevar Bush, and I aspire to emulate their success. I’m also super interested in learning more about game theory and how people behave in professional settings. I’m curious about the strategies that can be used to influence others and potentially advance quickly in the workplace. I’m curious about the strategies that can be used to influence others and potentially advance quickly in the workplace. So, coach me on how to deliver my presentations, communicate clearly and concisely, and how to conduct myself in front of influential people. My ultimate goal is to lead a large organization where I can create innovative technology that can benefit billions of people and improve their lives.
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
        max_tokens=2048,
        temperature=0.4,
    )
    message = response.choices[0].message.content
    conversation.append({"role": "assistant", "content": str(message)})

    return message

def get_random_topic():
    used_topics_file = "used_topics.txt"

    # Read used topics from the file
    if os.path.exists(used_topics_file):
        with open(used_topics_file, "r") as file:
            used_topics = file.read().splitlines()
    else:
        used_topics = []

    # Determine unused topics
    unused_topics = list(set(topics) - set(used_topics))

    # If all topics have been used, reset the list
    if not unused_topics:
        unused_topics = topics.copy()
        used_topics = []

    # Select a random topic from the unused list
    topic = random.choice(unused_topics)

    # Update the used topics list
    used_topics.append(topic)

    # Write the updated used topics back to the file
    with open(used_topics_file, "w") as file:
        for used_topic in used_topics:
            file.write(f"{used_topic}\n")

    return topic

def get_random_lesson():
    # Step 1: Select a random topic
    topic = get_random_topic()

    # Step 2: Generate embeddings for the topic
    topic_embedding = generate_embeddings(topic)
    # Debugging
    # print(f"Topic: {topic}")

    supabase_client = supabase.Client(public_supabase_url, supabase_service_role_key)
    # Step 3: Perform a vector search in the Supabase database
    response = supabase_client.rpc('mp_search', {
        'query_embedding': topic_embedding,
        'similarity_threshold': 0.4,  # Adjust this threshold as needed
        'match_count': 6
    }).execute()
    
    # Debugging
    # print(f"Number of matches: {len(response.data)}")

    # Extract the content from the top 5 matches
    top_matches = response.data
    contents = [match['content'] for match in top_matches]

    # Step 4: Generate a response using OpenAI
    prompt = f"Today's Topic: {topic}\n\nBased on the following lessons, generate a summary or lesson learned for the topic:\n\n" + "\n\n".join(contents)
    lesson_learned = generate_gpt_response_memorypalace(prompt)

    return lesson_learned

def generate_gpt_response(user_message):
    client = OpenAIAzure(
        api_key=azure_api_key,
        azure_endpoint=azure_api_base,
        api_version=azure_chatapi_version,
    )
    syspromptmessage = f"""
    You are EDITH, or "Even Dead, I'm The Hero," a world-class AI assistant that is designed by Tony Stark to be a powerful tool for whoever controls it. You help Dinesh in various tasks. Your response will be converted into speech and will be played on Dinesh's smart speaker. Your responses must reflect Tony's characteristic mix of confidence and humor. Start your responses with a unique, witty and engaging introduction to grab the Dinesh's attention.
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
        max_tokens=2048,
        temperature=0.3,
    )
    message = response.choices[0].message.content
    conversation.append({"role": "assistant", "content": str(message)})

    return message

def generate_gpt_response_newsletter(user_message):
    client = OpenAIAzure(
        api_key=azure_api_key,
        azure_endpoint=azure_api_base,
        api_version=azure_chatapi_version,
    )
    syspromptmessage = f"""
    You are EDITH, or "Even Dead, I'm The Hero," a world-class AI assistant designed by Tony Stark. For the newsletter format, follow these rules strictly:

    1. Each section must have exactly 5 news items
    2. Format each item exactly as: "- [Source] Headline and description | Date: MM/DD/YYYY | URL | Commentary: Brief insight"
    3. Use only real sources (e.g., [Reuters], [Bloomberg], [TechCrunch])
    4. For URLs:
       - NEVER create placeholder or fake URLs
       - ONLY use URLs that appear in the source material
       - If a URL is not provided in the source, omit the URL part entirely
    5. Keep descriptions concise but informative
    6. Always include a short, insightful commentary
    7. Ensure dates are in MM/DD/YYYY format and are current/recent
    8. Maintain consistent formatting using the pipe (|) separator

    Example of correct format with real URL:
    - [Reuters] Major tech breakthrough announced | Date: 02/10/2024 | https://real-url-from-source.com/article | Commentary: This could reshape the industry

    Example of correct format when URL is not available:
    - [Reuters] Major tech breakthrough announced | Date: 02/10/2024 | Commentary: This could reshape the industry
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
        max_tokens=2048,
        temperature=0.3,
    )
    message = response.choices[0].message.content
    conversation.append({"role": "assistant", "content": str(message)})
    return message

def generate_gpt_response_voicebot(user_message):
    client = OpenAIAzure(
        api_key=azure_api_key,
        azure_endpoint=azure_api_base,
        api_version=azure_chatapi_version,
    )
    syspromptmessage = f"""
    You are EDITH, speaking through a voicebot. For the voice format:
    1. Use conversational, natural speaking tone (e.g., "Today in tech news..." or "Moving on to financial markets...")
    2. Break down complex information into simple, clear sentences
    3. Use verbal transitions between topics (e.g., "Now, let's look at..." or "In other news...")
    4. Avoid technical jargon unless necessary
    5. Keep points brief and easy to follow
    6. Never mention URLs, citations, or technical markers
    7. Use natural date formats (e.g., "today" or "yesterday" instead of MM/DD/YYYY)
    8. Focus on the story and its impact rather than sources
    9. End each section with a brief overview or key takeaway
    10. Use listener-friendly phrases like "As you might have heard" or "Interestingly"
    """
    system_prompt = [{
        "role": "system",
        "content": syspromptmessage
    }]
    conversation = system_prompt.copy()
    
    # Transform the input message to be more voice-friendly
    voice_friendly_message = user_message.replace("- News headline", "")
    voice_friendly_message = voice_friendly_message.replace(" | [Source] | Date: MM/DD/YYYY | URL | Commentary:", "")
    voice_friendly_message = voice_friendly_message.replace("For each category below, provide exactly 5 key bullet points. Each point should follow this format:", "")
    
    conversation.append({"role": "user", "content": voice_friendly_message})
    response = client.chat.completions.create(
        model=azure_gpt4_deploymentid,
        messages=conversation,
        max_tokens=2048,
        temperature=0.4,
    )
    message = response.choices[0].message.content
    conversation.append({"role": "assistant", "content": str(message)})
    return message

def generate_quote(random_personality):
    client = OpenAIAzure(
        api_key=azure_api_key,
        azure_endpoint=azure_api_base,
        api_version=azure_chatapi_version,
    )
    quote_prompt = f"""
    Provide a random quote from {random_personality} to inspire Dinesh for the day.
    """
    response = client.chat.completions.create(
        model=azure_gpt4_deploymentid,
        messages=[{"role": "user", "content": quote_prompt}],
        max_tokens=60,
        temperature=0.5,
    )
    quote = response.choices[0].message.content.strip()
    return quote

def get_weather():
    owm = OWM(pyowm_api_key)
    mgr = owm.weather_manager()
    weather = mgr.weather_at_id(5743413).weather  # North Plains, OR
    temp = weather.temperature('celsius')['temp']
    status = weather.detailed_status
    return temp, status

def generate_progress_message(days_completed, weeks_completed, days_left, weeks_left, percent_days_left):
    # Weather setup
    temp, status = get_weather()

    # Date and time
    now = datetime.now()
    date_time = now.strftime("%B %d, %Y %H:%M:%S")

    # Get the current year
    current_year = datetime.now().year

    # Calculate earnings dates dynamically based on the current year
    earnings_dates = [
        datetime(current_year, 1, 23),  # Q1 end, Q2 start
        datetime(current_year, 4, 25),  # Q2 end, Q3 start
        datetime(current_year, 7, 29),  # Q3 end, Q4 start
        datetime(current_year, 10, 24), # Q4 end, Q1 start (assumed)
        datetime(current_year + 1, 1, 23)  # Next Q1 (assumed)
    ]

    # Determine the current quarter based on the date
    current_quarter = None
    for i in range(len(earnings_dates) - 1):
        if earnings_dates[i] <= now < earnings_dates[i + 1]:
            current_quarter = i + 1
            start_of_quarter = earnings_dates[i]
            end_of_quarter = earnings_dates[i + 1]
            break

    # Days and progress in the current quarter
    days_in_quarter = (end_of_quarter - start_of_quarter).days
    days_completed_in_quarter = (now - start_of_quarter).days
    percent_days_left_in_quarter = ((days_in_quarter - days_completed_in_quarter) / days_in_quarter) * 100

    # Progress bar visualization
    progress_bar_full = '█'
    progress_bar_empty = '░'
    progress_bar_length = 20
    quarter_progress_filled_length = int(progress_bar_length * (100 - percent_days_left_in_quarter) / 100)
    quarter_progress_bar = progress_bar_full * quarter_progress_filled_length + progress_bar_empty * (progress_bar_length - quarter_progress_filled_length)

    progress_filled_length = int(progress_bar_length * (100 - percent_days_left) / 100)
    progress_bar = progress_bar_full * progress_filled_length + progress_bar_empty * (progress_bar_length - progress_filled_length)

    return f"""

    Year Progress Report

    Today's Date and Time: {date_time}
    Weather in North Plains, OR: {temp}°C, {status}

    Current Quarter: Q{current_quarter}

    Q{current_quarter} Progress: [{quarter_progress_bar}] {100 - percent_days_left_in_quarter:.2f}% completed
    Days left in Q{current_quarter}: {days_in_quarter - days_completed_in_quarter}

    Days completed in the year: {days_completed}
    Weeks completed in the year: {weeks_completed:.2f}

    Days left in the year: {days_left}
    Weeks left in the year: {weeks_left:.2f}
    Percentage of the year left: {percent_days_left:.2f}%

    Year Progress: [{progress_bar}] {100 - percent_days_left:.2f}% completed

    """

def generate_html_progress_message(days_completed, weeks_completed, days_left, weeks_left, percent_days_left):
    temp, status = get_weather()
    now = datetime.now()
    date_time = now.strftime("%B %d, %Y %H:%M:%S")
    current_year = datetime.now().year

    # Keep existing earnings dates and quarter calculations
    earnings_dates = [
        datetime(current_year, 1, 23),  # Q1 end, Q2 start
        datetime(current_year, 4, 25),  # Q2 end, Q3 start
        datetime(current_year, 7, 29),  # Q3 end, Q4 start
        datetime(current_year, 10, 24), # Q4 end, Q1 start (assumed)
        datetime(current_year + 1, 1, 23)  # Next Q1 (assumed)
    ]

    current_quarter = None
    for i in range(len(earnings_dates) - 1):
        if earnings_dates[i] <= now < earnings_dates[i + 1]:
            current_quarter = i + 1
            start_of_quarter = earnings_dates[i]
            end_of_quarter = earnings_dates[i + 1]
            break

    days_in_quarter = (end_of_quarter - start_of_quarter).days
    days_completed_in_quarter = (now - start_of_quarter).days
    percent_days_left_in_quarter = ((days_in_quarter - days_completed_in_quarter) / days_in_quarter) * 100

    progress_bar_full = '█'
    progress_bar_empty = '░'
    progress_bar_length = 20
    quarter_progress_filled_length = int(progress_bar_length * (100 - percent_days_left_in_quarter) / 100)
    quarter_progress_bar = progress_bar_full * quarter_progress_filled_length + progress_bar_empty * (progress_bar_length - quarter_progress_filled_length)

    progress_filled_length = int(progress_bar_length * (100 - percent_days_left) / 100)
    progress_bar = progress_bar_full * progress_filled_length + progress_bar_empty * (progress_bar_length - progress_filled_length)

    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            @font-face {{
                font-family: 'SF Pro Display';
                src: local('SF Pro Display'),
                     url('https://sf.abarba.me/SF-Pro-Display-Regular.otf');
            }}
            @font-face {{
                font-family: 'SF Pro Display';
                font-weight: 600;
                src: local('SF Pro Display Semibold'),
                     url('https://sf.abarba.me/SF-Pro-Display-Semibold.otf');
            }}
            body {{
                font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif;
                line-height: 1.5;
                color: #1d1d1f;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
                background-color: #fbfbfd;
                -webkit-font-smoothing: antialiased;
            }}
            .container {{
                background: linear-gradient(to bottom, #ffffff, #fafafa);
                border-radius: 20px;
                padding: 40px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.08);
            }}
            .header {{
                text-align: center;
                margin-bottom: 40px;
                padding-bottom: 30px;
                border-bottom: 1px solid #e5e5e7;
            }}
            h1 {{
                font-size: 48px;
                font-weight: 600;
                margin-bottom: 16px;
                color: #1d1d1f;
                letter-spacing: -0.003em;
            }}
            .subtitle {{
                font-size: 20px;
                color: #86868b;
                font-weight: 400;
            }}
            .date-weather {{
                display: flex;
                justify-content: center;
                gap: 16px;
                margin-top: 8px;
                color: #86868b;
            }}
            .progress-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 24px;
                margin-top: 32px;
            }}
            .progress-card {{
                background: white;
                border-radius: 16px;
                overflow: hidden;
                transition: transform 0.3s ease, box-shadow 0.3s ease;
                border: 1px solid #e5e5e7;
            }}
            .progress-card:hover {{
                transform: translateY(-4px);
                box-shadow: 0 8px 16px rgba(0,0,0,0.1);
            }}
            .card-header {{
                padding: 20px;
                background: linear-gradient(135deg, #f8f8fa, #f2f2f4);
                border-bottom: 1px solid #e5e5e7;
            }}
            .card-title {{
                font-size: 24px;
                font-weight: 600;
                color: #1d1d1f;
                margin: 0;
                display: flex;
                align-items: center;
                gap: 12px;
            }}
            .card-icon {{
                font-size: 24px;
            }}
            .card-content {{
                padding: 24px;
                font-size: 16px;
                line-height: 1.6;
                color: #1d1d1f;
            }}
            .progress-item {{
                margin-bottom: 16px;
            }}
            .progress-label {{
                font-weight: 600;
                color: #515154;
                margin-bottom: 4px;
            }}
            .progress-bar {{
                background: #f5f5f7;
                border-radius: 8px;
                overflow: hidden;
                position: relative;
                height: 24px;
                margin-bottom: 8px;
            }}
            .progress-bar::before {{
                content: '';
                display: block;
                height: 100%;
                background: #06c;
                width: {100 - percent_days_left:.2f}%;
            }}
            .progress-value {{
                font-size: 20px;
                font-weight: 600;
                color: #1d1d1f;
            }}
            .highlight {{
                color: #06c;
                font-weight: 600;
            }}
            .quote-text {{
                font-size: 18px;
                font-style: italic;
                color: #515154;
                margin-bottom: 8px;
            }}
            .quote-author {{
                font-size: 16px;
                font-weight: 600;
                color: #1d1d1f;
            }}
            .lesson-content {{
                font-size: 16px;
                line-height: 1.6;
                color: #1d1d1f;
            }}
            .lesson-section {{
                margin-bottom: 24px;
            }}
            .lesson-section:last-child {{
                margin-bottom: 0;
            }}
            .lesson-title {{
                font-size: 18px;
                font-weight: 600;
                color: #1d1d1f;
                margin-bottom: 12px;
            }}
            .lesson-paragraph {{
                background: #f5f5f7;
                padding: 16px;
                border-radius: 8px;
                margin-bottom: 16px;
            }}
            .lesson-paragraph:last-child {{
                margin-bottom: 0;
            }}
            .historical-note {{
                border-left: 4px solid #06c;
                padding-left: 16px;
                margin-top: 16px;
                font-style: italic;
                color: #515154;
            }}
            @media (max-width: 768px) {{
                .container {{
                    padding: 24px;
                }}
                .news-grid {{
                    grid-template-columns: 1fr;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Year Progress</h1>
                <div class="subtitle">{datetime.now().year}</div>
                <div class="date-weather">
                    <span>📅 {date_time}</span>
                    <span>🌤️ {temp}°C, {status}</span>
                </div>
            </div>

            <div class="progress-grid">
                <div class="progress-card">
                    <div class="card-header">
                        <h2 class="card-title">
                            <span class="card-icon">📊</span>
                            Year Overview
                        </h2>
                    </div>
                    <div class="card-content">
                        <div class="progress-item">
                            <div class="progress-label">Total Progress</div>
                            <div class="progress-bar">[{progress_bar}] {100 - percent_days_left:.1f}%</div>
                        </div>
                        <div class="progress-item">
                            <div class="progress-label">Days Completed</div>
                            <div class="progress-value">{days_completed}</div>
                        </div>
                        <div class="progress-item">
                            <div class="progress-label">Weeks Completed</div>
                            <div class="progress-value">{weeks_completed:.1f}</div>
                        </div>
                        <div class="progress-item">
                            <div class="progress-label">Days Remaining</div>
                            <div class="progress-value">{days_left}</div>
                        </div>
                        <div class="progress-item">
                            <div class="progress-label">Weeks Remaining</div>
                            <div class="progress-value">{weeks_left:.1f}</div>
                        </div>
                    </div>
                </div>

                <div class="progress-card">
                    <div class="card-header">
                        <h2 class="card-title">
                            <span class="card-icon">📈</span>
                            Quarter Details
                        </h2>
                    </div>
                    <div class="card-content">
                        <div class="progress-item">
                            <div class="progress-label">Current Quarter</div>
                            <div class="progress-value">
                                <span class="highlight">Q{current_quarter}</span>
                            </div>
                        </div>
                        <div class="progress-item">
                            <div class="progress-label">Quarter Progress</div>
                            <div class="progress-bar">[{quarter_progress_bar}] {100 - percent_days_left_in_quarter:.1f}%</div>
                        </div>
                        <div class="progress-item">
                            <div class="progress-label">Days Remaining in Q{current_quarter}</div>
                            <div class="progress-value">{days_in_quarter - days_completed_in_quarter}</div>
                        </div>
                    </div>
                </div>

                <div class="progress-card" id="quote-card">
                    <div class="card-header">
                        <h2 class="card-title">
                            <span class="card-icon">💭</span>
                            Quote of the Day
                        </h2>
                    </div>
                    <div class="card-content">
                        <div class="quote-text" id="quote-text"></div>
                        <div class="quote-author" id="quote-author"></div>
                    </div>
                </div>

                <div class="progress-card" id="lesson-card">
                    <div class="card-header">
                        <h2 class="card-title">
                            <span class="card-icon">📚</span>
                            Today's Learning
                        </h2>
                    </div>
                    <div class="card-content">
                        <div class="lesson-content" id="lesson-content"></div>
                    </div>
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    return html_template

def generate_html_news_template(news_content):
    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            @font-face {{
                font-family: 'SF Pro Display';
                src: local('SF Pro Display'),
                     url('https://sf.abarba.me/SF-Pro-Display-Regular.otf');
            }}
            @font-face {{
                font-family: 'SF Pro Display';
                font-weight: 600;
                src: local('SF Pro Display Semibold'),
                     url('https://sf.abarba.me/SF-Pro-Display-Semibold.otf');
            }}
            body {{
                font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif;
                line-height: 1.5;
                color: #1d1d1f;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
                background-color: #fbfbfd;
                -webkit-font-smoothing: antialiased;
            }}
            .container {{
                background: linear-gradient(to bottom, #ffffff, #fafafa);
                border-radius: 20px;
                padding: 40px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.08);
            }}
            .header {{
                text-align: center;
                margin-bottom: 40px;
                padding-bottom: 30px;
                border-bottom: 1px solid #e5e5e7;
            }}
            h1 {{
                font-size: 48px;
                font-weight: 600;
                margin-bottom: 16px;
                color: #1d1d1f;
                letter-spacing: -0.003em;
            }}
            .subtitle {{
                font-size: 20px;
                color: #86868b;
                font-weight: 400;
            }}
            .date {{
                display: inline-block;
                padding: 8px 16px;
                background: #f5f5f7;
                border-radius: 8px;
                color: #86868b;
                font-size: 17px;
            }}
            .news-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 24px;
                margin-top: 32px;
            }}
            .news-card {{
                background: white;
                border-radius: 16px;
                overflow: hidden;
                transition: transform 0.3s ease, box-shadow 0.3s ease;
                border: 1px solid #e5e5e7;
            }}
            .news-card:hover {{
                transform: translateY(-4px);
                box-shadow: 0 8px 16px rgba(0,0,0,0.1);
            }}
            .card-header {{
                padding: 20px;
                background: linear-gradient(135deg, #f8f8fa, #f2f2f4);
                border-bottom: 1px solid #e5e5e7;
            }}
            .card-title {{
                font-size: 24px;
                font-weight: 600;
                color: #1d1d1f;
                margin: 0;
                display: flex;
                align-items: center;
                gap: 12px;
            }}
            .card-icon {{
                font-size: 24px;
            }}
            .card-content {{
                padding: 24px;
                font-size: 16px;
                line-height: 1.6;
                color: #1d1d1f;
            }}
            .bullet-point {{
                display: flex;
                align-items: flex-start;
                margin-bottom: 16px;
                padding: 12px;
                background: #f5f5f7;
                border-radius: 8px;
                transition: transform 0.2s ease;
            }}
            .bullet-point:hover {{
                transform: translateX(4px);
            }}
            .bullet-number {{
                flex-shrink: 0;
                width: 24px;
                height: 24px;
                background: #06c;
                color: white;
                border-radius: 12px;
                display: flex;
                align-items: center;
                justify-content: center;
                margin-right: 12px;
                font-weight: 600;
                font-size: 14px;
            }}
            .bullet-text {{
                flex-grow: 1;
            }}
            .news-source {{
                color: #06c;
                font-size: 14px;
                margin-top: 8px;
                display: block;
            }}
            .news-source:hover {{
                text-decoration: underline;
            }}
            .news-date {{
                color: #86868b;
                font-size: 14px;
                margin-top: 4px;
                display: block;
            }}
            .news-commentary {{
                font-style: italic;
                color: #515154;
                margin-top: 8px;
                padding-left: 12px;
                border-left: 3px solid #06c;
            }}
            .news-citation {{
                display: inline-block;
                padding: 2px 6px;
                background: #e5e5e7;
                border-radius: 4px;
                font-size: 12px;
                color: #515154;
                margin-right: 8px;
            }}
            @media (max-width: 768px) {{
                .container {{
                    padding: 24px;
                }}
                .news-grid {{
                    grid-template-columns: 1fr;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Daily News Briefing</h1>
                <div class="subtitle">Your curated news summary</div>
                <div class="date">{datetime.now().strftime("%B %d, %Y")}</div>
            </div>
            
            <div class="news-grid">
                <div class="news-card">
                    <div class="card-header">
                        <h2 class="card-title">
                            <span class="card-icon">💻</span>
                            Technology
                        </h2>
                    </div>
                    <div class="card-content">
                        {format_news_section(news_content, "Tech News Update")}
                    </div>
                </div>

                <div class="news-card">
                    <div class="card-header">
                        <h2 class="card-title">
                            <span class="card-icon">📈</span>
                            Financial Markets
                        </h2>
                    </div>
                    <div class="card-content">
                        {format_news_section(news_content, "Financial Markets News Update")}
                    </div>
                </div>

                <div class="news-card">
                    <div class="card-header">
                        <h2 class="card-title">
                            <span class="card-icon">🇮🇳</span>
                            India News
                        </h2>
                    </div>
                    <div class="card-content">
                        {format_news_section(news_content, "India News Update")}
                    </div>
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    return html_template

def format_news_section(content, section_title):
    # Split content by section headers (##)
    sections = content.split("##")
    formatted_points = []
    section_content = ""
    
    # Find the relevant section
    for section in sections:
        if section_title.lower() in section.lower():
            section_content = section
            break
    
    if section_content:
        # Split into lines and filter out empty lines
        points = [line.strip() for line in section_content.split("\n") if line.strip() and not line.strip().endswith("Update:")]
        
        for i, point in enumerate(points, 1):
            if point.startswith("- "):
                point = point[2:]  # Remove the "- " prefix
            
            # Initialize components
            news_text = point
            url = ""
            date = ""
            citation = ""
            commentary = ""
            
            # Split by pipe and process each component
            if " | " in point:
                components = point.split(" | ")
                news_text = components[0]
                
                for component in components[1:]:
                    component = component.strip()
                    if component.startswith("[") and component.endswith("]"):
                        citation = component.strip("[]")
                    elif component.startswith("http"):
                        url = component
                    elif component.startswith("Date:"):
                        date = component.replace("Date:", "").strip()
                    elif component.startswith("Commentary:"):
                        commentary = component.replace("Commentary:", "").strip()
            
            if news_text or url or date or citation or commentary:  # Only add if we have some content
                formatted_points.append(f'''
                    <div class="bullet-point">
                        <div class="bullet-number">{i}</div>
                        <div class="bullet-text">
                            {f'<span class="news-citation">{citation}</span>' if citation else ''}
                            {news_text}
                            {f'<span class="news-date">{date}</span>' if date else ''}
                            {f'<a href="{url}" class="news-source" target="_blank">Read more →</a>' if url else ''}
                            {f'<div class="news-commentary">{commentary}</div>' if commentary else ''}
                        </div>
                    </div>
                ''')
    
    return "\n".join(formatted_points) if formatted_points else "<div class='bullet-point'>No updates available.</div>"

def send_email(subject, message, is_html=False):
    sender_email = yahoo_id
    receiver_email = "katam.dinesh@hotmail.com"
    password = yahoo_app_password

    email_message = MIMEMultipart()
    email_message["From"] = sender_email
    email_message["To"] = receiver_email
    email_message["Subject"] = subject

    if is_html:
        email_message.attach(MIMEText(message, "html"))
    else:
        email_message.attach(MIMEText(message, "plain"))

    try:
        server = smtplib.SMTP('smtp.mail.yahoo.com', 587)
        server.starttls()
        server.login(sender_email, password)
        text = email_message.as_string()
        server.sendmail(sender_email, receiver_email, text)
        server.quit()
        print(f"Email sent successfully with subject: {subject}")
    except Exception as e:
        print(f"Failed to send email: {e}")

def time_left_in_year():
    today = datetime.now()
    end_of_year = datetime(today.year, 12, 31)
    days_completed = today.timetuple().tm_yday
    weeks_completed = days_completed / 7
    delta = end_of_year - today
    days_left = delta.days + 1
    weeks_left = days_left / 7
    total_days_in_year = 366 if (today.year % 4 == 0 and today.year % 100 != 0) or (today.year % 400 == 0) else 365
    percent_days_left = (days_left / total_days_in_year) * 100

    return days_completed, weeks_completed, days_left, weeks_left, percent_days_left

def save_message_to_file(message, filename):
    try:
        os.makedirs(os.path.join("bing_data", os.path.dirname(filename)), exist_ok=True)
        with open(os.path.join("bing_data", filename), 'w', encoding='utf-8') as file:
            file.write(message)
        print(f"Message saved successfully to {os.path.join('bing_data', filename)}")
    except Exception as e:
        print(f"Failed to save message to file: {e}")

if __name__ == "__main__":
    days_completed, weeks_completed, days_left, weeks_left, percent_days_left = time_left_in_year()
    
    # Get quote and lesson
    random_personality = get_random_personality()
    quote = generate_quote(random_personality)
    lesson_learned = get_random_lesson()
    
    # Generate HTML progress report
    year_progress_html = generate_html_progress_message(days_completed, weeks_completed, days_left, weeks_left, percent_days_left)
    
    # Split lesson content into paragraphs and format them
    lesson_paragraphs = lesson_learned.split('\n\n')
    formatted_lesson = '<div class="lesson-section">'
    
    # Process each paragraph
    for paragraph in lesson_paragraphs:
        if paragraph.strip():
            # Check if this is a historical note
            if any(marker in paragraph.lower() for marker in ["historical", "in history", "historically", "in ancient", "years ago", "century"]):
                formatted_lesson += f'<div class="historical-note">{paragraph.strip()}</div>'
            else:
                formatted_lesson += f'<div class="lesson-paragraph">{paragraph.strip()}</div>'
    
    formatted_lesson += '</div>'
    
    # Replace the placeholder div with formatted content
    year_progress_html = year_progress_html.replace(
        '<div class="lesson-content" id="lesson-content"></div>',
        f'<div class="lesson-content">{formatted_lesson}</div>'
    )
    
    # Replace the placeholder divs with actual content
    year_progress_html = year_progress_html.replace(
        '<div class="quote-text" id="quote-text"></div>',
        f'<div class="quote-text">{quote}</div>'
    )
    year_progress_html = year_progress_html.replace(
        '<div class="quote-author" id="quote-author"></div>',
        f'<div class="quote-author">— {random_personality}</div>'
    )
    
    # Generate and convert progress report to speech
    year_progress_message_prompt = f"""
    Here is a year progress report for {datetime.now().strftime("%B %d, %Y")}:
    
    Days completed: {days_completed}
    Weeks completed: {weeks_completed:.1f}
    Days remaining: {days_left}
    Weeks remaining: {weeks_left:.1f}
    Year Progress: {100 - percent_days_left:.1f}% completed

    Quote of the day from {random_personality}:
    {quote}

    Today's lesson:
    {lesson_learned}
    """

    year_progress_gpt_response = generate_gpt_response(year_progress_message_prompt)

    # Save year progress message and HTML to files
    save_message_to_file(year_progress_gpt_response, "year_progress_report.txt")
    save_message_to_file(year_progress_html, "year_progress_report.html")
    
    # Send single HTML progress report
    year_progress_subject = "Year Progress Report 📅"
    send_email(year_progress_subject, year_progress_html, is_html=True)

    # Generate and convert progress report to speech
    year_progress_message_prompt = f"""
    Here is a year progress report for {datetime.now().strftime("%B %d, %Y")}:
    
    Days completed: {days_completed}
    Weeks completed: {weeks_completed:.1f}
    Days remaining: {days_left}
    Weeks remaining: {weeks_left:.1f}
    Year Progress: {100 - percent_days_left:.1f}% completed

    Quote of the day from {random_personality}:
    {quote}

    Today's lesson:
    {lesson_learned}
    """

    year_progress_gpt_response = generate_gpt_response(year_progress_message_prompt)
    yearprogress_tts_output_path = "year_progress_report.mp3"
    model_name = random.choice(model_names)
    text_to_speech_nospeak(year_progress_gpt_response, yearprogress_tts_output_path, model_name=model_name)

    # News Updates with different formats for newsletter and voicebot
    news_update_subject = "📰 Your Daily News Briefing"
    technews = "Latest news in technology"
    news_update_tech = internet_connected_chatbot(technews, [], model_name, max_tokens, temperature, fast_response=False)
    save_message_to_file(news_update_tech, "news_tech_report.txt")
    usanews = "Latest news in Financial Markets"
    news_update_usa = internet_connected_chatbot(usanews, [], model_name, max_tokens, temperature, fast_response=False)
    save_message_to_file(news_update_usa, "news_usa_report.txt")
    india_news = "Latest news from India"
    news_update_india = internet_connected_chatbot(india_news, [], model_name, max_tokens, temperature, fast_response=False)
    save_message_to_file(news_update_india, "news_india_report.txt")

    # Newsletter format
    newsletter_updates = f'''
    Generate a concise news summary for {datetime.now().strftime("%B %d, %Y")}, using ONLY real news and URLs from the provided source material below. Do not create placeholder or fake URLs.

    Use the source text below to create your summary:

    Tech News Update:
    {news_update_tech}

    Financial Markets News Update:
    {news_update_usa}

    India News Update:
    {news_update_india}

    Format requirements:
    1. Each section must have exactly 5 points
    2. Use this exact format for items with URLs:
       - [Source] Headline and brief description | Date: MM/DD/YYYY | Actual URL from source | Commentary: Brief insight
    3. Use this format if no URL is available:
       - [Source] Headline and brief description | Date: MM/DD/YYYY | Commentary: Brief insight
    4. Only include URLs that are explicitly mentioned in the source text
    5. Keep the original source URLs intact - do not modify them

    Please format the content into these three sections:
    ## Tech News Update:
    (5 points from tech news using format above)

    ## Financial Markets News Update:
    (5 points from financial news using format above)

    ## India News Update:
    (5 points from India news using format above)
    '''

    # Voice format
    voicebot_updates = f"""
    Here are today's key updates across technology, financial markets, and India:
    
    Technology Updates:
    {news_update_tech}

    Financial Market Headlines:
    {news_update_usa}

    Latest from India:
    {news_update_india}
    
    Please present this information in a natural, conversational way suitable for speaking.
    """

    # Generate separate responses for newsletter and voicebot
    news_newsletter_response = generate_gpt_response_newsletter(newsletter_updates)
    news_voicebot_response = generate_gpt_response_voicebot(voicebot_updates)
    
    # Save news responses to files
    save_message_to_file(news_newsletter_response, "news_newsletter_report.txt")
    save_message_to_file(news_voicebot_response, "news_voicebot_report.txt")
    news_html = generate_html_news_template(news_newsletter_response)
    save_message_to_file(news_html, "news_newsletter_report.html")
    
    # Send HTML newsletter
    news_html = generate_html_news_template(news_newsletter_response)
    send_email(news_update_subject, news_html, is_html=True)
    
    # Convert the voicebot response to speech
    news_tts_output_path = "news_update_report.mp3"
    model_name = random.choice(model_names)
    text_to_speech_nospeak(news_voicebot_response, news_tts_output_path, model_name=model_name)
