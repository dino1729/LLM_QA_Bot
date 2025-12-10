import smtplib
import logging
import re
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pyowm import OWM
from config import config
from helper_functions.chat_generation_with_internet import internet_connected_chatbot
from helper_functions.audio_processors import text_to_speech_nospeak
from helper_functions.llm_client import get_client
from helper_functions.news_researcher import gather_daily_news
import random
import os

# Configure logging with timestamps
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(funcName)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# --- Configuration ---
# You can switch to "ollama" if preferred
LLM_PROVIDER = "litellm" 
# LLM_PROVIDER = "ollama"


yahoo_id = config.yahoo_id
yahoo_app_password = config.yahoo_app_password
pyowm_api_key = config.pyowm_api_key

temperature = config.temperature
max_tokens = config.max_tokens

# Updated model names for LiteLLM/Ollama
if LLM_PROVIDER == "litellm":
    model_names = ["LITELLM_SMART", "LITELLM_STRATEGIC"]
else:
    model_names = ["OLLAMA_SMART", "OLLAMA_STRATEGIC"]

# --- End Configuration ---

# List of topics
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
    "What is the 'Accusations Audit' technique, and how does it disarm resistance in adversarial conversations?",  
    "How do calibrated questions (e.g., *How am I supposed to do that?*) shift power dynamics in negotiations?",  
    "When should you use the 'Late-Night FM DJ Voice' to de-escalate tension during disagreements?",  
    "How can anchoring bias be leveraged to set favorable terms in salary or deal negotiations?",  
    "What are 'Black Swan' tactics for uncovering hidden information in negotiations?",  
    "How can active listening techniques improve conflict resolution in team settings?",  
    "What non-verbal cues (e.g., tone, body language) most impact persuasive communication?",  
    "How can I adapt my communication style to different personality types (e.g., assertive vs. analytical)?",  
    "What storytelling frameworks make complex ideas more compelling during presentations?",  
    "How do you balance assertiveness and empathy when delivering critical feedback?",  
    "What are strategies for managing difficult conversations (e.g., layoffs, project failures) with grace?",  
    "How can Nash Equilibrium concepts guide decision-making in workplace collaborations?",  
    "What real-world scenarios mimic the 'Chicken Game,' and how should you strategize in them?",  
    "How do Schelling Points (focal points) help teams reach consensus without direct communication?",  
    "When is tit-for-tat with forgiveness more effective than strict reciprocity in office politics?",  
    "How does backward induction in game theory apply to long-term career or project planning?",  
    "What are examples of zero-sum vs. positive-sum games in corporate negotiations?",  
    "How can Bayesian reasoning improve decision-making under uncertainty (e.g., mergers, market entry)?",  
    "How can Boyd's OODA Loop (Observe, Orient, Decide, Act) improve decision-making under pressure?",  
    "What game theory principles optimize resource allocation in cross-functional teams?",  
    "How can the 'MAD' (Mutually Assured Destruction) concept deter adversarial behavior in workplaces?", 
    "How does Conway's Law ('organizations design systems that mirror their communication structures') impact the efficiency of IP or product design?",  
    "What strategies can mitigate the negative effects of Conway's Law on modularity in IP design (e.g., reusable components)?",  
    "How can teams align their structure with IP design goals to leverage Conway's Law for better outcomes?",  
    "What are real-world examples of Conway's Law leading to inefficient or efficient IP architecture in tech companies?",  
    "How does cross-functional collaboration counteract siloed IP design as predicted by Conway's Law?",  
    "Why is communication architecture critical for scalable IP design under Conway's Law?",  
    "How can organizations use Conway's Law intentionally to improve reusability and scalability of IP blocks?",  
    "What metrics assess the impact of organizational structure (Conway's Law) on IP design quality and speed?",
    
    # Steve Jobs specific questions
    "What were Steve Jobs' key leadership principles at Apple?",
    "How did Steve Jobs' product design philosophy transform consumer electronics?",
    "What can engineers learn from Steve Jobs' approach to simplicity and user experience?",
    "How did Steve Jobs balance innovation with commercial viability?",
    
    # Elon Musk specific questions
    "What makes Elon Musk's approach to engineering challenges unique?",
    "How does Elon Musk manage multiple revolutionary companies simultaneously?",
    "What risk management strategies does Elon Musk employ in his ventures?",
    "How has Elon Musk's first principles thinking changed traditional industries?",
    
    # Jeff Bezos specific questions
    "What is the significance of Jeff Bezos' Day 1 philosophy at Amazon?",
    "How did Jeff Bezos' customer obsession shape Amazon's business model?",
    "What can be learned from Jeff Bezos' approach to long-term thinking?",
    "How does Jeff Bezos' decision-making framework handle uncertainty?",
    
    # Bill Gates specific questions
    "How did Bill Gates transition from technology leader to philanthropist?",
    "What made Bill Gates' product strategy at Microsoft so effective?",
    "How did Bill Gates foster a culture of technical excellence?",
    "What can we learn from Bill Gates' approach to global health challenges?",
    
    # Guns, Germs, and Steel
    "How did geographical factors determine which societies developed advanced technologies and conquered others?",
    "What does Jared Diamond's analysis reveal about environmental determinism in human development?",
    
    # The Rise and Fall of the Third Reich
    "What political and economic conditions in Germany enabled Hitler's rise to power?",
    "How did the Nazi regime's propaganda techniques create such effective mass manipulation?",
    
    # The Silk Roads
    "How did the ancient trade networks of the Silk Roads facilitate cultural exchange and technological diffusion?",
    "What does a Silk Roads perspective teach us about geopolitical power centers throughout history?",
    
    # 1491
    "What advanced civilizations existed in pre-Columbian Americas that challenge our historical narratives?",
    "How did indigenous American societies develop sophisticated agricultural and urban systems before European contact?",
    
    # The Age of Revolution
    "How did the dual revolutions (French and Industrial) fundamentally reshape European society?",
    "What economic and social factors drove the revolutionary changes across Europe from 1789-1848?",
    
    # The Ottoman Empire
    "What administrative innovations allowed the Ottoman Empire to successfully govern a diverse, multi-ethnic state?",
    "How did the Ottoman Empire's position between East and West influence its cultural development?",
    
    # Decline and Fall of the Roman Empire
    "What internal factors contributed most significantly to the Roman Empire's decline?",
    "How did the rise of Christianity influence the political transformation of the Roman Empire?",
    
    # The Warmth of Other Suns
    "How did the Great Migration of African Americans transform both Northern and Southern American society?",
    "What personal stories from the Great Migration reveal about systemic racism and individual resilience?",
    
    # The Peloponnesian War
    "What strategic and leadership lessons can be learned from Athens' and Sparta's conflict?",
    "How did democratic Athens' political system influence its military decisions during the war?",
    
    # The Making of the Atomic Bomb
    "What moral dilemmas faced scientists during the Manhattan Project, and how are they relevant today?",
    "How did the development of nuclear weapons transform the relationship between science and government?"
] 

# List of personalities
personalities = [
    # Historical figures and philosophers
    "Chanakya", "Sun Tzu", "Machiavelli", "Leonardo da Vinci", "Socrates", "Plato", "Aristotle",
    "Confucius", "Marcus Aurelius", "Friedrich Nietzsche", "Carl Jung", "Sigmund Freud",
    
    # Political and social leaders
    "Winston Churchill", "Abraham Lincoln", "Mahatma Gandhi", "Martin Luther King Jr.", "Nelson Mandela",
    
    # Scientists and mathematicians
    "Albert Einstein", "Isaac Newton", "Marie Curie", "Stephen Hawking", "Richard Feynman", "Nikola Tesla",
    "Galileo Galilei", "James Clerk Maxwell", "Charles Darwin",
    
    # Computer science and tech pioneers
    "Alan Turing", "Claude Shannon", "Ada Lovelace", "Grace Hopper", "Tim Berners-Lee",
    
    # Programming language creators
    "Linus Torvalds", "Guido van Rossum", "Dennis Ritchie",
    
    # Modern tech leaders
    "Bill Gates", "Steve Jobs", "Elon Musk", "Jeff Bezos", "Satya Nadella", "Tim Cook",
    "Lisa Su", "Larry Page", "Sergey Brin", "Mark Zuckerberg", "Jensen Huang",
    
    # Semiconductor pioneers
    "Gordon Moore", "Robert Noyce", "Andy Grove"
]

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
    """
    Generate a comprehensive daily lesson using LLM's knowledge
    No database required - leverages the model's training data
    """
    topic = get_random_topic()
    
    # Structured prompt for scannable wisdom sections
    prompt = f"""Topic: {topic}

YOU MUST START YOUR RESPONSE WITH "[KEY INSIGHT]" - this is mandatory.

Generate wisdom in EXACTLY this structured format:

[KEY INSIGHT]
Write 1-2 powerful sentences capturing the core wisdom about this topic.

[HISTORICAL]
Share ONE brief historical anecdote (2-3 sentences) that illustrates this principle.

[APPLICATION]
Write 1-2 sentences on how this applies to modern engineering or leadership.

CRITICAL RULES:
1. Your FIRST line MUST be "[KEY INSIGHT]" followed by your insight
2. Then "[HISTORICAL]" followed by your historical example
3. Then "[APPLICATION]" followed by your application
4. Keep each section brief - this is for a newsletter
5. Do NOT skip any section - all three are required"""
    
    lesson_learned = generate_lesson_response(prompt)
    return lesson_learned

def generate_lesson_response(user_message):
    """
    Generate a comprehensive lesson/learning content with historical context
    Uses LLM's knowledge directly - no external database required
    """
    logger.info(f"Generating lesson for topic: {user_message[:100]}...")
    
    # Use fast model to avoid reasoning artifacts
    client = get_client(provider=LLM_PROVIDER, model_tier="fast")
    
    syspromptmessage = f"""You are EDITH, an expert teacher helping Dinesh master timeless principles of success.

MANDATORY FORMAT - YOUR RESPONSE MUST LOOK EXACTLY LIKE THIS:
[KEY INSIGHT]
<your key insight here - 1-2 sentences>

[HISTORICAL]
<your historical example here - 2-3 sentences>

[APPLICATION]
<your application here - 1-2 sentences>

RULES:
1. START with "[KEY INSIGHT]" on the very first line - no exceptions
2. Include ALL THREE sections in order: KEY INSIGHT, HISTORICAL, APPLICATION
3. Keep each section brief (1-3 sentences)
4. NO introductions, NO meta-commentary - just the formatted content

Context: Dinesh is a Senior Analog Circuit Design Engineer who values first principles thinking."""
    
    conversation = [
        {"role": "system", "content": syspromptmessage},
        {"role": "user", "content": user_message}
    ]
    
    try:
        message = client.chat_completion(
            messages=conversation,
            max_tokens=4000,
            temperature=0.65
        )
        
        logger.debug(f"Raw LLM response for lesson (first 300 chars): {message[:300] if message else 'EMPTY'}")
        logger.debug(f"Full raw response length: {len(message) if message else 0}")
        
        if not message or len(message.strip()) == 0:
            logger.warning("LLM returned empty response for lesson generation")
            return _get_fallback_lesson()
        
        # Aggressive cleaning of reasoning artifacts
        cleaned = message.strip()
        
        # Remove common reasoning patterns at the start
        reasoning_patterns = [
            "the user wants", "user wants", "user says", "user is asking",
            "we need to", "we need", "we must", "let me provide", "let me",
            "here's", "here is", "i'll provide", "i'll", "i will",
            "task:", "topic:", "provide a", "generate", "create a lesson",
            "write a lesson", "this lesson", "this response", "my response",
            "to answer this", "in response", "based on"
        ]
        
        # Split into lines and find first substantial content line
        lines = cleaned.split('\n')
        start_idx = 0
        
        for i, line in enumerate(lines):
            line_lower = line.lower().strip()
            # Skip empty lines and lines starting with reasoning patterns
            if line.strip():
                # Check if line starts with reasoning pattern
                has_reasoning = any(line_lower.startswith(pattern) for pattern in reasoning_patterns)
                # Check if line has substantial content
                has_substance = len(line.strip()) > 30 and not line.strip().endswith(':')
                
                if not has_reasoning and has_substance:
                    start_idx = i
                    logger.debug(f"Starting content at line {i}: {line[:50]}")
                    break
        
        cleaned = '\n'.join(lines[start_idx:]).strip()
        
        # If still has artifacts at the very beginning, try harder
        first_para = cleaned.split('\n\n')[0] if '\n\n' in cleaned else cleaned[:250]
        if any(pattern in first_para.lower() for pattern in reasoning_patterns):
            logger.debug("Found reasoning patterns in first paragraph, using second paragraph")
            # Use second paragraph if available
            paragraphs = cleaned.split('\n\n')
            if len(paragraphs) > 1:
                cleaned = '\n\n'.join(paragraphs[1:]).strip()
        
        # Final validation - ensure we have meaningful content
        if cleaned and len(cleaned) > 200:
            logger.info(f"Successfully generated lesson ({len(cleaned)} chars)")
            return cleaned
        else:
            logger.warning(f"Generated lesson too short ({len(cleaned) if cleaned else 0} chars), using fallback")
            return _get_fallback_lesson()
        
    except Exception as e:
        logger.error(f"Error generating lesson: {e}", exc_info=True)
        return _get_fallback_lesson()


def _get_fallback_lesson():
    """Return a fallback lesson when generation fails"""
    fallback = """The pursuit of mastery requires understanding that excellence is not a destination but a continuous journey. Ancient philosophers like Aristotle understood this, coining the term "eudaimonia" to describe the flourishing that comes from living up to one's potential through disciplined practice.

Consider the example of Leonardo da Vinci, who kept detailed notebooks throughout his life, documenting not just his artistic techniques but his observations of nature, engineering concepts, and philosophical musings. This habit of systematic learning and documentation allowed him to make connections across domains that others missed.

For modern engineers and leaders, this translates to three practical principles: First, maintain a learning system - whether notebooks, digital tools, or structured reflection time. Second, seek cross-domain knowledge, as innovation often happens at the intersection of fields. Third, embrace deliberate practice over mere repetition, focusing on the areas where improvement is most needed.

The historical parallel to the Roman aqueduct engineers is instructive: they built systems that lasted millennia not through revolutionary innovation alone, but through meticulous attention to fundamentals, redundancy in design, and deep understanding of the materials and forces they worked with."""
    
    logger.info("Using fallback lesson content")
    return fallback

def generate_gpt_response(user_message):
    """Generate a speech-friendly response from EDITH"""
    client = get_client(provider=LLM_PROVIDER, model_tier="smart")
    
    syspromptmessage = f"""You are EDITH, Tony Stark's AI assistant, speaking to Dinesh through his smart speaker.

Your response will be converted to speech (TTS), so you MUST:
- Write in flowing, conversational prose - NO formatting whatsoever
- Use Tony's confidence and wit
- Start with a unique, attention-grabbing greeting

CRITICAL TTS FORMATTING RULES - The text will be read aloud, so NEVER use:
- Asterisks (*) for emphasis or bullets - TTS reads these as "asterisk"
- Markdown formatting (**, __, ##, etc.)
- Bullet points or numbered lists
- Dashes at the start of lines
- Special characters or symbols
- Section headers

Instead of formatting, use natural speech patterns:
- For emphasis, use word choice and sentence structure
- For lists, say "First... Second... And finally..."
- For sections, use verbal transitions like "Now, regarding the lesson..."

Write ONLY plain text that sounds natural when spoken aloud. No visual formatting of any kind.

Do NOT include meta-commentary like "The user wants..." - just speak directly to Dinesh.
"""
    
    conversation = [
        {"role": "system", "content": syspromptmessage},
        {"role": "user", "content": str(user_message)}
    ]
    
    message = client.chat_completion(
        messages=conversation,
        max_tokens=2500,
        temperature=0.4
    )
    
    # Clean up reasoning artifacts
    cleaned = message.strip()
    
    if cleaned.lower().startswith(("the user", "user wants", "we need", "let me", "task:")):
        lines = cleaned.split('\n')
        start_idx = 0
        for i, line in enumerate(lines):
            if line.strip() and not any(line.lower().startswith(prefix) for prefix in 
                ["the user", "user wants", "we need", "let me", "task:", "provide", "respond with"]):
                start_idx = i
                break
        cleaned = '\n'.join(lines[start_idx:]).strip()
    
    return cleaned

def generate_gpt_response_newsletter(user_message):
    """
    Generate newsletter-formatted news content with error handling
    Returns formatted content or fallback summary if generation fails
    """
    try:
        client = get_client(provider=LLM_PROVIDER, model_tier="smart")
        
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
        9. ALWAYS start each section with the exact header format: "## Section Name Update:"

    Example of correct format with real URL:
    - [Reuters] Major tech breakthrough announced | Date: 02/10/2024 | https://real-url-from-source.com/article | Commentary: This could reshape the industry

    Example of correct format when URL is not available:
    - [Reuters] Major tech breakthrough announced | Date: 02/10/2024 | Commentary: This could reshape the industry
    """
        system_prompt = [{"role": "system", "content": syspromptmessage}]
        conversation = system_prompt.copy()
        conversation.append({"role": "user", "content": str(user_message)})
        
        message = client.chat_completion(
            messages=conversation,
            max_tokens=3000,
            temperature=0.3
        )
        
        # Verify we got meaningful content
        if not message or len(message.strip()) < 100:
            print("Warning: Newsletter generation returned minimal content")
            return ""
            
        return message
        
    except Exception as e:
        print(f"Error generating newsletter: {e}")
        return ""

def generate_gpt_response_voicebot(user_message):
    """
    Generate voice-friendly news content with error handling
    Returns formatted content or fallback summary if generation fails
    """
    try:
        client = get_client(provider=LLM_PROVIDER, model_tier="smart")
        
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
    
CRITICAL: You MUST generate a substantial voice script (at least 500 words). Start speaking immediately.
    """
        system_prompt = [{"role": "system", "content": syspromptmessage}]
        conversation = system_prompt.copy()
        
        # Transform the input message to be more voice-friendly
        voice_friendly_message = user_message.replace("- News headline", "")
        voice_friendly_message = voice_friendly_message.replace(" | [Source] | Date: MM/DD/YYYY | URL | Commentary:", "")
        voice_friendly_message = voice_friendly_message.replace("For each category below, provide exactly 5 key bullet points. Each point should follow this format:", "")
        
        conversation.append({"role": "user", "content": voice_friendly_message})
        
        message = client.chat_completion(
            messages=conversation,
            max_tokens=2048,
            temperature=0.4
        )
        
        # Verify we got meaningful content
        if not message or len(message.strip()) < 100:
            logger.warning(f"Voicebot generation returned minimal content ({len(message) if message else 0} chars)")
            return _get_fallback_voicebot_script()
            
        return message
        
    except Exception as e:
        logger.error(f"Error generating voicebot script: {e}")
        return _get_fallback_voicebot_script()

def _get_fallback_voicebot_script():
    """Return a fallback voicebot script when generation fails"""
    from datetime import datetime
    today = datetime.now().strftime("%B %d, %Y")
    return f"""Good morning, Dinesh. Here's your news briefing for {today}.

In technology news today, the AI industry continues to evolve rapidly with major developments across infrastructure and applications. Companies are investing heavily in AI capabilities while navigating an increasingly competitive landscape.

Moving to financial markets, investors are watching key economic indicators closely. The markets are showing mixed signals as we head into the final weeks of the year.

From India, the economy continues to show resilience with strong growth numbers. The Reserve Bank is balancing growth support with inflation management.

That's your quick briefing for today. Stay informed, stay ahead."""

def generate_quote(random_personality):
    """Generate an inspirational quote from the given personality"""
    import re
    
    logger.info(f"Generating quote for personality: {random_personality}")
    
    # Use fast model for simple quote generation to avoid reasoning artifacts
    client = get_client(provider=LLM_PROVIDER, model_tier="fast")
    
    conversation = [
        {"role": "system", "content": f"You provide famous quotes from {random_personality}. Reply with ONLY the quote in quotation marks - no introduction, no explanation, just the quote."},
        {"role": "user", "content": f"Give me an inspirational quote from {random_personality}"}
    ]
    
    try:
        message = client.chat_completion(
            messages=conversation,
            max_tokens=250,
            temperature=0.8
        )
        
        logger.debug(f"Raw LLM response for quote (first 200 chars): {message[:200] if message else 'EMPTY'}")
        logger.debug(f"Full raw response length: {len(message) if message else 0}")
        
        if not message or len(message.strip()) == 0:
            logger.warning("LLM returned empty response for quote generation")
            return _get_fallback_quote(random_personality)
        
        cleaned = message.strip()
        
        # Find any quoted text
        quote_matches = re.findall(r'"([^"]+)"', cleaned)
        logger.debug(f"Found {len(quote_matches)} quoted segments in response")
        
        if quote_matches:
            # Get the longest quote (likely the actual quote, not a fragment)
            best_quote = max(quote_matches, key=len)
            logger.debug(f"Best quote candidate (length {len(best_quote)}): {best_quote[:100]}")
            # Make sure it's substantial
            if len(best_quote) > 15:
                result = f'"{best_quote}"'
                logger.info(f"Successfully extracted quote: {result[:80]}...")
                return result
        
        # If no quotes found, look for substantial content without meta-text
        lines = [l.strip() for l in cleaned.split('\n') if l.strip()]
        logger.debug(f"Searching {len(lines)} lines for usable content")
        
        for line in lines:
            line_lower = line.lower()
            # Skip lines with meta-text
            if any(skip in line_lower for skip in [
                "the user", "provide", "respond", "generate", "here is", "here's",
                "famous quote", "quote by", "quote from", "said:", "once said"
            ]):
                logger.debug(f"Skipping meta-text line: {line[:50]}")
                continue
            
            # If we found a substantial line, use it
            if len(line) > 20:
                # Add quotes if not present
                if not (line.startswith('"') and line.endswith('"')):
                    line = f'"{line}"'
                logger.info(f"Using line as quote: {line[:80]}...")
                return line
        
        # Last resort: use the whole response if it's reasonable
        if 20 < len(cleaned) < 300 and "user" not in cleaned.lower():
            if not (cleaned.startswith('"') and cleaned.endswith('"')):
                cleaned = f'"{cleaned}"'
            logger.info(f"Using cleaned response as quote: {cleaned[:80]}...")
            return cleaned
        
        logger.warning(f"Could not extract valid quote from response. Using fallback.")
        return _get_fallback_quote(random_personality)
        
    except Exception as e:
        logger.error(f"Error generating quote: {e}", exc_info=True)
        return _get_fallback_quote(random_personality)


def _get_fallback_quote(personality):
    """Return a fallback quote when generation fails"""
    fallback_quotes = {
        "default": '"The only way to do great work is to love what you do."',
        "Steve Jobs": '"Stay hungry, stay foolish."',
        "Albert Einstein": '"Imagination is more important than knowledge."',
        "Mahatma Gandhi": '"Be the change you wish to see in the world."',
        "Martin Luther King Jr.": '"Darkness cannot drive out darkness; only light can do that."',
        "Winston Churchill": '"Success is not final, failure is not fatal: it is the courage to continue that counts."',
        "Nelson Mandela": '"It always seems impossible until it is done."',
    }
    quote = fallback_quotes.get(personality, fallback_quotes["default"])
    logger.info(f"Using fallback quote for {personality}: {quote}")
    return quote

def get_weather():
    try:
        owm = OWM(pyowm_api_key)
        mgr = owm.weather_manager()
        weather = mgr.weather_at_id(5743413).weather  # North Plains, OR
        temp = weather.temperature('celsius')['temp']
        status = weather.detailed_status
        return temp, status
    except Exception as e:
        print(f"Error fetching weather: {e}")
        return "N/A", "Unknown"

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

    # If current quarter is not found (e.g. at very end/start of year), default to 4 or 1
    if current_quarter is None:
        current_quarter = 4
        start_of_quarter = earnings_dates[3]
        end_of_quarter = earnings_dates[4]
        # Adjust for year boundaries if needed, but for simplicity:
        if now < earnings_dates[0]:
            current_quarter = 1
            start_of_quarter = datetime(current_year-1, 10, 24) # Approx
            end_of_quarter = earnings_dates[0]

    # Days and progress in the current quarter
    days_in_quarter = (end_of_quarter - start_of_quarter).days
    days_completed_in_quarter = (now - start_of_quarter).days
    if days_in_quarter == 0: days_in_quarter = 1 # avoid zero division
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

def get_luxurious_css():
    """Shared CSS for the luxurious dark theme"""
    return """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,600;0,700;1,400&family=Inter:wght@300;400;500;600&display=swap');
            
            :root {
                --bg-color: #0F1115;
                --card-bg: rgba(30, 34, 42, 0.6);
                --card-border: rgba(255, 255, 255, 0.08);
                --text-primary: #EDEDED;
                --text-secondary: #9CA3AF;
                --accent-gold: #D4AF37;
                --accent-glow: rgba(212, 175, 55, 0.3);
                --progress-bg: rgba(255, 255, 255, 0.1);
                --font-display: 'Playfair Display', serif;
                --font-body: 'Inter', sans-serif;
            }

            body {
                margin: 0;
                padding: 0;
                background-color: #0F1115; /* Fallback */
                background-color: var(--bg-color);
                color: #EDEDED;
                font-family: 'Inter', sans-serif;
                -webkit-font-smoothing: antialiased;
            }

            .container {
                max-width: 600px;
                margin: 0 auto;
                padding: 24px;
            }

            /* Header */
            .header {
                text-align: center;
                padding: 40px 0;
                border-bottom: 1px solid rgba(255, 255, 255, 0.08);
                margin-bottom: 32px;
            }

            h1 {
                font-family: 'Playfair Display', serif;
                font-size: 32px;
                font-weight: 700;
                color: #D4AF37;
                margin: 0 0 8px 0;
                letter-spacing: 0.5px;
            }

            .subtitle {
                color: #9CA3AF;
                font-size: 14px;
                font-family: 'Inter', sans-serif;
            }

            .date-badge {
                display: inline-block;
                font-size: 13px;
                text-transform: uppercase;
                letter-spacing: 1.5px;
                color: #9CA3AF;
                border: 1px solid rgba(255, 255, 255, 0.08);
                padding: 6px 16px;
                border-radius: 50px;
                background: rgba(255,255,255,0.03);
                margin-top: 12px;
            }

            /* Cards */
            .card {
                background: rgba(30, 34, 42, 0.6);
                border: 1px solid rgba(255, 255, 255, 0.08);
                border-radius: 16px;
                padding: 24px;
                margin-bottom: 24px;
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
            }

            .card-header {
                display: flex;
                align-items: center;
                justify-content: space-between;
                margin-bottom: 20px;
                border-bottom: 1px solid rgba(255, 255, 255, 0.08);
                padding-bottom: 12px;
            }

            .card-title {
                font-family: 'Playfair Display', serif;
                font-size: 20px;
                color: #EDEDED;
                margin: 0;
            }
            
            .card-icon {
                margin-right: 8px;
            }

            /* Grid Layout for Stats */
            .stats-grid {
                display: flex;
                gap: 12px;
                margin-bottom: 24px;
            }
            
            .stat-card {
                flex: 1;
                background: rgba(30, 34, 42, 0.6);
                border: 1px solid rgba(255, 255, 255, 0.08);
                border-radius: 16px;
                padding: 16px;
                text-align: center;
            }

            /* Progress Bars */
            .progress-item {
                margin-bottom: 20px;
            }

            .progress-label {
                display: flex;
                justify-content: space-between;
                font-size: 14px;
                color: #9CA3AF;
                margin-bottom: 8px;
            }

            .progress-track {
                height: 6px;
                background: rgba(255, 255, 255, 0.1);
                border-radius: 3px;
                overflow: hidden;
            }

            .progress-fill {
                height: 100%;
                background: linear-gradient(90deg, #D4AF37, #F2D06B);
                border-radius: 3px;
                box-shadow: 0 0 10px rgba(212, 175, 55, 0.3);
            }

            .highlight {
                color: #D4AF37;
                font-weight: 600;
            }

            /* Quote */
            .quote-text {
                font-family: 'Playfair Display', serif;
                font-size: 22px;
                line-height: 1.5;
                color: #fff;
                margin-bottom: 16px;
                text-align: center;
                font-style: italic;
            }

            .quote-author {
                font-family: 'Inter', sans-serif;
                font-size: 14px;
                color: #D4AF37;
                font-weight: 500;
                text-align: center;
            }

            /* Lesson - Legacy styles kept for compatibility */
            .lesson-content {
                font-size: 15px;
                line-height: 1.7;
                color: #9CA3AF;
            }
            .lesson-paragraph {
                margin-bottom: 16px;
            }
            .historical-note {
                border-left: 2px solid #D4AF37;
                padding-left: 12px;
                margin: 16px 0;
                color: #EDEDED;
                background: rgba(212, 175, 55, 0.03);
                padding: 12px;
                border-radius: 0 8px 8px 0;
            }

            /* Wisdom Grid - Structured Daily Wisdom */
            .wisdom-grid {
                display: flex;
                flex-direction: column;
                gap: 16px;
            }

            .wisdom-section {
                padding: 16px;
                border-radius: 12px;
                background: rgba(255, 255, 255, 0.02);
            }

            .wisdom-label {
                font-family: 'Inter', sans-serif;
                font-size: 10px;
                font-weight: 600;
                letter-spacing: 1.5px;
                text-transform: uppercase;
                margin-bottom: 8px;
                color: #6B7280;
            }

            .wisdom-text {
                font-family: 'Inter', sans-serif;
                font-size: 14px;
                line-height: 1.6;
                color: #9CA3AF;
            }

            /* Key Insight - Most prominent */
            .wisdom-insight {
                background: rgba(212, 175, 55, 0.08);
                border: 1px solid rgba(212, 175, 55, 0.2);
            }

            .wisdom-insight .wisdom-label {
                color: #D4AF37;
            }

            .wisdom-insight .wisdom-text {
                font-family: 'Playfair Display', serif;
                font-size: 17px;
                font-weight: 500;
                color: #EDEDED;
                line-height: 1.5;
            }

            /* Historical Context - Subtle callout */
            .wisdom-historical {
                border-left: 2px solid #D4AF37;
                background: rgba(30, 34, 42, 0.4);
                padding-left: 16px;
            }

            .wisdom-historical .wisdom-label {
                color: #D4AF37;
                opacity: 0.8;
            }

            .wisdom-historical .wisdom-text {
                font-style: italic;
                font-size: 13px;
                color: #9CA3AF;
            }

            /* Application - Clean and direct */
            .wisdom-application {
                background: transparent;
                border: 1px dashed rgba(255, 255, 255, 0.1);
            }

            .wisdom-application .wisdom-text {
                font-size: 13px;
                color: #9CA3AF;
            }

            /* News Items */
            .news-item {
                padding-bottom: 24px;
                margin-bottom: 24px;
                border-bottom: 1px solid rgba(255, 255, 255, 0.08);
            }

            .news-item:last-child {
                border-bottom: none;
                margin-bottom: 0;
                padding-bottom: 0;
            }

            .news-source {
                font-size: 11px;
                text-transform: uppercase;
                letter-spacing: 1px;
                color: #D4AF37;
                margin-bottom: 6px;
                display: block;
                font-weight: 600;
            }

            .news-headline {
                font-size: 18px;
                font-family: 'Playfair Display', serif;
                font-weight: 600;
                line-height: 1.4;
                color: #EDEDED;
                margin: 0 0 10px 0;
            }

            .news-headline a {
                color: inherit;
                text-decoration: none;
                border-bottom: 1px dotted rgba(255,255,255,0.3);
            }

            .news-commentary {
                font-size: 14px;
                color: #9CA3AF;
                line-height: 1.6;
                background: rgba(255,255,255,0.03);
                padding: 16px;
                border-radius: 8px;
                border-left: 2px solid #D4AF37;
            }
            
            .news-citation {
                font-size: 11px;
                color: #6B7280;
                margin-right: 6px;
            }

            footer {
                text-align: center;
                padding: 40px 0;
                color: #6B7280;
                font-size: 12px;
                border-top: 1px solid rgba(255, 255, 255, 0.08);
                margin-top: 40px;
            }
        </style>
    """

def generate_html_progress_message(days_completed, weeks_completed, days_left, weeks_left, percent_days_left):
    temp, status = get_weather()
    now = datetime.now()
    date_time = now.strftime("%B %d, %Y")
    current_year = now.year
    
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
            
    if current_quarter is None:
        current_quarter = 4
        start_of_quarter = earnings_dates[3]
        end_of_quarter = earnings_dates[4]

    days_in_quarter = (end_of_quarter - start_of_quarter).days
    days_completed_in_quarter = (now - start_of_quarter).days
    if days_in_quarter == 0: days_in_quarter = 1
    percent_days_left_in_quarter = ((days_in_quarter - days_completed_in_quarter) / days_in_quarter) * 100
    q_progress = 100 - percent_days_left_in_quarter
    year_progress = 100 - percent_days_left

    html_template = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Year Progress</title>
        {get_luxurious_css()}
    </head>
    <body style="background-color: #0F1115; color: #EDEDED;">
        <div class="container">
            <div class="header">
                <h1>Year Progress</h1>
                <div class="date-badge">{date_time}</div>
            </div>

            <!-- Quick Stats -->
            <div class="stats-grid">
                <div class="stat-card">
                    <div style="font-size: 24px; margin-bottom: 4px;">🌤️</div>
                    <div style="font-size: 14px; color: #9CA3AF;">{temp}°C, {status}</div>
                </div>
                <div class="stat-card">
                    <div style="font-size: 24px; margin-bottom: 4px;">📅</div>
                    <div style="font-size: 14px; color: #9CA3AF;">Day {days_completed}</div>
                </div>
            </div>

            <!-- Progress Section -->
            <div class="card">
                <div class="card-header">
                    <h2 class="card-title">Temporal Status</h2>
                    <span style="color: #D4AF37; font-size: 12px;">{year_progress:.1f}% Complete</span>
                </div>
                
                <div class="progress-item">
                    <div class="progress-label">
                        <span>Year {current_year}</span>
                        <span>{days_completed} / {365 if current_year % 4 != 0 else 366} Days</span>
                    </div>
                    <div class="progress-track">
                        <div class="progress-fill" style="width: {year_progress}%"></div>
                    </div>
                    <div style="text-align: right; font-size: 12px; color: #6B7280; margin-top: 4px;">
                        {weeks_left:.1f} weeks remaining
                    </div>
                </div>

                <div class="progress-item">
                    <div class="progress-label">
                        <span>Quarter {current_quarter}</span>
                        <span>{days_completed_in_quarter} / {days_in_quarter} Days</span>
                    </div>
                    <div class="progress-track">
                        <div class="progress-fill" style="width: {q_progress}%"></div>
                    </div>
                </div>
            </div>

            <div class="card" id="quote-card">
                <div class="card-header" style="justify-content: center; border-bottom: none; padding-bottom: 0;">
                    <div style="font-size: 24px; color: #D4AF37;">❝</div>
                </div>
                <div style="padding: 0 20px 20px 20px;">
                    <div class="quote-text" id="quote-text"></div>
                    <div class="quote-author" id="quote-author"></div>
                </div>
            </div>

            <div class="card" id="lesson-card">
                <div class="card-header">
                    <h2 class="card-title">Daily Wisdom</h2>
                </div>
                <div class="lesson-content" id="lesson-content"></div>
            </div>

            <footer>
                 Generated by EDITH • {current_year}
            </footer>
        </div>
    </body>
    </html>
    """
    return html_template

def generate_html_news_template(news_content):
    now = datetime.now()
    
    html_template = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Daily Briefing</title>
        {get_luxurious_css()}
    </head>
    <body style="background-color: #0F1115; color: #EDEDED;">
        <div class="container">
            <div class="header">
                <h1>Daily Briefing</h1>
                <div class="date-badge">{now.strftime("%B %d, %Y")}</div>
            </div>
            
            <div class="card">
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

            <div class="card">
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

            <div class="card">
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
            
            <footer>
                 Generated by EDITH • {now.year}
            </footer>
        </div>
    </body>
    </html>
    """
    return html_template

def format_news_section(content, section_title):
    """
    Format news section for HTML display using the luxurious design
    """
    sections = content.split("##")
    section_content = ""
    
    for section in sections:
        if section_title.lower() in section.lower():
            section_content = section
            break
    
    formatted_items = []
    
    if section_content:
        lines = [line.strip() for line in section_content.split("\n") if line.strip()]
        
        points = []
        for line in lines:
            if line.startswith("- "):
                points.append(line[2:])
            elif not line.lower().endswith(("update:", "update")):
                if len(line) > 20 and not line.startswith("#"):
                    points.append(line)
        
        for i, point in enumerate(points[:5], 1):
            news_text = point
            url = ""
            date = ""
            citation = ""
            commentary = ""
            
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
            
            if news_text.startswith("[") and "]" in news_text:
                end_bracket = news_text.index("]")
                citation = news_text[1:end_bracket]
                news_text = news_text[end_bracket+1:].strip()
            
            # Format using new CSS classes
            headlines = news_text
            if url:
                headlines = f'<a href="{url}" target="_blank">{news_text}</a>'
                
            item_html = f'''
                <div class="news-item">
                    <span class="news-source">{citation if citation else "News Update"}</span>
                    <h3 class="news-headline">{headlines}</h3>
                    {f'<div class="news-commentary">{commentary}</div>' if commentary else ''}
                </div>
            '''
            formatted_items.append(item_html)
            
    if not formatted_items:
        return '<div style="padding: 20px; color: #9CA3AF;">No updates available for this section.</div>'
        
    return "\n".join(formatted_items)

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
    logger.info("="*80)
    logger.info("STARTING YEAR PROGRESS AND NEWS REPORTER")
    logger.info("="*80)
    
    days_completed, weeks_completed, days_left, weeks_left, percent_days_left = time_left_in_year()
    logger.info(f"Year progress: {100 - percent_days_left:.1f}% complete, {days_left} days remaining")
    
    # Get quote and lesson
    logger.info("Generating quote and lesson...")
    random_personality = get_random_personality()
    logger.info(f"Selected personality: {random_personality}")
    
    quote = generate_quote(random_personality)
    logger.info(f"Quote result: {'SUCCESS' if quote else 'EMPTY'} - {quote[:80] if quote else 'N/A'}...")
    
    lesson_learned = get_random_lesson()
    logger.info(f"Lesson result: {'SUCCESS' if lesson_learned else 'EMPTY'} - {len(lesson_learned) if lesson_learned else 0} chars")
    
    # Generate HTML progress report
    year_progress_html = generate_html_progress_message(days_completed, weeks_completed, days_left, weeks_left, percent_days_left)
    
    # Parse structured lesson content with section markers
    formatted_lesson = '<div class="wisdom-grid">'
    
    # Extract sections using markers
    key_insight_match = re.search(r'\[KEY INSIGHT\]\s*(.*?)(?=\[HISTORICAL\]|\[APPLICATION\]|$)', lesson_learned, re.DOTALL | re.IGNORECASE)
    historical_match = re.search(r'\[HISTORICAL\]\s*(.*?)(?=\[KEY INSIGHT\]|\[APPLICATION\]|$)', lesson_learned, re.DOTALL | re.IGNORECASE)
    application_match = re.search(r'\[APPLICATION\]\s*(.*?)(?=\[KEY INSIGHT\]|\[HISTORICAL\]|$)', lesson_learned, re.DOTALL | re.IGNORECASE)
    
    # Key Insight section - with fallback if marker not found
    insight_text = None
    if key_insight_match and key_insight_match.group(1).strip():
        insight_text = key_insight_match.group(1).strip()
    else:
        # Fallback: check for text before the first marker
        first_marker_pos = len(lesson_learned)
        for marker in [r'\[HISTORICAL\]', r'\[APPLICATION\]']:
            match = re.search(marker, lesson_learned, re.IGNORECASE)
            if match and match.start() < first_marker_pos:
                first_marker_pos = match.start()
        
        if first_marker_pos > 0:
            pre_text = lesson_learned[:first_marker_pos].strip()
            if len(pre_text) > 30:
                insight_text = pre_text
                logger.info("Using pre-marker text as KEY INSIGHT fallback")
        
        # If still no insight, generate one from the first sentence of historical
        if not insight_text and historical_match and historical_match.group(1).strip():
            hist_text = historical_match.group(1).strip()
            # Extract a summary principle from historical context
            insight_text = "Timeless principles reveal themselves through historical patterns - understanding the past illuminates the path forward."
            logger.info("Using default KEY INSIGHT as fallback")
    
    if insight_text:
        formatted_lesson += f'''
        <div class="wisdom-section wisdom-insight">
            <div class="wisdom-label">KEY INSIGHT</div>
            <div class="wisdom-text">{insight_text}</div>
        </div>'''
    
    # Historical section
    if historical_match and historical_match.group(1).strip():
        historical_text = historical_match.group(1).strip()
        formatted_lesson += f'''
        <div class="wisdom-section wisdom-historical">
            <div class="wisdom-label">HISTORICAL CONTEXT</div>
            <div class="wisdom-text">{historical_text}</div>
        </div>'''
    
    # Application section
    if application_match and application_match.group(1).strip():
        application_text = application_match.group(1).strip()
        formatted_lesson += f'''
        <div class="wisdom-section wisdom-application">
            <div class="wisdom-label">APPLICATION</div>
            <div class="wisdom-text">{application_text}</div>
        </div>'''
    
    # Fallback if no markers found (legacy format)
    if not (key_insight_match or historical_match or application_match):
        logger.warning("No section markers found in lesson, using fallback formatting")
        paragraphs = lesson_learned.split('\n\n')
        for i, para in enumerate(paragraphs[:3]):  # Limit to 3 paragraphs
            if para.strip():
                section_class = ['wisdom-insight', 'wisdom-historical', 'wisdom-application'][min(i, 2)]
                section_label = ['KEY INSIGHT', 'HISTORICAL CONTEXT', 'APPLICATION'][min(i, 2)]
                formatted_lesson += f'''
        <div class="wisdom-section {section_class}">
            <div class="wisdom-label">{section_label}</div>
            <div class="wisdom-text">{para.strip()}</div>
        </div>'''
    
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
    
    # Select a random model name from the configured list (mapped to voices in audio_processors)
    # Since we are using LITELLM_SMART/STRATEGIC which might not be in VOICE_MAP, 
    # audio_processors.py will fallback to DEFAULT voice.
    model_name = random.choice(model_names)
    text_to_speech_nospeak(year_progress_gpt_response, yearprogress_tts_output_path, model_name=model_name)

    # News updates with different formats for the newsletter and voicebot
    news_update_subject = "📰 Your Daily News Briefing"
    
    # Use the first model from our list for consistent news generation
    news_model_name = model_names[0]
    
    print("\n" + "="*80)
    print("FETCHING NEWS UPDATES")
    print("="*80)
    
    # Determine provider from model name
    if LLM_PROVIDER == "litellm":
        news_provider = "litellm"
    else:
        news_provider = "ollama"
    
    print("\n📰 Fetching Technology News...")
    news_update_tech = gather_daily_news(
        category="technology",
        max_sources=10,  # Increased from 5 for better diversity
        aggregator_limit=1,
        freshness_hours=24,
        provider=news_provider
    )
    save_message_to_file(news_update_tech, "news_tech_report.txt")
    print(f"✓ Tech news: {len(news_update_tech)} characters")
    
    print("\n📈 Fetching Financial Markets News...")
    news_update_usa = gather_daily_news(
        category="financial",
        max_sources=10,  # Increased from 5 for better diversity
        aggregator_limit=1,
        freshness_hours=24,
        provider=news_provider
    )
    save_message_to_file(news_update_usa, "news_usa_report.txt")
    print(f"✓ Financial news: {len(news_update_usa)} characters")
    
    print("\n🇮🇳 Fetching India News...")
    news_update_india = gather_daily_news(
        category="india",
        max_sources=10,  # Increased from 5 for better diversity
        aggregator_limit=0,  # No aggregator for India news
        freshness_hours=24,
        provider=news_provider
    )
    save_message_to_file(news_update_india, "news_india_report.txt")
    print(f"✓ India news: {len(news_update_india)} characters")

    print("\n" + "="*80)
    print("GENERATING NEWSLETTER")
    print("="*80)

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
       - [Source] Headline and brief description | Date: MM/DD/YYYY | Actual URL from source | Commentary: [Concise summary of implications, analysis, or context - 3-5 substantive sentences]
    3. Use this format if no URL is available:
       - [Source] Headline and brief description | Date: MM/DD/YYYY | Commentary: [Concise summary of implications, analysis, or context - 3-5 substantive sentences]
    4. Only include URLs that are explicitly mentioned in the source text
    5. Keep the original source URLs intact - do not modify them
    6. The commentary section must provide meaningful analysis, context, or implications - not just a restatement of the headline

    Please format the content into these three sections:
    ## Tech News Update:
    (5 points from tech news using format above)

    ## Financial Markets News Update:
    (5 points from financial news using format above)

    ## India News Update:
    (5 points from India news using format above)
    '''

    print("\n📝 Generating newsletter format...")
    news_newsletter_response = generate_gpt_response_newsletter(newsletter_updates)
    print(f"✓ Newsletter generated: {len(news_newsletter_response)} characters")

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

    print("\n🎙️ Generating voicebot format...")
    news_voicebot_response = generate_gpt_response_voicebot(voicebot_updates)
    print(f"✓ Voicebot script generated: {len(news_voicebot_response)} characters")
    
    # Save news responses to files
    print("\n💾 Saving newsletter files...")
    save_message_to_file(news_newsletter_response, "news_newsletter_report.txt")
    save_message_to_file(news_voicebot_response, "news_voicebot_report.txt")
    
    # Generate HTML newsletter
    print("\n🎨 Generating HTML newsletter...")
    news_html = generate_html_news_template(news_newsletter_response)
    save_message_to_file(news_html, "news_newsletter_report.html")
    print("✓ HTML newsletter saved")
    
    # Send HTML newsletter
    print("\n📧 Sending newsletter email...")
    send_email(news_update_subject, news_html, is_html=True)
    
    # Convert the voicebot response to speech
    print("\n🔊 Converting news to speech...")
    news_tts_output_path = "news_update_report.mp3"
    model_name = random.choice(model_names)
    text_to_speech_nospeak(news_voicebot_response, news_tts_output_path, model_name=model_name)
    print(f"✓ News audio saved to {news_tts_output_path}")
    
    print("\n" + "="*80)
    print("✓ ALL TASKS COMPLETED SUCCESSFULLY!")
    print("="*80)
