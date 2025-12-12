import smtplib
import logging
import re
import json
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path
from typing import Dict, List, Any, Optional
from pyowm import OWM
from config import config
from helper_functions.chat_generation_with_internet import internet_connected_chatbot
from helper_functions.audio_processors import text_to_speech_nospeak
from helper_functions.llm_client import get_client
from helper_functions.news_researcher import gather_daily_news
import random
import os
import argparse

# ============================================================================
# OUTPUT DIRECTORY AND SCHEMA VERSION
# ============================================================================
OUTPUT_DIR = Path("newsletter_research_data")
BUNDLE_SCHEMA_VERSION = "1.0.0"

# ============================================================================
# CACHING CONFIGURATION
# ============================================================================
NEWS_CACHE_FILE = OUTPUT_DIR / "news_cache.json"
CACHE_MAX_AGE_HOURS = 6  # Cache expires after 6 hours

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

# ============================================================================
# BUNDLE JSON HELPERS
# ============================================================================

def compute_quarter_progress() -> Dict[str, Any]:
    """
    Compute current fiscal quarter progress based on NVIDIA earnings dates.
    Returns dict with quarter number, days completed/left, and percent complete.
    """
    now = datetime.now()
    current_year = now.year
    
    # NVIDIA earnings-based quarter boundaries
    earnings_dates = [
        datetime(current_year, 1, 23),      # Q1 end, Q2 start
        datetime(current_year, 4, 25),      # Q2 end, Q3 start
        datetime(current_year, 7, 29),      # Q3 end, Q4 start
        datetime(current_year, 10, 24),     # Q4 end, Q1 start
        datetime(current_year + 1, 1, 23)   # Next Q1
    ]
    
    current_quarter = None
    start_of_quarter = None
    end_of_quarter = None
    
    for i in range(len(earnings_dates) - 1):
        if earnings_dates[i] <= now < earnings_dates[i + 1]:
            current_quarter = i + 1
            start_of_quarter = earnings_dates[i]
            end_of_quarter = earnings_dates[i + 1]
            break
    
    # Handle edge cases at year boundaries
    if current_quarter is None:
        current_quarter = 4
        start_of_quarter = earnings_dates[3]
        end_of_quarter = earnings_dates[4]
        if now < earnings_dates[0]:
            current_quarter = 1
            start_of_quarter = datetime(current_year - 1, 10, 24)
            end_of_quarter = earnings_dates[0]
    
    days_in_quarter = (end_of_quarter - start_of_quarter).days
    days_completed_in_quarter = (now - start_of_quarter).days
    if days_in_quarter == 0:
        days_in_quarter = 1
    days_left_in_quarter = days_in_quarter - days_completed_in_quarter
    percent_complete = ((days_completed_in_quarter) / days_in_quarter) * 100
    
    return {
        "current_quarter": current_quarter,
        "days_in_quarter": days_in_quarter,
        "days_completed_in_quarter": days_completed_in_quarter,
        "days_left_in_quarter": days_left_in_quarter,
        "percent_complete": round(percent_complete, 2)
    }


def parse_lesson_to_dict(lesson_text: str, topic: str = "") -> Dict[str, Any]:
    """
    Parse lesson text with [KEY INSIGHT], [HISTORICAL], [APPLICATION] markers
    into a structured dictionary.
    """
    result = {
        "topic": topic,
        "key_insight": "",
        "historical": "",
        "application": "",
        "raw_text": lesson_text
    }
    
    if not lesson_text:
        return result
    
    # Extract sections using markers
    key_insight_match = re.search(
        r'\[KEY INSIGHT\]\s*(.*?)(?=\[HISTORICAL\]|\[APPLICATION\]|$)',
        lesson_text, re.DOTALL | re.IGNORECASE
    )
    historical_match = re.search(
        r'\[HISTORICAL\]\s*(.*?)(?=\[KEY INSIGHT\]|\[APPLICATION\]|$)',
        lesson_text, re.DOTALL | re.IGNORECASE
    )
    application_match = re.search(
        r'\[APPLICATION\]\s*(.*?)(?=\[KEY INSIGHT\]|\[HISTORICAL\]|$)',
        lesson_text, re.DOTALL | re.IGNORECASE
    )
    
    if key_insight_match and key_insight_match.group(1).strip():
        result["key_insight"] = key_insight_match.group(1).strip()
    else:
        # Fallback: check for text before the first marker
        first_marker_pos = len(lesson_text)
        for marker in [r'\[HISTORICAL\]', r'\[APPLICATION\]']:
            match = re.search(marker, lesson_text, re.IGNORECASE)
            if match and match.start() < first_marker_pos:
                first_marker_pos = match.start()
        if first_marker_pos > 0:
            pre_text = lesson_text[:first_marker_pos].strip()
            if len(pre_text) > 30:
                result["key_insight"] = pre_text
        # If still no insight, use a default
        if not result["key_insight"]:
            result["key_insight"] = "Timeless principles reveal themselves through historical patterns - understanding the past illuminates the path forward."
    
    if historical_match and historical_match.group(1).strip():
        result["historical"] = historical_match.group(1).strip()
    
    if application_match and application_match.group(1).strip():
        result["application"] = application_match.group(1).strip()
    
    return result


def build_daily_bundle(
    days_completed: int,
    weeks_completed: float,
    days_left: int,
    weeks_left: float,
    percent_days_left: float,
    weather_data: Dict[str, Any],
    quote_text: str,
    quote_author: str,
    lesson_dict: Dict[str, Any],
    news_raw_sources: Dict[str, str],
    newsletter_sections: Dict[str, List[Dict[str, Any]]],
    voicebot_script: str
) -> Dict[str, Any]:
    """
    Build the complete daily bundle dictionary conforming to the schema.
    """
    now = datetime.now()
    current_year = now.year
    total_days_in_year = 366 if (current_year % 4 == 0 and current_year % 100 != 0) or (current_year % 400 == 0) else 365
    
    quarter_data = compute_quarter_progress()
    
    bundle = {
        "meta": {
            "schema_version": BUNDLE_SCHEMA_VERSION,
            "generated_at_iso": now.isoformat(),
            "date_iso": now.strftime("%Y-%m-%d"),
            "date_formatted": now.strftime("%B %d, %Y"),
            "day_of_week": now.strftime("%A"),
            "llm_provider": LLM_PROVIDER,
            "model_tiers_used": model_names
        },
        "progress": {
            "time": {
                "year": current_year,
                "total_days_in_year": total_days_in_year,
                "days_completed": days_completed,
                "days_left": days_left,
                "weeks_completed": round(weeks_completed, 2),
                "weeks_left": round(weeks_left, 2),
                "percent_complete": round(100 - percent_days_left, 2)
            },
            "quarter": quarter_data,
            "weather": weather_data,
            "quote": {
                "text": quote_text,
                "author": quote_author
            },
            "lesson": lesson_dict
        },
        "news": {
            "raw_sources": news_raw_sources,
            "newsletter": {
                "sections": newsletter_sections
            },
            "voicebot": {
                "script": voicebot_script
            }
        }
    }
    
    return bundle


def write_bundle_json(bundle: Dict[str, Any], output_dir: Path = OUTPUT_DIR) -> Path:
    """
    Write the bundle to both a dated file and a 'latest' file.
    Returns the path to the dated file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    date_str = bundle["meta"]["date_iso"]
    dated_filename = f"daily_bundle_{date_str}.json"
    latest_filename = "daily_bundle_latest.json"
    
    dated_path = output_dir / dated_filename
    latest_path = output_dir / latest_filename
    
    # Write both files
    for path in [dated_path, latest_path]:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(bundle, f, indent=2, ensure_ascii=False)
        logger.info(f"Bundle JSON written to {path}")
    
    return dated_path


def load_bundle_json(path: Path) -> Dict[str, Any]:
    """
    Load a bundle JSON from disk.
    """
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


# ============================================================================
# NEWS CACHING FUNCTIONS
# ============================================================================

def save_news_cache(news_data: Dict[str, str], cache_file: Path = NEWS_CACHE_FILE) -> bool:
    """
    Save news data to cache with timestamp.
    Returns True if successful.
    """
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    
    cache_entry = {
        "timestamp": datetime.now().isoformat(),
        "date_iso": datetime.now().strftime("%Y-%m-%d"),
        "news": news_data
    }
    
    try:
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache_entry, f, indent=2, ensure_ascii=False)
        logger.info(f"News cache saved to {cache_file}")
        return True
    except Exception as e:
        logger.error(f"Failed to save news cache: {e}")
        return False


def load_news_cache(cache_file: Path = NEWS_CACHE_FILE, max_age_hours: int = CACHE_MAX_AGE_HOURS) -> Optional[Dict[str, str]]:
    """
    Load news data from cache if available and fresh.
    Returns None if cache is missing, expired, or invalid.
    """
    if not cache_file.exists():
        logger.info("No news cache found")
        return None
    
    try:
        with open(cache_file, 'r', encoding='utf-8') as f:
            cache_entry = json.load(f)
        
        # Check timestamp
        cache_time = datetime.fromisoformat(cache_entry["timestamp"])
        age_hours = (datetime.now() - cache_time).total_seconds() / 3600
        
        if age_hours > max_age_hours:
            logger.info(f"News cache expired ({age_hours:.1f} hours old, max is {max_age_hours})")
            return None
        
        # Check if same day (for daily news)
        if cache_entry.get("date_iso") != datetime.now().strftime("%Y-%m-%d"):
            logger.info("News cache is from a different day")
            return None
        
        logger.info(f"Loaded news cache ({age_hours:.1f} hours old)")
        return cache_entry["news"]
        
    except Exception as e:
        logger.warning(f"Failed to load news cache: {e}")
        return None


def is_cache_valid(cache_file: Path = NEWS_CACHE_FILE, max_age_hours: int = CACHE_MAX_AGE_HOURS) -> bool:
    """
    Check if cache exists and is still valid.
    """
    return load_news_cache(cache_file, max_age_hours) is not None


def get_cache_info(cache_file: Path = NEWS_CACHE_FILE) -> Optional[Dict[str, Any]]:
    """
    Get information about the cache without loading full data.
    """
    if not cache_file.exists():
        return None
    
    try:
        with open(cache_file, 'r', encoding='utf-8') as f:
            cache_entry = json.load(f)
        
        cache_time = datetime.fromisoformat(cache_entry["timestamp"])
        age_hours = (datetime.now() - cache_time).total_seconds() / 3600
        
        return {
            "exists": True,
            "timestamp": cache_entry["timestamp"],
            "date_iso": cache_entry.get("date_iso"),
            "age_hours": round(age_hours, 2),
            "has_tech": "technology" in cache_entry.get("news", {}),
            "has_financial": "financial" in cache_entry.get("news", {}),
            "has_india": "india" in cache_entry.get("news", {}),
        }
    except Exception as e:
        return {"exists": True, "error": str(e)}


def parse_newsletter_item(item_text: str) -> Dict[str, Any]:
    """
    Parse a single newsletter item from text format to structured dict.
    Expected format: "[Source] Headline | Date: MM/DD/YYYY | URL | Commentary: text"
    """
    result = {
        "source": "",
        "headline": "",
        "date_mmddyyyy": "",
        "url": "",
        "commentary": ""
    }
    
    if not item_text:
        return result
    
    # Remove leading "- " if present
    item_text = item_text.strip()
    if item_text.startswith("- "):
        item_text = item_text[2:]
    
    # Extract source from beginning [Source]
    if item_text.startswith("[") and "]" in item_text:
        end_bracket = item_text.index("]")
        result["source"] = item_text[1:end_bracket]
        item_text = item_text[end_bracket + 1:].strip()
    
    # Split by pipe separator
    if " | " in item_text:
        components = item_text.split(" | ")
        result["headline"] = components[0].strip()
        
        for component in components[1:]:
            component = component.strip()
            if component.startswith("Date:"):
                result["date_mmddyyyy"] = component.replace("Date:", "").strip()
            elif component.startswith("http"):
                result["url"] = component
            elif component.startswith("Commentary:"):
                result["commentary"] = component.replace("Commentary:", "").strip()
    else:
        result["headline"] = item_text
    
    return result


def parse_newsletter_text_to_sections(newsletter_text: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Parse the full newsletter text into structured sections dict.
    Returns dict with keys: tech, financial, india
    Each value is a list of item dicts.
    """
    sections = {
        "tech": [],
        "financial": [],
        "india": []
    }
    
    if not newsletter_text:
        return sections
    
    # Map section headers to keys
    section_map = {
        "tech news update": "tech",
        "financial markets news update": "financial",
        "india news update": "india"
    }
    
    # Split by ## headers
    parts = newsletter_text.split("##")
    
    for part in parts:
        part = part.strip()
        if not part:
            continue
        
        # Find which section this belongs to
        section_key = None
        for header, key in section_map.items():
            if header in part.lower():
                section_key = key
                break
        
        if not section_key:
            continue
        
        # Extract items (lines starting with -)
        lines = part.split("\n")
        for line in lines:
            line = line.strip()
            if line.startswith("- "):
                item = parse_newsletter_item(line)
                if item["headline"]:  # Only add if we got a headline
                    sections[section_key].append(item)
    
    return sections


def generate_fallback_newsletter_sections() -> Dict[str, List[Dict[str, Any]]]:
    """
    Generate fallback newsletter sections when LLM generation fails.
    """
    today = datetime.now().strftime("%m/%d/%Y")
    
    fallback_item = {
        "source": "Update",
        "headline": "News update temporarily unavailable. Check back later for the latest updates.",
        "date_mmddyyyy": today,
        "url": "",
        "commentary": "We're working to bring you the latest news. Please check back soon."
    }
    
    return {
        "tech": [fallback_item],
        "financial": [fallback_item],
        "india": [fallback_item]
    }


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

def get_random_lesson() -> tuple:
    """
    Generate a comprehensive daily lesson using LLM's knowledge.
    No database required - leverages the model's training data.
    
    Returns:
        tuple: (topic, lesson_text) - the topic and the generated lesson
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
    return topic, lesson_learned

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

def generate_gpt_response_newsletter(user_message: str) -> str:
    """
    Generate newsletter-formatted news content with error handling.
    Returns formatted text content or empty string if generation fails.
    """
    try:
        client = get_client(provider=LLM_PROVIDER, model_tier="smart")
        
        syspromptmessage = """You are EDITH, or "Even Dead, I'm The Hero," a world-class AI assistant designed by Tony Stark. For the newsletter format, follow these rules strictly:

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
            logger.warning("Newsletter generation returned minimal content")
            return ""
            
        return message
        
    except Exception as e:
        logger.error(f"Error generating newsletter: {e}")
        return ""


def generate_newsletter_sections(
    news_tech: str,
    news_financial: str,
    news_india: str
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Generate newsletter sections as structured data.
    Uses LLM generation with parsing, falling back to structured fallback if needed.
    
    Returns dict with keys: tech, financial, india
    Each value is a list of news item dicts.
    """
    today = datetime.now().strftime("%B %d, %Y")
    
    newsletter_prompt = f'''
    Generate a concise news summary for {today}, using ONLY real news and URLs from the provided source material below. Do not create placeholder or fake URLs.

    Use the source text below to create your summary:

    Tech News Update:
    {news_tech}

    Financial Markets News Update:
    {news_financial}

    India News Update:
    {news_india}

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
    
    # Generate text-based newsletter
    newsletter_text = generate_gpt_response_newsletter(newsletter_prompt)
    
    if not newsletter_text:
        logger.warning("Newsletter generation failed, using fallback sections")
        return generate_fallback_newsletter_sections()
    
    # Parse text to structured sections
    sections = parse_newsletter_text_to_sections(newsletter_text)
    
    # Validate that we have content in each section
    has_content = all(len(sections[key]) > 0 for key in ["tech", "financial", "india"])
    
    if not has_content:
        logger.warning("Parsed newsletter sections incomplete, using fallback")
        return generate_fallback_newsletter_sections()
    
    logger.info(f"Successfully generated newsletter sections: tech={len(sections['tech'])}, financial={len(sections['financial'])}, india={len(sections['india'])}")
    return sections

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
    """
    Ultra-premium, mobile-first CSS for the newsletter.
    Follows the Architecture Spec: 8px spacing scale, 44px tap targets,
    dark mode support, and fluid responsive design.
    """
    return """
        <style>
            /* Google Fonts with system fallbacks */
            @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,600;0,700;1,400&family=Inter:wght@300;400;500;600&display=swap');

            /* ============================================
               DESIGN TOKENS - 8px Spacing Scale
               ============================================ */
            :root {
                /* Spacing Scale (8px base) */
                --space-xs: 8px;
                --space-sm: 16px;
                --space-md: 24px;
                --space-lg: 32px;
                --space-xl: 48px;
                --space-2xl: 64px;

                /* Colors - Dark Theme */
                --bg-primary: #0A0B0D;
                --bg-secondary: #12141A;
                --bg-card: rgba(22, 25, 32, 0.85);
                --bg-card-hover: rgba(28, 32, 42, 0.9);
                --border-subtle: rgba(255, 255, 255, 0.06);
                --border-medium: rgba(255, 255, 255, 0.1);

                /* Text Colors */
                --text-primary: #F5F5F7;
                --text-secondary: #A1A1AA;
                --text-muted: #6B7280;

                /* Accent Colors */
                --accent-gold: #D4AF37;
                --accent-gold-light: #E8C959;
                --accent-gold-dark: #B8962E;
                --accent-glow: rgba(212, 175, 55, 0.15);
                --accent-glow-strong: rgba(212, 175, 55, 0.3);

                /* Typography */
                --font-display: 'Playfair Display', Georgia, 'Times New Roman', serif;
                --font-body: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;

                /* Sizing */
                --tap-target-min: 44px;
                --border-radius-sm: 8px;
                --border-radius-md: 12px;
                --border-radius-lg: 16px;
                --border-radius-xl: 24px;

                /* Shadows */
                --shadow-sm: 0 2px 8px rgba(0, 0, 0, 0.15);
                --shadow-md: 0 4px 16px rgba(0, 0, 0, 0.2);
                --shadow-lg: 0 8px 32px rgba(0, 0, 0, 0.25);
                --shadow-glow: 0 0 20px var(--accent-glow);

                /* Transitions */
                --transition-fast: 150ms ease;
                --transition-base: 250ms ease;
                --transition-slow: 400ms ease;
            }

            /* ============================================
               RESET & BASE STYLES (Mobile-First)
               ============================================ */
            *, *::before, *::after {
                box-sizing: border-box;
            }

            body {
                margin: 0;
                padding: 0;
                background: var(--bg-primary);
                background-image:
                    radial-gradient(ellipse at top, rgba(212, 175, 55, 0.03) 0%, transparent 50%),
                    linear-gradient(180deg, var(--bg-primary) 0%, var(--bg-secondary) 100%);
                background-attachment: fixed;
                color: var(--text-primary);
                font-family: var(--font-body);
                font-size: 16px;
                line-height: 1.6;
                -webkit-font-smoothing: antialiased;
                -moz-osx-font-smoothing: grayscale;
                min-height: 100vh;
            }

            /* ============================================
               CONTAINER - Fluid with Max Width
               ============================================ */
            .container {
                width: 100%;
                max-width: 640px;
                margin: 0 auto;
                padding: var(--space-sm);
            }

            /* ============================================
               HEADER
               ============================================ */
            .header {
                text-align: center;
                padding: var(--space-lg) 0 var(--space-md);
                margin-bottom: var(--space-md);
                position: relative;
            }

            .header::after {
                content: '';
                position: absolute;
                bottom: 0;
                left: 50%;
                transform: translateX(-50%);
                width: 60px;
                height: 2px;
                background: linear-gradient(90deg, transparent, var(--accent-gold), transparent);
            }

            h1 {
                font-family: var(--font-display);
                font-size: 28px;
                font-weight: 700;
                color: var(--accent-gold);
                margin: 0 0 var(--space-xs) 0;
                letter-spacing: 0.5px;
                text-shadow: 0 2px 20px var(--accent-glow);
            }

            .subtitle {
                color: var(--text-secondary);
                font-size: 14px;
                font-family: var(--font-body);
                font-weight: 400;
            }

            .date-badge {
                display: inline-flex;
                align-items: center;
                justify-content: center;
                min-height: var(--tap-target-min);
                font-size: 11px;
                text-transform: uppercase;
                letter-spacing: 2px;
                color: var(--text-secondary);
                border: 1px solid var(--border-subtle);
                padding: var(--space-xs) var(--space-md);
                border-radius: 100px;
                background: rgba(255, 255, 255, 0.02);
                margin-top: var(--space-sm);
                transition: all var(--transition-base);
            }

            /* ============================================
               THEME TOGGLE (Browser Preview Only)
               NOTE: Most email clients strip <script> and/or block buttons.
               This is safe to include (no-op in email), and works in browsers.
               ============================================ */
            .theme-toggle {
                position: absolute;
                top: var(--space-sm);
                right: var(--space-sm);
                width: var(--tap-target-min);
                height: var(--tap-target-min);
                display: inline-flex;
                align-items: center;
                justify-content: center;
                border-radius: 999px;
                border: 1px solid var(--border-medium);
                background: rgba(255, 255, 255, 0.03);
                color: var(--text-primary);
                cursor: pointer;
                font-size: 16px;
                line-height: 1;
                user-select: none;
                -webkit-tap-highlight-color: transparent;
                transition: transform var(--transition-fast), background var(--transition-fast);
            }

            .theme-toggle:hover {
                background: rgba(255, 255, 255, 0.06);
            }

            .theme-toggle:active {
                transform: scale(0.98);
            }

            /* ============================================
               CARDS - Glass Morphism Effect
               ============================================ */
            .card {
                background: var(--bg-card);
                backdrop-filter: blur(20px);
                -webkit-backdrop-filter: blur(20px);
                border: 1px solid var(--border-subtle);
                border-radius: var(--border-radius-lg);
                padding: var(--space-md);
                margin-bottom: var(--space-md);
                box-shadow: var(--shadow-md);
                transition: all var(--transition-base);
                animation: fadeInUp 0.5s ease forwards;
                opacity: 0;
            }

            .card:nth-child(1) { animation-delay: 0.1s; }
            .card:nth-child(2) { animation-delay: 0.2s; }
            .card:nth-child(3) { animation-delay: 0.3s; }
            .card:nth-child(4) { animation-delay: 0.4s; }
            .card:nth-child(5) { animation-delay: 0.5s; }

            @keyframes fadeInUp {
                from {
                    opacity: 0;
                    transform: translateY(20px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }

            .card-header {
                display: flex;
                align-items: center;
                justify-content: space-between;
                flex-wrap: wrap;
                gap: var(--space-xs);
                margin-bottom: var(--space-sm);
                padding-bottom: var(--space-sm);
                border-bottom: 1px solid var(--border-subtle);
            }

            .card-title {
                font-family: var(--font-display);
                font-size: 20px;
                font-weight: 600;
                color: var(--text-primary);
                margin: 0;
                display: flex;
                align-items: center;
                gap: var(--space-xs);
            }

            .card-icon {
                font-size: 20px;
            }

            .card-meta {
                font-size: 12px;
                color: var(--accent-gold);
                font-weight: 500;
            }

            /* ============================================
               STATS GRID - Mobile Stack, Desktop Row
               ============================================ */
            .stats-grid {
                display: grid;
                grid-template-columns: 1fr;
                gap: var(--space-sm);
                margin-bottom: var(--space-md);
            }

            .stat-card {
                background: var(--bg-card);
                border: 1px solid var(--border-subtle);
                border-radius: var(--border-radius-md);
                padding: var(--space-sm);
                text-align: center;
                transition: all var(--transition-base);
            }

            .stat-card:active {
                transform: scale(0.98);
            }

            .stat-icon {
                font-size: 28px;
                margin-bottom: var(--space-xs);
                display: block;
            }

            .stat-value {
                font-size: 14px;
                color: var(--text-secondary);
                line-height: 1.4;
            }

            /* ============================================
               PROGRESS BARS - Animated
               ============================================ */
            .progress-item {
                margin-bottom: var(--space-md);
            }

            .progress-item:last-child {
                margin-bottom: 0;
            }

            .progress-label {
                display: flex;
                justify-content: space-between;
                align-items: center;
                flex-wrap: wrap;
                gap: var(--space-xs);
                font-size: 14px;
                color: var(--text-secondary);
                margin-bottom: var(--space-xs);
            }

            .progress-label strong {
                color: var(--text-primary);
                font-weight: 500;
            }

            .progress-track {
                height: 8px;
                background: rgba(255, 255, 255, 0.08);
                border-radius: 100px;
                overflow: hidden;
                position: relative;
            }

            .progress-fill {
                height: 100%;
                background: linear-gradient(90deg, var(--accent-gold-dark), var(--accent-gold), var(--accent-gold-light));
                border-radius: 100px;
                position: relative;
                animation: progressGrow 1.5s ease forwards;
                transform-origin: left;
            }

            @keyframes progressGrow {
                from { transform: scaleX(0); }
                to { transform: scaleX(1); }
            }

            .progress-fill::after {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
                animation: shimmer 2s infinite;
            }

            @keyframes shimmer {
                0% { transform: translateX(-100%); }
                100% { transform: translateX(100%); }
            }

            .progress-meta {
                text-align: right;
                font-size: 12px;
                color: var(--text-muted);
                margin-top: 6px;
            }

            .highlight {
                color: var(--accent-gold);
                font-weight: 600;
            }

            /* ============================================
               QUOTE SECTION - Premium Typography
               ============================================ */
            .quote-card {
                position: relative;
                padding: var(--space-lg) var(--space-md);
                text-align: center;
            }

            .quote-icon {
                font-size: 48px;
                color: var(--accent-gold);
                opacity: 0.6;
                margin-bottom: var(--space-sm);
                text-shadow: 0 4px 20px var(--accent-glow);
            }

            .quote-text {
                font-family: var(--font-display);
                font-size: 20px;
                line-height: 1.6;
                color: var(--text-primary);
                margin: 0 0 var(--space-sm) 0;
                font-style: italic;
                font-weight: 400;
            }

            .quote-author {
                font-family: var(--font-body);
                font-size: 13px;
                color: var(--accent-gold);
                font-weight: 500;
                letter-spacing: 0.5px;
            }

            /* ============================================
               WISDOM GRID - Daily Insights
               ============================================ */
            .wisdom-grid {
                display: flex;
                flex-direction: column;
                gap: var(--space-sm);
            }

            .wisdom-section {
                padding: var(--space-sm);
                border-radius: var(--border-radius-md);
                transition: all var(--transition-base);
            }

            .wisdom-label {
                font-family: var(--font-body);
                font-size: 10px;
                font-weight: 600;
                letter-spacing: 1.5px;
                text-transform: uppercase;
                margin-bottom: var(--space-xs);
                color: var(--text-muted);
            }

            .wisdom-text {
                font-family: var(--font-body);
                font-size: 14px;
                line-height: 1.7;
                color: var(--text-secondary);
                margin: 0;
            }

            /* Key Insight - Hero treatment */
            .wisdom-insight {
                background: linear-gradient(135deg, rgba(212, 175, 55, 0.1) 0%, rgba(212, 175, 55, 0.05) 100%);
                border: 1px solid rgba(212, 175, 55, 0.2);
                padding: var(--space-md);
            }

            .wisdom-insight .wisdom-label {
                color: var(--accent-gold);
            }

            .wisdom-insight .wisdom-text {
                font-family: var(--font-display);
                font-size: 17px;
                font-weight: 500;
                color: var(--text-primary);
                line-height: 1.6;
            }

            /* Historical Context */
            .wisdom-historical {
                border-left: 3px solid var(--accent-gold);
                background: rgba(255, 255, 255, 0.02);
                padding-left: var(--space-sm);
                margin-left: var(--space-xs);
            }

            .wisdom-historical .wisdom-label {
                color: var(--accent-gold);
                opacity: 0.8;
            }

            .wisdom-historical .wisdom-text {
                font-style: italic;
                font-size: 14px;
            }

            /* Application */
            .wisdom-application {
                background: transparent;
                border: 1px dashed var(--border-medium);
            }

            .wisdom-application .wisdom-text {
                font-size: 14px;
            }

            /* ============================================
               NEWS ITEMS - Touch Optimized
               ============================================ */
            .news-item {
                padding: var(--space-md) 0;
                border-bottom: 1px solid var(--border-subtle);
            }

            .news-item:first-child {
                padding-top: 0;
            }

            .news-item:last-child {
                border-bottom: none;
                padding-bottom: 0;
            }

            .news-source {
                display: inline-block;
                font-size: 10px;
                text-transform: uppercase;
                letter-spacing: 1.5px;
                color: var(--accent-gold);
                font-weight: 600;
                margin-bottom: var(--space-xs);
                padding: 4px 8px;
                background: var(--accent-glow);
                border-radius: 4px;
            }

            .news-headline {
                font-size: 17px;
                font-family: var(--font-display);
                font-weight: 600;
                line-height: 1.4;
                color: var(--text-primary);
                margin: 0 0 var(--space-sm) 0;
            }

            .news-headline a {
                color: inherit;
                text-decoration: none;
                display: block;
                min-height: var(--tap-target-min);
                padding: var(--space-xs) 0;
                transition: color var(--transition-fast);
            }

            .news-headline a:hover,
            .news-headline a:focus {
                color: var(--accent-gold);
            }

            .news-headline a:active {
                opacity: 0.8;
            }

            .news-commentary {
                font-size: 14px;
                color: var(--text-secondary);
                line-height: 1.7;
                background: rgba(255, 255, 255, 0.02);
                padding: var(--space-sm);
                border-radius: var(--border-radius-sm);
                border-left: 3px solid var(--accent-gold);
                margin-top: var(--space-sm);
            }

            /* Bulletproof CTA Button - Compact size */
            .news-cta {
                display: inline-flex;
                align-items: center;
                justify-content: center;
                height: 32px;
                padding: 0 var(--space-sm);
                margin-top: var(--space-xs);
                font-family: var(--font-body);
                font-size: 12px;
                font-weight: 500;
                letter-spacing: 0.3px;
                color: var(--bg-primary);
                background: linear-gradient(135deg, var(--accent-gold) 0%, var(--accent-gold-light) 100%);
                border: none;
                border-radius: 6px;
                text-decoration: none;
                cursor: pointer;
                transition: all var(--transition-base);
                box-shadow: 0 2px 8px var(--accent-glow);
            }

            .news-cta:hover,
            .news-cta:focus {
                transform: translateY(-1px);
                box-shadow: 0 4px 12px var(--accent-glow-strong);
            }

            .news-cta:active {
                transform: translateY(0);
            }

            /* ============================================
               FOOTER
               ============================================ */
            footer {
                text-align: center;
                padding: var(--space-lg) var(--space-sm);
                color: var(--text-muted);
                font-size: 12px;
                border-top: 1px solid var(--border-subtle);
                margin-top: var(--space-lg);
            }

            footer a {
                color: var(--accent-gold);
                text-decoration: none;
            }

            /* ============================================
               RESPONSIVE - Tablet & Desktop
               ============================================ */
            @media screen and (min-width: 480px) {
                .container {
                    padding: var(--space-md);
                }

                h1 {
                    font-size: 32px;
                }

                .stats-grid {
                    grid-template-columns: repeat(2, 1fr);
                }

                .card {
                    padding: var(--space-lg);
                }

                .quote-text {
                    font-size: 22px;
                }

                .wisdom-insight .wisdom-text {
                    font-size: 18px;
                }

                .news-headline {
                    font-size: 18px;
                }
            }

            @media screen and (min-width: 768px) {
                .container {
                    padding: var(--space-lg);
                }

                h1 {
                    font-size: 36px;
                }

                .header {
                    padding: var(--space-xl) 0 var(--space-lg);
                }

                .quote-text {
                    font-size: 24px;
                    padding: 0 var(--space-md);
                }

                .card {
                    border-radius: var(--border-radius-xl);
                }
            }

            /* ============================================
               DARK MODE SUPPORT (System Preference)
               ============================================ */
            @media (prefers-color-scheme: light) {
                :root {
                    --bg-primary: #FAFAFA;
                    --bg-secondary: #F5F5F5;
                    --bg-card: rgba(255, 255, 255, 0.9);
                    --bg-card-hover: rgba(255, 255, 255, 0.95);
                    --border-subtle: rgba(0, 0, 0, 0.06);
                    --border-medium: rgba(0, 0, 0, 0.1);
                    --text-primary: #1A1A1A;
                    --text-secondary: #4A4A4A;
                    --text-muted: #8A8A8A;
                    --accent-gold: #B8962E;
                    --accent-gold-light: #D4AF37;
                    --accent-gold-dark: #8B7023;
                    --shadow-sm: 0 2px 8px rgba(0, 0, 0, 0.08);
                    --shadow-md: 0 4px 16px rgba(0, 0, 0, 0.1);
                    --shadow-lg: 0 8px 32px rgba(0, 0, 0, 0.12);
                }

                body {
                    background-image:
                        radial-gradient(ellipse at top, rgba(184, 150, 46, 0.05) 0%, transparent 50%),
                        linear-gradient(180deg, var(--bg-primary) 0%, var(--bg-secondary) 100%);
                }

                .news-cta {
                    color: #FFFFFF;
                }
            }

            /* ============================================
               MANUAL THEME OVERRIDE (data-theme)
               data-theme always wins over system preference.
               This enables a browser toggle, while remaining safe for email.
               ============================================ */
            :root[data-theme="dark"] {
                --bg-primary: #0A0B0D;
                --bg-secondary: #12141A;
                --bg-card: rgba(22, 25, 32, 0.85);
                --bg-card-hover: rgba(28, 32, 42, 0.9);
                --border-subtle: rgba(255, 255, 255, 0.06);
                --border-medium: rgba(255, 255, 255, 0.1);
                --text-primary: #F5F5F7;
                --text-secondary: #A1A1AA;
                --text-muted: #6B7280;
                --accent-gold: #D4AF37;
                --accent-gold-light: #E8C959;
                --accent-gold-dark: #B8962E;
                --shadow-sm: 0 2px 8px rgba(0, 0, 0, 0.15);
                --shadow-md: 0 4px 16px rgba(0, 0, 0, 0.2);
                --shadow-lg: 0 8px 32px rgba(0, 0, 0, 0.25);
            }

            :root[data-theme="dark"] body {
                background-image:
                    radial-gradient(ellipse at top, rgba(212, 175, 55, 0.03) 0%, transparent 50%),
                    linear-gradient(180deg, var(--bg-primary) 0%, var(--bg-secondary) 100%);
            }

            :root[data-theme="light"] {
                --bg-primary: #FAFAFA;
                --bg-secondary: #F5F5F5;
                --bg-card: rgba(255, 255, 255, 0.9);
                --bg-card-hover: rgba(255, 255, 255, 0.95);
                --border-subtle: rgba(0, 0, 0, 0.06);
                --border-medium: rgba(0, 0, 0, 0.1);
                --text-primary: #1A1A1A;
                --text-secondary: #4A4A4A;
                --text-muted: #8A8A8A;
                --accent-gold: #B8962E;
                --accent-gold-light: #D4AF37;
                --accent-gold-dark: #8B7023;
                --shadow-sm: 0 2px 8px rgba(0, 0, 0, 0.08);
                --shadow-md: 0 4px 16px rgba(0, 0, 0, 0.1);
                --shadow-lg: 0 8px 32px rgba(0, 0, 0, 0.12);
            }

            :root[data-theme="light"] body {
                background-image:
                    radial-gradient(ellipse at top, rgba(184, 150, 46, 0.05) 0%, transparent 50%),
                    linear-gradient(180deg, var(--bg-primary) 0%, var(--bg-secondary) 100%);
            }

            :root[data-theme="light"] .news-cta {
                color: #FFFFFF;
            }

            /* ============================================
               ACCESSIBILITY & FOCUS STATES
               ============================================ */
            @media (prefers-reduced-motion: reduce) {
                *, *::before, *::after {
                    animation-duration: 0.01ms !important;
                    animation-iteration-count: 1 !important;
                    transition-duration: 0.01ms !important;
                }
            }

            a:focus-visible,
            button:focus-visible,
            .news-cta:focus-visible {
                outline: 2px solid var(--accent-gold);
                outline-offset: 2px;
            }

            /* High contrast mode support */
            @media (prefers-contrast: high) {
                :root {
                    --border-subtle: rgba(255, 255, 255, 0.2);
                    --border-medium: rgba(255, 255, 255, 0.3);
                }
            }

            /* ============================================
               LEGACY SUPPORT
               ============================================ */
            .lesson-content {
                font-size: 15px;
                line-height: 1.7;
                color: var(--text-secondary);
            }

            .lesson-paragraph {
                margin-bottom: var(--space-sm);
            }

            .historical-note {
                border-left: 3px solid var(--accent-gold);
                padding: var(--space-sm);
                margin: var(--space-sm) 0;
                color: var(--text-primary);
                background: var(--accent-glow);
                border-radius: 0 var(--border-radius-sm) var(--border-radius-sm) 0;
            }
        </style>
    """


def get_theme_toggle_script():
    """
    Returns a tiny, dependency-free theme toggle script.

    Email reality check:
    - Most email clients strip <script> entirely, and some strip <button>.
    - This is therefore "best effort" and primarily benefits browser previews / hosted web views.
    - The newsletter still supports automatic dark/light via prefers-color-scheme without this script.
    """
    return """
        <script>
            (function () {
                var STORAGE_KEY = "edith_newsletter_theme";
                var root = document.documentElement;
                var btn = document.getElementById("theme-toggle");
                if (!root || !btn) return;

                function systemTheme() {
                    try {
                        return window.matchMedia &&
                            window.matchMedia("(prefers-color-scheme: light)").matches
                            ? "light"
                            : "dark";
                    } catch (e) {
                        return "dark";
                    }
                }

                function currentTheme() {
                    return root.getAttribute("data-theme") || systemTheme();
                }

                function updateButton(theme) {
                    var isDark = theme === "dark";
                    btn.textContent = isDark ? "☾" : "☀";
                    btn.setAttribute("aria-pressed", isDark ? "true" : "false");
                    var title = isDark ? "Switch to light mode" : "Switch to dark mode";
                    btn.setAttribute("title", title);
                    btn.setAttribute("aria-label", title);
                }

                function setTheme(theme, persist) {
                    if (theme !== "light" && theme !== "dark") return;
                    root.setAttribute("data-theme", theme);
                    updateButton(theme);
                    if (persist) {
                        try { localStorage.setItem(STORAGE_KEY, theme); } catch (e) {}
                    }
                }

                function clearThemeOverride() {
                    root.removeAttribute("data-theme");
                    try { localStorage.removeItem(STORAGE_KEY); } catch (e) {}
                    updateButton(systemTheme());
                }

                // Initialize: honor stored override if present, otherwise just match system for button state.
                var stored = null;
                try { stored = localStorage.getItem(STORAGE_KEY); } catch (e) {}

                if (stored === "light" || stored === "dark") {
                    setTheme(stored, false);
                } else {
                    updateButton(systemTheme());
                }

                // Click to toggle; Shift/Alt-click to reset back to system preference.
                btn.addEventListener("click", function (e) {
                    if (e && (e.shiftKey || e.altKey)) {
                        clearThemeOverride();
                        return;
                    }
                    var next = currentTheme() === "dark" ? "light" : "dark";
                    setTheme(next, true);
                });
            })();
        </script>
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

    # Calculate total days in year for accurate display
    total_days_in_year = 366 if (current_year % 4 == 0 and current_year % 100 != 0) or (current_year % 400 == 0) else 365

    html_template = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=5.0">
        <meta name="theme-color" content="#0A0B0D">
        <meta name="color-scheme" content="dark light">
        <title>Year Progress Report</title>
        {get_luxurious_css()}
    </head>
    <body>
        <div class="container">
            <!-- Header Section -->
            <header class="header">
                <!-- Theme toggle works in browser previews / hosted pages.
                     Most email clients strip scripts, so this is best-effort. -->
                <button class="theme-toggle" id="theme-toggle" type="button" aria-label="Toggle theme" aria-pressed="false" title="Toggle theme">☾</button>
                <h1>Year Progress</h1>
                <p class="subtitle">Your daily temporal briefing</p>
                <div class="date-badge">{date_time}</div>
            </header>

            <!-- Quick Stats Grid -->
            <section class="stats-grid" aria-label="Quick statistics">
                <div class="stat-card">
                    <span class="stat-icon" aria-hidden="true">🌤️</span>
                    <div class="stat-value">
                        <strong>{temp}°C</strong><br>
                        {status}
                    </div>
                </div>
                <div class="stat-card">
                    <span class="stat-icon" aria-hidden="true">📅</span>
                    <div class="stat-value">
                        <strong>Day {days_completed}</strong><br>
                        of {total_days_in_year}
                    </div>
                </div>
            </section>

            <!-- Progress Section -->
            <article class="card" aria-label="Year and quarter progress">
                <div class="card-header">
                    <h2 class="card-title">
                        <span class="card-icon" aria-hidden="true">⏱️</span>
                        Temporal Status
                    </h2>
                    <span class="card-meta">{year_progress:.1f}% Complete</span>
                </div>

                <div class="progress-item">
                    <div class="progress-label">
                        <strong>Year {current_year}</strong>
                        <span>{days_completed} / {total_days_in_year} Days</span>
                    </div>
                    <div class="progress-track" role="progressbar" aria-valuenow="{year_progress:.0f}" aria-valuemin="0" aria-valuemax="100">
                        <div class="progress-fill" style="width: {year_progress}%"></div>
                    </div>
                    <div class="progress-meta">
                        <span class="highlight">{weeks_left:.1f}</span> weeks remaining
                    </div>
                </div>

                <div class="progress-item">
                    <div class="progress-label">
                        <strong>Q{current_quarter}</strong>
                        <span>{days_completed_in_quarter} / {days_in_quarter} Days</span>
                    </div>
                    <div class="progress-track" role="progressbar" aria-valuenow="{q_progress:.0f}" aria-valuemin="0" aria-valuemax="100">
                        <div class="progress-fill" style="width: {q_progress}%"></div>
                    </div>
                    <div class="progress-meta">
                        <span class="highlight">{days_in_quarter - days_completed_in_quarter}</span> days until Q{current_quarter + 1 if current_quarter < 4 else 1}
                    </div>
                </div>
            </article>

            <!-- Quote Card -->
            <article class="card quote-card" id="quote-card" aria-label="Quote of the day">
                <div class="quote-icon" aria-hidden="true">❝</div>
                <blockquote>
                    <p class="quote-text" id="quote-text"></p>
                    <cite class="quote-author" id="quote-author"></cite>
                </blockquote>
            </article>

            <!-- Daily Wisdom Card -->
            <article class="card" id="lesson-card" aria-label="Daily wisdom">
                <div class="card-header">
                    <h2 class="card-title">
                        <span class="card-icon" aria-hidden="true">💡</span>
                        Daily Wisdom
                    </h2>
                </div>
                <div class="lesson-content" id="lesson-content"></div>
            </article>

            <!-- Footer -->
            <footer>
                <p>Generated by <strong>EDITH</strong> • {current_year}</p>
                <p style="margin-top: 8px; font-size: 11px;">Even Dead, I'm The Hero</p>
            </footer>
        </div>
        {get_theme_toggle_script()}
    </body>
    </html>
    """
    return html_template

def generate_html_news_template(news_content):
    """Generate premium mobile-first HTML newsletter for daily news briefing."""
    now = datetime.now()
    date_formatted = now.strftime("%B %d, %Y")
    day_of_week = now.strftime("%A")

    html_template = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=5.0">
        <meta name="theme-color" content="#0A0B0D">
        <meta name="color-scheme" content="dark light">
        <title>Daily Briefing - {date_formatted}</title>
        {get_luxurious_css()}
    </head>
    <body>
        <div class="container">
            <!-- Header Section -->
            <header class="header">
                <!-- Theme toggle works in browser previews / hosted pages.
                     Most email clients strip scripts, so this is best-effort. -->
                <button class="theme-toggle" id="theme-toggle" type="button" aria-label="Toggle theme" aria-pressed="false" title="Toggle theme">☾</button>
                <h1>Daily Briefing</h1>
                <p class="subtitle">{day_of_week}'s Essential Updates</p>
                <div class="date-badge">{date_formatted}</div>
            </header>

            <!-- Technology Section -->
            <article class="card" aria-label="Technology news">
                <div class="card-header">
                    <h2 class="card-title">
                        <span class="card-icon" aria-hidden="true">💻</span>
                        Technology
                    </h2>
                    <span class="card-meta">Tech & AI</span>
                </div>
                <div class="card-content">
                    {format_news_section(news_content, "Tech News Update")}
                </div>
            </article>

            <!-- Financial Markets Section -->
            <article class="card" aria-label="Financial markets news">
                <div class="card-header">
                    <h2 class="card-title">
                        <span class="card-icon" aria-hidden="true">📈</span>
                        Markets
                    </h2>
                    <span class="card-meta">Finance</span>
                </div>
                <div class="card-content">
                    {format_news_section(news_content, "Financial Markets News Update")}
                </div>
            </article>

            <!-- India News Section -->
            <article class="card" aria-label="India news">
                <div class="card-header">
                    <h2 class="card-title">
                        <span class="card-icon" aria-hidden="true" style="display: inline-flex; align-items: center;">
                            <svg width="20" height="14" viewBox="0 0 20 14" style="border-radius: 2px; box-shadow: 0 1px 2px rgba(0,0,0,0.2);">
                                <rect width="20" height="4.67" fill="#FF9933"/>
                                <rect y="4.67" width="20" height="4.67" fill="#FFFFFF"/>
                                <rect y="9.33" width="20" height="4.67" fill="#138808"/>
                                <circle cx="10" cy="7" r="1.8" fill="#000080"/>
                            </svg>
                        </span>
                        India
                    </h2>
                    <span class="card-meta">Regional</span>
                </div>
                <div class="card-content">
                    {format_news_section(news_content, "India News Update")}
                </div>
            </article>

            <!-- Footer -->
            <footer>
                <p>Generated by <strong>EDITH</strong> • {now.year}</p>
                <p style="margin-top: 8px; font-size: 11px;">Even Dead, I'm The Hero</p>
            </footer>
        </div>
        {get_theme_toggle_script()}
    </body>
    </html>
    """
    return html_template

def format_news_section(content, section_title):
    """
    Format news section for HTML display using the premium mobile-first design.
    Features:
    - Touch-optimized tap targets (44px minimum)
    - Bulletproof CTA buttons
    - Semantic HTML structure
    - Accessible link text
    """
    import html as html_module  # For escaping user content

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

        for point in points[:5]:
            news_text = point
            url = ""
            news_date = ""
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
                        news_date = component.replace("Date:", "").strip()
                    elif component.startswith("Commentary:"):
                        commentary = component.replace("Commentary:", "").strip()

            # Extract citation from beginning of news_text if present
            if news_text.startswith("[") and "]" in news_text:
                end_bracket = news_text.index("]")
                citation = news_text[1:end_bracket]
                news_text = news_text[end_bracket + 1:].strip()

            # Sanitize text content for HTML
            safe_news_text = html_module.escape(news_text) if news_text else ""
            safe_citation = html_module.escape(citation) if citation else "News Update"
            safe_commentary = html_module.escape(commentary) if commentary else ""

            # Build headline with proper touch target
            if url:
                # Link with bulletproof tap target
                headline_html = f'''
                    <h3 class="news-headline">
                        <a href="{html_module.escape(url)}" target="_blank" rel="noopener noreferrer">
                            {safe_news_text}
                        </a>
                    </h3>'''
            else:
                headline_html = f'<h3 class="news-headline">{safe_news_text}</h3>'

            # Build date display if available
            date_html = ""
            if news_date:
                date_html = f'<time class="news-date" style="display: block; font-size: 12px; color: var(--text-muted); margin-bottom: 8px;">{html_module.escape(news_date)}</time>'

            # Build commentary section
            commentary_html = ""
            if safe_commentary:
                commentary_html = f'<div class="news-commentary">{safe_commentary}</div>'

            # Build bulletproof CTA button if URL exists
            cta_html = ""
            if url:
                cta_html = f'''
                    <a href="{html_module.escape(url)}"
                       target="_blank"
                       rel="noopener noreferrer"
                       class="news-cta"
                       aria-label="Read full article about {safe_news_text[:50]}...">
                        Read More →
                    </a>'''

            # Assemble complete news item
            item_html = f'''
                <article class="news-item">
                    <span class="news-source">{safe_citation}</span>
                    {date_html}
                    {headline_html}
                    {commentary_html}
                    {cta_html}
                </article>
            '''
            formatted_items.append(item_html)

    if not formatted_items:
        return '''
            <div style="padding: var(--space-md); text-align: center; color: var(--text-muted);">
                <p style="margin: 0;">No updates available for this section.</p>
                <p style="margin-top: 8px; font-size: 12px;">Check back later for the latest news.</p>
            </div>
        '''

    return "\n".join(formatted_items)


# ============================================================================
# BUNDLE-BASED HTML RENDERING FUNCTIONS
# ============================================================================

def format_news_items_html(items: List[Dict[str, Any]]) -> str:
    """
    Format a list of structured news items to HTML.
    Each item should have: source, headline, date_mmddyyyy, url (optional), commentary
    """
    import html as html_module
    
    if not items:
        return '''
            <div style="padding: var(--space-md); text-align: center; color: var(--text-muted);">
                <p style="margin: 0;">No updates available for this section.</p>
                <p style="margin-top: 8px; font-size: 12px;">Check back later for the latest news.</p>
            </div>
        '''
    
    formatted_items = []
    
    for item in items[:5]:  # Limit to 5 items
        source = html_module.escape(item.get("source", "Update"))
        headline = html_module.escape(item.get("headline", ""))
        date_str = html_module.escape(item.get("date_mmddyyyy", ""))
        url = item.get("url", "")
        commentary = html_module.escape(item.get("commentary", ""))
        
        if not headline:
            continue
        
        # Build headline with proper touch target
        if url:
            headline_html = f'''
                <h3 class="news-headline">
                    <a href="{html_module.escape(url)}" target="_blank" rel="noopener noreferrer">
                        {headline}
                    </a>
                </h3>'''
        else:
            headline_html = f'<h3 class="news-headline">{headline}</h3>'
        
        # Build date display if available
        date_html = ""
        if date_str:
            date_html = f'<time class="news-date" style="display: block; font-size: 12px; color: var(--text-muted); margin-bottom: 8px;">{date_str}</time>'
        
        # Build commentary section
        commentary_html = ""
        if commentary:
            commentary_html = f'<div class="news-commentary">{commentary}</div>'
        
        # Build bulletproof CTA button if URL exists
        cta_html = ""
        if url:
            cta_html = f'''
                <a href="{html_module.escape(url)}"
                   target="_blank"
                   rel="noopener noreferrer"
                   class="news-cta"
                   aria-label="Read full article about {headline[:50]}...">
                    Read More →
                </a>'''
        
        item_html = f'''
            <article class="news-item">
                <span class="news-source">{source}</span>
                {date_html}
                {headline_html}
                {commentary_html}
                {cta_html}
            </article>
        '''
        formatted_items.append(item_html)
    
    return "\n".join(formatted_items) if formatted_items else '''
        <div style="padding: var(--space-md); text-align: center; color: var(--text-muted);">
            <p style="margin: 0;">No updates available for this section.</p>
        </div>
    '''


def render_lesson_html(lesson: Dict[str, Any]) -> str:
    """
    Render lesson content from structured dict to HTML.
    """
    import html as html_module
    
    html_parts = ['<div class="wisdom-grid">']
    
    # Key Insight section
    key_insight = lesson.get("key_insight", "")
    if key_insight:
        safe_insight = html_module.escape(key_insight)
        html_parts.append(f'''
        <div class="wisdom-section wisdom-insight">
            <div class="wisdom-label">KEY INSIGHT</div>
            <div class="wisdom-text">{safe_insight}</div>
        </div>''')
    
    # Historical section
    historical = lesson.get("historical", "")
    if historical:
        safe_historical = html_module.escape(historical)
        html_parts.append(f'''
        <div class="wisdom-section wisdom-historical">
            <div class="wisdom-label">HISTORICAL CONTEXT</div>
            <div class="wisdom-text">{safe_historical}</div>
        </div>''')
    
    # Application section
    application = lesson.get("application", "")
    if application:
        safe_application = html_module.escape(application)
        html_parts.append(f'''
        <div class="wisdom-section wisdom-application">
            <div class="wisdom-label">APPLICATION</div>
            <div class="wisdom-text">{safe_application}</div>
        </div>''')
    
    html_parts.append('</div>')
    return '\n'.join(html_parts)


def render_year_progress_html_from_bundle(bundle: Dict[str, Any]) -> str:
    """
    Render the year progress HTML from bundle data.
    No placeholder replacement - data is directly embedded.
    """
    import html as html_module
    
    meta = bundle["meta"]
    progress = bundle["progress"]
    time_data = progress["time"]
    quarter_data = progress["quarter"]
    weather = progress["weather"]
    quote = progress["quote"]
    lesson = progress["lesson"]
    
    date_formatted = meta["date_formatted"]
    current_year = time_data["year"]
    days_completed = time_data["days_completed"]
    total_days_in_year = time_data["total_days_in_year"]
    year_progress = time_data["percent_complete"]
    weeks_left = time_data["weeks_left"]
    
    current_quarter = quarter_data["current_quarter"]
    days_in_quarter = quarter_data["days_in_quarter"]
    days_completed_in_quarter = quarter_data["days_completed_in_quarter"]
    days_left_in_quarter = quarter_data["days_left_in_quarter"]
    q_progress = quarter_data["percent_complete"]
    
    temp = weather.get("temp_c", "N/A")
    status = html_module.escape(weather.get("status", "Unknown"))
    
    quote_text = html_module.escape(quote.get("text", ""))
    quote_author = html_module.escape(quote.get("author", ""))
    
    lesson_html = render_lesson_html(lesson)
    
    html_template = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=5.0">
        <meta name="theme-color" content="#0A0B0D">
        <meta name="color-scheme" content="dark light">
        <title>Year Progress Report</title>
        {get_luxurious_css()}
    </head>
    <body>
        <div class="container">
            <!-- Header Section -->
            <header class="header">
                <button class="theme-toggle" id="theme-toggle" type="button" aria-label="Toggle theme" aria-pressed="false" title="Toggle theme">☾</button>
                <h1>Year Progress</h1>
                <p class="subtitle">Your daily temporal briefing</p>
                <div class="date-badge">{date_formatted}</div>
            </header>

            <!-- Quick Stats Grid -->
            <section class="stats-grid" aria-label="Quick statistics">
                <div class="stat-card">
                    <span class="stat-icon" aria-hidden="true">🌤️</span>
                    <div class="stat-value">
                        <strong>{temp}°C</strong><br>
                        {status}
                    </div>
                </div>
                <div class="stat-card">
                    <span class="stat-icon" aria-hidden="true">📅</span>
                    <div class="stat-value">
                        <strong>Day {days_completed}</strong><br>
                        of {total_days_in_year}
                    </div>
                </div>
            </section>

            <!-- Progress Section -->
            <article class="card" aria-label="Year and quarter progress">
                <div class="card-header">
                    <h2 class="card-title">
                        <span class="card-icon" aria-hidden="true">⏱️</span>
                        Temporal Status
                    </h2>
                    <span class="card-meta">{year_progress:.1f}% Complete</span>
                </div>

                <div class="progress-item">
                    <div class="progress-label">
                        <strong>Year {current_year}</strong>
                        <span>{days_completed} / {total_days_in_year} Days</span>
                    </div>
                    <div class="progress-track" role="progressbar" aria-valuenow="{year_progress:.0f}" aria-valuemin="0" aria-valuemax="100">
                        <div class="progress-fill" style="width: {year_progress}%"></div>
                    </div>
                    <div class="progress-meta">
                        <span class="highlight">{weeks_left:.1f}</span> weeks remaining
                    </div>
                </div>

                <div class="progress-item">
                    <div class="progress-label">
                        <strong>Q{current_quarter}</strong>
                        <span>{days_completed_in_quarter} / {days_in_quarter} Days</span>
                    </div>
                    <div class="progress-track" role="progressbar" aria-valuenow="{q_progress:.0f}" aria-valuemin="0" aria-valuemax="100">
                        <div class="progress-fill" style="width: {q_progress}%"></div>
                    </div>
                    <div class="progress-meta">
                        <span class="highlight">{days_left_in_quarter}</span> days until Q{current_quarter + 1 if current_quarter < 4 else 1}
                    </div>
                </div>
            </article>

            <!-- Quote Card -->
            <article class="card quote-card" aria-label="Quote of the day">
                <div class="quote-icon" aria-hidden="true">❝</div>
                <blockquote>
                    <p class="quote-text">{quote_text}</p>
                    <cite class="quote-author">— {quote_author}</cite>
                </blockquote>
            </article>

            <!-- Daily Wisdom Card -->
            <article class="card" aria-label="Daily wisdom">
                <div class="card-header">
                    <h2 class="card-title">
                        <span class="card-icon" aria-hidden="true">💡</span>
                        Daily Wisdom
                    </h2>
                </div>
                <div class="lesson-content">{lesson_html}</div>
            </article>

            <!-- Footer -->
            <footer>
                <p>Generated by <strong>EDITH</strong> • {current_year}</p>
                <p style="margin-top: 8px; font-size: 11px;">Even Dead, I'm The Hero</p>
            </footer>
        </div>
        {get_theme_toggle_script()}
    </body>
    </html>
    """
    return html_template


def render_newsletter_html_from_bundle(bundle: Dict[str, Any]) -> str:
    """
    Render the newsletter HTML from bundle data.
    Uses structured news sections directly - no text parsing.
    """
    meta = bundle["meta"]
    news = bundle["news"]
    newsletter = news["newsletter"]
    sections = newsletter["sections"]
    
    date_formatted = meta["date_formatted"]
    day_of_week = meta["day_of_week"]
    current_year = datetime.now().year
    
    tech_html = format_news_items_html(sections.get("tech", []))
    financial_html = format_news_items_html(sections.get("financial", []))
    india_html = format_news_items_html(sections.get("india", []))
    
    html_template = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=5.0">
        <meta name="theme-color" content="#0A0B0D">
        <meta name="color-scheme" content="dark light">
        <title>Daily Briefing - {date_formatted}</title>
        {get_luxurious_css()}
    </head>
    <body>
        <div class="container">
            <!-- Header Section -->
            <header class="header">
                <button class="theme-toggle" id="theme-toggle" type="button" aria-label="Toggle theme" aria-pressed="false" title="Toggle theme">☾</button>
                <h1>Daily Briefing</h1>
                <p class="subtitle">{day_of_week}'s Essential Updates</p>
                <div class="date-badge">{date_formatted}</div>
            </header>

            <!-- Technology Section -->
            <article class="card" aria-label="Technology news">
                <div class="card-header">
                    <h2 class="card-title">
                        <span class="card-icon" aria-hidden="true">💻</span>
                        Technology
                    </h2>
                    <span class="card-meta">Tech & AI</span>
                </div>
                <div class="card-content">
                    {tech_html}
                </div>
            </article>

            <!-- Financial Markets Section -->
            <article class="card" aria-label="Financial markets news">
                <div class="card-header">
                    <h2 class="card-title">
                        <span class="card-icon" aria-hidden="true">📈</span>
                        Markets
                    </h2>
                    <span class="card-meta">Finance</span>
                </div>
                <div class="card-content">
                    {financial_html}
                </div>
            </article>

            <!-- India News Section -->
            <article class="card" aria-label="India news">
                <div class="card-header">
                    <h2 class="card-title">
                        <span class="card-icon" aria-hidden="true" style="display: inline-flex; align-items: center;">
                            <svg width="20" height="14" viewBox="0 0 20 14" style="border-radius: 2px; box-shadow: 0 1px 2px rgba(0,0,0,0.2);">
                                <rect width="20" height="4.67" fill="#FF9933"/>
                                <rect y="4.67" width="20" height="4.67" fill="#FFFFFF"/>
                                <rect y="9.33" width="20" height="4.67" fill="#138808"/>
                                <circle cx="10" cy="7" r="1.8" fill="#000080"/>
                            </svg>
                        </span>
                        India
                    </h2>
                    <span class="card-meta">Regional</span>
                </div>
                <div class="card-content">
                    {india_html}
                </div>
            </article>

            <!-- Footer -->
            <footer>
                <p>Generated by <strong>EDITH</strong> • {current_year}</p>
                <p style="margin-top: 8px; font-size: 11px;">Even Dead, I'm The Hero</p>
            </footer>
        </div>
        {get_theme_toggle_script()}
    </body>
    </html>
    """
    return html_template


def send_email(subject, message, is_html=False, skip_send=False) -> bool:
    """
    Send email via Yahoo SMTP.
    
    Args:
        subject: Email subject
        message: Email body content
        is_html: Whether message is HTML
        skip_send: If True, skip actual sending (for testing)
    
    Returns:
        True if email was sent successfully, False otherwise
    """
    if skip_send:
        logger.info(f"[SKIP] Would send email: {subject}")
        print(f"[SKIP] Email skipped (testing mode): {subject}")
        return True
    
    sender_email = yahoo_id
    receiver_email = "katam.dinesh@hotmail.com"
    password = yahoo_app_password
    
    # Validate credentials
    if not sender_email or not password:
        logger.error("Email credentials not configured in config.yml")
        print("ERROR: Email credentials missing - check yahoo_id and yahoo_app_password in config.yml")
        return False

    email_message = MIMEMultipart()
    email_message["From"] = sender_email
    email_message["To"] = receiver_email
    email_message["Subject"] = subject

    if is_html:
        email_message.attach(MIMEText(message, "html", "utf-8"))
    else:
        email_message.attach(MIMEText(message, "plain", "utf-8"))

    try:
        logger.info(f"Connecting to Yahoo SMTP server...")
        server = smtplib.SMTP('smtp.mail.yahoo.com', 587, timeout=30)
        server.set_debuglevel(0)  # Set to 1 for verbose SMTP debug
        
        logger.info("Starting TLS...")
        server.starttls()
        
        logger.info(f"Logging in as {sender_email}...")
        server.login(sender_email, password)
        
        logger.info(f"Sending email to {receiver_email}...")
        text = email_message.as_string()
        server.sendmail(sender_email, receiver_email, text)
        
        server.quit()
        logger.info(f"Email sent successfully: {subject}")
        print(f"✓ Email sent: {subject} → {receiver_email}")
        return True
        
    except smtplib.SMTPAuthenticationError as e:
        logger.error(f"SMTP Authentication failed: {e}")
        print(f"ERROR: Email authentication failed - check Yahoo app password")
        print(f"  Hint: Generate an app password at https://login.yahoo.com/account/security")
        return False
    except smtplib.SMTPConnectError as e:
        logger.error(f"SMTP Connection failed: {e}")
        print(f"ERROR: Could not connect to Yahoo SMTP server")
        return False
    except smtplib.SMTPException as e:
        logger.error(f"SMTP Error: {e}")
        print(f"ERROR: SMTP error - {e}")
        return False
    except Exception as e:
        logger.error(f"Failed to send email: {e}", exc_info=True)
        print(f"ERROR: Failed to send email - {e}")
        return False

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

def save_to_output_dir(content: str, filename: str, output_dir: Path = OUTPUT_DIR) -> Path:
    """
    Save content to a file in the output directory.
    Returns the path to the saved file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / filename
    
    # Handle subdirectories in filename
    if "/" in filename:
        filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    logger.info(f"Saved to {filepath}")
    return filepath

def parse_arguments():
    """Parse command-line arguments for testing and production modes."""
    parser = argparse.ArgumentParser(
        description="Year Progress and News Reporter - Generate daily newsletters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full production run
  python year_progress_and_news_reporter_litellm.py

  # Fast testing mode (use cached news, skip email/audio)
  python year_progress_and_news_reporter_litellm.py --test

  # Use cached news but still send emails
  python year_progress_and_news_reporter_litellm.py --use-cache

  # Generate HTML only (skip email and audio)
  python year_progress_and_news_reporter_litellm.py --skip-email --skip-audio

  # Force refresh news cache
  python year_progress_and_news_reporter_litellm.py --refresh-cache

  # Check cache status
  python year_progress_and_news_reporter_litellm.py --cache-info
        """
    )
    
    parser.add_argument(
        '--test', '-t',
        action='store_true',
        help='Testing mode: use cache, skip email and audio (same as --use-cache --skip-email --skip-audio)'
    )
    parser.add_argument(
        '--use-cache', '-c',
        action='store_true',
        help='Use cached news data if available (speeds up iteration)'
    )
    parser.add_argument(
        '--refresh-cache',
        action='store_true',
        help='Force refresh of news cache even if valid cache exists'
    )
    parser.add_argument(
        '--skip-email', '-e',
        action='store_true',
        help='Skip sending emails (for testing)'
    )
    parser.add_argument(
        '--skip-audio', '-a',
        action='store_true',
        help='Skip generating TTS audio (for testing)'
    )
    parser.add_argument(
        '--cache-info',
        action='store_true',
        help='Show cache status and exit'
    )
    parser.add_argument(
        '--html-only',
        action='store_true',
        help='Only regenerate HTML from latest bundle (fastest for design iteration)'
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    
    # Handle --test shortcut
    if args.test:
        args.use_cache = True
        args.skip_email = True
        args.skip_audio = True
    
    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Handle --cache-info
    if args.cache_info:
        cache_info = get_cache_info()
        print("\n" + "=" * 60)
        print("NEWS CACHE STATUS")
        print("=" * 60)
        if cache_info is None:
            print("  Status: No cache found")
            print(f"  Location: {NEWS_CACHE_FILE}")
        elif "error" in cache_info:
            print(f"  Status: Cache exists but invalid ({cache_info['error']})")
        else:
            print(f"  Status: {'VALID' if cache_info['age_hours'] < CACHE_MAX_AGE_HOURS else 'EXPIRED'}")
            print(f"  Location: {NEWS_CACHE_FILE}")
            print(f"  Timestamp: {cache_info['timestamp']}")
            print(f"  Date: {cache_info['date_iso']}")
            print(f"  Age: {cache_info['age_hours']:.1f} hours (max: {CACHE_MAX_AGE_HOURS})")
            print(f"  Has tech news: {cache_info['has_tech']}")
            print(f"  Has financial news: {cache_info['has_financial']}")
            print(f"  Has India news: {cache_info['has_india']}")
        print("=" * 60)
        exit(0)
    
    # Handle --html-only (regenerate from existing bundle)
    if args.html_only:
        print("\n" + "=" * 60)
        print("HTML-ONLY MODE - Regenerating from latest bundle")
        print("=" * 60)
        
        latest_bundle_path = OUTPUT_DIR / "daily_bundle_latest.json"
        if not latest_bundle_path.exists():
            print(f"ERROR: No bundle found at {latest_bundle_path}")
            print("Run without --html-only first to generate a bundle.")
            exit(1)
        
        bundle = load_bundle_json(latest_bundle_path)
        print(f"✓ Loaded bundle from {bundle['meta']['date_iso']}")
        
        # Render HTML
        year_progress_html = render_year_progress_html_from_bundle(bundle)
        progress_html_path = save_to_output_dir(year_progress_html, "year_progress_report.html")
        print(f"✓ Progress HTML: {progress_html_path}")
        
        newsletter_html = render_newsletter_html_from_bundle(bundle)
        newsletter_html_path = save_to_output_dir(newsletter_html, "news_newsletter_report.html")
        print(f"✓ Newsletter HTML: {newsletter_html_path}")
        
        print("\n✓ HTML regeneration complete!")
        print(f"  View: {OUTPUT_DIR}/news_newsletter_report.html")
        exit(0)
    
    # ==========================================================================
    # MAIN EXECUTION
    # ==========================================================================
    logger.info("=" * 80)
    logger.info("STARTING YEAR PROGRESS AND NEWS REPORTER (JSON-BACKED)")
    logger.info("=" * 80)
    
    mode_flags = []
    if args.use_cache:
        mode_flags.append("USE_CACHE")
    if args.skip_email:
        mode_flags.append("SKIP_EMAIL")
    if args.skip_audio:
        mode_flags.append("SKIP_AUDIO")
    if args.refresh_cache:
        mode_flags.append("REFRESH_CACHE")
    
    if mode_flags:
        print(f"\n⚙️  Mode: {', '.join(mode_flags)}")
    
    logger.info(f"Output directory: {OUTPUT_DIR.absolute()}")
    
    # =========================================================================
    # STEP 1: GATHER TIME/PROGRESS DATA
    # =========================================================================
    days_completed, weeks_completed, days_left, weeks_left, percent_days_left = time_left_in_year()
    logger.info(f"Year progress: {100 - percent_days_left:.1f}% complete, {days_left} days remaining")
    
    # =========================================================================
    # STEP 2: GET WEATHER
    # =========================================================================
    logger.info("Fetching weather data...")
    temp, status = get_weather()
    weather_data = {
        "temp_c": temp,
        "status": status,
        "location": "North Plains, OR"
    }
    logger.info(f"Weather: {temp}°C, {status}")
    
    # =========================================================================
    # STEP 3: GENERATE QUOTE AND LESSON
    # =========================================================================
    logger.info("Generating quote and lesson...")
    random_personality = get_random_personality()
    logger.info(f"Selected personality: {random_personality}")
    
    quote_text = generate_quote(random_personality)
    logger.info(f"Quote result: {'SUCCESS' if quote_text else 'EMPTY'} - {quote_text[:80] if quote_text else 'N/A'}...")
    
    # Get the topic and lesson together (get_random_lesson returns both)
    topic, lesson_raw = get_random_lesson()
    logger.info(f"Topic: {topic[:50]}...")
    logger.info(f"Lesson result: {'SUCCESS' if lesson_raw else 'EMPTY'} - {len(lesson_raw) if lesson_raw else 0} chars")
    
    # Parse lesson to structured dict
    lesson_dict = parse_lesson_to_dict(lesson_raw, topic)
    
    # =========================================================================
    # STEP 4: FETCH NEWS FROM ALL CATEGORIES (WITH CACHING)
    # =========================================================================
    print("\n" + "=" * 80)
    print("FETCHING NEWS UPDATES")
    print("=" * 80)
    
    news_raw_sources = None
    
    # Try to use cache if requested
    if args.use_cache and not args.refresh_cache:
        cached_news = load_news_cache()
        if cached_news:
            print("\n📦 Using cached news data...")
            news_raw_sources = cached_news
            print(f"✓ Tech news: {len(news_raw_sources.get('technology', ''))} characters (cached)")
            print(f"✓ Financial news: {len(news_raw_sources.get('financial', ''))} characters (cached)")
            print(f"✓ India news: {len(news_raw_sources.get('india', ''))} characters (cached)")
    
    # Fetch fresh news if no cache or cache not used
    if news_raw_sources is None:
        news_provider = "litellm" if LLM_PROVIDER == "litellm" else "ollama"
        
        print("\n📰 Fetching Technology News...")
        news_update_tech = gather_daily_news(
            category="technology",
            max_sources=10,
            aggregator_limit=1,
            freshness_hours=24,
            provider=news_provider
        )
        print(f"✓ Tech news: {len(news_update_tech)} characters")
        
        print("\n📈 Fetching Financial Markets News...")
        news_update_financial = gather_daily_news(
            category="financial",
            max_sources=10,
            aggregator_limit=1,
            freshness_hours=24,
            provider=news_provider
        )
        print(f"✓ Financial news: {len(news_update_financial)} characters")
        
        print("\n🇮🇳 Fetching India News...")
        news_update_india = gather_daily_news(
            category="india",
            max_sources=10,
            aggregator_limit=0,
            freshness_hours=24,
            provider=news_provider
        )
        print(f"✓ India news: {len(news_update_india)} characters")
        
        news_raw_sources = {
            "technology": news_update_tech,
            "financial": news_update_financial,
            "india": news_update_india
        }
        
        # Save to cache for future use
        print("\n💾 Saving news to cache...")
        save_news_cache(news_raw_sources)
    
    # Extract for convenience
    news_update_tech = news_raw_sources.get("technology", "")
    news_update_financial = news_raw_sources.get("financial", "")
    news_update_india = news_raw_sources.get("india", "")
    
    # =========================================================================
    # STEP 5: GENERATE NEWSLETTER SECTIONS (STRUCTURED)
    # =========================================================================
    print("\n" + "=" * 80)
    print("GENERATING NEWSLETTER")
    print("=" * 80)
    
    print("\n📝 Generating newsletter sections...")
    newsletter_sections = generate_newsletter_sections(
        news_update_tech,
        news_update_financial,
        news_update_india
    )
    print(f"✓ Newsletter sections: tech={len(newsletter_sections['tech'])}, financial={len(newsletter_sections['financial'])}, india={len(newsletter_sections['india'])}")
    
    # =========================================================================
    # STEP 6: GENERATE VOICEBOT SCRIPT
    # =========================================================================
    print("\n🎙️ Generating voicebot script...")
    voicebot_prompt = f"""
    Here are today's key updates across technology, financial markets, and India:
    
    Technology Updates:
    {news_update_tech}

    Financial Market Headlines:
    {news_update_financial}

    Latest from India:
    {news_update_india}
    
    Please present this information in a natural, conversational way suitable for speaking.
    """
    voicebot_script = generate_gpt_response_voicebot(voicebot_prompt)
    print(f"✓ Voicebot script: {len(voicebot_script)} characters")
    
    # =========================================================================
    # STEP 7: BUILD THE COMPLETE BUNDLE
    # =========================================================================
    print("\n📦 Building daily bundle...")
    bundle = build_daily_bundle(
        days_completed=days_completed,
        weeks_completed=weeks_completed,
        days_left=days_left,
        weeks_left=weeks_left,
        percent_days_left=percent_days_left,
        weather_data=weather_data,
        quote_text=quote_text,
        quote_author=random_personality,
        lesson_dict=lesson_dict,
        news_raw_sources=news_raw_sources,
        newsletter_sections=newsletter_sections,
        voicebot_script=voicebot_script
    )
    print("✓ Bundle built successfully")
    
    # =========================================================================
    # STEP 8: WRITE BUNDLE JSON FILES
    # =========================================================================
    print("\n💾 Writing JSON bundle files...")
    bundle_path = write_bundle_json(bundle, OUTPUT_DIR)
    print(f"✓ Bundle saved to: {bundle_path}")
    print(f"✓ Latest bundle: {OUTPUT_DIR / 'daily_bundle_latest.json'}")
    
    # =========================================================================
    # STEP 9: RENDER HTML FROM BUNDLE
    # =========================================================================
    print("\n🎨 Rendering HTML from bundle...")
    
    # Render year progress HTML
    year_progress_html = render_year_progress_html_from_bundle(bundle)
    progress_html_path = save_to_output_dir(year_progress_html, "year_progress_report.html")
    print(f"✓ Progress HTML: {progress_html_path}")
    
    # Render newsletter HTML
    newsletter_html = render_newsletter_html_from_bundle(bundle)
    newsletter_html_path = save_to_output_dir(newsletter_html, "news_newsletter_report.html")
    print(f"✓ Newsletter HTML: {newsletter_html_path}")
    
    # =========================================================================
    # STEP 10: SEND EMAILS
    # =========================================================================
    print("\n📧 Sending emails...")
    
    year_progress_subject = "Year Progress Report 📅"
    send_email(year_progress_subject, year_progress_html, is_html=True, skip_send=args.skip_email)
    
    news_update_subject = "📰 Your Daily News Briefing"
    send_email(news_update_subject, newsletter_html, is_html=True, skip_send=args.skip_email)
    
    # =========================================================================
    # STEP 11: GENERATE TTS AUDIO
    # =========================================================================
    if args.skip_audio:
        print("\n🔊 Audio generation skipped (--skip-audio)")
    else:
        print("\n🔊 Generating audio files...")
        
        # Generate speech-friendly progress summary
        year_progress_message_prompt = f"""
        Here is a year progress report for {datetime.now().strftime("%B %d, %Y")}:
        
        Days completed: {days_completed}
        Weeks completed: {weeks_completed:.1f}
        Days remaining: {days_left}
        Weeks remaining: {weeks_left:.1f}
        Year Progress: {100 - percent_days_left:.1f}% completed

        Quote of the day from {random_personality}:
        {quote_text}

        Today's lesson:
        {lesson_raw}
        """
        
        year_progress_gpt_response = generate_gpt_response(year_progress_message_prompt)
        yearprogress_tts_output_path = str(OUTPUT_DIR / "year_progress_report.mp3")
        model_name = random.choice(model_names)
        tts_result = text_to_speech_nospeak(year_progress_gpt_response, yearprogress_tts_output_path, model_name=model_name, speed=1.5)
        if tts_result:
            print(f"✓ Progress audio: {yearprogress_tts_output_path}")
        else:
            print(f"⚠ Progress audio: TTS unavailable (requires NVIDIA Riva service)")
        
        # Generate news audio
        news_tts_output_path = str(OUTPUT_DIR / "news_update_report.mp3")
        model_name = random.choice(model_names)
        tts_result = text_to_speech_nospeak(voicebot_script, news_tts_output_path, model_name=model_name, speed=1.5)
        if tts_result:
            print(f"✓ News audio: {news_tts_output_path}")
        else:
            print(f"⚠ News audio: TTS unavailable (requires NVIDIA Riva service)")
    
    # =========================================================================
    # DONE
    # =========================================================================
    print("\n" + "=" * 80)
    print("✓ ALL TASKS COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"\nOutput files in {OUTPUT_DIR}:")
    print(f"  - daily_bundle_{bundle['meta']['date_iso']}.json")
    print(f"  - daily_bundle_latest.json")
    print(f"  - year_progress_report.html")
    print(f"  - news_newsletter_report.html")
    
    if args.skip_email:
        print("\n📧 Emails were skipped (use without --skip-email to send)")
    if args.skip_audio:
        print("\n🔊 Audio was skipped (use without --skip-audio to generate)")
    
    print(f"\n💡 Quick commands:")
    print(f"  View newsletter: firefox {OUTPUT_DIR}/news_newsletter_report.html")
    print(f"  Test mode:       python {__file__} --test")
    print(f"  HTML only:       python {__file__} --html-only")
