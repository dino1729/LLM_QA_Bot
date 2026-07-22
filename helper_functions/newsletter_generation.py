import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from helper_functions.llm_client import get_client
from helper_functions.newsletter_parsing import (
    generate_fallback_newsletter_sections,
    parse_newsletter_text_to_sections,
)

logger = logging.getLogger(__name__)


def _extract_prompt_value(prompt: str, label: str) -> str:
    """Extract a single-line value from the progress prompt."""
    match = re.search(rf"^{re.escape(label)}:\s*(.+)$", prompt, re.MULTILINE)
    if match:
        return match.group(1).strip()
    inline_match = re.search(rf"\b{re.escape(label)}:\s*([^\n.]+(?:\.[^\n.]*)?)", prompt)
    return inline_match.group(1).strip() if inline_match else ""


def _extract_prompt_block(prompt: str, start_label: str, stop_labels: List[str]) -> str:
    """Extract a block from the progress prompt until the next known section."""
    start = prompt.find(start_label)
    if start < 0:
        return ""
    block = prompt[start + len(start_label):]
    stop_positions = [block.find(label) for label in stop_labels if block.find(label) >= 0]
    if stop_positions:
        block = block[:min(stop_positions)]
    return block.strip()


def _build_progress_script_fallback(user_message: str) -> str:
    """Build a deterministic spoken script when the LLM returns unusable text."""
    prompt = str(user_message or "")

    days_completed = _extract_prompt_value(prompt, "- Days completed")
    days_remaining = _extract_prompt_value(prompt, "- Days remaining")
    percent_complete = _extract_prompt_value(prompt, "- Percent complete")
    location = _extract_prompt_value(prompt, "- Location")
    temperature = _extract_prompt_value(prompt, "- Temperature")
    conditions = _extract_prompt_value(prompt, "- Conditions")

    quote_match = re.search(r'INSPIRATIONAL QUOTE:\s*\n"?(.+?)"?\s*-\s*(.+)', prompt, re.DOTALL)
    quote_text = ""
    quote_author = ""
    if quote_match:
        quote_text = quote_match.group(1).splitlines()[0].strip().strip('"')
        quote_author = quote_match.group(2).splitlines()[0].strip()

    lesson_topic = _extract_prompt_value(prompt, "MEMORY PALACE WISDOM - Topic")
    lesson_insight = _extract_prompt_value(prompt, "Key Insight")
    if not lesson_insight:
        lesson_insight = _extract_prompt_value(prompt, "Curated Insight")
    if not lesson_topic:
        lesson_topic = _extract_prompt_value(prompt, "DAILY LESSON - Topic")

    weather_part = ""
    if location or temperature or conditions:
        weather_part = (
            f" In {location or 'your area'}, it is {temperature or 'currently updating'}"
            f" with {conditions or 'conditions worth checking before you head out'}."
        )

    quote_part = ""
    if quote_text:
        quote_text = quote_text.rstrip(".!?")
        quote_part = f" Today's quote from {quote_author or 'the archive'} is: {quote_text}."

    lesson_part = ""
    if lesson_insight:
        topic_text = f" on {lesson_topic}" if lesson_topic else ""
        lesson_part = f" The lesson{topic_text} is this: {lesson_insight}"

    progress_part = ""
    if days_completed or days_remaining or percent_complete:
        progress_part = (
            f" You've completed {days_completed or 'another meaningful stretch'}"
            f" of the year, with {days_remaining or 'time still ahead'} remaining."
            f" That puts the year at {percent_complete or 'a real checkpoint'} complete."
        )

    return (
        "Dinesh, quick systems check from EDITH."
        f"{progress_part}{weather_part}{quote_part}{lesson_part}"
        " Use it as a practical cue today: protect attention, choose the next high-leverage action, and keep compounding."
    ).strip()


def _is_unusable_generated_response(text: str) -> bool:
    """Return True when a model response should not be sent to TTS/output."""
    normalized = re.sub(r"\s+", " ", (text or "")).strip().lower()
    if not normalized:
        return True
    meta_only = (
        "the user wants",
        "user wants",
        "we need",
        "let me",
        "task:",
    )
    return any(normalized.startswith(prefix) for prefix in meta_only) and len(normalized.split()) < 12


def load_persona_content(voice_persona: str, personas_dir: str = "personas") -> str | None:
    """Load persona instructions from file if available.

    Args:
        voice_persona: Voice/persona name (e.g., "rick_sanchez")
        personas_dir: Directory containing persona text files

    Returns:
        Persona file content if found, None otherwise
    """
    persona_path = Path(personas_dir) / f"{voice_persona}.txt"
    if persona_path.exists():
        logger.info(f"Loading persona file: {persona_path}")
        return persona_path.read_text()
    logger.debug(f"No persona file found at {persona_path}, using name-based prompt")
    return None


def generate_gpt_response(user_message: str, llm_provider: str, model_tier: str, voice_persona: str = None, use_chatterbox: bool = False, model_name: str = None) -> str:
    """Generate a speech-friendly response from EDITH (or persona when using Chatterbox)."""
    client = get_client(provider=llm_provider, model_tier=model_tier, model_name=model_name)

    # When Chatterbox is enabled with a persona, try to load detailed persona file
    if use_chatterbox and voice_persona:
        persona_content = load_persona_content(voice_persona)
        if persona_content:
            # Use rich persona file content with context about the listener
            identity_intro = f"{persona_content}\n\nYou are speaking to Dinesh through his smart speaker."
            style_instruction = "Stay completely in character throughout."
        else:
            # Fallback to simple name-based prompt when no persona file exists
            persona_name = voice_persona.replace('_', ' ').title()
            identity_intro = f"You are {persona_name}, speaking to Dinesh through his smart speaker."
            style_instruction = f"Fully embody {persona_name}'s personality, vocabulary, rhythm, and speaking style."
    else:
        persona_name = "EDITH"
        identity_intro = "You are EDITH, Tony Stark's AI assistant, speaking to Dinesh through his smart speaker."
        if voice_persona:
            identity_intro += f" You are speaking as {voice_persona.replace('_', ' ').title()}."
        style_instruction = "Use Tony's confidence and wit, blended with the essence of the target persona if provided."
    
    # Chatterbox TTS supports paralinguistic tags for more natural speech
    chatterbox_instruction = ""
    if use_chatterbox:
        chatterbox_instruction = """

PARALINGUISTIC EXPRESSION TAGS (Chatterbox TTS):
You can use these special tags to add natural vocal expressions:
- [laugh] - for moments of genuine amusement or joy
- [chuckle] - for lighter, subtle humor or ironic observations
- [cough] - for dramatic pauses or clearing throat before important points

Use these sparingly and naturally - maybe 2-4 times total in a response.
Examples:
- "And get this [chuckle] the AI actually beat its own creator at chess."
- "[laugh] I know, it sounds crazy, but the numbers don't lie."
- "Now [cough] this next part is important..."
"""
    
    syspromptmessage = f"""{identity_intro}

Your response will be converted to speech (TTS), so you MUST:
- Write in flowing, conversational prose - NO formatting whatsoever
- {style_instruction}
- Start with a unique, attention-grabbing greeting
- Adopt your signature vocabulary, rhythm, and style throughout.

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
{chatterbox_instruction}"""

    conversation = [
        {"role": "system", "content": syspromptmessage},
        {"role": "user", "content": str(user_message)},
    ]

    message = client.chat_completion(
        messages=conversation,
        max_tokens=2500,
        temperature=0.4,
    )

    cleaned = (message or "").strip()

    if _is_unusable_generated_response(cleaned):
        logger.warning("LLM returned an unusable progress script; using deterministic fallback")
        return _build_progress_script_fallback(user_message)

    if cleaned.lower().startswith(
        ("the user", "user wants", "we need", "let me", "task:")
    ):
        lines = cleaned.split("\n")
        start_idx = 0
        for i, line in enumerate(lines):
            if line.strip() and not any(
                line.lower().startswith(prefix)
                for prefix in [
                    "the user",
                    "user wants",
                    "we need",
                    "let me",
                    "task:",
                    "provide",
                    "respond with",
                ]
            ):
                start_idx = i
                break
        cleaned = "\n".join(lines[start_idx:]).strip()

    if _is_unusable_generated_response(cleaned):
        logger.warning("Cleaned progress script was unusable; using deterministic fallback")
        return _build_progress_script_fallback(user_message)

    return cleaned


def generate_gpt_response_newsletter(user_message: str, llm_provider: str, model_tier: str, model_name: str = None) -> str:
    """
    Generate newsletter-formatted news content with error handling.
    Returns formatted text content or empty string if generation fails.
    """
    try:
        client = get_client(provider=llm_provider, model_tier=model_tier, model_name=model_name)

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
        conversation = [{"role": "system", "content": syspromptmessage}]
        conversation.append({"role": "user", "content": str(user_message)})

        message = client.chat_completion(
            messages=conversation,
            max_tokens=3000,
            temperature=0.3,
        )

        if not message or len(message.strip()) < 100:
            logger.warning("Newsletter generation returned minimal content")
            return ""

        return message

    except Exception as e:
        logger.error("Error generating newsletter: %s", e)
        return ""


def generate_newsletter_sections(
    news_tech: str, news_financial: str, news_india: str, llm_provider: str, model_tier: str, model_name: str = None
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Generate newsletter sections as structured data.
    Uses LLM generation with parsing, falling back to structured fallback if needed.
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

    newsletter_text = generate_gpt_response_newsletter(newsletter_prompt, llm_provider, model_tier, model_name=model_name)

    if not newsletter_text:
        logger.warning("Newsletter generation failed, using fallback sections")
        return generate_fallback_newsletter_sections()

    sections = parse_newsletter_text_to_sections(newsletter_text)
    has_content = all(len(sections[key]) > 0 for key in ["tech", "financial", "india"])

    if not has_content:
        logger.warning("Parsed newsletter sections incomplete, using fallback")
        return generate_fallback_newsletter_sections()

    logger.info(
        "Successfully generated newsletter sections: tech=%s, financial=%s, india=%s",
        len(sections["tech"]),
        len(sections["financial"]),
        len(sections["india"]),
    )
    return sections


def generate_gpt_response_voicebot(user_message: str, llm_provider: str, model_tier: str, voice_persona: str = None, use_chatterbox: bool = False, model_name: str = None) -> str:
    """
    Generate voice-friendly news content with error handling.
    Returns formatted content or fallback summary if generation fails.
    """
    try:
        client = get_client(provider=llm_provider, model_tier=model_tier, model_name=model_name)

        # When Chatterbox is enabled with a persona, try to load detailed persona file
        if use_chatterbox and voice_persona:
            persona_content = load_persona_content(voice_persona)
            if persona_content:
                # Use rich persona file content with context about the delivery format
                identity_intro = f"{persona_content}\n\nYou are delivering a news briefing through a voicebot to Dinesh."
                style_instruction = "Stay completely in character throughout the news delivery."
            else:
                # Fallback to simple name-based prompt when no persona file exists
                persona_name = voice_persona.replace('_', ' ').title()
                identity_intro = f"You are {persona_name}, delivering a news briefing through a voicebot."
                style_instruction = f"Fully embody {persona_name}'s personality, vocabulary, rhythm, and speaking style throughout."
        else:
            identity_intro = "You are EDITH, speaking through a voicebot."
            if voice_persona:
                identity_intro += f" You are speaking as {voice_persona.replace('_', ' ').title()}."
            style_instruction = "If a persona is provided, adopt their signature vocabulary, rhythm, and style (e.g., if Morgan Freeman, use wise, calm, and rhythmic phrasing)."
        
        # Chatterbox TTS supports paralinguistic tags for more expressive delivery
        chatterbox_instruction = ""
        if use_chatterbox:
            chatterbox_instruction = """

PARALINGUISTIC EXPRESSION TAGS (Chatterbox TTS):
You can use these special tags to add natural vocal expressions throughout your delivery:
- [laugh] - for moments of genuine amusement, surprising news, or ironic situations
- [chuckle] - for lighter humor, witty observations, or amusing market quirks
- [cough] - for dramatic pauses before big reveals or clearing throat before important points

Use these naturally throughout the script - aim for 4-6 uses across the full briefing.
They make the delivery feel more human and engaging. Place them where a real news anchor might naturally react.
Examples:
- "And here's the kicker [chuckle] the stock actually went UP after the scandal."
- "[laugh] Yes, you heard that right - a trillion dollars."
- "Now [cough] pay attention to this next part..."
"""
        
        syspromptmessage = f"""
{identity_intro} For the voice format:
    1. Use conversational, natural speaking tone (e.g., "Today in tech news..." or "Moving on to financial markets...")
    2. {style_instruction}
    3. Break down complex information into simple, clear sentences
    4. Use verbal transitions between topics (e.g., "Now, let's look at..." or "In other news...")
    5. Avoid technical jargon unless necessary
    6. Keep points brief and easy to follow
    7. Never mention URLs, citations, or technical markers
    8. Use natural date formats (e.g., "today" or "yesterday" instead of MM/DD/YYYY)
    9. Focus on the story and its impact rather than sources
    10. End each section with a brief overview or key takeaway
    11. Use listener-friendly phrases like "As you might have heard" or "Interestingly"
{chatterbox_instruction}
CRITICAL: You MUST generate a substantial voice script (at least 500 words). Start speaking immediately.
    """
        conversation = [{"role": "system", "content": syspromptmessage}]

        voice_friendly_message = user_message.replace("- News headline", "")
        voice_friendly_message = voice_friendly_message.replace(
            " | [Source] | Date: MM/DD/YYYY | URL | Commentary:", ""
        )
        voice_friendly_message = voice_friendly_message.replace(
            "For each category below, provide exactly 5 key bullet points. Each point should follow this format:",
            "",
        )

        conversation.append({"role": "user", "content": voice_friendly_message})

        message = client.chat_completion(
            messages=conversation,
            max_tokens=2048,
            temperature=0.4,
        )

        if not message or len(message.strip()) < 100:
            logger.warning(
                "Voicebot generation returned minimal content (%s chars)",
                len(message) if message else 0,
            )
            return _get_fallback_voicebot_script()

        return message

    except Exception as e:
        logger.error("Error generating voicebot script: %s", e)
        return _get_fallback_voicebot_script()


def _get_fallback_voicebot_script():
    """Return a fallback voicebot script when generation fails."""
    today = datetime.now().strftime("%B %d, %Y")
    return f"""Good morning, Dinesh. Here's your news briefing for {today}.

In technology news today, the AI industry continues to evolve rapidly with major developments across infrastructure and applications. Companies are investing heavily in AI capabilities while navigating an increasingly competitive landscape.

Moving to financial markets, investors are watching key economic indicators closely. The markets are showing mixed signals as we head into the final weeks of the year.

From India, the economy continues to show resilience with strong growth numbers. The Reserve Bank is balancing growth support with inflation management.

That's your quick briefing for today. Stay informed, stay ahead."""
