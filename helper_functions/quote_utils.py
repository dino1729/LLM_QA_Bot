import logging
import re

from helper_functions.llm_client import get_client

logger = logging.getLogger(__name__)


def generate_quote(random_personality: str, llm_provider: str) -> str:
    """Generate an inspirational quote from the given personality."""
    logger.info("Generating quote for personality: %s", random_personality)
    client = get_client(provider=llm_provider, model_tier="fast")

    conversation = [
        {
            "role": "system",
            "content": f"You provide famous quotes from {random_personality}. "
            f"Reply with ONLY the quote in quotation marks - no introduction, no explanation, just the quote.",
        },
        {"role": "user", "content": f"Give me an inspirational quote from {random_personality}"},
    ]

    try:
        message = client.chat_completion(
            messages=conversation,
            max_tokens=250,
            temperature=0.8,
        )

        logger.debug(
            "Raw LLM response for quote (first 200 chars): %s",
            message[:200] if message else "EMPTY",
        )
        logger.debug("Full raw response length: %s", len(message) if message else 0)

        if not message or len(message.strip()) == 0:
            logger.warning("LLM returned empty response for quote generation")
            return _get_fallback_quote(random_personality)

        cleaned = message.strip()
        quote_matches = re.findall(r'"([^"]+)"', cleaned)
        logger.debug("Found %s quoted segments in response", len(quote_matches))

        if quote_matches:
            best_quote = max(quote_matches, key=len)
            logger.debug("Best quote candidate (length %s): %s", len(best_quote), best_quote[:100])
            if len(best_quote) > 15:
                result = f'"{best_quote}"'
                logger.info("Successfully extracted quote: %s...", result[:80])
                return result

        lines = [l.strip() for l in cleaned.split("\n") if l.strip()]
        logger.debug("Searching %s lines for usable content", len(lines))

        for line in lines:
            line_lower = line.lower()
            if any(
                skip in line_lower
                for skip in [
                    "the user",
                    "provide",
                    "respond",
                    "generate",
                    "here is",
                    "here's",
                    "famous quote",
                    "quote by",
                    "quote from",
                    "said:",
                    "once said",
                ]
            ):
                logger.debug("Skipping meta-text line: %s", line[:50])
                continue

            if len(line) > 20:
                if not (line.startswith('"') and line.endswith('"')):
                    line = f'"{line}"'
                logger.info("Using line as quote: %s...", line[:80])
                return line

        if 20 < len(cleaned) < 300 and "user" not in cleaned.lower():
            if not (cleaned.startswith('"') and cleaned.endswith('"')):
                cleaned = f'"{cleaned}"'
            logger.info("Using cleaned response as quote: %s...", cleaned[:80])
            return cleaned

        logger.warning("Could not extract valid quote from response. Using fallback.")
        return _get_fallback_quote(random_personality)

    except Exception as e:
        logger.error("Error generating quote: %s", e, exc_info=True)
        return _get_fallback_quote(random_personality)


def _get_fallback_quote(personality: str) -> str:
    """Return a fallback quote when generation fails."""
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
    logger.info("Using fallback quote for %s: %s", personality, quote)
    return quote
