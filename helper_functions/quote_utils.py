import logging
import re

from helper_functions.llm_client import get_client

logger = logging.getLogger(__name__)

_META_TEXT_PREFIXES = (
    "the user",
    "provide",
    "respond",
    "generate",
)
_META_TEXT_MARKERS = (
    "famous quote",
    "quote by",
    "quote from",
    "said:",
    "once said",
)
_AUTHOR_SUFFIX_RE = re.compile(r"\s*[-–—]\s+[A-Z][A-Za-z0-9.&' -]{1,80}$")


def _normalize_quotes(text: str) -> str:
    """Replace Unicode curly/smart quotes with straight ASCII quotes."""
    for ch in ("\u201c", "\u201d", "\u201e", "\u201f"):  # double curly quotes
        text = text.replace(ch, '"')
    for ch in ("\u2018", "\u2019", "\u201a", "\u201b"):  # single curly quotes
        text = text.replace(ch, "'")
    return text


def _is_meta_text(text: str) -> bool:
    """Identify instruction/meta text that should not be treated as a quote."""
    normalized = " ".join(text.lower().split()).strip()
    if normalized.startswith(_META_TEXT_PREFIXES):
        return True

    if (normalized.startswith("here is") or normalized.startswith("here's")) and "quote" in normalized:
        return True

    return any(
        marker in normalized for marker in _META_TEXT_MARKERS
    )


def _clean_quote_candidate(text: str) -> str:
    """Normalize whitespace and trim wrapper text from a quote candidate."""
    candidate = " ".join(text.split()).strip()
    candidate = re.sub(r"^(?:>\s*)+", "", candidate)
    candidate = _AUTHOR_SUFFIX_RE.sub("", candidate).strip()

    if candidate.startswith('"') and not candidate.endswith('"'):
        candidate = candidate[1:].strip()
    elif candidate.endswith('"') and not candidate.startswith('"'):
        candidate = candidate[:-1].strip()
    else:
        candidate = candidate.strip('"').strip()

    return candidate


def _recover_multiline_quote(lines: list[str]) -> str | None:
    """Join broken multi-line quote output when the model omits a closing quote."""
    if len(lines) < 2:
        return None

    joined = " ".join(lines).strip()
    if joined.count('"') != 1 and not any(line.startswith('"') and not line.endswith('"') for line in lines):
        return None

    candidate = _clean_quote_candidate(joined)
    if len(candidate) <= 20 or _is_meta_text(candidate):
        return None

    return f'"{candidate}"'


def generate_quote(random_personality: str, llm_provider: str, model_tier: str, model_name: str = None) -> str:
    """Generate an inspirational quote from the given personality."""
    logger.info("Generating quote for personality: %s", random_personality)
    client = get_client(provider=llm_provider, model_tier=model_tier, model_name=model_name)

    conversation = [
        {
            "role": "system",
            "content": f"You provide famous quotes from {random_personality}. "
            f"Reply with ONLY the quote in quotation marks on a single line - no introduction, no explanation, just the quote.",
        },
        {"role": "user", "content": f"Give me an inspirational quote from {random_personality}"},
    ]

    try:
        message = client.chat_completion(
            messages=conversation,
            max_tokens=1024,
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

        # Normalize curly/smart quotes to straight quotes before any parsing
        cleaned = _normalize_quotes(message.strip())

        # Try regex on newline-collapsed text so multi-line quotes aren't split
        collapsed = " ".join(cleaned.split())
        quote_matches = re.findall(r'"([^"]+)"', collapsed)
        logger.debug("Found %s quoted segments in response", len(quote_matches))

        if quote_matches:
            best_quote = max(quote_matches, key=len)
            logger.debug("Best quote candidate (length %s): %s", len(best_quote), best_quote[:100])
            if len(best_quote) > 15:
                result = f'"{best_quote}"'
                logger.info("Successfully extracted quote: %s...", result[:80])
                return result

        # Fallback: scan individual lines (preserving newlines so meta-text
        # on separate lines can be skipped)
        lines = [l.strip() for l in cleaned.split("\n") if l.strip()]
        logger.debug("Searching %s lines for usable content", len(lines))
        content_lines = [line for line in lines if not _is_meta_text(line)]

        recovered_multiline = _recover_multiline_quote(content_lines)
        if recovered_multiline:
            logger.info("Recovered quote from malformed multi-line response: %s...", recovered_multiline[:80])
            return recovered_multiline

        for line in content_lines:
            if len(content_lines) > 1 and line.startswith('"') and not line.endswith('"'):
                logger.debug("Skipping partial quote line in favor of joined candidate: %s", line[:50])
                continue

            if _is_meta_text(line):
                logger.debug("Skipping meta-text line: %s", line[:50])
                continue

            candidate = _clean_quote_candidate(line)
            if len(candidate) > 20:
                result = f'"{candidate}"'
                logger.info("Using line as quote: %s...", result[:80])
                return result

        fallback_source = " ".join(content_lines).strip() if content_lines else ""
        fallback_candidate = _clean_quote_candidate(fallback_source)
        if 20 < len(fallback_candidate) < 300 and not _is_meta_text(fallback_candidate):
            result = f'"{fallback_candidate}"'
            logger.info("Using cleaned response as quote: %s...", result[:80])
            return result

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
