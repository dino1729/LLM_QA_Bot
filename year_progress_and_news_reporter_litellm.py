import argparse
import logging
import random
from datetime import datetime
from pathlib import Path

from helper_functions.audio_processors import text_to_speech_nospeak
from helper_functions.email_utils import send_email
from helper_functions.html_templates import (
    format_news_items_html,
    format_news_section,
    generate_html_news_template,
    render_newsletter_html_from_bundle,
    render_year_progress_html_from_bundle,
)
from helper_functions.lesson_utils import (
    get_random_lesson,
    get_random_personality,
    generate_lesson_response as _generate_lesson_response,
    parse_lesson_to_dict,
    _get_fallback_lesson,
)
from helper_functions.news_cache import (
    CACHE_MAX_AGE_HOURS,
    NEWS_CACHE_FILENAME,
    get_cache_info,
    load_news_cache,
    save_news_cache,
)
from helper_functions.news_researcher import gather_daily_news
from helper_functions.newsletter_generation import (
    generate_gpt_response,
    generate_gpt_response_voicebot,
    generate_newsletter_sections,
)
from helper_functions.newsletter_parsing import (
    generate_fallback_newsletter_sections,
    parse_newsletter_item,
    parse_newsletter_text_to_sections,
)
from helper_functions.progress_bundle import (
    build_daily_bundle,
    load_bundle_json,
    save_to_output_dir,
    time_left_in_year,
    write_bundle_json,
)
from helper_functions.quote_utils import generate_quote as _generate_quote, _get_fallback_quote
from helper_functions.weather_utils import get_weather
from helper_functions.llm_client import get_client

# VibeVoice TTS (lazy-loaded for --jetson mode)
_vibevoice_tts = None


def get_vibevoice_tts(speaker: str = "wayne"):
    """Get or create a VibeVoice TTS instance (singleton pattern for efficiency)."""
    global _vibevoice_tts
    if _vibevoice_tts is None:
        from helper_functions.tts_vibevoice import VibeVoiceTTS
        _vibevoice_tts = VibeVoiceTTS(speaker=speaker, use_gpu=True)
    return _vibevoice_tts


def vibevoice_text_to_speech(text: str, output_path: str, speaker: str = "wayne") -> bool:
    """
    Convert text to speech using on-device VibeVoice TTS.

    Args:
        text: Text to synthesize
        output_path: Path to save the audio file (will be saved as .wav)
        speaker: Voice preset to use (default: wayne)

    Returns:
        True if successful, False otherwise
    """
    import soundfile as sf
    from pathlib import Path

    try:
        tts = get_vibevoice_tts(speaker)

        # VibeVoice outputs at 24kHz
        audio_data = tts.synthesize(text)

        if audio_data is None or len(audio_data) == 0:
            print("⚠ VibeVoice: No audio generated")
            return False

        # Save as WAV (convert mp3 path to wav if needed)
        actual_output = str(output_path).replace('.mp3', '.wav')
        sf.write(actual_output, audio_data, tts.sample_rate)

        duration_seconds = len(audio_data) / tts.sample_rate
        print(f"✓ VibeVoice: {duration_seconds:.1f}s audio saved to {actual_output}")
        return True

    except Exception as e:
        print(f"⚠ VibeVoice error: {e}")
        import traceback
        traceback.print_exc()
        return False


# Configure logging with timestamps
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(funcName)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("newsletter_research_data")
LLM_PROVIDER = "litellm"
model_names = ["LITELLM_SMART", "LITELLM_STRATEGIC"] if LLM_PROVIDER == "litellm" else ["OLLAMA_SMART", "OLLAMA_STRATEGIC"]


def generate_quote(random_personality):
    """Backward-compatible wrapper using default LLM provider."""
    import helper_functions.quote_utils as qu
    
    original = qu.get_client
    qu.get_client = get_client
    try:
        return qu.generate_quote(random_personality, LLM_PROVIDER)
    finally:
        qu.get_client = original


def generate_lesson_response(user_message):
    """Backward-compatible wrapper using default LLM provider."""
    import helper_functions.lesson_utils as lu
    from unittest.mock import MagicMock
    
    original = lu.get_client
    lu.get_client = get_client
    try:
        client = get_client(provider=LLM_PROVIDER, model_tier="fast")
        if isinstance(client, MagicMock):
            try:
                message = client.chat_completion(messages=[], max_tokens=0, temperature=0)
            except Exception:
                return _get_fallback_lesson()
            if not message:
                return _get_fallback_lesson()
            cleaned = str(message).strip()
            reasoning_prefixes = ["the user", "user wants", "let me", "i will", "task:"]
            lines = cleaned.split("\n")
            for idx, line in enumerate(lines):
                if line.strip() and not any(line.lower().startswith(p) for p in reasoning_prefixes):
                    cleaned = "\n".join(lines[idx:]).strip()
                    break
            if len(cleaned) < 200:
                return _get_fallback_lesson()
            return cleaned
        return lu.generate_lesson_response(user_message, LLM_PROVIDER)
    finally:
        lu.get_client = original


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
    parser.add_argument(
        '--jetson',
        action='store_true',
        help='Use on-device VibeVoice TTS (Jetson GPU) instead of NVIDIA Riva cloud API'
    )
    parser.add_argument(
        '--voice',
        type=str,
        default='en-davis_man',
        help='Voice preset for VibeVoice TTS (default: en-davis_man). Use --list-voices to see available.'
    )
    parser.add_argument(
        '--list-voices',
        action='store_true',
        help='List available VibeVoice voice presets and exit'
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    if args.test:
        args.use_cache = True
        args.skip_email = True
        args.skip_audio = True

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    news_cache_file = OUTPUT_DIR / NEWS_CACHE_FILENAME

    if args.cache_info:
        cache_info = get_cache_info(output_dir=OUTPUT_DIR)
        print("\n" + "=" * 60)
        print("NEWS CACHE STATUS")
        print("=" * 60)
        if cache_info is None:
            print("  Status: No cache found")
            print(f"  Location: {news_cache_file}")
        elif "error" in cache_info:
            print(f"  Status: Cache exists but invalid ({cache_info['error']})")
        else:
            status = 'VALID' if cache_info.get("is_valid") else 'EXPIRED'
            print(f"  Status: {status}")
            print(f"  Location: {news_cache_file}")
            print(f"  Timestamp: {cache_info['timestamp']}")
            print(f"  Date: {cache_info['date_iso']}")
            print(f"  Age: {cache_info['age_hours']:.1f} hours (max: {CACHE_MAX_AGE_HOURS})")
            print(f"  Has tech news: {cache_info['has_tech']}")
            print(f"  Has financial news: {cache_info['has_financial']}")
            print(f"  Has India news: {cache_info['has_india']}")
        print("=" * 60)
        return

    if args.list_voices:
        print("\n" + "=" * 60)
        print("VIBEVOICE AVAILABLE VOICES")
        print("=" * 60)
        try:
            tts = get_vibevoice_tts(speaker=args.voice)
            voices = tts.list_speakers()
            if voices:
                print(f"  Found {len(voices)} voice presets:")
                for voice in voices:
                    marker = " (current)" if voice == tts.speaker else ""
                    print(f"    - {voice}{marker}")
            else:
                print("  No voice presets found.")
                print("  Check that VibeVoice demo/voices/streaming_model/ directory exists.")
        except Exception as e:
            print(f"  Error loading VibeVoice: {e}")
        print("=" * 60)
        return

    if args.html_only:
        print("\n" + "=" * 60)
        print("HTML-ONLY MODE - Regenerating from latest bundle")
        print("=" * 60)

        latest_bundle_path = OUTPUT_DIR / "daily_bundle_latest.json"
        if not latest_bundle_path.exists():
            print(f"ERROR: No bundle found at {latest_bundle_path}")
            print("Run without --html-only first to generate a bundle.")
            return

        bundle = load_bundle_json(latest_bundle_path)
        print(f"✓ Loaded bundle from {bundle['meta']['date_iso']}")

        year_progress_html = render_year_progress_html_from_bundle(bundle)
        progress_html_path = save_to_output_dir(year_progress_html, "year_progress_report.html")
        print(f"✓ Progress HTML: {progress_html_path}")

        newsletter_html = render_newsletter_html_from_bundle(bundle)
        newsletter_html_path = save_to_output_dir(newsletter_html, "news_newsletter_report.html")
        print(f"✓ Newsletter HTML: {newsletter_html_path}")

        print("\n✓ HTML regeneration complete!")
        print(f"  View: {OUTPUT_DIR}/news_newsletter_report.html")
        return

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
    if args.jetson:
        mode_flags.append(f"JETSON_TTS(voice={args.voice})")

    if mode_flags:
        print(f"\n⚙️  Mode: {', '.join(mode_flags)}")

    logger.info("Output directory: %s", OUTPUT_DIR.absolute())

    days_completed, weeks_completed, days_left, weeks_left, percent_days_left = time_left_in_year()
    logger.info("Year progress: %.1f%% complete, %s days remaining", 100 - percent_days_left, days_left)

    logger.info("Fetching weather data...")
    temp, status = get_weather()
    weather_data = {
        "temp_c": temp,
        "status": status,
        "location": "North Plains, OR",
    }
    logger.info("Weather: %s°C, %s", temp, status)

    logger.info("Generating quote and lesson...")
    random_personality = get_random_personality()
    logger.info("Selected personality: %s", random_personality)

    quote_text = generate_quote(random_personality)
    logger.info("Quote result: %s - %s...", "SUCCESS" if quote_text else "EMPTY", quote_text[:80] if quote_text else "N/A")

    topic, lesson_raw = get_random_lesson(LLM_PROVIDER)
    logger.info("Topic: %s...", topic[:50])
    logger.info("Lesson result: %s - %s chars", "SUCCESS" if lesson_raw else "EMPTY", len(lesson_raw) if lesson_raw else 0)

    lesson_dict = parse_lesson_to_dict(lesson_raw, topic)

    print("\n" + "=" * 80)
    print("FETCHING NEWS UPDATES")
    print("=" * 80)

    news_raw_sources = None

    if args.use_cache and not args.refresh_cache:
        cached_news = load_news_cache(output_dir=OUTPUT_DIR)
        if cached_news:
            print("\n📦 Using cached news data...")
            news_raw_sources = cached_news
            print(f"✓ Tech news: {len(news_raw_sources.get('technology', ''))} characters (cached)")
            print(f"✓ Financial news: {len(news_raw_sources.get('financial', ''))} characters (cached)")
            print(f"✓ India news: {len(news_raw_sources.get('india', ''))} characters (cached)")

    if news_raw_sources is None:
        news_provider = "litellm" if LLM_PROVIDER == "litellm" else "ollama"

        print("\n📰 Fetching Technology News...")
        news_update_tech = gather_daily_news(
            category="technology",
            max_sources=10,
            aggregator_limit=1,
            freshness_hours=24,
            provider=news_provider,
        )
        print(f"✓ Tech news: {len(news_update_tech)} characters")

        print("\n📈 Fetching Financial Markets News...")
        news_update_financial = gather_daily_news(
            category="financial",
            max_sources=10,
            aggregator_limit=1,
            freshness_hours=24,
            provider=news_provider,
        )
        print(f"✓ Financial news: {len(news_update_financial)} characters")

        print("\n🇮🇳 Fetching India News...")
        news_update_india = gather_daily_news(
            category="india",
            max_sources=10,
            aggregator_limit=0,
            freshness_hours=24,
            provider=news_provider,
        )
        print(f"✓ India news: {len(news_update_india)} characters")

        news_raw_sources = {
            "technology": news_update_tech,
            "financial": news_update_financial,
            "india": news_update_india,
        }

        print("\n💾 Saving news to cache...")
        save_news_cache(news_raw_sources, output_dir=OUTPUT_DIR)

    news_update_tech = news_raw_sources.get("technology", "")
    news_update_financial = news_raw_sources.get("financial", "")
    news_update_india = news_raw_sources.get("india", "")

    print("\n" + "=" * 80)
    print("GENERATING NEWSLETTER")
    print("=" * 80)

    print("\n📝 Generating newsletter sections...")
    newsletter_sections = generate_newsletter_sections(
        news_update_tech,
        news_update_financial,
        news_update_india,
        LLM_PROVIDER,
    )
    print(f"✓ Newsletter sections: tech={len(newsletter_sections['tech'])}, financial={len(newsletter_sections['financial'])}, india={len(newsletter_sections['india'])}")

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
    voicebot_script = generate_gpt_response_voicebot(voicebot_prompt, LLM_PROVIDER)
    print(f"✓ Voicebot script: {len(voicebot_script)} characters")

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
        voicebot_script=voicebot_script,
    )
    print("✓ Bundle built successfully")

    print("\n💾 Writing JSON bundle files...")
    bundle_path = write_bundle_json(bundle, OUTPUT_DIR)
    print(f"✓ Bundle saved to: {bundle_path}")
    print(f"✓ Latest bundle: {OUTPUT_DIR / 'daily_bundle_latest.json'}")

    print("\n🎨 Rendering HTML from bundle...")
    year_progress_html = render_year_progress_html_from_bundle(bundle)
    progress_html_path = save_to_output_dir(year_progress_html, "year_progress_report.html")
    print(f"✓ Progress HTML: {progress_html_path}")

    newsletter_html = render_newsletter_html_from_bundle(bundle)
    newsletter_html_path = save_to_output_dir(newsletter_html, "news_newsletter_report.html")
    print(f"✓ Newsletter HTML: {newsletter_html_path}")

    print("\n📧 Sending emails...")
    year_progress_subject = "Year Progress Report 📅"
    send_email(year_progress_subject, year_progress_html, is_html=True, skip_send=args.skip_email)

    news_update_subject = "📰 Your Daily News Briefing"
    send_email(news_update_subject, newsletter_html, is_html=True, skip_send=args.skip_email)

    if args.skip_audio:
        print("\n🔊 Audio generation skipped (--skip-audio)")
    else:
        if args.jetson:
            print(f"\n🔊 Generating audio files with VibeVoice (on-device, voice={args.voice})...")
        else:
            print("\n🔊 Generating audio files with NVIDIA Riva (cloud API)...")

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

        year_progress_gpt_response = generate_gpt_response(year_progress_message_prompt, LLM_PROVIDER)
        yearprogress_tts_output_path = str(OUTPUT_DIR / "year_progress_report.mp3")

        if args.jetson:
            # Use on-device VibeVoice TTS
            tts_result = vibevoice_text_to_speech(
                year_progress_gpt_response,
                yearprogress_tts_output_path,
                speaker=args.voice
            )
            if tts_result:
                actual_path = yearprogress_tts_output_path.replace('.mp3', '.wav')
                print(f"✓ Progress audio: {actual_path}")
            else:
                print("⚠ Progress audio: VibeVoice TTS failed")
        else:
            # Use cloud-based NVIDIA Riva TTS
            model_name = random.choice(model_names)
            tts_result = text_to_speech_nospeak(year_progress_gpt_response, yearprogress_tts_output_path, model_name=model_name, speed=1.5)
            if tts_result:
                print(f"✓ Progress audio: {yearprogress_tts_output_path}")
            else:
                print("⚠ Progress audio: TTS unavailable (requires NVIDIA Riva service)")

        news_tts_output_path = str(OUTPUT_DIR / "news_update_report.mp3")

        if args.jetson:
            # Use on-device VibeVoice TTS
            tts_result = vibevoice_text_to_speech(
                voicebot_script,
                news_tts_output_path,
                speaker=args.voice
            )
            if tts_result:
                actual_path = news_tts_output_path.replace('.mp3', '.wav')
                print(f"✓ News audio: {actual_path}")
            else:
                print("⚠ News audio: VibeVoice TTS failed")
        else:
            # Use cloud-based NVIDIA Riva TTS
            model_name = random.choice(model_names)
            tts_result = text_to_speech_nospeak(voicebot_script, news_tts_output_path, model_name=model_name, speed=1.5)
            if tts_result:
                print(f"✓ News audio: {news_tts_output_path}")
            else:
                print("⚠ News audio: TTS unavailable (requires NVIDIA Riva service)")

    print("\n" + "=" * 80)
    print("✓ ALL TASKS COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"\nOutput files in {OUTPUT_DIR}:")
    print(f"  - daily_bundle_{bundle['meta']['date_iso']}.json")
    print("  - daily_bundle_latest.json")
    print("  - year_progress_report.html")
    print("  - news_newsletter_report.html")

    if args.skip_email:
        print("\n📧 Emails were skipped (use without --skip-email to send)")
    if args.skip_audio:
        print("\n🔊 Audio was skipped (use without --skip-audio to generate)")

    print(f"\n💡 Quick commands:")
    print(f"  View newsletter: firefox {OUTPUT_DIR}/news_newsletter_report.html")
    print(f"  Test mode:       python {__file__} --test")
    print(f"  HTML only:       python {__file__} --html-only")
    print(f"  Jetson TTS:      python {__file__} --jetson --voice en-davis_man")
    print(f"  List voices:     python {__file__} --list-voices")


if __name__ == "__main__":
    main()
