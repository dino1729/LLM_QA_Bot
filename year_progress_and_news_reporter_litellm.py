"""
Year Progress and News Reporter (LiteLLM Version)
================================================

This script orchestrates the generation of a daily newsletter bundle. It performs the following tasks:
1. Calculates year and quarter progress.
2. Fetches weather data.
3. Selects a random personality and generates a motivational quote.
4. Generates a daily lesson using an LLM.
5. Gathers and aggregates news from Technology, Finance, and India categories.
6. Summarizes news into newsletter sections and a voice-friendly script.
7. (Optional) Generates a two-persona podcast dialogue based on the news briefing.
8. Persists all state into a versioned JSON bundle for caching and future reference.
9. Renders HTML reports and sends emails.
10. Generates high-quality TTS audio using NVIDIA Riva, VibeVoice, or Chatterbox (GPU).

CLI OPTIONS:
------------
--test, -t              Testing mode: use cache, skip email and audio.
--use-cache, -c         Use cached news data (raw source text) if available.
--full-cache            Use full bundle cache (skip all LLM generation including transcripts).
--refresh-cache         Force refresh of news cache even if valid cache exists.
--skip-email, -e        Skip sending emails.
--skip-audio, -a        Skip generating TTS audio.
--skip-news-audio       Skip generating news TTS audio only (still does progress audio).
--progress-only, -p     Run year progress flow only, skip all news generation.
--cache-info            Show cache status and exit.
--html-only             Only regenerate HTML from the latest bundle (no API calls).
--review                Review generated transcripts and wait for confirmation before audio synthesis.

TTS OPTIONS (Chatterbox is the default):
--riva-tts              Use NVIDIA Riva cloud TTS instead of default Chatterbox.
--local-tts             Use on-device VibeVoice TTS (local GPU) instead of Chatterbox.
--chatterbox-tts        Explicit Chatterbox TTS (default behavior, kept for compatibility).
                        LLM text generation includes paralinguistic tags
                        ([laugh], [chuckle], [cough]) for expressive speech synthesis.
--podcast               Transform the news briefing into a two-persona dialogue (Rick and Morty style).
--voice VOICE           Voice preset for VibeVoice TTS (default: en-davis_man).
--progress-voice VOICE  Voice for year progress report (Chatterbox mode).
--news-voice VOICE      Voice for news briefing (Chatterbox mode).
--list-voices           List available VibeVoice voice presets and exit.
--list-chatterbox-voices List available Chatterbox voice files in voices/ folder and exit.
"""

import argparse
import logging
import random
from datetime import datetime
from pathlib import Path

# Helper functions for audio, email, and HTML rendering
from helper_functions.audio_processors import text_to_speech_nospeak
from helper_functions.email_utils import send_email
from helper_functions.html_templates import (
    format_news_items_html,
    format_news_section,
    generate_html_news_template,
    render_newsletter_html_from_bundle,
    render_year_progress_html_from_bundle,
)
# Helper functions for lessons, personalities, and fallbacks
from helper_functions.lesson_utils import (
    get_random_lesson,
    get_random_personality,
    generate_lesson_response as _generate_lesson_response,
    parse_lesson_to_dict,
    _get_fallback_lesson,
)
# Cache management for news sources
from helper_functions.news_cache import (
    CACHE_MAX_AGE_HOURS,
    NEWS_CACHE_FILENAME,
    get_cache_info,
    load_news_cache,
    save_news_cache,
)
from helper_functions.news_researcher import gather_daily_news
# LLM generation and parsing
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
# State management and bundle building
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

# VibeVoice TTS instances (lazy-loaded for --local-tts mode)
# Cache one TTS instance per voice for efficiency
_vibevoice_tts_cache = {}

# Chatterbox TTS instances (lazy-loaded for --chatterbox-tts mode)
# Cache one TTS instance per model type
_chatterbox_tts_cache = {}


def get_chatterbox_tts(model_type: str = "turbo"):
    """
    Get or create a Chatterbox TTS instance for the specified model type.
    Uses a cache to reuse TTS instances for the same model, improving performance.
    """
    global _chatterbox_tts_cache

    if model_type not in _chatterbox_tts_cache:
        from helper_functions.tts_chatterbox import ChatterboxTTS
        from config import config

        tts = ChatterboxTTS(
            model_type=model_type,
            cfg_weight=config.chatterbox_tts_cfg_weight,
            exaggeration=config.chatterbox_tts_exaggeration,
            device=config.chatterbox_tts_device,
        )
        _chatterbox_tts_cache[model_type] = tts

    return _chatterbox_tts_cache[model_type]


def chatterbox_text_to_speech(
    text: str,
    output_path: str,
    voice_name: str = None,
    cfg_weight: float = None,
    exaggeration: float = None,
) -> bool:
    """
    Convert text to speech using Chatterbox TTS with voice cloning.
    Uses the shared split_text_for_chatterbox() function for sentence splitting.
    
    Args:
        text: The full transcript to synthesize.
        output_path: Path to save the resulting .wav file.
        voice_name: The voice file to clone (must exist in voices/{voice_name}.wav).
        cfg_weight: Classifier-free guidance weight (0.0-1.0).
        exaggeration: Speech exaggeration level (0.0-1.0).
    
    Returns:
        True if audio was successfully generated and saved, False otherwise.
    """
    from pathlib import Path
    from config import config
    from helper_functions.tts_chatterbox import split_text_for_chatterbox
    import numpy as np

    try:
        tts = get_chatterbox_tts(model_type=config.chatterbox_tts_model_type)

        # Use provided voice_name, or fall back to config defaults
        actual_voice = voice_name or config.newsletter_progress_voice or config.chatterbox_tts_default_voice
        if not actual_voice:
            raise ValueError(
                "No voice specified for Chatterbox TTS. Please provide one of:\n"
                "  1. --progress-voice or --news-voice CLI argument\n"
                "  2. 'newsletter_progress_voice' or 'newsletter_news_voice' in config.yml\n"
                "  3. 'chatterbox_tts_default_voice' in config.yml"
            )
        
        # Construct voice prompt path
        voice_path = Path(f"voices/{actual_voice}.wav")
        if not voice_path.exists():
            print(f"⚠ Voice file not found: {voice_path}")
            return False

        print(f"  Using Chatterbox voice: {actual_voice}")
        
        # Use shared sentence splitting function
        chunks = split_text_for_chatterbox(text, max_chars=300)
        if not chunks:
            print("⚠ Chatterbox: No text to synthesize after splitting")
            return False
        
        combined_audio = []
        print(f"  Synthesizing {len(chunks)} segments...")

        for chunk in chunks:
            # Each chunk is already properly sized by split_text_for_chatterbox
            audio_chunk = tts.synthesize(
                chunk,
                audio_prompt_path=str(voice_path),
                cfg_weight=cfg_weight,
                exaggeration=exaggeration,
            )
            
            if audio_chunk is not None and len(audio_chunk) > 0:
                combined_audio.append(audio_chunk)
                # Add a 0.1s silence buffer between segments for natural pacing
                silence = np.zeros(int(tts.sample_rate * 0.1), dtype=np.float32)
                combined_audio.append(silence)

        if not combined_audio:
            print("⚠ Chatterbox: No audio generated for any segments")
            return False

        # Concatenate all generated audio arrays
        final_audio = np.concatenate(combined_audio)

        # Ensure output is WAV (Chatterbox produces WAV directly)
        actual_output = str(output_path).replace('.mp3', '.wav')
        success = tts.save_audio(final_audio, actual_output)

        if success:
            duration = len(final_audio) / tts.sample_rate
            print(f"✓ Chatterbox: {duration:.1f}s audio saved to {actual_output} (voice: {actual_voice})")
            return True
        else:
            return False

    except Exception as e:
        print(f"⚠ Chatterbox error: {e}")
        return False


def get_vibevoice_tts(speaker: str = "wayne"):
    """
    Get or create a VibeVoice TTS instance for the specified speaker.
    Allows voice switching on the same instance for efficiency.
    """
    global _vibevoice_tts_cache

    speaker_key = speaker.lower()

    if speaker_key in _vibevoice_tts_cache:
        tts = _vibevoice_tts_cache[speaker_key]
        if tts.speaker != speaker_key:
            tts.speaker = speaker_key
            tts._load_voice_prompt()
        return tts

    from helper_functions.tts_vibevoice import VibeVoiceTTS
    tts = VibeVoiceTTS(speaker=speaker, use_gpu=True)
    _vibevoice_tts_cache[speaker_key] = tts
    return tts


def vibevoice_text_to_speech(
    text: str,
    output_path: str,
    speaker: str = "wayne",
    temperature: float = 1.1,
    top_p: float = 0.95,
    cfg_scale: float = 2.0,
    excitement_level: str = "high",
    speed_multiplier: float = 1.15
) -> bool:
    """
    Convert text to speech using on-device VibeVoice TTS.
    Optimized for excitement and varied speed.
    """
    import soundfile as sf
    from pathlib import Path
    from scipy import signal
    import numpy as np

    # Excitement presets mapping
    excitement_presets = {
        "low": {"temperature": 0.7, "top_p": 0.85, "cfg_scale": 1.5},
        "medium": {"temperature": 0.9, "top_p": 0.9, "cfg_scale": 1.8},
        "high": {"temperature": 1.1, "top_p": 0.95, "cfg_scale": 2.0},
        "very_high": {"temperature": 1.3, "top_p": 0.98, "cfg_scale": 2.5},
    }

    if excitement_level in excitement_presets:
        preset = excitement_presets[excitement_level]
        temperature = preset["temperature"]
        top_p = preset["top_p"]
        cfg_scale = preset["cfg_scale"]

    try:
        tts = get_vibevoice_tts(speaker)
        formatted_text = text

        audio_data = tts.synthesize(
            formatted_text,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            cfg_scale=cfg_scale
        )

        if audio_data is None or len(audio_data) == 0:
            print("⚠ VibeVoice: No audio generated")
            return False

        # Apply time-stretching if multiplier is not 1.0
        if speed_multiplier != 1.0:
            try:
                import pyrubberband as pyrb
                audio_data = pyrb.time_stretch(audio_data, tts.sample_rate, speed_multiplier)
            except ImportError:
                try:
                    import librosa
                    audio_data = librosa.effects.time_stretch(y=audio_data, sr=tts.sample_rate, rate=speed_multiplier)
                except ImportError:
                    num_samples = int(len(audio_data) / speed_multiplier)
                    audio_data = signal.resample(audio_data, num_samples)

        actual_output = str(output_path).replace('.mp3', '.wav')
        sf.write(actual_output, audio_data, tts.sample_rate)

        duration = len(audio_data) / tts.sample_rate
        print(f"✓ VibeVoice: {duration:.1f}s audio saved to {actual_output}")
        return True

    except Exception as e:
        print(f"⚠ VibeVoice error: {e}")
        return False


# Logging configuration
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(funcName)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("newsletter_research_data")
LLM_PROVIDER = "litellm"
model_names = ["LITELLM_SMART", "LITELLM_STRATEGIC"] if LLM_PROVIDER == "litellm" else ["OLLAMA_SMART", "OLLAMA_STRATEGIC"]


def generate_quote(random_personality, model_tier: str):
    """Wrapper for quote generation using the unified LLM client."""
    import helper_functions.quote_utils as qu
    original = qu.get_client
    qu.get_client = get_client
    try:
        return qu.generate_quote(random_personality, LLM_PROVIDER, model_tier=model_tier)
    finally:
        qu.get_client = original


def generate_lesson_response(user_message, model_tier: str):
    """Wrapper for lesson generation using the unified LLM client."""
    import helper_functions.lesson_utils as lu
    original = lu.get_client
    lu.get_client = get_client
    try:
        return lu.generate_lesson_response(user_message, LLM_PROVIDER, model_tier=model_tier)
    finally:
        lu.get_client = original


def parse_arguments():
    """Defines and parses CLI arguments for the reporter."""
    parser = argparse.ArgumentParser(
        description="Year Progress and News Reporter - Generate daily newsletters",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--test', '-t', action='store_true', help='Testing mode: use cache, skip email/audio')
    parser.add_argument('--use-cache', '-c', action='store_true', help='Use cached news data if available')
    parser.add_argument('--full-cache', action='store_true', help='Use full bundle cache (skip all LLM generation)')
    parser.add_argument('--refresh-cache', action='store_true', help='Force refresh of news cache')
    parser.add_argument('--skip-email', '-e', action='store_true', help='Skip sending emails')
    parser.add_argument('--skip-audio', '-a', action='store_true', help='Skip generating TTS audio')
    parser.add_argument('--skip-news-audio', action='store_true', help='Skip news TTS audio only')
    parser.add_argument('--progress-only', '-p', action='store_true', help='Run year progress flow only, skip all news generation')
    parser.add_argument('--cache-info', action='store_true', help='Show cache status and exit')
    parser.add_argument('--html-only', action='store_true', help='Regenerate HTML from latest bundle')
    parser.add_argument('--review', action='store_true', help='Review transcripts before audio generation')
    parser.add_argument('--local-tts', action='store_true', help='Use VibeVoice local TTS')
    parser.add_argument('--chatterbox-tts', action='store_true', help='Use Chatterbox TTS (default, kept for compatibility)')
    parser.add_argument('--riva-tts', action='store_true', help='Use NVIDIA Riva cloud TTS instead of Chatterbox')
    parser.add_argument('--podcast', action='store_true', help='Generate two-persona news podcast dialogue')
    parser.add_argument('--voice', type=str, default='en-davis_man', help='VibeVoice preset')
    parser.add_argument('--progress-voice', type=str, default=None, help='Chatterbox progress voice')
    parser.add_argument('--news-voice', type=str, default=None, help='Chatterbox news voice')
    parser.add_argument('--list-voices', action='store_true', help='List VibeVoice presets')
    parser.add_argument('--list-chatterbox-voices', action='store_true', help='List Chatterbox voices')

    return parser.parse_args()


def main():
    args = parse_arguments()

    if args.test:
        args.use_cache = True
        args.skip_email = True
        args.skip_audio = True

    # Chatterbox is now the default TTS - use_chatterbox is True unless riva-tts or local-tts is specified
    use_chatterbox = not args.riva_tts and not args.local_tts

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    news_cache_file = OUTPUT_DIR / NEWS_CACHE_FILENAME

    # Handle informational flags
    if args.cache_info:
        cache_info = get_cache_info(output_dir=OUTPUT_DIR)
        print("\n" + "=" * 60 + "\nNEWS CACHE STATUS\n" + "=" * 60)
        if cache_info:
            print(f"  Status: {'VALID' if cache_info.get('is_valid') else 'EXPIRED'}")
            print(f"  Age: {cache_info['age_hours']:.1f} hours")
        else:
            print("  Status: No cache found")
        return

    if args.list_chatterbox_voices:
        print("\n" + "=" * 60 + "\nCHATTERBOX AVAILABLE VOICES\n" + "=" * 60)
        voices_dir = Path("voices")
        if voices_dir.exists():
            for v in sorted(voices_dir.glob("*.wav")):
                print(f"    - {v.stem}")
        return

    # Handle --html-only mode (No state generation, just rendering)
    if args.html_only:
        latest_bundle_path = OUTPUT_DIR / "daily_bundle_latest.json"
        if not latest_bundle_path.exists():
            print(f"ERROR: No bundle found at {latest_bundle_path}")
            return
        bundle = load_bundle_json(latest_bundle_path)
        year_progress_html = render_year_progress_html_from_bundle(bundle)
        save_to_output_dir(year_progress_html, "year_progress_report.html")
        newsletter_html = render_newsletter_html_from_bundle(bundle)
        save_to_output_dir(newsletter_html, "news_newsletter_report.html")
        print("✓ HTML regeneration complete!")
        return

    logger.info("=" * 80)
    logger.info("STARTING YEAR PROGRESS AND NEWS REPORTER")
    logger.info("=" * 80)

    # State variables
    quote_text = None
    quote_author = None
    lesson_dict = None
    lesson_raw = None
    news_raw_sources = None
    newsletter_sections = None
    voicebot_script = None
    year_progress_gpt_response = None
    podcast_turns = None

    # [CACHING] Load state from bundle if --full-cache is used
    if args.full_cache:
        print("\n" + "=" * 80 + "\nFULL CACHE MODE - Loading state\n" + "=" * 80)
        latest_bundle_path = OUTPUT_DIR / "daily_bundle_latest.json"
        if latest_bundle_path.exists():
            try:
                bundle = load_bundle_json(latest_bundle_path)
                quote_text = bundle["progress"]["quote"]["text"]
                quote_author = bundle["progress"]["quote"]["author"]
                lesson_dict = bundle["progress"]["lesson"]
                lesson_raw = str(lesson_dict)
                news_raw_sources = bundle["news"]["raw_sources"]
                newsletter_sections = bundle["news"]["newsletter"]["sections"]
                voicebot_script = bundle["news"]["newsletter"]["voicebot_script"]
                
                from helper_functions.dialogue_engine import DialogueTurn
                cached_turns = bundle["news"]["newsletter"].get("podcast_transcript", [])
                if cached_turns:
                    podcast_turns = [DialogueTurn.from_dict(t) for t in cached_turns]
                
                year_progress_gpt_response = bundle["progress"].get("year_progress_script", None)
                print(f"✓ Loaded full state from {bundle['meta']['date_iso']}")
            except Exception as e:
                print(f"⚠ Failed to load full cache: {e}")

    # Core data fetching
    days_completed, weeks_completed, days_left, weeks_left, percent_days_left = time_left_in_year()
    temp, status = get_weather()
    weather_data = {"temp_c": temp, "status": status, "location": "North Plains, OR"}

    # Import config early for accessing tier settings
    from config import config
    
    # Generate or use cached Quote & Lesson
    if not quote_text or not quote_author:
        random_personality = get_random_personality()
        quote_author = random_personality
        print(f"📝 Generating quote using LLM tier: {config.newsletter_progress_llm_tier}")
        quote_text = generate_quote(random_personality, model_tier=config.newsletter_progress_llm_tier)
    else:
        random_personality = quote_author

    if not lesson_dict:
        topic, lesson_raw = get_random_lesson(LLM_PROVIDER, model_tier=config.newsletter_progress_llm_tier)
        lesson_dict = parse_lesson_to_dict(lesson_raw, topic)

    # [NEWS GENERATION] Skip entirely if --progress-only mode
    news_update_tech = ""
    news_update_financial = ""
    news_update_india = ""
    is_podcast = False

    if not args.progress_only:
        # [CACHING] Handle raw news sources
        if news_raw_sources is None:
            if args.use_cache and not args.refresh_cache:
                news_raw_sources = load_news_cache(output_dir=OUTPUT_DIR)

        if news_raw_sources is None:
            news_provider = "litellm" if LLM_PROVIDER == "litellm" else "ollama"
            news_raw_sources = {
                "technology": gather_daily_news(category="technology", provider=news_provider),
                "financial": gather_daily_news(category="financial", provider=news_provider),
                "india": gather_daily_news(category="india", provider=news_provider),
            }
            save_news_cache(news_raw_sources, output_dir=OUTPUT_DIR)

        news_update_tech = news_raw_sources.get("technology", "")
        news_update_financial = news_raw_sources.get("financial", "")
        news_update_india = news_raw_sources.get("india", "")

        # [CACHING] Newsletter Sections Generation
        if not newsletter_sections:
            print(f"📝 Generating newsletter sections using LLM tier: {config.newsletter_news_llm_tier}")
            newsletter_sections = generate_newsletter_sections(news_update_tech, news_update_financial, news_update_india, LLM_PROVIDER, config.newsletter_news_llm_tier)

        # [CACHING] Voicebot script (Single Voice)
        if not voicebot_script:
            voicebot_prompt = f"Transform news into an anchor-style script:\n{news_update_tech}\n{news_update_financial}\n{news_update_india}"
            # Use CLI arg first, then config, for persona (voice cloning uses same persona for text generation)
            news_persona = (args.news_voice or config.newsletter_news_voice) if use_chatterbox else None
            print(f"📝 Generating news voicebot script using LLM tier: {config.newsletter_news_llm_tier}, persona: {news_persona}")
            voicebot_script = generate_gpt_response_voicebot(
                voicebot_prompt, LLM_PROVIDER, config.newsletter_news_llm_tier,
                voice_persona=news_persona,
                use_chatterbox=use_chatterbox
            )

        # [CACHING] Podcast Dialogue Generation (Two Personas)
        is_podcast = args.podcast or config.podcast_enabled

        if is_podcast and not podcast_turns:
            print("\n🎙️ Generating two-persona podcast dialogue...")
            from helper_functions.dialogue_engine import DialogueEngine
            engine = DialogueEngine(config, use_chatterbox=use_chatterbox)
            context = f"NEWS BRIEFING TRANSCRIPT:\n{voicebot_script}\n\nRAW DATA:\n{news_update_tech}\n{news_update_financial}\n{news_update_india}"
            topic = f"News Briefing for {datetime.now().strftime('%B %d, %Y')}"
            podcast_turns = engine.run_conversation(topic, context)
    else:
        print("\n📅 Progress-only mode: skipping news generation")

    # [CACHING] Year Progress script
    if not year_progress_gpt_response:
        # Use CLI arg first, then config, for persona (voice cloning uses same persona for text generation)
        progress_persona = (args.progress_voice or config.newsletter_progress_voice) if use_chatterbox else None
        
        # Build comprehensive prompt with all progress data
        lesson_text = ""
        if lesson_dict:
            lesson_text = f"""
DAILY LESSON - Topic: {lesson_dict.get('topic', 'Leadership')}
Key Insight: {lesson_dict.get('key_insight', '')}
Historical Example: {lesson_dict.get('historical', '')}
Application: {lesson_dict.get('application', '')}
"""
        
        progress_prompt = f"""Create a spoken year progress announcement that covers ALL of the following:

YEAR PROGRESS:
- Days completed: {days_completed} out of 365
- Days remaining: {days_left}
- Weeks completed: {weeks_completed}
- Percent complete: {round(100 - percent_days_left, 1)}%

WEATHER TODAY:
- Location: {weather_data.get('location', 'Unknown')}
- Temperature: {weather_data.get('temp_c', 'N/A')}°C
- Conditions: {weather_data.get('status', 'Unknown')}

INSPIRATIONAL QUOTE:
"{quote_text}" - {quote_author}
{lesson_text}
YOU MUST include and discuss:
1. The year progress statistics (days completed, days left, percentage)
2. Today's weather conditions
3. The inspirational quote and its meaning
4. The daily lesson - explain the key insight, share the historical example, and describe how to apply it

Make it conversational, motivating, and tie everything together naturally."""
        
        print(f"📝 Generating year progress script using LLM tier: {config.newsletter_progress_llm_tier}, persona: {progress_persona}")
        year_progress_gpt_response = generate_gpt_response(
            progress_prompt, LLM_PROVIDER, config.newsletter_progress_llm_tier,
            voice_persona=progress_persona,
            use_chatterbox=use_chatterbox
        )

    # Build and Save the Daily Bundle (The primary cache source)
    podcast_transcript_dicts = [t.to_dict() for t in podcast_turns] if podcast_turns else []
    bundle = build_daily_bundle(
        days_completed, weeks_completed, days_left, weeks_left, percent_days_left,
        weather_data, quote_text, quote_author, lesson_dict, news_raw_sources, 
        newsletter_sections, voicebot_script, year_progress_gpt_response, podcast_transcript_dicts
    )
    write_bundle_json(bundle, OUTPUT_DIR)

    # Render HTML
    year_progress_html = render_year_progress_html_from_bundle(bundle)
    save_to_output_dir(year_progress_html, "year_progress_report.html")

    newsletter_html = None
    if not args.progress_only:
        newsletter_html = render_newsletter_html_from_bundle(bundle)
        save_to_output_dir(newsletter_html, "news_newsletter_report.html")

    # [REVIEW] Handle transcript review before final steps
    if args.review:
        print("\n" + "=" * 80 + "\nTRANSCRIPT REVIEW\n" + "=" * 80)
        print(f"--- PROGRESS ---\n{year_progress_gpt_response}")
        if not args.progress_only and voicebot_script:
            print(f"\n--- NEWS ---\n{voicebot_script}")
            if podcast_turns:
                print("\n--- PODCAST ---\n" + "\n".join([f"[{t.character_id}]: {t.text}" for t in podcast_turns]))
        input("\nPress Enter to proceed with audio and email...")

    # Email reporting
    send_email("Year Progress Report 📅", year_progress_html, is_html=True, skip_send=args.skip_email)
    if not args.progress_only and newsletter_html:
        send_email("Your Daily News Briefing 📰", newsletter_html, is_html=True, skip_send=args.skip_email)

    # [TTS] Audio generation phase - Chatterbox is the default
    if not args.skip_audio:
        if args.riva_tts:
            # NVIDIA Riva cloud TTS
            text_to_speech_nospeak(year_progress_gpt_response, str(OUTPUT_DIR / "year_progress_report.mp3"), speed=1.5)
            if not args.progress_only and voicebot_script:
                text_to_speech_nospeak(voicebot_script, str(OUTPUT_DIR / "news_update_report.mp3"), speed=1.5)
        elif args.local_tts:
            # VibeVoice local TTS
            vibevoice_text_to_speech(year_progress_gpt_response, str(OUTPUT_DIR / "year_progress_report.wav"), speaker="en-frank_man")
            if not args.progress_only and voicebot_script:
                vibevoice_text_to_speech(voicebot_script, str(OUTPUT_DIR / "news_update_report.wav"), speaker="en-mike_man")
        else:
            # Chatterbox TTS (default) - GPU-accelerated voice cloning
            progress_voice = args.progress_voice or config.newsletter_progress_voice
            chatterbox_text_to_speech(year_progress_gpt_response, str(OUTPUT_DIR / "year_progress_report.wav"), voice_name=progress_voice)

            # Generate News Audio (Podcast or Single) - skip if progress-only
            if not args.progress_only and not args.skip_news_audio:
                if is_podcast and podcast_turns:
                    from helper_functions.podcast_orchestrator import PodcastOrchestrator
                    orchestrator = PodcastOrchestrator(config, use_chatterbox=True)
                    context = f"Transcript:\n{voicebot_script}"
                    orchestrator.generate_podcast("News Podcast", context, output_filename="news_update_report.wav", pregenerated_turns=podcast_turns)
                elif voicebot_script:
                    news_voice = args.news_voice or config.newsletter_news_voice
                    chatterbox_text_to_speech(voicebot_script, str(OUTPUT_DIR / "news_update_report.wav"), voice_name=news_voice)

    print("\n" + "=" * 80 + "\n✓ ALL TASKS COMPLETED SUCCESSFULLY!\n" + "=" * 80)


if __name__ == "__main__":
    main()
