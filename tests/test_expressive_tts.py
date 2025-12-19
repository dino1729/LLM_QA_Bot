#!/usr/bin/env python3
"""Test script to demonstrate expressive VibeVoice TTS with different voices."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import soundfile as sf
from helper_functions.tts_vibevoice import VibeVoiceTTS

OUTPUT_DIR = Path(__file__).parent.parent / "newsletter_research_data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("EXPRESSIVE VIBEVOICE TTS DEMO")
print("=" * 80)

# Test configurations
test_cases = [
    {
        "name": "Low Excitement - Calm News",
        "voice": "en-davis_man",
        "text": "Good evening. Here's a brief market update. The S&P 500 closed slightly higher today, gaining point-two percent.",
        "excitement": "low",
        "filename": "test_low_excitement.wav"
    },
    {
        "name": "Medium Excitement - Regular Report",
        "voice": "en-carter_man",
        "text": "Good morning! Let's take a look at today's technology headlines. Apple has announced a new product launch event for next month.",
        "excitement": "medium",
        "filename": "test_medium_excitement.wav"
    },
    {
        "name": "High Excitement - Breaking News",
        "voice": "en-frank_man",
        "text": "Breaking news! A major breakthrough in artificial intelligence research has just been announced! Scientists have achieved remarkable results!",
        "excitement": "high",
        "filename": "test_high_excitement.wav"
    },
    {
        "name": "Very High Excitement - Celebration",
        "voice": "en-mike_man",
        "text": "Wow! This is absolutely incredible news! The team has exceeded all expectations! What an amazing achievement! This is truly game-changing!",
        "excitement": "very_high",
        "filename": "test_very_high_excitement.wav"
    },
    {
        "name": "Female Voice - Professional",
        "voice": "en-emma_woman",
        "text": "Good afternoon, everyone! Today's progress report shows excellent results. We're making great strides toward our goals!",
        "excitement": "high",
        "filename": "test_female_professional.wav"
    },
]

# Excitement parameter mappings
excitement_params = {
    "low": {"temperature": 0.7, "top_p": 0.85, "cfg_scale": 1.5},
    "medium": {"temperature": 0.9, "top_p": 0.9, "cfg_scale": 1.8},
    "high": {"temperature": 1.1, "top_p": 0.95, "cfg_scale": 2.0},
    "very_high": {"temperature": 1.3, "top_p": 0.98, "cfg_scale": 2.5},
}

# Run tests
for i, test in enumerate(test_cases, 1):
    print(f"\n{i}. {test['name']}")
    print(f"   Voice: {test['voice']}")
    print(f"   Text: {test['text'][:70]}...")
    print(f"   Excitement: {test['excitement']}")

    params = excitement_params[test['excitement']]
    print(f"   Parameters: temp={params['temperature']}, top_p={params['top_p']}, cfg={params['cfg_scale']}")

    try:
        # Initialize TTS with the specific voice
        tts = VibeVoiceTTS(speaker=test['voice'], use_gpu=True)

        # Generate audio with expressive parameters
        audio_data = tts.synthesize(
            test['text'],
            temperature=params['temperature'],
            top_p=params['top_p'],
            do_sample=True,
            cfg_scale=params['cfg_scale']
        )

        if audio_data is not None and len(audio_data) > 0:
            output_path = OUTPUT_DIR / test['filename']
            sf.write(output_path, audio_data, tts.sample_rate)
            duration = len(audio_data) / tts.sample_rate
            print(f"   ✓ Success! Generated {duration:.2f}s audio → {output_path}")
        else:
            print(f"   ✗ Failed to generate audio")

    except Exception as e:
        print(f"   ✗ Error: {e}")

print("\n" + "=" * 80)
print("✓ TEST COMPLETE!")
print("=" * 80)
print(f"\nAudio files saved to: {OUTPUT_DIR.absolute()}")
print("\nYou can listen to the different excitement levels and voices to compare!")
print("\nVoice Characteristics:")
print("  - en-davis_man: Deep, authoritative")
print("  - en-carter_man: Warm, friendly")
print("  - en-frank_man: Energetic, dynamic")
print("  - en-mike_man: Professional, clear")
print("  - en-emma_woman: Confident, engaging")
