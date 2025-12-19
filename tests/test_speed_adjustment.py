#!/usr/bin/env python3
"""Test speed adjustment for VibeVoice TTS."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from year_progress_and_news_reporter_litellm import vibevoice_text_to_speech
import soundfile as sf

OUTPUT_DIR = Path(__file__).parent.parent / "newsletter_research_data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("SPEED ADJUSTMENT TEST")
print("=" * 80)

test_text = """
Good morning! This is a test of the speed adjustment feature.
We're going to generate the same audio at different speeds so you can compare them.
Notice how the faster version saves time while maintaining clarity and quality!
"""

# Test different speeds
speeds = [
    (1.0, "Normal speed (original)"),
    (1.15, "15% faster (new default - high quality)"),
    (1.25, "25% faster (old default)"),
    (1.5, "50% faster"),
]

print(f"\nGenerating audio at different speeds...")
print(f"Text length: {len(test_text)} characters\n")

for speed, description in speeds:
    filename = f"test_speed_{speed:.2f}x.wav"
    print(f"\n{description}")
    print(f"  Speed multiplier: {speed}x")
    print(f"  Output: {filename}")

    result = vibevoice_text_to_speech(
        test_text,
        str(OUTPUT_DIR / filename),
        speaker="en-mike_man",
        excitement_level="high",
        speed_multiplier=speed
    )

    if result:
        # Read the file to get actual duration
        audio, sr = sf.read(OUTPUT_DIR / filename)
        actual_duration = len(audio) / sr
        print(f"  ✓ Generated: {actual_duration:.1f}s")
    else:
        print(f"  ✗ Failed")

print("\n" + "=" * 80)
print("✓ SPEED TEST COMPLETE!")
print("=" * 80)
print(f"\nAudio files saved to: {OUTPUT_DIR.absolute()}")
print("\nCompare the different speeds:")
print("  1. test_speed_1.00x.wav - Normal speed (baseline)")
print("  2. test_speed_1.15x.wav - 15% faster (NEW default - high quality)")
print("  3. test_speed_1.25x.wav - 25% faster (old default)")
print("  4. test_speed_1.50x.wav - 50% faster (maximum recommended)")
print("\nThe 1.15x speed provides excellent quality with pyrubberband time stretching!")
