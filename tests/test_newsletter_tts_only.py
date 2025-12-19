#!/usr/bin/env python3
"""Test the expressive TTS functionality for newsletter audio generation."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from year_progress_and_news_reporter_litellm import vibevoice_text_to_speech

OUTPUT_DIR = Path(__file__).parent.parent / "newsletter_research_data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("NEWSLETTER TTS TEST - EXPRESSIVE VOICES")
print("=" * 80)

# Test 1: Year Progress Report (Very High Excitement)
print("\n1. Year Progress Report - VERY HIGH EXCITEMENT")
print("   Voice: en-frank_man (energetic)")
year_progress_text = """
Hey there! Welcome to your year progress update for December 18th, 2025!

Let me tell you, we're absolutely CRUSHING it this year! We've completed a whopping 352 days -
that's 50 full weeks of amazing progress! Can you believe it?

And guess what? We still have 13 incredible days left - that's almost 2 full weeks to finish
the year strong! We're at 96.4% completion, which means we're in the home stretch!

Here's an inspiring quote to keep you motivated: "The only way to do great work is to love what you do!"

Now, let's talk about today's lesson on resilience and determination. The key insight here is that
success comes from consistent effort and believing in yourself, even when things get tough.
History has shown us time and time again that those who persevere are the ones who achieve
greatness!

So let's make these final days count! Stay focused, stay motivated, and let's finish this year
with a BANG!
"""

result = vibevoice_text_to_speech(
    year_progress_text,
    str(OUTPUT_DIR / "test_year_progress.mp3"),
    speaker="en-frank_man",
    excitement_level="very_high"
)
print(f"   Result: {'✓ Success!' if result else '✗ Failed'}")

# Test 2: News Briefing (High Excitement)
print("\n2. News Briefing - HIGH EXCITEMENT")
print("   Voice: en-mike_man (professional)")
news_text = """
Good morning, everyone! Here's what's making headlines today!

First up in technology - we've got some INCREDIBLE developments in AI! Major tech companies are
announcing breakthrough innovations that could transform how we work and live. This is truly
game-changing stuff, folks!

Now turning to financial markets - what an exciting day on Wall Street! The markets are showing
strong performance with tech stocks leading the charge. Investors are optimistic about the future,
and you can really feel the momentum building!

And finally, the latest from India - amazing news from the subcontinent! Innovation and growth
are at an all-time high, with new initiatives launching across multiple sectors. It's wonderful
to see such dynamic progress!

That's your briefing for today - stay informed, stay engaged, and have an absolutely fantastic day!
"""

result = vibevoice_text_to_speech(
    news_text,
    str(OUTPUT_DIR / "test_news_briefing.mp3"),
    speaker="en-mike_man",
    excitement_level="high"
)
print(f"   Result: {'✓ Success!' if result else '✗ Failed'}")

# Test 3: Calm Professional Voice (Medium Excitement)
print("\n3. Professional Update - MEDIUM EXCITEMENT")
print("   Voice: en-carter_man (warm, friendly)")
professional_text = """
Good afternoon! Let me share today's key updates with you.

We're seeing consistent progress across all our initiatives. The team has been working diligently,
and the results are starting to show. Market conditions remain favorable, and we're well-positioned
for continued growth.

On the technology front, we're implementing new solutions that will improve efficiency. The financial
outlook remains positive, and stakeholder confidence is strong.

Overall, it's been a productive period, and we're on track to meet our objectives. Thank you for
your continued engagement and support!
"""

result = vibevoice_text_to_speech(
    professional_text,
    str(OUTPUT_DIR / "test_professional.mp3"),
    speaker="en-carter_man",
    excitement_level="medium"
)
print(f"   Result: {'✓ Success!' if result else '✗ Failed'}")

# Test 4: Female Voice (High Excitement)
print("\n4. Motivational Message - HIGH EXCITEMENT (Female Voice)")
print("   Voice: en-emma_woman (confident, engaging)")
motivational_text = """
Hello everyone! What an amazing day to be alive and making progress!

I'm so excited to share that we're crushing our goals this year! Every single day brings new
opportunities, and we're seizing them with both hands!

Remember, success isn't just about the destination - it's about enjoying the journey and celebrating
every milestone along the way! And wow, have we hit some incredible milestones!

Keep that energy high, stay positive, and let's continue this amazing momentum! You've got this,
and I believe in you!
"""

result = vibevoice_text_to_speech(
    motivational_text,
    str(OUTPUT_DIR / "test_motivational.mp3"),
    speaker="en-emma_woman",
    excitement_level="high"
)
print(f"   Result: {'✓ Success!' if result else '✗ Failed'}")

print("\n" + "=" * 80)
print("✓ TTS TEST COMPLETE!")
print("=" * 80)
print(f"\nAudio files saved to: {OUTPUT_DIR.absolute()}")
print("\nListen to the different voices and excitement levels:")
print("  1. test_year_progress.wav - Very High Excitement (en-frank_man)")
print("  2. test_news_briefing.wav - High Excitement (en-mike_man)")
print("  3. test_professional.wav - Medium Excitement (en-carter_man)")
print("  4. test_motivational.wav - High Excitement (en-emma_woman)")
