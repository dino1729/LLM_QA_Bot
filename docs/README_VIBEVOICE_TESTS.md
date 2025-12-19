# VibeVoice TTS Tests

This directory contains test scripts for the expressive VibeVoice TTS implementation with GPU acceleration.

## Test Scripts

### Core Functionality Tests

**test_vibevoice_gpu.py**
- Verifies GPU acceleration is working on RTX 5090
- Checks CUDA availability and device placement
- Monitors GPU memory usage
- Confirms model is running on cuda:0 with bfloat16 precision

**test_expressive_tts.py**
- Demonstrates different excitement levels (low, medium, high, very_high)
- Tests multiple voice presets (male and female)
- Generates audio samples for each combination
- Shows expressiveness parameter effects

**test_speed_adjustment.py**
- Tests pitch-preserving speed adjustment (1.0x, 1.25x, 1.5x)
- Uses librosa's time_stretch for quality time-stretching
- Compares audio duration at different speeds
- Verifies pitch preservation

**test_newsletter_tts_only.py**
- Tests complete newsletter audio generation workflow
- Year progress report (very_high excitement, en-frank_man)
- News briefing (high excitement, en-mike_man)
- Professional update (medium excitement, en-carter_man)
- Motivational message (high excitement, en-emma_woman)

## Running Tests

From the project root:
```bash
# Run specific tests
python tests/test_vibevoice_gpu.py
python tests/test_expressive_tts.py
python tests/test_speed_adjustment.py
python tests/test_newsletter_tts_only.py
```

From the tests directory:
```bash
cd tests
python test_vibevoice_gpu.py
python test_expressive_tts.py
python test_speed_adjustment.py
python test_newsletter_tts_only.py
```

## Output

All tests save audio files to `newsletter_research_data/` in the project root.

## Key Features Tested

1. **GPU Acceleration**: RTX 5090 running at 6x faster than realtime
2. **Expressiveness Control**: Temperature (0.7-1.3), top_p (0.85-0.98), cfg_scale (1.5-2.5)
3. **Multiple Voices**: 5 English voices + 20 international languages
4. **Speed Adjustment**: 1.25x default with pitch preservation via librosa
5. **Quality**: Maintains natural speech at all excitement levels and speeds

## Performance Metrics

- RTF (Real-Time Factor): ~0.16x (6x faster than realtime)
- GPU Memory: ~1.9 GB (6% of 32GB VRAM)
- Speed savings: 36% time reduction at 1.25x speed
- Pitch preservation: Maintained across all speed adjustments using librosa.effects.time_stretch
