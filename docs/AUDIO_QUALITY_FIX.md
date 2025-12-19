# Audio Quality Fix for VibeVoice TTS

## Problem
The VibeVoice TTS generated audio with increased playback speed (1.25x) was causing vocals to fade into background noise. The issue was caused by the phase vocoder algorithm in `librosa.effects.time_stretch()`, which introduces artifacts at higher speeds.

## Solution
Implemented a multi-tier approach for high-quality time stretching:

### 1. Primary Solution: pyrubberband
- Uses the Rubber Band library (studio-quality time stretching)
- Preserves audio quality without artifacts
- Requires `rubberband-cli` system package

### 2. Fallback: librosa
- Uses phase vocoder algorithm
- May introduce minor artifacts at higher speeds
- Better than simple resampling

### 3. Last Resort: scipy resampling
- Changes pitch along with speed
- Only used if neither pyrubberband nor librosa is available

## Changes Made

### 1. Updated `year_progress_and_news_reporter_litellm.py`
- Changed default speed from 1.25x to 1.15x (better quality/speed balance)
- Implemented pyrubberband as primary time-stretching method
- Added fallback chain for compatibility
- Updated documentation

### 2. Updated `requirements.txt`
- Added `pyrubberband` for high-quality time stretching
- Added `librosa` as fallback
- Added `scipy` for signal processing

### 3. System Dependencies
- Installed `rubberband-cli` package (required by pyrubberband)

```bash
sudo apt-get install rubberband-cli
```

### 4. Updated Test Files
- Modified `tests/test_speed_adjustment.py` to include 1.15x speed test
- Updated test output messages

## Installation

To use the improved audio quality:

```bash
# Install Python packages
pip install pyrubberband librosa scipy

# Install system dependency
sudo apt-get install rubberband-cli
```

## Results

Generated test files in `newsletter_research_data/`:
- `test_speed_1.00x.wav` - Normal speed (baseline)
- `test_speed_1.15x.wav` - 15% faster (NEW default - high quality)
- `test_speed_1.25x.wav` - 25% faster (old default)
- `test_speed_1.50x.wav` - 50% faster (maximum recommended)

The 1.15x speed with pyrubberband provides excellent quality without the vocal fading issue.

## Technical Details

### Time Stretching Quality Comparison

| Method | Quality | Artifacts | Speed | Pitch Preservation |
|--------|---------|-----------|-------|-------------------|
| pyrubberband | Excellent | None | Fast | Yes |
| librosa | Good | Minor at >1.2x | Medium | Yes |
| scipy resample | Poor | Pitch shift | Fast | No |

### Code Example

```python
# Try pyrubberband first (highest quality)
import pyrubberband as pyrb
audio_data = pyrb.time_stretch(audio_data, sample_rate, speed_multiplier)
```

## Verification

Run the speed adjustment test to verify the fix:

```bash
python tests/test_speed_adjustment.py
```

Listen to the generated files to compare quality at different speeds.

