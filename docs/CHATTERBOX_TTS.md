# Chatterbox TTS Integration

This document describes the Chatterbox TTS integration for GPU-accelerated, on-device text-to-speech synthesis.

## Overview

**Chatterbox TTS** by Resemble AI is a state-of-the-art text-to-speech system with three model variants:

1. **Chatterbox-Turbo** (350M params) - Default
   - Sub-200ms latency for production use
   - Supports paralinguistic tags: `[laugh]`, `[chuckle]`, `[cough]`
   - Requires reference audio for voice cloning (10 seconds recommended)
   - English only

2. **Chatterbox-Standard** (500M params)
   - Creative controls via CFG weight and exaggeration parameters
   - English only

3. **Chatterbox-Multilingual** (500M params)
   - Supports 23+ languages
   - Zero-shot voice cloning

## Installation

### For RTX 50-series GPUs (RTX 5090, 5080, etc.)

RTX 50-series uses Blackwell architecture (sm_120) which requires PyTorch 2.9.1+ with CUDA 12.8:

```bash
# Activate your virtual environment
source venv/bin/activate

# 1. Install PyTorch with CUDA 12.8 support
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 2. Install Chatterbox TTS and dependencies
pip install --no-deps chatterbox-tts
pip install torchcodec perth s3tokenizer conformer==0.3.2 diffusers==0.29.0 pykakasi==2.3.0 resemble-perth==1.0.1
pip install omegaconf pyloudnorm spacy-pkuseg

# 3. Install numpy compatible version
pip install "numpy>=1.24.0,<2.0.0" --no-build-isolation
```

### For RTX 40-series and older GPUs

```bash
source venv/bin/activate

pip install chatterbox-tts torchaudio torchcodec
pip install perth s3tokenizer conformer==0.3.2 diffusers==0.29.0 pykakasi==2.3.0 resemble-perth==1.0.1
pip install omegaconf pyloudnorm spacy-pkuseg
```

**Note**: You may see version conflict warnings with torch, transformers, gradio, etc. These are expected and safe to ignore - the newer versions installed in your environment are backward compatible with Chatterbox TTS.

## Configuration

Add these settings to `config/config.yml`:

```yaml
# Chatterbox TTS Configuration (GPU-accelerated on-device TTS)
chatterbox_tts_model_type: "turbo"  # Options: "turbo", "standard", "multilingual"
chatterbox_tts_cfg_weight: 0.5      # Lower values (0.3) for faster speech
chatterbox_tts_exaggeration: 0.5    # Higher values (0.7+) for dramatic speech
chatterbox_tts_audio_prompt_path: "/path/to/reference_audio.wav"  # Required for turbo model
```

## Usage

### Basic Usage (Python API)

```python
from helper_functions.tts_chatterbox import get_chatterbox_tts

# Initialize TTS (uses config.yml settings)
tts = get_chatterbox_tts(model_type="turbo")

# Synthesize speech with paralinguistic tags
text = "Hi there [chuckle], this is a test of the Chatterbox TTS system."
audio_data = tts.synthesize(
    text,
    audio_prompt_path="/path/to/reference_audio.wav"  # 10s recommended
)

# Save to file
tts.save_audio(audio_data, "output.wav")

# Or synthesize directly to file
tts.synthesize_to_file(
    text,
    "output.wav",
    audio_prompt_path="/path/to/reference_audio.wav"
)
```

### Multilingual Usage

```python
from helper_functions.tts_chatterbox import get_chatterbox_tts

tts = get_chatterbox_tts(model_type="multilingual")

# French
audio_fr = tts.synthesize("Bonjour, comment ça va?", language_id="fr")
tts.save_audio(audio_fr, "output_french.wav")

# Chinese
audio_zh = tts.synthesize("你好，今天天气真不错。", language_id="zh")
tts.save_audio(audio_zh, "output_chinese.wav")

# Supported languages
languages = tts.get_supported_languages()
print(languages)  # ['ar', 'da', 'de', 'el', 'en', 'es', 'fi', 'fr', ...]
```

### Advanced Parameters

```python
# Control speech characteristics
audio = tts.synthesize(
    text,
    audio_prompt_path="/path/to/reference.wav",
    cfg_weight=0.3,      # Lower for faster speech
    exaggeration=0.7,    # Higher for dramatic/expressive speech
)
```

### Test Script

A standalone test script is provided:

```bash
# Activate venv
source venv/bin/activate

# Test with turbo model (requires reference audio)
python test_chatterbox_tts.py \
    --text "Hello world [chuckle]" \
    --audio-prompt path/to/reference.wav \
    --output test_output.wav \
    --play

# Test with multilingual model
python test_chatterbox_tts.py \
    --model multilingual \
    --text "Bonjour le monde" \
    --language fr \
    --output french_output.wav

# Custom parameters
python test_chatterbox_tts.py \
    --text "This is dramatic [laugh]" \
    --audio-prompt path/to/reference.wav \
    --cfg-weight 0.3 \
    --exaggeration 0.8 \
    --play
```

## Paralinguistic Tags (Turbo Model Only)

The Turbo model supports emotional and non-speech sounds:

- `[laugh]` - Natural laughter
- `[chuckle]` - Light chuckle
- `[cough]` - Cough sound

Example:
```python
text = "Well [chuckle], that's interesting [laugh]. Let me think about that."
```

## Performance

### GPU Memory Usage

- **Turbo**: ~2-3 GB VRAM
- **Standard/Multilingual**: ~4-5 GB VRAM

### Latency

- **Turbo**: Sub-200ms first-chunk latency
- **Standard/Multilingual**: ~300-500ms

Example output:
```
-> GPU detected: NVIDIA GeForce RTX 5090
-> GPU memory: 32.00 GB
-> Model loaded on GPU successfully
-> GPU memory allocated: 2.34 GB
-> Generated 5.23s audio in 0.18s (RTF: 0.03x)
```

## Integration with Existing Workflows

### Newsletter Generation

Modify `year_progress_and_news_reporter_litellm.py` to use Chatterbox:

```python
from helper_functions.tts_chatterbox import get_chatterbox_tts

# Initialize TTS
tts = get_chatterbox_tts(model_type="turbo")

# Generate newsletter audio
tts.synthesize_to_file(
    newsletter_text,
    "newsletter_audio.wav",
    audio_prompt_path=config.chatterbox_tts_audio_prompt_path
)
```

### Voice Bot Applications

```python
from helper_functions.tts_chatterbox import get_chatterbox_tts

tts = get_chatterbox_tts()

# Generate response with emotion
response = "I understand your concern [pause]. Let me help you with that."
audio = tts.synthesize(response, audio_prompt_path="voice_reference.wav")
tts.play_audio_with_amplitude(audio)  # Play with amplitude visualization
```

## Troubleshooting

### CPU Inference on WSL2

For detailed instructions on setting up Chatterbox TTS for CPU inference (Intel processors, WSL2), including virtual environment best practices and package installation order, see:

**📖 [TROUBLESHOOTING_CHATTERBOX_TTS_CPU.md](./TROUBLESHOOTING_CHATTERBOX_TTS_CPU.md)**

This comprehensive guide covers:
- Virtual environment setup and common pitfalls
- Correct PyTorch CPU installation order
- Preventing mixed Python version issues
- Performance expectations for CPU inference
- Complete troubleshooting steps

### CUDA Errors on RTX 50-series

If you see `CUDA error: no kernel image is available`:

1. Verify PyTorch version:
   ```bash
   python -c "import torch; print(torch.__version__)"
   ```
   Should show `2.9.1+` or `2.11.0+` with `cu128`

2. Check CUDA version:
   ```bash
   python -c "import torch; print(torch.version.cuda)"
   ```
   Should show `12.8` or higher

3. Reinstall PyTorch if needed:
   ```bash
   pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
   ```

### Version Conflicts

When you see pip warnings like:
```
chatterbox-tts 0.1.6 requires torch==2.6.0, but you have torch 2.11.0
```

These are **safe to ignore**. PyTorch 2.11 is backward compatible with code written for 2.6.0. The same applies to transformers, gradio, and other dependencies.

### Missing Reference Audio

For the Turbo model, if you don't have reference audio:

1. Record a 10-second sample of the desired voice
2. Or use a pre-existing audio file
3. Set path in `config.yml` or pass to `audio_prompt_path` parameter

### Out of Memory

If you run out of GPU memory:

1. Use the Turbo model (smaller footprint)
2. Clear GPU cache before loading:
   ```python
   import torch
   torch.cuda.empty_cache()
   ```
3. Close other GPU applications

## API Reference

See `helper_functions/tts_chatterbox.py` for full API documentation.

### ChatterboxTTS Class

```python
class ChatterboxTTS:
    def __init__(
        model_type: str = "turbo",
        model_path: Optional[str] = None,
        cfg_weight: float = 0.5,
        exaggeration: float = 0.5,
    )

    def synthesize(
        text: str,
        audio_prompt_path: Optional[str] = None,
        language_id: Optional[str] = None,
        cfg_weight: Optional[float] = None,
        exaggeration: Optional[float] = None,
    ) -> np.ndarray

    def save_audio(audio_data: np.ndarray, output_path: str) -> bool

    def synthesize_to_file(...) -> bool

    def get_supported_languages() -> List[str]

    def get_gpu_memory_info() -> dict
```

### Helper Function

```python
def get_chatterbox_tts(
    model_type: str = "turbo",
    **kwargs
) -> ChatterboxTTS
```

Returns a cached TTS instance to avoid reloading models.

## License

Chatterbox TTS is released under the MIT License by Resemble AI.

## Resources

- **HuggingFace Model**: https://huggingface.co/ResembleAI/chatterbox-turbo
- **Demo Page**: https://resemble-ai.github.io/chatterbox_turbo_demopage/
- **GitHub**: https://github.com/resemble-ai/chatterbox
- **Discord Community**: https://discord.gg/rJq9cRJBJ6
