# VibeVoice TTS: Voice Configuration Guide

This repo integrates Microsoft’s **VibeVoice-Realtime** streaming TTS model (first audio chunk in ~300ms on capable hardware) via `helper_functions/tts_vibevoice.py`. VibeVoice-Realtime is a **single-speaker** model; “voice” in practice means selecting a **speaker preset** (an embedded voice prompt) plus optional **generation/expressiveness** knobs.

Upstream references:
- Repo: https://github.com/microsoft/VibeVoice
- Realtime model doc: https://github.com/microsoft/VibeVoice/blob/main/docs/vibevoice-realtime-0.5b.md

## How “voices” work (speaker presets)

VibeVoice ships speaker identity as prebuilt **voice prompt** files (`.pt`) stored under:
- `VibeVoice/demo/voices/streaming_model/**/*.pt`

This repo’s wrapper scans that directory and builds an in-memory map:
- Key: preset filename (lowercased, no extension), e.g. `en-davis_man`
- Value: absolute path to the `.pt` file

When you synthesize speech, VibeVoice uses:
- `processor.process_input_with_cached_prompt(text=..., cached_prompt=<voice_prompt>)`
- `model.generate(..., all_prefilled_outputs=<voice_prompt>)`

Upstream notes that voice prompts are provided in an embedded format to mitigate deepfake risks and reduce first-chunk latency; custom voice creation is not generally exposed.

## Selecting a voice in this repository

**CLI (newsletter generator / TTS batch runs)**
- Use on-device VibeVoice: `python year_progress_and_news_reporter_litellm.py --local-tts`
- Pick a voice preset: `python year_progress_and_news_reporter_litellm.py --local-tts --voice en-davis_man`
- List installed presets: `python year_progress_and_news_reporter_litellm.py --list-voices`

**Python API**
- `VibeVoiceTTS(speaker="en-davis_man", use_gpu=True)` (see `helper_functions/tts_vibevoice.py`)
- If the exact preset name doesn’t match, the wrapper attempts a partial match; otherwise it falls back to the first available preset.

## Installing more voices (experimental upstream packs)

Upstream provides a script to download additional “experimental” voice packs (multilingual + extra English styles) into `demo/voices/streaming_model/experimental_voices/`:
- `bash VibeVoice/demo/download_experimental_voices.sh`

After installing, re-run `--list-voices` to confirm the `.pt` files are discoverable.

## Voice-related knobs you can tune here

These are the controls exposed by this repo’s wrapper and calling code (not all are present in upstream demo defaults):

- `speaker` (voice identity): preset name like `en-davis_man` (mapped to a `.pt` prompt).
- `model_path`: defaults to `microsoft/VibeVoice-Realtime-0.5B` (Hugging Face id or local path).
- `device` / `use_gpu`: `use_gpu=True` forces `cuda` when available; otherwise `cpu`/`mps` is supported.
- `cfg_scale` (Classifier-Free Guidance): higher often increases adherence/strength of the conditioning; defaults vary (`VibeVoiceTTS.cfg_scale`, and `year_progress_and_news_reporter_litellm.py` overrides per “excitement” preset).
- `num_inference_steps`: diffusion steps via `model.set_ddpm_inference_steps(num_steps=...)` (more steps can improve quality at the cost of speed).

## Expressiveness vs. “voice”

The following affect **delivery** (prosody/variation), not identity:
- `do_sample`: deterministic (`False`) vs more varied (`True`)
- `temperature`: higher can sound more expressive/less stable
- `top_p`: lower can sound more focused/less varied
- `excitement_level` presets in `year_progress_and_news_reporter_litellm.py`: maps to a coordinated set of `(temperature, top_p, cfg_scale)`

This repo also applies optional post-processing:
- `speed_multiplier`: time-stretching (prefers `pyrubberband`, falls back to `librosa` or resampling)

