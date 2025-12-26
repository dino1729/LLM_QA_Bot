#!/usr/bin/env python3
"""
Test Chatterbox TTS with persona-based voice cloning.

This script generates text using a specific persona and synthesizes it
with the corresponding voice reference audio.

Configuration Required (in config.yml):
- chatterbox_tts_model_type: TTS model type (turbo, standard, multilingual)
- chatterbox_tts_device: Device for inference (cuda, cpu, macos)
- chatterbox_tts_default_voice: Default voice to use
- default_analyzers_provider: LLM provider (litellm, ollama)
- default_llm_tier: LLM tier (fast, smart, strategic)
"""

import sys
from pathlib import Path

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from helper_functions.tts_chatterbox import get_chatterbox_tts, split_text_for_chatterbox
from helper_functions.llm_client import get_client
from config import config


def main():
    # Validate required config
    if not config.chatterbox_tts_default_voice:
        print("ERROR: 'chatterbox_tts_default_voice' must be set in config.yml")
        return 1
    if not config.chatterbox_tts_model_type:
        print("ERROR: 'chatterbox_tts_model_type' must be set in config.yml")
        return 1
    if not config.chatterbox_tts_device:
        print("ERROR: 'chatterbox_tts_device' must be set in config.yml")
        return 1
    if not config.default_analyzers_provider:
        print("ERROR: 'default_analyzers_provider' must be set in config.yml (under llm_tiers)")
        return 1
    
    # Get test persona from config (use default voice as the persona name)
    persona_name = config.chatterbox_tts_default_voice
    voice_file = f"voices/{persona_name}.wav"
    persona_file = f"personas/{persona_name}.txt"
    
    # Get LLM settings from config
    llm_provider = config.default_analyzers_provider
    llm_tier = config.default_llm_tier or "fast"
    
    # Get TTS settings from config
    tts_model_type = config.chatterbox_tts_model_type
    tts_device = config.chatterbox_tts_device

    print("=" * 70)
    print("Chatterbox TTS + Persona Test")
    print("=" * 70)
    print(f"Persona: {persona_name} (from config.chatterbox_tts_default_voice)")
    print(f"Voice file: {voice_file}")
    print(f"LLM Provider: {llm_provider} (from config)")
    print(f"LLM Tier: {llm_tier} (from config)")
    print(f"TTS Model: {tts_model_type} (from config)")
    print(f"TTS Device: {tts_device} (from config)")
    print("=" * 70)
    print()

    # Check if persona file exists
    if not Path(persona_file).exists():
        print(f"WARNING: Persona file not found: {persona_file}")
        print("Using generic system prompt instead.")
        persona_text = f"You are {persona_name.replace('_', ' ').title()}. Speak in your characteristic style."
    else:
        # Read persona
        print("Loading persona...")
        with open(persona_file, 'r') as f:
            persona_text = f.read()

    # Generate persona-style text using LLM
    print(f"Generating {persona_name} response using LLM...")
    client = get_client(provider=llm_provider, model_tier=llm_tier)

    prompt = (
        "Describe, in your usual style, how you would teach an alien who just landed on Earth "
        "to order coffee at a noisy Parisian café using only text-to-speech and elaborate gesturing."
    )
    messages = [
        {"role": "system", "content": persona_text},
        {"role": "user", "content": prompt}
    ]

    response = client.chat_completion(messages=messages, max_tokens=200, temperature=0.9)
    generated_text = response.strip()

    print(f"\nGenerated text:\n{'-' * 70}")
    print(generated_text)
    print('-' * 70)
    print()

    # Initialize Chatterbox TTS using config settings
    print(f"Initializing Chatterbox TTS ({tts_model_type} model on {tts_device})...")
    tts = get_chatterbox_tts(model_type=tts_model_type, device=tts_device)

    # Get GPU info
    gpu_info = tts.get_gpu_memory_info()
    if gpu_info:
        print(f"GPU: {gpu_info['allocated_gb']:.2f} GB allocated / {gpu_info['total_gb']:.2f} GB total")
    print()

    # Show how text will be split for synthesis
    chunks = split_text_for_chatterbox(generated_text, max_chars=300)
    print(f"Text will be split into {len(chunks)} segment(s):")
    for i, chunk in enumerate(chunks, 1):
        print(f"  [{i}] {chunk[:60]}{'...' if len(chunk) > 60 else ''}")
    print()

    # Synthesize with configured voice
    print(f"Synthesizing speech with {persona_name} voice...")
    output_file = f"test_output_{persona_name}.wav"

    success = tts.synthesize_to_file(
        generated_text,
        output_file,
        audio_prompt_path=voice_file
    )

    if success:
        print(f"\n{'=' * 70}")
        print(f"✓ SUCCESS! Audio saved to: {output_file}")
        print(f"{'=' * 70}")
        print()

        # Ask if user wants to play it
        print("Playing audio...")
        import soundfile as sf
        import sounddevice as sd

        audio_data, sample_rate = sf.read(output_file)
        sd.play(audio_data, sample_rate)
        sd.wait()
        print("Playback complete!")

    else:
        print("\nERROR: Failed to synthesize speech")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
