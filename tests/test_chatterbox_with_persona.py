#!/usr/bin/env python3
"""
Test Chatterbox TTS with persona-based voice cloning.

This script generates text using a specific persona and synthesizes it
with the corresponding voice reference audio.
"""

import sys
from pathlib import Path

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from helper_functions.tts_chatterbox import get_chatterbox_tts
from helper_functions.llm_client import get_client
from config import config


def main():
    # Test with Rick Sanchez persona and voice
    persona_name = "rick_sanchez"
    voice_file = f"voices/{persona_name}.wav"
    persona_file = f"personas/{persona_name}.txt"

    print("=" * 70)
    print("Chatterbox TTS + Persona Test")
    print("=" * 70)
    print(f"Persona: {persona_name}")
    print(f"Voice file: {voice_file}")
    print("=" * 70)
    print()

    # Read persona
    print("Loading persona...")
    with open(persona_file, 'r') as f:
        persona_text = f.read()

    # Generate Rick-style text using LLM
    print("Generating Rick Sanchez response using LLM...")
    client = get_client(provider="litellm", model_tier="fast")

    prompt = "Explain why text-to-speech technology is important, but in your usual style."
    messages = [
        {"role": "system", "content": persona_text},
        {"role": "user", "content": prompt}
    ]

    response = client.chat_completion(messages=messages, max_tokens=200, temperature=0.9)
    rick_text = response.strip()

    print(f"\nGenerated text:\n{'-' * 70}")
    print(rick_text)
    print('-' * 70)
    print()

    # Initialize Chatterbox TTS
    print("Initializing Chatterbox TTS (Turbo model)...")
    tts = get_chatterbox_tts(model_type="turbo")

    # Get GPU info
    gpu_info = tts.get_gpu_memory_info()
    if gpu_info:
        print(f"GPU: {gpu_info['allocated_gb']:.2f} GB allocated / {gpu_info['total_gb']:.2f} GB total")
    print()

    # Synthesize with Rick's voice
    print("Synthesizing speech with Rick Sanchez voice...")
    output_file = f"test_output_{persona_name}.wav"

    success = tts.synthesize_to_file(
        rick_text,
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
