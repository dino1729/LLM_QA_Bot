#!/usr/bin/env python3
"""
Test script for Chatterbox TTS integration.

This script demonstrates how to use the Chatterbox TTS module with GPU acceleration.

Usage:
    python test_chatterbox_tts.py [--text "Your text here"] [--model turbo|standard|multilingual]

For Turbo model (requires reference audio):
    python test_chatterbox_tts.py --text "Hello world [chuckle]" --audio-prompt path/to/reference.wav

For Multilingual model:
    python test_chatterbox_tts.py --model multilingual --text "Bonjour le monde" --language fr
"""

import argparse
import sys
from pathlib import Path

# Add the parent directory to the path to import helper_functions
sys.path.insert(0, str(Path(__file__).parent.parent))

from helper_functions.tts_chatterbox import ChatterboxTTS, get_chatterbox_tts
from config import config


def main():
    parser = argparse.ArgumentParser(description="Test Chatterbox TTS integration")
    parser.add_argument(
        "--text",
        type=str,
        default="Hi there [chuckle], this is a test of the Chatterbox TTS system. It sounds pretty natural, doesn't it?",
        help="Text to synthesize"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["turbo", "standard", "multilingual"],
        default=None,
        help="Model type to use (default: from config.yml)"
    )
    parser.add_argument(
        "--audio-prompt",
        type=str,
        default=None,
        help="Path to reference audio for voice cloning (required for turbo model)"
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        help="Language code for multilingual model (e.g., 'en', 'fr', 'zh')"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="chatterbox_test_output.wav",
        help="Output audio file path"
    )
    parser.add_argument(
        "--cfg-weight",
        type=float,
        default=None,
        help="CFG weight (0.0-1.0, lower=faster speech)"
    )
    parser.add_argument(
        "--exaggeration",
        type=float,
        default=None,
        help="Exaggeration level (0.0-1.0, higher=more dramatic)"
    )
    parser.add_argument(
        "--play",
        action="store_true",
        help="Play audio after synthesis"
    )

    args = parser.parse_args()

    # Get model type from config if not specified
    model_type = args.model or config.chatterbox_tts_model_type

    # Get audio prompt from config if not specified and using turbo model
    audio_prompt_path = args.audio_prompt or config.chatterbox_tts_audio_prompt_path

    print("=" * 60)
    print(f"Chatterbox TTS Test")
    print("=" * 60)
    print(f"Model type: {model_type}")
    print(f"Text: {args.text}")
    if model_type == "turbo":
        if not audio_prompt_path:
            print("\nERROR: Turbo model requires --audio-prompt or chatterbox_tts_audio_prompt_path in config.yml")
            print("Please provide a reference audio file (10 seconds recommended)")
            return 1
        print(f"Audio prompt: {audio_prompt_path}")
    if model_type == "multilingual":
        print(f"Language: {args.language}")
    print(f"Output: {args.output}")
    print("=" * 60)
    print()

    try:
        # Create TTS instance
        print("Initializing Chatterbox TTS...")
        tts = get_chatterbox_tts(
            model_type=model_type,
            cfg_weight=args.cfg_weight or config.chatterbox_tts_cfg_weight,
            exaggeration=args.exaggeration or config.chatterbox_tts_exaggeration,
        )

        # Get GPU memory info
        gpu_info = tts.get_gpu_memory_info()
        if gpu_info:
            print(f"GPU memory allocated: {gpu_info['allocated_gb']:.2f} GB")
            print(f"GPU memory reserved: {gpu_info['reserved_gb']:.2f} GB")
            print(f"GPU memory total: {gpu_info['total_gb']:.2f} GB")
            print()

        # Synthesize speech
        print("Synthesizing speech...")
        kwargs = {}
        if model_type == "turbo":
            kwargs["audio_prompt_path"] = audio_prompt_path
        elif model_type == "multilingual":
            kwargs["language_id"] = args.language

        success = tts.synthesize_to_file(args.text, args.output, **kwargs)

        if success:
            print(f"\nSuccess! Audio saved to: {args.output}")

            # Play audio if requested
            if args.play:
                print("\nPlaying audio...")
                import soundfile as sf
                import sounddevice as sd
                audio_data, sample_rate = sf.read(args.output)
                sd.play(audio_data, sample_rate)
                sd.wait()
                print("Playback complete.")
        else:
            print("\nERROR: Failed to synthesize speech")
            return 1

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
