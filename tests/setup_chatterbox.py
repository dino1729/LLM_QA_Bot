#!/usr/bin/env python3
"""
Setup script for Chatterbox TTS - handles HuggingFace authentication and model download.
"""

import sys
import os


def main():
    print("=" * 70)
    print("Chatterbox TTS Setup")
    print("=" * 70)
    print()

    # Check if HF_TOKEN is set in environment
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    if not hf_token:
        print("HuggingFace token not found in environment.")
        print()
        print("To use Chatterbox TTS, you need to:")
        print()
        print("1. Get a HuggingFace token:")
        print("   - Go to https://huggingface.co/settings/tokens")
        print("   - Create a new token (Read access is sufficient)")
        print()
        print("2. Set the token in your environment:")
        print("   export HF_TOKEN='your_token_here'")
        print()
        print("   Or add to your ~/.bashrc or ~/.zshrc:")
        print("   echo 'export HF_TOKEN=\"your_token_here\"' >> ~/.bashrc")
        print()
        print("3. Or login via CLI:")
        print("   source venv/bin/activate")
        print("   hf auth login")
        print()
        return 1

    print("✓ HuggingFace token found in environment")
    print()

    # Test model download
    print("Testing model download from HuggingFace...")
    print("(This may take a few minutes on first run)")
    print()

    try:
        from huggingface_hub import snapshot_download

        model_name = "ResembleAI/chatterbox-turbo"
        print(f"Downloading model: {model_name}")

        local_path = snapshot_download(
            repo_id=model_name,
            token=hf_token
        )

        print(f"✓ Model downloaded successfully to: {local_path}")
        print()
        print("=" * 70)
        print("Setup complete! You can now use Chatterbox TTS.")
        print("=" * 70)
        print()
        print("Try running:")
        print("  python test_chatterbox_with_persona.py")
        print()
        return 0

    except Exception as e:
        print(f"✗ Error downloading model: {e}")
        print()
        print("Please check:")
        print("1. Your HuggingFace token is valid")
        print("2. You have internet connection")
        print("3. You have enough disk space (~2GB for turbo model)")
        return 1


if __name__ == "__main__":
    sys.exit(main())
