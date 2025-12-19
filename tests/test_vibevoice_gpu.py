#!/usr/bin/env python3
"""Test script to verify VibeVoice TTS is using GPU acceleration."""

import torch
from helper_functions.tts_vibevoice import VibeVoiceTTS

print("=" * 80)
print("VIBEVOICE GPU TEST")
print("=" * 80)

# Check PyTorch CUDA availability
print(f"\n1. PyTorch CUDA Configuration:")
print(f"   CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Compute Capability: sm_{torch.cuda.get_device_capability(0)[0]}{torch.cuda.get_device_capability(0)[1]}")
    print(f"   CUDA Version: {torch.version.cuda}")

# Initialize VibeVoice with GPU
print(f"\n2. Initializing VibeVoice TTS:")
print(f"   Forcing GPU mode (use_gpu=True)...")

tts = VibeVoiceTTS(speaker="en-davis_man", use_gpu=True)

# Verify device placement
print(f"\n3. Device Verification:")
print(f"   Selected device: {tts.device}")
print(f"   Model device: {next(tts._model.parameters()).device}")
print(f"   Model dtype: {next(tts._model.parameters()).dtype}")

# Check GPU memory usage before synthesis
if torch.cuda.is_available():
    print(f"\n4. GPU Memory Status (before synthesis):")
    print(f"   Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"   Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

# Test synthesis
print(f"\n5. Running test synthesis:")
test_text = "Hello, this is a test of VibeVoice text to speech running on the RTX 5090."
print(f"   Text: '{test_text}'")

audio_data = tts.synthesize(test_text)

if audio_data is not None and len(audio_data) > 0:
    print(f"   ✓ Success! Generated {len(audio_data)} audio samples")
    print(f"   Duration: {len(audio_data) / tts.sample_rate:.2f} seconds")
else:
    print(f"   ✗ Failed to generate audio")

# Check GPU memory usage after synthesis
if torch.cuda.is_available():
    print(f"\n6. GPU Memory Status (after synthesis):")
    print(f"   Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"   Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

# Final verification
print(f"\n7. Final Verification:")
if tts.device == "cuda" and torch.cuda.is_available():
    model_device = str(next(tts._model.parameters()).device)
    if "cuda" in model_device:
        print(f"   ✓ CONFIRMED: VibeVoice is running on GPU!")
        print(f"   ✓ Model is on: {model_device}")
    else:
        print(f"   ✗ WARNING: Model is on CPU despite cuda device setting!")
        print(f"   ✗ Model device: {model_device}")
else:
    print(f"   ✗ WARNING: VibeVoice is NOT using GPU!")
    print(f"   Device: {tts.device}")

print("\n" + "=" * 80)
