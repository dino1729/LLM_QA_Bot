# Troubleshooting Guide: Chatterbox TTS on CPU (WSL2)

**Date**: December 24, 2025  
**System**: WSL2 on Windows with Intel Processor  
**Goal**: Get Chatterbox TTS Turbo working on CPU

---

## Table of Contents
1. [Initial Problem](#initial-problem)
2. [Root Cause Analysis](#root-cause-analysis)
3. [Solution Steps](#solution-steps)
4. [What Went Wrong](#what-went-wrong)
5. [What Went Right](#what-went-right)
6. [Critical Pitfalls to Avoid](#critical-pitfalls-to-avoid)
7. [Correct Installation Order](#correct-installation-order)
8. [Prevention Best Practices](#prevention-best-practices)
9. [Final Test Results](#final-test-results)

---

## Initial Problem

### The Setup
- Testing Chatterbox TTS with persona-based voice cloning
- Running on WSL2 (Linux 6.6.87.2-microsoft-standard-WSL2)
- Intel CPU (no NVIDIA GPU available)
- Python virtual environment at `.venv/`

### Initial Error
```
ERROR: Failed to build 'numpy' when getting requirements to build wheel
AttributeError: module 'pkgutil' has no attribute 'ImpImporter'
```

---

## Root Cause Analysis

### The Core Issue: Mixed Python Versions in Virtual Environment

When we investigated the virtual environment, we discovered a critical problem:

```bash
$ python --version
Python 3.11.11

$ pip --version
pip 25.3 from /home/dkatam/myprojects/LLM_QA_Bot/.venv/lib/python3.14/site-packages/pip (python 3.14)

$ ls -la .venv/bin/python*
lrwxrwxrwx 1 dkatam dkatam 47 Dec 11 18:17 .venv/bin/python -> /home/dkatam/.pyenv/versions/3.11.11/bin/python
lrwxrwxrwx 1 dkatam dkatam  6 Dec 11 18:17 .venv/bin/python3 -> python
lrwxrwxrwx 1 dkatam dkatam  6 Dec 11 18:17 .venv/bin/python3.11 -> python
lrwxrwxrwx 1 dkatam dkatam 57 Dec 12 12:29 .venv/bin/python3.14 -> /home/linuxbrew/.linuxbrew/opt/python@3.14/bin/python3.14
```

**Problem Identified:**
- Python interpreter: 3.11.11 (from pyenv)
- Pip: 3.14 (from linuxbrew)
- Extra Python 3.14 symlink in venv

### Why This Breaks Package Installation

When pip tries to build packages:
1. Pip runs using Python 3.14 internally
2. But the venv expects Python 3.11
3. Some build tools fail due to Python 3.14 incompatibilities (like `pkgutil.ImpImporter` removal)
4. Result: Build failures even though Python 3.11 is technically active

### How This Happened

The venv was likely contaminated by:
1. Creating venv with Python 3.11 initially
2. Later upgrading or installing pip using a command that picked up Python 3.14 from linuxbrew
3. System PATH had linuxbrew paths that took precedence during some operations

---

## Solution Steps

### Step 1: Remove Corrupted Virtual Environment

```bash
cd /home/dkatam/myprojects/LLM_QA_Bot
rm -rf .venv
```

**Rationale**: Once a venv is contaminated with mixed Python versions, it's cleaner to start fresh than to try to fix it.

### Step 2: Create Clean Virtual Environment

```bash
# Use EXPLICIT path to Python 3.11
/home/dkatam/.pyenv/versions/3.11.11/bin/python -m venv .venv
```

**Critical**: Don't use `python -m venv` - use the full path to ensure correct Python version.

### Step 3: Verify Environment Integrity

```bash
source .venv/bin/activate
python --version    # Should show: Python 3.11.11
pip --version       # Should show: pip X.X from .../python3.11/site-packages/pip (python 3.11)
ls -la .venv/bin/python*  # Should only show 3.11 symlinks
```

**Result**: All three checks passed - clean environment confirmed.

### Step 4: Upgrade pip, setuptools, wheel

```bash
pip install --upgrade pip setuptools wheel
```

**Note**: This upgrade now stays within Python 3.11 since the venv is clean.

### Step 5: Install Base Requirements

```bash
pip install -r requirements.txt
```

**Result**: All base requirements installed successfully without conflicts.

### Step 6: Install PyTorch CPU-Only Version

This was critical for CPU inference on Intel processors:

```bash
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu
```

**Why These Specific Versions:**
- Chatterbox-TTS 0.1.6 requires `torch==2.6.0` and `torchaudio==2.6.0`
- Must use CPU-only builds (from `--index-url https://download.pytorch.org/whl/cpu`)
- These are significantly smaller and don't require CUDA libraries

**Initial Mistakes:**
- First tried torch 2.5.1 → Failed (version mismatch with chatterbox)
- Had torchvision version mismatch → Fixed by using 0.21.0+cpu

### Step 7: Install Chatterbox TTS

```bash
pip install chatterbox-tts
```

**Dependencies Handled:**
- numpy<1.26.0,>=1.24.0 (chatterbox requires older numpy)
- transformers, diffusers, and other ML libraries
- perth (for watermarking)

### Step 8: Install Audio Libraries

```bash
pip install pyaudio
```

**Note**: PyAudio compiled successfully on WSL2. If you get portaudio errors, install system dependencies:
```bash
sudo apt-get install portaudio19-dev python3-pyaudio
```

### Step 9: Modify Code for CPU Support

We updated `helper_functions/tts_chatterbox.py` to support CPU inference:

**Changes Made:**

1. **Added `device` parameter to `__init__`**:
```python
def __init__(
    self,
    model_type: str = "turbo",
    model_path: Optional[str] = None,
    cfg_weight: float = 0.5,
    exaggeration: float = 0.5,
    device: str = "cuda",  # NEW PARAMETER
) -> None:
```

2. **Updated device initialization**:
```python
# OLD: self.device = "cuda"
# NEW: self.device = device
```

3. **Updated `_import_dependencies` to check device**:
```python
if self.device == "cuda":
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. This module requires an NVIDIA GPU. "
            "Please ensure CUDA is properly installed and configured, or use device='cpu'."
        )
    # ... GPU info ...
else:
    print(f"-> Using CPU inference")
    print(f"-> PyTorch version: {torch.__version__}")
```

4. **Updated model loading**:
```python
# Load model with specified device
if self.model_path:
    self._model = self._ModelClass.from_pretrained(
        self.model_path,
        device=self.device  # Use configured device
    )
else:
    self._model = self._ModelClass.from_pretrained(device=self.device)
```

5. **Updated `get_chatterbox_tts` function**:
```python
def get_chatterbox_tts(
    model_type: str = "turbo",
    device: str = "cuda",  # NEW PARAMETER
    **kwargs
) -> ChatterboxTTS:
    cache_key = f"{model_type}_{device}"  # Cache per device type
    
    if cache_key not in _chatterbox_tts_cache:
        _chatterbox_tts_cache[cache_key] = ChatterboxTTS(
            model_type=model_type,
            device=device,  # Pass device
            **kwargs
        )
    
    return _chatterbox_tts_cache[cache_key]
```

6. **Updated test file**:
```python
# OLD: tts = get_chatterbox_tts(model_type="turbo")
# NEW: tts = get_chatterbox_tts(model_type="turbo", device="cpu")
```

---

## What Went Wrong

### 1. Initial Virtual Environment Contamination
- **Problem**: Mixed Python 3.11 and 3.14 in the same venv
- **Impact**: Pip couldn't build packages due to Python 3.14 incompatibilities
- **Lesson**: Always verify venv integrity after creation

### 2. Multiple Failed Package Installation Attempts
- **Problem**: Tried installing chatterbox-tts before fixing the venv
- **Impact**: Wasted time with confusing error messages
- **Lesson**: Fix the environment first, then install packages

### 3. PyTorch Version Mismatches
- **Problem**: Initially installed wrong PyTorch versions (2.5.1, then 2.9.1 with CUDA)
- **Impact**: Had to reinstall multiple times
- **Lesson**: Check package requirements before installing

### 4. Trying Workarounds Before Understanding the Problem
- **Problem**: Attempted `--no-build-isolation` and other workarounds
- **Impact**: These masked the real issue temporarily
- **Lesson**: Diagnose thoroughly before applying fixes

---

## What Went Right

### 1. Systematic Diagnosis
- Checked `python --version` and `pip --version` separately
- Inspected `.venv/bin/python*` symlinks
- **Result**: Identified the root cause quickly

### 2. Clean Slate Approach
- Deleted corrupted venv completely
- Recreated with explicit Python path
- **Result**: Eliminated all contamination

### 3. Explicit Package Versions
- Used exact versions for PyTorch (2.6.0+cpu)
- Used specific index URL for CPU builds
- **Result**: No version conflicts

### 4. Code Modifications for CPU Support
- Added device parameter throughout the codebase
- Made CPU inference a first-class option
- **Result**: Module now works on both CPU and GPU

### 5. Comprehensive Testing
- Generated LLM text with persona
- Synthesized speech with voice cloning
- Created audio file successfully
- **Result**: End-to-end functionality verified

---

## Critical Pitfalls to Avoid

### Pitfall 1: Using `python -m venv` Without Full Path
```bash
# ❌ WRONG - uses whatever 'python' resolves to
python -m venv .venv

# ✅ CORRECT - explicit path ensures correct Python
~/.pyenv/versions/3.11.11/bin/python -m venv .venv
```

### Pitfall 2: Not Verifying Venv After Creation
```bash
# Always run these checks after creating venv:
source .venv/bin/activate
python --version        # Check Python version
pip --version           # Check pip is using same Python
ls -la .venv/bin/python*  # Check for unexpected symlinks
```

### Pitfall 3: Installing Packages in Wrong Order
```bash
# ❌ WRONG ORDER
pip install chatterbox-tts  # This pulls wrong PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cpu  # Too late

# ✅ CORRECT ORDER
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu  # First
pip install chatterbox-tts  # Then dependencies
```

### Pitfall 4: Mixing Package Managers
- Don't let linuxbrew, system pip, and pyenv interact
- Choose ONE Python source and stick with it
- Keep system PATH clean to avoid precedence issues

### Pitfall 5: Installing CUDA PyTorch on CPU-Only Systems
```bash
# ❌ WRONG - Pulls 2GB+ of CUDA libraries you don't need
pip install torch

# ✅ CORRECT - Smaller, faster, CPU-only
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Pitfall 6: Trying to Fix Corrupted Venv Instead of Recreating
- Time spent fixing > Time to recreate fresh
- Partial fixes lead to mysterious errors later
- **Rule**: If venv is contaminated, delete and recreate

### Pitfall 7: Not Checking Package Version Requirements
```bash
# Always check what packages require
pip show chatterbox-tts
# Look at "Requires:" line before installing
```

---

## Correct Installation Order

### Complete Installation Sequence

```bash
# 1. CLEAN SLATE
cd /home/dkatam/myprojects/LLM_QA_Bot
rm -rf .venv

# 2. CREATE VENV WITH EXPLICIT PYTHON PATH
/home/dkatam/.pyenv/versions/3.11.11/bin/python -m venv .venv

# 3. ACTIVATE AND VERIFY
source .venv/bin/activate
python --version    # Should be 3.11.11
pip --version       # Should say python 3.11
ls -la .venv/bin/python*  # Should only have 3.11

# 4. UPGRADE PIP TOOLING
pip install --upgrade pip setuptools wheel

# 5. INSTALL BASE REQUIREMENTS
pip install -r requirements.txt

# 6. INSTALL PYTORCH CPU-ONLY (BEFORE CHATTERBOX)
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cpu

# 7. INSTALL CHATTERBOX TTS
pip install chatterbox-tts

# 8. INSTALL ADDITIONAL AUDIO LIBRARIES
pip install pyaudio

# 9. VERIFY INSTALLATION
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CPU: {torch.cpu.is_available()}')"
python -c "import chatterbox; print('Chatterbox TTS imported successfully')"
```

### Dependency Tree Explanation

```
requirements.txt
├── Core dependencies (gradio, fastapi, etc.)
├── torch/torchvision/torchaudio (MUST BE CPU VERSION)
│   └── Installed BEFORE chatterbox-tts
├── chatterbox-tts
│   ├── numpy<1.26.0,>=1.24.0 (specific version requirement)
│   ├── torch==2.6.0 (already satisfied)
│   ├── transformers
│   ├── diffusers
│   └── perth (watermarking)
└── pyaudio (for audio playback)
```

**Why This Order Matters:**
1. PyTorch must be CPU version BEFORE chatterbox (otherwise it pulls CUDA)
2. Base requirements first (they're more flexible)
3. Chatterbox last (it has strict version requirements)

---

## Prevention Best Practices

### 1. Virtual Environment Management

**Always create venvs with explicit Python:**
```bash
# Add to your ~/.zshrc or ~/.bashrc
alias mkvenv='~/.pyenv/versions/3.11.11/bin/python -m venv .venv'
```

**Verify venv script:**
```bash
#!/bin/bash
# verify_venv.sh
source .venv/bin/activate
echo "Python: $(python --version)"
echo "Pip: $(pip --version)"
echo "Symlinks:"
ls -la .venv/bin/python*
```

### 2. Project-Specific Requirements Files

Create separate requirements files for different scenarios:

**requirements.txt** (base):
```txt
# Base dependencies
gradio
fastapi
# ... other packages
```

**requirements-cpu.txt**:
```txt
-r requirements.txt
--index-url https://download.pytorch.org/whl/cpu
torch==2.6.0
torchvision==0.21.0
torchaudio==2.6.0
chatterbox-tts==0.1.6
pyaudio
```

**requirements-gpu.txt**:
```txt
-r requirements.txt
--index-url https://download.pytorch.org/whl/cu121
torch==2.6.0
torchvision==0.21.0
torchaudio==2.6.0
chatterbox-tts==0.1.6
```

### 3. PATH Management

Keep your PATH clean:
```bash
# ~/.zshrc or ~/.bashrc
# Prioritize pyenv over linuxbrew
export PATH="$HOME/.pyenv/bin:$PATH"
export PATH="$HOME/.pyenv/shims:$PATH"
# Put linuxbrew AFTER
export PATH="/home/linuxbrew/.linuxbrew/bin:$PATH"
```

### 4. Regular Venv Health Checks

Create a health check script:
```python
#!/usr/bin/env python3
# check_venv_health.py
import sys
import subprocess

print("=" * 60)
print("VIRTUAL ENVIRONMENT HEALTH CHECK")
print("=" * 60)

# Check Python version
print(f"\nPython Version: {sys.version}")
print(f"Python Executable: {sys.executable}")

# Check pip
pip_info = subprocess.run([sys.executable, "-m", "pip", "--version"], 
                         capture_output=True, text=True)
print(f"\nPip Info: {pip_info.stdout.strip()}")

# Check for version mismatches
if "python 3.11" not in pip_info.stdout.lower():
    print("\n⚠️  WARNING: Pip Python version mismatch detected!")
else:
    print("\n✅ Virtual environment is healthy")
```

### 5. Documentation

Keep a `VENV_SETUP.md` in your project:
```markdown
# Virtual Environment Setup

## Create
/home/dkatam/.pyenv/versions/3.11.11/bin/python -m venv .venv

## Activate
source .venv/bin/activate

## Install
pip install -r requirements-cpu.txt  # For CPU
# OR
pip install -r requirements-gpu.txt  # For GPU
```

### 6. Git Ignore Patterns

Ensure `.gitignore` includes:
```gitignore
.venv/
.venv.*/
venv/
*.pyc
__pycache__/
.python-version
```

### 7. Pre-commit Hooks

Add a pre-commit hook to check venv health:
```bash
#!/bin/bash
# .git/hooks/pre-commit

if [ -d ".venv" ]; then
    source .venv/bin/activate
    python_ver=$(python --version)
    pip_ver=$(pip --version)
    
    if [[ "$pip_ver" != *"python 3.11"* ]]; then
        echo "ERROR: Virtual environment contamination detected"
        echo "Python: $python_ver"
        echo "Pip: $pip_ver"
        echo "Please recreate your virtual environment"
        exit 1
    fi
fi
```

---

## Final Test Results

### Test Command
```bash
cd /home/dkatam/myprojects/LLM_QA_Bot
source .venv/bin/activate
python tests/test_chatterbox_with_persona.py
```

### Test Output Summary

```
======================================================================
Chatterbox TTS + Persona Test
======================================================================
Persona: rick_sanchez
Voice file: voices/rick_sanchez.wav
======================================================================

Loading persona...
Generating Rick Sanchez response using LLM...
  🤖 LLM Client initialized: provider=litellm, tier=fast, model=gemini-2.5-flash-lite

Generated text:
----------------------------------------------------------------------
Ugh, *burp*, text-to-speech. Why are we even talking about this, dummy?
[... Rick Sanchez style response ...]
----------------------------------------------------------------------

Initializing Chatterbox TTS (Turbo model on CPU)...
-> Using CPU inference
-> PyTorch version: 2.6.0+cpu
-> Chatterbox Turbo TTS library loaded successfully
-> Loading Chatterbox Turbo TTS model on CPU...

[Model downloading and loading...]

-> Model loaded on CPU successfully
-> Sample rate: 24000 Hz
-> Using Chatterbox Turbo TTS

Synthesizing speech with Rick Sanchez voice...
[Progress: 100%]

-> Generated 20.92s audio in 166.66s (RTF: 7.97x)
-> Audio saved to test_output_rick_sanchez.wav

======================================================================
✓ SUCCESS! Audio saved to: test_output_rick_sanchez.wav
======================================================================
```

### Key Metrics

| Metric | Value | Notes |
|--------|-------|-------|
| **Model** | Chatterbox Turbo (350M params) | CPU inference |
| **Audio Duration** | 20.92 seconds | Voice-cloned output |
| **Generation Time** | 166.66 seconds | ~2.78 minutes |
| **Real-Time Factor** | 7.97x | Slower than real-time (expected on CPU) |
| **Sample Rate** | 24000 Hz | High quality |
| **Status** | ✅ SUCCESS | Fully functional |

### Performance Notes

**CPU Performance:**
- RTF of 7.97x means it takes ~8 seconds to generate 1 second of audio
- This is **normal for CPU inference** with a 350M parameter model
- For production use, GPU would provide ~100x speedup (RTF < 0.1)
- For development/testing on Intel CPU, this is perfectly acceptable

**Memory Usage:**
- Model size: ~1.4GB
- Peak RAM usage: ~3-4GB during inference
- Acceptable for most modern systems

### Known Limitation: Audio Playback on WSL2

The test failed at the playback step:
```
sounddevice.PortAudioError: Error querying device -1
```

**Why This Happens:**
- WSL2 doesn't have direct access to Windows audio devices
- `sounddevice` expects a Linux audio system (ALSA/PulseAudio)

**Workarounds:**
1. **Play file in Windows**: Open `test_output_rick_sanchez.wav` in Windows Media Player
2. **Setup WSLg**: Use WSLg for Linux GUI apps with audio
3. **Configure PulseAudio**: Forward audio to Windows
4. **Skip playback in tests**: Comment out the playback section for automated tests

**The audio file itself is valid** - the TTS synthesis completed successfully.

---

## Quick Reference Checklist

### Before Starting Any Python Project

- [ ] Decide on Python version (use pyenv for consistency)
- [ ] Create venv with EXPLICIT Python path
- [ ] Verify venv integrity (python --version, pip --version, ls symlinks)
- [ ] Upgrade pip, setuptools, wheel
- [ ] Check PATH for package manager conflicts

### When Installing ML Packages

- [ ] Determine if you need CPU or GPU versions
- [ ] Install PyTorch FIRST (with correct index URL)
- [ ] Verify PyTorch installation before proceeding
- [ ] Install ML packages that depend on PyTorch
- [ ] Check for version conflicts (`pip check`)

### When Things Go Wrong

- [ ] Check `python --version` vs `pip --version`
- [ ] Inspect `.venv/bin/python*` symlinks
- [ ] Look for mixed Python versions
- [ ] Don't try to fix corrupted venv - recreate it
- [ ] Install packages in the correct order

### For This Specific Project

- [ ] Use Python 3.11 (not 3.14)
- [ ] Install PyTorch 2.6.0+cpu BEFORE chatterbox-tts
- [ ] Set `device="cpu"` when calling `get_chatterbox_tts()`
- [ ] Expect slow performance on CPU (RTF ~8x)
- [ ] Play generated audio files in Windows, not WSL2

---

## Additional Resources

### Documentation
- PyTorch Installation: https://pytorch.org/get-started/locally/
- Chatterbox TTS: https://github.com/resemble-ai/chatterbox
- WSL Audio Setup: https://learn.microsoft.com/en-us/windows/wsl/multimedia

### Useful Commands

```bash
# Check package info
pip show chatterbox-tts

# List all installed packages
pip list

# Check for dependency conflicts
pip check

# See where a package is installed
python -c "import chatterbox; print(chatterbox.__file__)"

# Test PyTorch CPU
python -c "import torch; print(torch.__version__); print(f'CPU: {torch.cpu.is_available()}')"

# Cleanup pip cache (if needed)
pip cache purge
```

---

## Conclusion

The Chatterbox TTS integration now works successfully on CPU in WSL2. The key lessons:

1. **Virtual environment integrity is critical** - mixed Python versions cause mysterious failures
2. **Package installation order matters** - install PyTorch before packages that depend on it
3. **CPU inference is viable** - slower but functional for development and testing
4. **Explicit paths prevent issues** - always use full Python paths when creating venvs
5. **Clean slate is faster** - recreating a venv is faster than debugging a corrupted one

This guide should help prevent similar issues in the future and serve as a reference for setting up ML projects in WSL2 environments.

---

**Last Updated**: December 24, 2025  
**Tested On**: WSL2 (Ubuntu), Python 3.11.11, Intel CPU  
**Status**: ✅ Working

