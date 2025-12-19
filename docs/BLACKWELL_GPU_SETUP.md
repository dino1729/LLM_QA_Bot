# PyTorch Setup for NVIDIA Blackwell GPUs (RTX 5090, 5080, etc.)

This guide covers the specific requirements for running PyTorch-based applications (like VibeVoice TTS) on NVIDIA Blackwell architecture GPUs.

## Problem

Standard PyTorch releases (as of December 2024) do not support NVIDIA Blackwell GPUs due to compute capability limitations:

- **Blackwell GPUs**: Compute capability `sm_120` (12.0)
- **PyTorch stable releases**: Support up to `sm_90` (compute capability 9.0)
- **Error symptom**: `CUDA error: no kernel image is available for execution on the device`

## Requirements

### Minimum Software Versions

- **NVIDIA Driver**: R570 or higher
- **CUDA**: 12.8 or higher
- **PyTorch**: 2.11.0-dev (nightly builds) with CUDA 12.8

### Hardware

- NVIDIA RTX 5090, RTX 5080, or other Blackwell-based GPUs
- Sufficient VRAM for your workload (VibeVoice uses ~2GB)

## Installation Steps

### 1. Verify NVIDIA Driver

Check your current driver version:
```bash
nvidia-smi --query-gpu=driver_version --format=csv,noheader
```

**Expected output**: `570.00` or higher (e.g., `590.44.01`)

If your driver is older than R570:
- Download the latest driver from [NVIDIA Driver Downloads](https://www.nvidia.com/download/index.aspx)
- Install and reboot your system

### 2. Uninstall Old PyTorch (if needed)

If you have an existing PyTorch installation:
```bash
pip uninstall torch torchvision torchaudio -y
```

### 3. Install PyTorch Nightly with CUDA 12.8

```bash
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

This will install:
- `torch-2.11.0.dev<date>+cu128` or later
- All required CUDA 12.8 libraries (cublas, cudnn, etc.)
- Latest Triton compiler with Blackwell support

**Installation time**: ~5-10 minutes (downloads ~2GB of packages)

### 4. Verify Installation

Run these verification commands:

```bash
# Check PyTorch version
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
# Expected: 2.11.0.dev20251218+cu128 (or similar)

# Check CUDA version
python -c "import torch; print(f'CUDA: {torch.version.cuda}')"
# Expected: 12.8

# Check GPU detection
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
# Expected: NVIDIA GeForce RTX 5090

# Check compute capability
python -c "import torch; print(f'Compute: {torch.cuda.get_device_capability(0)}')"
# Expected: (12, 0)
```

All checks should pass without errors.

## Troubleshooting

### Error: "CUDA error: no kernel image is available"

This means PyTorch doesn't have CUDA kernels compiled for your GPU's compute capability.

**Solution**: Ensure you're using PyTorch nightly with CUDA 12.8 (see step 3 above)

### Error: "NVIDIA GeForce RTX 5090 is not compatible with PyTorch"

This warning appears when using stable PyTorch releases. Upgrade to nightly builds.

### Performance Expectations

With proper setup, you should see:
- **VibeVoice TTS**: ~6x realtime (RTF 0.16-0.18)
- **GPU Memory**: ~2GB VRAM for VibeVoice
- **Generation speed**: ~10 seconds for 60 seconds of audio

If performance is significantly lower:
1. Check GPU utilization: `nvidia-smi dmon -s u`
2. Verify bfloat16 is being used (automatic on Blackwell)
3. Ensure no CPU fallback is occurring

## Alternative: Docker Container

NVIDIA provides pre-built containers with PyTorch + CUDA 12.8:

```bash
docker pull nvcr.io/nvidia/pytorch:25.01-py3
docker run --gpus all -it --rm nvcr.io/nvidia/pytorch:25.01-py3
```

This container includes:
- PyTorch with CUDA 12.8 support
- All necessary CUDA libraries
- Optimized for Blackwell architecture

## Testing VibeVoice TTS

After installation, test VibeVoice with:

```bash
cd /path/to/LLM_QA_Bot
python tests/test_vibevoice_gpu.py
```

**Expected output**:
```
✓ GPU detected: NVIDIA GeForce RTX 5090
✓ CUDA version: 12.8
✓ VibeVoice model loaded on cuda:0
✓ Generated audio in X.Xs (RTF: 0.1Xx)
```

## References

- [NVIDIA Blackwell Software Migration Guide](https://forums.developer.nvidia.com/t/software-migration-guide-for-nvidia-blackwell-rtx-gpus-a-guide-to-cuda-12-8-pytorch-tensorrt-and-llama-cpp/321330)
- [PyTorch Nightly Builds](https://pytorch.org/get-started/locally/)
- [CUDA 12.8 Release Notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/)

## Update Schedule

- PyTorch stable releases with CUDA 12.8 support: Expected Q1 2025
- Until then, use nightly builds for Blackwell GPU support

## Notes

- Nightly builds are updated daily and may have bugs
- For production use, pin to a specific nightly version: `torch==2.11.0.dev20251218+cu128`
- Monitor [PyTorch GitHub](https://github.com/pytorch/pytorch) for stable release announcements
- WSL2 users: Follow the same installation steps (works identically to native Linux)
