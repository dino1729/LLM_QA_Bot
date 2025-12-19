"""Text-to-speech using Microsoft VibeVoice with GPU acceleration."""

from __future__ import annotations

import copy
import glob
import os
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pyaudio


class _DictWithAttrs:
    """Wrapper to make a dict accessible via both keys and attributes.

    Used to wrap cached voice prompt dicts so they can be accessed like
    model output objects (with .past_key_values etc.)
    """

    def __init__(self, data: dict):
        self._data = data

    def __getattr__(self, name: str):
        if name.startswith('_'):
            return object.__getattribute__(self, name)
        try:
            return self._data[name]
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value):
        self._data[key] = value

    def __contains__(self, key):
        return key in self._data

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()

    def items(self):
        return self._data.items()


class VibeVoiceTTS:
    """Generate audio using Microsoft VibeVoice with GPU acceleration.

    VibeVoice is a realtime streaming TTS model with ~300ms first-chunk latency.
    It provides high-quality, expressive speech synthesis with multiple speaker
    voices available.
    """

    def __init__(
        self,
        speaker: str = "wayne",
        device: str = "cpu",
        use_gpu: bool = True,
        model_path: str = "microsoft/VibeVoice-Realtime-0.5B",
        cfg_scale: float = 1.5,
        num_inference_steps: int = 5,
        vibevoice_path: Optional[str] = None,
    ) -> None:
        """Initialize VibeVoice TTS.

        Args:
            speaker: Speaker name to use (default: wayne). Run list_speakers() for available.
            device: Device to use ('cpu', 'cuda', or 'mps'). Overridden by use_gpu.
            use_gpu: If True, forces CUDA device for maximum speed.
            model_path: HuggingFace model path or local path.
            cfg_scale: Classifier-free guidance scale (default: 1.5).
            num_inference_steps: Number of diffusion inference steps (default: 5).
            vibevoice_path: Path to VibeVoice installation (auto-detected if None).
        """
        self.speaker = speaker.lower()
        self.model_path = model_path
        self.cfg_scale = cfg_scale
        self.num_inference_steps = num_inference_steps
        self.sample_rate = 24000  # VibeVoice outputs at 24kHz

        # Determine device
        if use_gpu:
            self.device = "cuda"
        else:
            self.device = device

        # Find VibeVoice installation path
        self.vibevoice_path = vibevoice_path or self._find_vibevoice_path()

        # Lazy-load model components
        self._model = None
        self._processor = None
        self._voice_prompt = None
        self._torch = None
        self._available_voices: Dict[str, str] = {}

        # Import and validate
        self._import_dependencies()
        self._scan_voice_presets()
        self._initialize_model()

    def _find_vibevoice_path(self) -> Optional[str]:
        """Find the VibeVoice installation directory."""
        possible_paths = [
            Path("/home/dino/myprojects/VibeVoice"),
            Path.home() / "myprojects" / "VibeVoice",
            Path.cwd().parent / "VibeVoice",
            Path.cwd() / "VibeVoice",
        ]

        try:
            import vibevoice
            pkg_path = Path(vibevoice.__file__).parent.parent
            possible_paths.insert(0, pkg_path)
        except Exception:
            pass

        for path in possible_paths:
            if path.exists() and (path / "demo" / "voices").exists():
                return str(path)

        return None

    def _scan_voice_presets(self) -> None:
        """Scan for available voice preset files."""
        self._available_voices = {}

        if not self.vibevoice_path:
            print("-> Warning: VibeVoice path not found, voice presets unavailable")
            return

        voices_dir = os.path.join(self.vibevoice_path, "demo", "voices", "streaming_model")

        if not os.path.exists(voices_dir):
            print(f"-> Warning: Voices directory not found at {voices_dir}")
            return

        pt_files = glob.glob(os.path.join(voices_dir, "**", "*.pt"), recursive=True)

        for pt_file in pt_files:
            name = os.path.splitext(os.path.basename(pt_file))[0].lower()
            self._available_voices[name] = os.path.abspath(pt_file)

        self._available_voices = dict(sorted(self._available_voices.items()))

        if self._available_voices:
            print(f"-> Found {len(self._available_voices)} voice presets")
            print(f"-> Available voices: {', '.join(self._available_voices.keys())}")

    def _import_dependencies(self) -> None:
        """Import required dependencies."""
        try:
            import torch
            self._torch = torch

            if self.device == "cuda":
                if not torch.cuda.is_available():
                    print("-> Warning: CUDA requested but not available, falling back to CPU")
                    self.device = "cpu"
                else:
                    print(f"-> GPU (CUDA) available: {torch.cuda.get_device_name(0)}")
                    print(f"-> CUDA version: {torch.version.cuda}")
                    # Jetson devices need CUDA memory caching disabled
                    gpu_name = torch.cuda.get_device_name(0).lower()
                    if "orin" in gpu_name or "jetson" in gpu_name:
                        os.environ.setdefault("PYTORCH_NO_CUDA_MEMORY_CACHING", "1")

        except ImportError as e:
            raise ImportError("PyTorch not installed. Run: pip install torch") from e

        try:
            from vibevoice.modular.modeling_vibevoice_streaming_inference import (
                VibeVoiceStreamingForConditionalGenerationInference,
            )
            from vibevoice.processor.vibevoice_streaming_processor import (
                VibeVoiceStreamingProcessor,
            )
            self._ModelClass = VibeVoiceStreamingForConditionalGenerationInference
            self._ProcessorClass = VibeVoiceStreamingProcessor
            print("-> VibeVoice library loaded successfully")
        except ImportError as e:
            print("-> Error: VibeVoice not installed. Install with:")
            print("   git clone https://github.com/microsoft/VibeVoice.git")
            print("   cd VibeVoice && pip install -e .")
            raise ImportError("VibeVoice not installed.") from e

    def _is_jetson(self) -> bool:
        """Check if running on NVIDIA Jetson platform."""
        try:
            gpu_name = self._torch.cuda.get_device_name(0).lower()
            return "orin" in gpu_name or "jetson" in gpu_name
        except Exception:
            return False

    def _get_dtype_and_attention(self) -> tuple:
        """Get appropriate dtype and attention implementation for device."""
        if self.device == "cuda":
            # Jetson devices prefer float16 over bfloat16
            if self._is_jetson():
                print("-> Detected Jetson platform, using float16 for optimal performance")
                return self._torch.float16, "sdpa"
            try:
                import flash_attn  # noqa: F401
                print("-> Using Flash Attention 2 for optimal GPU performance")
                return self._torch.bfloat16, "flash_attention_2"
            except ImportError:
                print("-> Flash Attention 2 not installed, using SDPA (still fast on GPU)")
                return self._torch.bfloat16, "sdpa"
        elif self.device == "mps":
            return self._torch.float32, "sdpa"
        else:
            return self._torch.float32, "sdpa"

    def _initialize_model(self) -> None:
        """Initialize the VibeVoice model and processor."""
        if self._model is not None:
            return

        print(f"-> Loading VibeVoice model on device: {self.device}")

        # Disable transformers' caching_allocator_warmup for Jetson compatibility
        # This prevents OOM errors during model loading on memory-constrained devices
        if self._is_jetson():
            try:
                import transformers.modeling_utils as mu
                original_warmup = getattr(mu, '_original_caching_allocator_warmup', None)
                if original_warmup is None:
                    mu._original_caching_allocator_warmup = mu.caching_allocator_warmup
                    def no_warmup(*args, **kwargs):
                        return
                    mu.caching_allocator_warmup = no_warmup
                    print("-> Disabled caching_allocator_warmup for Jetson")
            except Exception:
                pass

        dtype, attn_impl = self._get_dtype_and_attention()

        try:
            # Clear GPU memory before loading on Jetson
            if self._is_jetson() and self.device == "cuda":
                self._torch.cuda.empty_cache()
                import gc
                gc.collect()
                print("-> Cleared GPU memory cache")

            self._processor = self._ProcessorClass.from_pretrained(self.model_path)

            if self.device == "cuda":
                if self._is_jetson():
                    # On Jetson, use accelerate's auto device mapping with memory limits
                    # This handles unified memory architecture better than direct .to()
                    max_memory = {0: '3GB', 'cpu': '6GB'}
                    print(f"-> Using accelerate device_map with max_memory={max_memory}")
                    self._model = self._ModelClass.from_pretrained(
                        self.model_path,
                        torch_dtype=dtype,
                        device_map="auto",
                        max_memory=max_memory,
                        attn_implementation=attn_impl,
                        low_cpu_mem_usage=True,
                    )
                else:
                    self._model = self._ModelClass.from_pretrained(
                        self.model_path,
                        torch_dtype=dtype,
                        device_map="cuda",
                        attn_implementation=attn_impl,
                        low_cpu_mem_usage=True,
                    )
            elif self.device == "mps":
                self._model = self._ModelClass.from_pretrained(
                    self.model_path,
                    torch_dtype=dtype,
                    device_map=None,
                    attn_implementation=attn_impl,
                    low_cpu_mem_usage=True,
                )
                self._model.to("mps")
            else:
                self._model = self._ModelClass.from_pretrained(
                    self.model_path,
                    torch_dtype=dtype,
                    device_map="cpu",
                    attn_implementation=attn_impl,
                    low_cpu_mem_usage=True,
                )

            self._model.eval()
            self._model.set_ddpm_inference_steps(num_steps=self.num_inference_steps)

            self._load_voice_prompt()

            print(f"-> Using VibeVoice TTS with speaker: {self.speaker}")
            print(f"-> Model loaded on {self.device.upper()} with {dtype}")

        except Exception as e:
            print(f"-> Error initializing VibeVoice: {e}")
            raise

    def _load_voice_prompt(self) -> None:
        """Load the voice prompt for the selected speaker."""
        if not self._available_voices:
            print("-> Warning: No voice presets available")
            self._voice_prompt = None
            return

        voice_path = self._available_voices.get(self.speaker)

        if not voice_path:
            for name, path in self._available_voices.items():
                if self.speaker in name or name in self.speaker:
                    voice_path = path
                    self.speaker = name
                    break

        if not voice_path and self._available_voices:
            first_voice = list(self._available_voices.keys())[0]
            voice_path = self._available_voices[first_voice]
            print(f"-> Warning: Voice '{self.speaker}' not found, using '{first_voice}'")
            self.speaker = first_voice

        if voice_path:
            try:
                target_device = self.device if self.device != "cpu" else "cpu"
                self._voice_prompt = self._torch.load(
                    voice_path,
                    map_location=target_device,
                    weights_only=False,
                )
                # Convert voice prompt tensors to model dtype for consistency
                if self._voice_prompt is not None and self._model is not None:
                    model_dtype = next(self._model.parameters()).dtype
                    self._voice_prompt = self._convert_prompt_dtype(
                        self._voice_prompt, model_dtype, target_device
                    )
                print(f"-> Loaded voice prompt: {os.path.basename(voice_path)}")
            except Exception as e:
                print(f"-> Warning: Failed to load voice prompt: {e}")
                self._voice_prompt = None

    def _convert_prompt_dtype(self, prompt, dtype, device):
        """Recursively convert prompt tensors to specified dtype and device."""
        if isinstance(prompt, self._torch.Tensor):
            if prompt.is_floating_point():
                return prompt.to(dtype=dtype, device=device)
            else:
                return prompt.to(device=device)
        elif isinstance(prompt, dict):
            # Check if this is an outputs-like dict that needs attribute access
            if 'last_hidden_state' in prompt and 'past_key_values' in prompt:
                # This is a model output dict - convert and wrap for attribute access
                converted = {}
                for k, v in prompt.items():
                    converted[k] = self._convert_prompt_dtype(v, dtype, device)
                return _DictWithAttrs(converted)
            else:
                return {k: self._convert_prompt_dtype(v, dtype, device) for k, v in prompt.items()}
        elif isinstance(prompt, (list, tuple)):
            converted = [self._convert_prompt_dtype(v, dtype, device) for v in prompt]
            return type(prompt)(converted)
        else:
            # Handle DynamicCache and similar objects with key_cache/value_cache
            if hasattr(prompt, 'key_cache') and hasattr(prompt, 'value_cache'):
                for i in range(len(prompt.key_cache)):
                    if prompt.key_cache[i].is_floating_point():
                        prompt.key_cache[i] = prompt.key_cache[i].to(dtype=dtype, device=device)
                    else:
                        prompt.key_cache[i] = prompt.key_cache[i].to(device=device)
                    if prompt.value_cache[i].is_floating_point():
                        prompt.value_cache[i] = prompt.value_cache[i].to(dtype=dtype, device=device)
                    else:
                        prompt.value_cache[i] = prompt.value_cache[i].to(device=device)
            return prompt

    def synthesize(
        self,
        text: str,
        temperature: float = 0.9,
        top_p: float = 0.9,
        do_sample: bool = True,
        cfg_scale: Optional[float] = None,
    ) -> np.ndarray:
        """Synthesize speech from text with expressive control.

        Args:
            text: Text to synthesize
            temperature: Sampling temperature (0.1-1.5). Higher = more expressive/varied (default: 0.9)
            top_p: Nucleus sampling threshold (0.5-1.0). Lower = more focused (default: 0.9)
            do_sample: Enable sampling for expressiveness. False = deterministic (default: True)
            cfg_scale: Classifier-free guidance scale override (default: use model's default)

        Returns:
            Audio data as float32 numpy array
        """
        if not text.strip():
            return np.zeros(0, dtype=np.float32)

        if self._model is None:
            self._initialize_model()

        text = text.replace("'", "'").replace('"', '"').replace('"', '"')

        # Use provided cfg_scale or fall back to instance default
        effective_cfg_scale = cfg_scale if cfg_scale is not None else self.cfg_scale

        try:
            inputs = self._processor.process_input_with_cached_prompt(
                text=text,
                cached_prompt=self._voice_prompt,
                padding=True,
                return_tensors="pt",
                return_attention_mask=True,
            )

            target_device = self.device if self.device != "cpu" else "cpu"
            for k, v in inputs.items():
                if self._torch.is_tensor(v):
                    inputs[k] = v.to(target_device)

            start_time = time.time()
            with self._torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=None,
                    cfg_scale=effective_cfg_scale,
                    tokenizer=self._processor.tokenizer,
                    generation_config={
                        'do_sample': do_sample,
                        'temperature': temperature if do_sample else 1.0,
                        'top_p': top_p if do_sample else 1.0,
                    },
                    verbose=False,
                    all_prefilled_outputs=copy.deepcopy(self._voice_prompt) if self._voice_prompt else None,
                )
            gen_time = time.time() - start_time

            if outputs.speech_outputs and outputs.speech_outputs[0] is not None:
                speech = outputs.speech_outputs[0]

                if hasattr(speech, 'cpu'):
                    # Convert bfloat16 to float32 before numpy conversion
                    audio_np = speech.float().cpu().numpy()
                else:
                    audio_np = np.array(speech, dtype=np.float32)

                audio_np = audio_np.squeeze()

                audio_duration = len(audio_np) / self.sample_rate
                rtf = gen_time / audio_duration if audio_duration > 0 else float('inf')
                print(f"-> Generated {audio_duration:.2f}s audio in {gen_time:.2f}s (RTF: {rtf:.2f}x)")

                max_val = np.abs(audio_np).max()
                if max_val > 1.0:
                    audio_np = audio_np / max_val

                return audio_np
            else:
                print("-> Warning: No audio output generated")
                return np.zeros(0, dtype=np.float32)

        except Exception as e:
            print(f"-> Error synthesizing speech: {e}")
            import traceback
            traceback.print_exc()
            return np.zeros(0, dtype=np.float32)

    def _chunk_audio(self, audio_data: np.ndarray, chunk_size: int) -> List[np.ndarray]:
        """Split audio into chunks for playback with amplitude monitoring."""
        return [audio_data[i : i + chunk_size] for i in range(0, len(audio_data), chunk_size)]

    def play_audio_with_amplitude(
        self,
        audio_data: np.ndarray,
        amplitude_callback: Optional[Callable[[float], None]] = None,
        chunk_duration: float = 0.02,
    ) -> None:
        """Play audio with optional amplitude callback for animation."""
        if audio_data is None or len(audio_data) == 0:
            if amplitude_callback:
                amplitude_callback(0.0)
            return

        audio_float = audio_data.astype(np.float32, copy=False)
        chunk_size = max(1, int(self.sample_rate * chunk_duration))
        chunks = self._chunk_audio(audio_float, chunk_size)
        if not chunks:
            return

        rms_values = [float(np.sqrt(np.mean(np.square(chunk)) + 1e-8)) for chunk in chunks]
        max_rms = max(rms_values) or 1.0
        normalized_levels = [min(rms / max_rms, 1.0) for rms in rms_values]

        audio_int16 = np.clip(audio_float * 32767.0, -32768, 32767).astype(np.int16)

        pa = pyaudio.PyAudio()
        stream = None

        try:
            stream = pa.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                output=True,
                frames_per_buffer=4096,
            )

            cursor = 0
            for chunk, level in zip(chunks, normalized_levels):
                frames = audio_int16[cursor : cursor + len(chunk)]
                stream.write(frames.tobytes())
                cursor += len(chunk)
                if amplitude_callback:
                    amplitude_callback(level)
        finally:
            if stream is not None:
                try:
                    output_latency = stream.get_output_latency()
                except Exception:
                    output_latency = 0.0

                drain_wait = (
                    max(output_latency, chunk_duration)
                    if output_latency and output_latency > 0
                    else chunk_duration
                )
                if drain_wait > 0:
                    time.sleep(drain_wait)

                if amplitude_callback:
                    amplitude_callback(0.0)
                stream.stop_stream()
                stream.close()
            elif amplitude_callback:
                amplitude_callback(0.0)
            pa.terminate()

    def list_speakers(self) -> List[str]:
        """Return list of available speaker voices."""
        return list(self._available_voices.keys())


__all__ = ["VibeVoiceTTS"]
