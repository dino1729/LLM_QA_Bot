"""Text-to-speech using ResembleAI Chatterbox TTS with GPU acceleration.

This module provides integration with Chatterbox-Turbo, a 350M parameter TTS model
optimized for low-latency voice agents with sub-200ms generation time.

Key features:
- Paralinguistic tags support: [laugh], [chuckle], [cough]
- Zero-shot voice cloning with reference audio
- Built-in Perth watermarking for responsible AI
- Multiple model options: Turbo, Standard, Multilingual
"""

from __future__ import annotations

import os
import time
from typing import List, Optional, Callable

import numpy as np
import pyaudio


class ChatterboxTTS:
    """Generate audio using ResembleAI Chatterbox TTS with GPU acceleration.

    Chatterbox-Turbo is a realtime TTS model with sub-200ms latency.
    It provides high-quality, expressive speech synthesis with paralinguistic
    tags and voice cloning capabilities.

    This implementation is optimized for NVIDIA GPU inference.
    """

    def __init__(
        self,
        model_type: str = "turbo",
        model_path: Optional[str] = None,
        cfg_weight: float = 0.5,
        exaggeration: float = 0.5,
    ) -> None:
        """Initialize Chatterbox TTS for GPU inference.

        Args:
            model_type: Model variant to use ('turbo', 'standard', 'multilingual').
                - 'turbo': 350M params, English, paralinguistic tags, low latency
                - 'standard': 500M params, English, creative controls
                - 'multilingual': 500M params, 23+ languages
            model_path: Custom model path (optional, uses default pretrained if None).
            cfg_weight: Classifier-free guidance weight (0.0-1.0, default: 0.5).
                Lower values produce faster speech.
            exaggeration: Speech exaggeration level (0.0-1.0, default: 0.5).
                Higher values produce more dramatic/expressive speech.
        """
        self.model_type = model_type.lower()
        if self.model_type not in ["turbo", "standard", "multilingual"]:
            raise ValueError(f"Invalid model_type: {model_type}. Must be 'turbo', 'standard', or 'multilingual'")

        self.model_path = model_path
        self.cfg_weight = cfg_weight
        self.exaggeration = exaggeration
        self.device = "cuda"  # Always use CUDA for GPU inference

        # Lazy-load model components
        self._model = None
        self._torch = None
        self._torchaudio = None
        self.sample_rate = None  # Set during model initialization

        # Import and validate
        self._import_dependencies()
        self._initialize_model()

    def _import_dependencies(self) -> None:
        """Import required dependencies and validate GPU availability."""
        try:
            import torch
            self._torch = torch

            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA is not available. This module requires an NVIDIA GPU. "
                    "Please ensure CUDA is properly installed and configured."
                )

            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # Convert to GB
            print(f"-> GPU detected: {gpu_name}")
            print(f"-> GPU memory: {gpu_memory:.2f} GB")
            print(f"-> CUDA version: {torch.version.cuda}")
            print(f"-> PyTorch version: {torch.__version__}")

        except ImportError as e:
            raise ImportError(
                "PyTorch not installed. Run: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
            ) from e

        try:
            if self.model_type == "turbo":
                from chatterbox.tts_turbo import ChatterboxTurboTTS
                self._ModelClass = ChatterboxTurboTTS
            elif self.model_type == "multilingual":
                from chatterbox.mtl_tts import ChatterboxMultilingualTTS
                self._ModelClass = ChatterboxMultilingualTTS
            else:  # standard
                from chatterbox.tts import ChatterboxTTS as StandardChatterboxTTS
                self._ModelClass = StandardChatterboxTTS
            print(f"-> Chatterbox {self.model_type.capitalize()} TTS library loaded successfully")
        except ImportError as e:
            print(f"-> Import error: {str(e)}")
            if "chatterbox" in str(e).lower():
                print("-> Error: Chatterbox TTS not installed. Install with:")
                print("   pip install chatterbox-tts")
                print("   or from source:")
                print("   git clone https://github.com/resemble-ai/chatterbox.git")
                print("   cd chatterbox && pip install -e .")
            raise ImportError(f"Failed to import Chatterbox TTS: {str(e)}") from e

    def _initialize_model(self) -> None:
        """Initialize the Chatterbox TTS model on GPU."""
        if self._model is not None:
            return

        print(f"-> Loading Chatterbox {self.model_type.capitalize()} TTS model on GPU...")

        try:
            # Clear GPU cache before loading
            if self._torch.cuda.is_available():
                self._torch.cuda.empty_cache()
                print("-> Cleared GPU memory cache")

            # Load the model from pretrained or custom path (always on CUDA)
            if self.model_path:
                self._model = self._ModelClass.from_pretrained(
                    self.model_path,
                    device="cuda"
                )
            else:
                self._model = self._ModelClass.from_pretrained(device="cuda")

            # Get sample rate from model
            self.sample_rate = self._model.sr

            # Get GPU memory usage after loading
            if self._torch.cuda.is_available():
                memory_allocated = self._torch.cuda.memory_allocated(0) / (1024**3)  # GB
                memory_reserved = self._torch.cuda.memory_reserved(0) / (1024**3)  # GB
                print(f"-> Model loaded on GPU successfully")
                print(f"-> GPU memory allocated: {memory_allocated:.2f} GB")
                print(f"-> GPU memory reserved: {memory_reserved:.2f} GB")

            print(f"-> Sample rate: {self.sample_rate} Hz")
            print(f"-> Using Chatterbox {self.model_type.capitalize()} TTS")

        except Exception as e:
            print(f"-> Error initializing Chatterbox TTS: {e}")
            raise

    def synthesize(
        self,
        text: str,
        audio_prompt_path: Optional[str] = None,
        language_id: Optional[str] = None,
        cfg_weight: Optional[float] = None,
        exaggeration: Optional[float] = None,
    ) -> np.ndarray:
        """Synthesize speech from text with expressive control.

        Args:
            text: Text to synthesize. For Turbo model, can include paralinguistic tags
                like [laugh], [chuckle], [cough].
            audio_prompt_path: Path to reference audio for voice cloning (10s recommended).
                Required for Turbo model.
            language_id: Language code for multilingual model (e.g., "en", "fr", "zh").
                Only used with multilingual model.
            cfg_weight: Classifier-free guidance weight override (default: use model's default).
                Lower values (0.3) for faster speech, higher (0.7) for more controlled.
            exaggeration: Exaggeration level override (default: use model's default).
                Higher values (0.7+) for dramatic/expressive speech.

        Returns:
            Audio data as float32 numpy array

        Raises:
            ValueError: If Turbo model is used without audio_prompt_path.
        """
        if not text.strip():
            return np.zeros(0, dtype=np.float32)

        if self._model is None:
            self._initialize_model()

        # Validate inputs based on model type
        if self.model_type == "turbo" and not audio_prompt_path:
            raise ValueError("Turbo model requires audio_prompt_path for voice cloning")

        # Use provided parameters or fall back to instance defaults
        effective_cfg_weight = cfg_weight if cfg_weight is not None else self.cfg_weight
        effective_exaggeration = exaggeration if exaggeration is not None else self.exaggeration

        try:
            start_time = time.time()

            # Generate audio based on model type (all on GPU)
            with self._torch.amp.autocast('cuda'):  # Use automatic mixed precision for efficiency
                if self.model_type == "turbo":
                    wav = self._model.generate(
                        text,
                        audio_prompt_path=audio_prompt_path,
                        cfg_weight=effective_cfg_weight,
                        exaggeration=effective_exaggeration,
                    )
                elif self.model_type == "multilingual":
                    if not language_id:
                        raise ValueError("Multilingual model requires language_id parameter")
                    wav = self._model.generate(
                        text,
                        language_id=language_id,
                    )
                else:  # standard
                    wav = self._model.generate(
                        text,
                        cfg_weight=effective_cfg_weight,
                        exaggeration=effective_exaggeration,
                    )

            gen_time = time.time() - start_time

            # Convert to numpy array (move from GPU to CPU)
            if hasattr(wav, 'cpu'):
                audio_np = wav.cpu().numpy()
            else:
                audio_np = np.array(wav, dtype=np.float32)

            audio_np = audio_np.squeeze()

            # Calculate real-time factor
            audio_duration = len(audio_np) / self.sample_rate
            rtf = gen_time / audio_duration if audio_duration > 0 else float('inf')
            print(f"-> Generated {audio_duration:.2f}s audio in {gen_time:.2f}s (RTF: {rtf:.2f}x)")

            # Normalize audio if needed
            max_val = np.abs(audio_np).max()
            if max_val > 1.0:
                audio_np = audio_np / max_val

            return audio_np

        except Exception as e:
            print(f"-> Error synthesizing speech: {e}")
            import traceback
            traceback.print_exc()
            return np.zeros(0, dtype=np.float32)

    def save_audio(self, audio_data: np.ndarray, output_path: str) -> bool:
        """Save audio data to file.

        Args:
            audio_data: Audio data as numpy array
            output_path: Path to save audio file (will be saved as WAV)

        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure output path has .wav extension
            if not output_path.endswith('.wav'):
                output_path = os.path.splitext(output_path)[0] + '.wav'

            # Use soundfile for reliable audio saving (avoids torchcodec/FFmpeg issues)
            import soundfile as sf
            sf.write(output_path, audio_data, self.sample_rate, subtype='PCM_16')

            print(f"-> Audio saved to {output_path}")
            return True

        except Exception as e:
            print(f"-> Error saving audio: {e}")
            return False

    def synthesize_to_file(
        self,
        text: str,
        output_path: str,
        audio_prompt_path: Optional[str] = None,
        language_id: Optional[str] = None,
        cfg_weight: Optional[float] = None,
        exaggeration: Optional[float] = None,
    ) -> bool:
        """Synthesize speech and save directly to file.

        Args:
            text: Text to synthesize
            output_path: Path to save audio file
            audio_prompt_path: Path to reference audio for voice cloning
            language_id: Language code for multilingual model
            cfg_weight: Classifier-free guidance weight override
            exaggeration: Exaggeration level override

        Returns:
            True if successful, False otherwise
        """
        audio_data = self.synthesize(
            text,
            audio_prompt_path=audio_prompt_path,
            language_id=language_id,
            cfg_weight=cfg_weight,
            exaggeration=exaggeration,
        )

        if audio_data is None or len(audio_data) == 0:
            return False

        return self.save_audio(audio_data, output_path)

    def _chunk_audio(self, audio_data: np.ndarray, chunk_size: int) -> List[np.ndarray]:
        """Split audio into chunks for playback with amplitude monitoring."""
        return [audio_data[i : i + chunk_size] for i in range(0, len(audio_data), chunk_size)]

    def play_audio_with_amplitude(
        self,
        audio_data: np.ndarray,
        amplitude_callback: Optional[Callable[[float], None]] = None,
        chunk_duration: float = 0.02,
    ) -> None:
        """Play audio with optional amplitude callback for animation.

        Args:
            audio_data: Audio data to play
            amplitude_callback: Optional callback function that receives amplitude (0.0-1.0)
            chunk_duration: Duration of each chunk in seconds (default: 0.02s = 20ms)
        """
        if audio_data is None or len(audio_data) == 0:
            if amplitude_callback:
                amplitude_callback(0.0)
            return

        audio_float = audio_data.astype(np.float32, copy=False)
        chunk_size = max(1, int(self.sample_rate * chunk_duration))
        chunks = self._chunk_audio(audio_float, chunk_size)
        if not chunks:
            return

        # Calculate normalized amplitude levels
        rms_values = [float(np.sqrt(np.mean(np.square(chunk)) + 1e-8)) for chunk in chunks]
        max_rms = max(rms_values) or 1.0
        normalized_levels = [min(rms / max_rms, 1.0) for rms in rms_values]

        # Convert to int16 for playback
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

    def get_supported_languages(self) -> List[str]:
        """Return list of supported languages for multilingual model.

        Returns:
            List of language codes, or empty list for non-multilingual models
        """
        if self.model_type == "multilingual":
            return [
                "ar", "da", "de", "el", "en", "es", "fi", "fr", "he", "hi",
                "it", "ja", "ko", "ms", "nl", "no", "pl", "pt", "ru", "sv",
                "sw", "tr", "zh"
            ]
        return []

    def get_gpu_memory_info(self) -> dict:
        """Get current GPU memory usage information.

        Returns:
            Dictionary with GPU memory statistics in GB
        """
        if self._torch and self._torch.cuda.is_available():
            return {
                "allocated_gb": self._torch.cuda.memory_allocated(0) / (1024**3),
                "reserved_gb": self._torch.cuda.memory_reserved(0) / (1024**3),
                "total_gb": self._torch.cuda.get_device_properties(0).total_memory / (1024**3),
            }
        return {}


# Cache for model instances (one per model type)
_chatterbox_tts_cache = {}


def get_chatterbox_tts(
    model_type: str = "turbo",
    **kwargs
) -> ChatterboxTTS:
    """Get or create a cached Chatterbox TTS instance.

    This function maintains a cache of TTS instances to avoid reloading models.
    One instance is cached per model type. All models run on GPU.

    Args:
        model_type: Model variant ('turbo', 'standard', 'multilingual')
        **kwargs: Additional arguments passed to ChatterboxTTS constructor

    Returns:
        ChatterboxTTS instance
    """
    cache_key = model_type

    if cache_key not in _chatterbox_tts_cache:
        _chatterbox_tts_cache[cache_key] = ChatterboxTTS(
            model_type=model_type,
            **kwargs
        )

    return _chatterbox_tts_cache[cache_key]


__all__ = ["ChatterboxTTS", "get_chatterbox_tts"]
