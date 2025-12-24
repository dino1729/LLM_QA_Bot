"""
Tests for Chatterbox TTS integration.

These tests verify the Chatterbox TTS module functionality including
initialization, synthesis, and voice cloning capabilities.

Note: These tests intentionally use specific model types ("turbo", "multilingual")
and devices ("cuda", "cpu") to test model-specific behavior. This is correct
for unit tests - we're testing the ChatterboxTTS class handles each model type
correctly, not that the config values are used.

For integration tests that verify config loading, use fixtures from conftest.py.
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from helper_functions.tts_chatterbox import ChatterboxTTS, get_chatterbox_tts
from config import config


class TestChatterboxTTSInitialization:
    """Tests for ChatterboxTTS initialization and setup."""

    @patch('helper_functions.tts_chatterbox.ChatterboxTTS._import_dependencies')
    @patch('helper_functions.tts_chatterbox.ChatterboxTTS._initialize_model')
    def test_init_turbo_model(self, mock_init_model, mock_import_deps):
        """Test initialization with turbo model."""
        tts = ChatterboxTTS(model_type="turbo")

        assert tts.model_type == "turbo"
        assert tts.device == "cuda"
        assert tts.cfg_weight == 0.5
        assert tts.exaggeration == 0.5
        mock_import_deps.assert_called_once()
        mock_init_model.assert_called_once()

    @patch('helper_functions.tts_chatterbox.ChatterboxTTS._import_dependencies')
    @patch('helper_functions.tts_chatterbox.ChatterboxTTS._initialize_model')
    def test_init_multilingual_model(self, mock_init_model, mock_import_deps):
        """Test initialization with multilingual model."""
        tts = ChatterboxTTS(model_type="multilingual")

        assert tts.model_type == "multilingual"
        assert tts.device == "cuda"

    @patch('helper_functions.tts_chatterbox.ChatterboxTTS._import_dependencies')
    @patch('helper_functions.tts_chatterbox.ChatterboxTTS._initialize_model')
    def test_init_invalid_model_type(self, mock_init_model, mock_import_deps):
        """Test initialization with invalid model type."""
        with pytest.raises(ValueError, match="Invalid model_type"):
            ChatterboxTTS(model_type="invalid")


class TestChatterboxTTSSynthesis:
    """Tests for audio synthesis functionality."""

    @pytest.fixture
    def mock_tts(self):
        """Create a mock TTS instance with necessary components."""
        with patch('helper_functions.tts_chatterbox.ChatterboxTTS._import_dependencies'), \
             patch('helper_functions.tts_chatterbox.ChatterboxTTS._initialize_model'):

            tts = ChatterboxTTS(model_type="turbo")

            # Mock torch
            tts._torch = Mock()
            tts._torch.cuda.is_available.return_value = True
            tts._torch.amp.autocast = MagicMock()

            # Mock model
            tts._model = Mock()
            tts.sample_rate = 24000

            return tts

    def test_synthesize_empty_text(self, mock_tts):
        """Test synthesis with empty text."""
        result = mock_tts.synthesize("")

        assert isinstance(result, np.ndarray)
        assert len(result) == 0

    def test_synthesize_turbo_without_audio_prompt(self, mock_tts):
        """Test turbo model synthesis without required audio prompt."""
        mock_tts.model_type = "turbo"

        with pytest.raises(ValueError, match="Turbo model requires audio_prompt_path"):
            mock_tts.synthesize("Test text")

    def test_synthesize_multilingual_without_language_id(self, mock_tts):
        """Test multilingual model synthesis without language ID."""
        mock_tts.model_type = "multilingual"

        # The exception is caught and returns empty array
        result = mock_tts.synthesize("Test text")
        assert isinstance(result, np.ndarray)
        assert len(result) == 0  # Empty because of error

    def test_synthesize_success(self, mock_tts):
        """Test successful audio synthesis."""
        # Setup mock model output
        mock_audio = np.random.randn(24000).astype(np.float32)
        mock_wav = Mock()
        mock_wav.cpu.return_value.numpy.return_value = mock_audio

        mock_tts._model.generate.return_value = mock_wav
        mock_tts._torch.amp.autocast.return_value.__enter__ = Mock()
        mock_tts._torch.amp.autocast.return_value.__exit__ = Mock()

        # Test synthesis
        result = mock_tts.synthesize(
            "Test text",
            audio_prompt_path="test.wav"
        )

        assert isinstance(result, np.ndarray)
        assert len(result) > 0
        mock_tts._model.generate.assert_called_once()


class TestChatterboxTTSAudioSaving:
    """Tests for audio file saving functionality."""

    @pytest.fixture
    def mock_tts(self):
        """Create a mock TTS instance."""
        with patch('helper_functions.tts_chatterbox.ChatterboxTTS._import_dependencies'), \
             patch('helper_functions.tts_chatterbox.ChatterboxTTS._initialize_model'):

            tts = ChatterboxTTS(model_type="turbo")
            tts.sample_rate = 24000
            return tts

    @patch('soundfile.write')
    def test_save_audio_success(self, mock_sf_write, mock_tts):
        """Test successful audio saving."""
        audio_data = np.random.randn(24000).astype(np.float32)

        result = mock_tts.save_audio(audio_data, "test_output.wav")

        assert result is True
        mock_sf_write.assert_called_once()

    @patch('soundfile.write')
    def test_save_audio_adds_wav_extension(self, mock_sf_write, mock_tts):
        """Test that .wav extension is added if missing."""
        audio_data = np.random.randn(24000).astype(np.float32)

        mock_tts.save_audio(audio_data, "test_output.mp3")

        # Check that the call was made with .wav extension
        call_args = mock_sf_write.call_args
        assert call_args[0][0].endswith('.wav')


class TestChatterboxTTSCaching:
    """Tests for model caching functionality."""

    @patch('helper_functions.tts_chatterbox.ChatterboxTTS')
    def test_get_chatterbox_tts_caches_instance(self, mock_chatterbox_class):
        """Test that get_chatterbox_tts caches instances.
        
        Uses explicit model_type/device to test caching logic.
        """
        # Clear cache first
        import helper_functions.tts_chatterbox as tts_module
        tts_module._chatterbox_tts_cache.clear()

        # Get instance twice with explicit params (testing cache key logic)
        tts1 = get_chatterbox_tts(model_type="turbo", device="cpu")
        tts2 = get_chatterbox_tts(model_type="turbo", device="cpu")

        # Should only create one instance
        assert mock_chatterbox_class.call_count == 1
        assert tts1 is tts2

    @patch('helper_functions.tts_chatterbox.ChatterboxTTS')
    def test_get_chatterbox_tts_different_models(self, mock_chatterbox_class):
        """Test that different model types create separate cache entries.
        
        Uses explicit model types to verify cache key includes model_type.
        """
        # Clear cache first
        import helper_functions.tts_chatterbox as tts_module
        tts_module._chatterbox_tts_cache.clear()

        # Get different model types (testing different cache keys)
        tts1 = get_chatterbox_tts(model_type="turbo", device="cpu")
        tts2 = get_chatterbox_tts(model_type="multilingual", device="cpu")

        # Should create two instances
        assert mock_chatterbox_class.call_count == 2
    
    @patch('helper_functions.tts_chatterbox.ChatterboxTTS')
    def test_get_chatterbox_tts_uses_config_defaults(self, mock_chatterbox_class):
        """Test that get_chatterbox_tts uses config values when no params provided.
        
        This test verifies the integration with config.yml.
        """
        # Clear cache first
        import helper_functions.tts_chatterbox as tts_module
        tts_module._chatterbox_tts_cache.clear()

        # Skip test if config values not set
        if not config.chatterbox_tts_model_type or not config.chatterbox_tts_device:
            pytest.skip("Requires chatterbox_tts_model_type and chatterbox_tts_device in config.yml")

        # Get instance without params - should use config values
        tts = get_chatterbox_tts()

        # Verify ChatterboxTTS was called with config values
        mock_chatterbox_class.assert_called_once_with(
            model_type=config.chatterbox_tts_model_type,
            device=config.chatterbox_tts_device
        )


class TestChatterboxTTSMultilingual:
    """Tests for multilingual functionality."""

    @pytest.fixture
    def mock_multilingual_tts(self):
        """Create a mock multilingual TTS instance."""
        with patch('helper_functions.tts_chatterbox.ChatterboxTTS._import_dependencies'), \
             patch('helper_functions.tts_chatterbox.ChatterboxTTS._initialize_model'):

            tts = ChatterboxTTS(model_type="multilingual")
            return tts

    def test_get_supported_languages(self, mock_multilingual_tts):
        """Test getting supported languages for multilingual model."""
        languages = mock_multilingual_tts.get_supported_languages()

        assert isinstance(languages, list)
        assert len(languages) > 0
        assert "en" in languages
        assert "fr" in languages
        assert "zh" in languages

    def test_get_supported_languages_non_multilingual(self):
        """Test that non-multilingual models return empty language list."""
        with patch('helper_functions.tts_chatterbox.ChatterboxTTS._import_dependencies'), \
             patch('helper_functions.tts_chatterbox.ChatterboxTTS._initialize_model'):

            tts = ChatterboxTTS(model_type="turbo")
            languages = tts.get_supported_languages()

            assert languages == []


class TestChatterboxTTSGPUInfo:
    """Tests for GPU information retrieval."""

    @pytest.fixture
    def mock_tts_with_torch(self):
        """Create a mock TTS instance with torch."""
        with patch('helper_functions.tts_chatterbox.ChatterboxTTS._import_dependencies'), \
             patch('helper_functions.tts_chatterbox.ChatterboxTTS._initialize_model'):

            tts = ChatterboxTTS(model_type="turbo")
            tts._torch = Mock()
            return tts

    def test_get_gpu_memory_info_available(self, mock_tts_with_torch):
        """Test GPU memory info when CUDA is available."""
        mock_tts_with_torch._torch.cuda.is_available.return_value = True
        mock_tts_with_torch._torch.cuda.memory_allocated.return_value = 4 * 1024**3  # 4 GB
        mock_tts_with_torch._torch.cuda.memory_reserved.return_value = 5 * 1024**3  # 5 GB

        mock_props = Mock()
        mock_props.total_memory = 32 * 1024**3  # 32 GB
        mock_tts_with_torch._torch.cuda.get_device_properties.return_value = mock_props

        info = mock_tts_with_torch.get_gpu_memory_info()

        assert 'allocated_gb' in info
        assert 'reserved_gb' in info
        assert 'total_gb' in info
        assert info['allocated_gb'] == pytest.approx(4.0, rel=0.1)
        assert info['reserved_gb'] == pytest.approx(5.0, rel=0.1)
        assert info['total_gb'] == pytest.approx(32.0, rel=0.1)

    def test_get_gpu_memory_info_unavailable(self, mock_tts_with_torch):
        """Test GPU memory info when CUDA is not available."""
        mock_tts_with_torch._torch.cuda.is_available.return_value = False

        info = mock_tts_with_torch.get_gpu_memory_info()

        assert info == {}
