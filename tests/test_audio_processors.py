"""
Unit tests for helper_functions/audio_processors.py
Tests for NVIDIA Riva-based audio transcription and text-to-speech
"""
import os
import pytest
from unittest.mock import Mock, MagicMock, patch, mock_open
from helper_functions import audio_processors


class TestTranscribeAudio:
    """Tests for transcribe_audio() function"""
    
    @patch('helper_functions.audio_processors.RIVA_AVAILABLE', True)
    @patch('helper_functions.audio_processors.asr_service')
    def test_transcribe_audio_success(self, mock_asr_service, tmp_path):
        """Test successful audio transcription"""
        # Create a fake audio file
        audio_file = tmp_path / "test_audio.wav"
        audio_file.write_bytes(b"fake audio data")
        
        # Mock ASR response
        mock_response = Mock()
        mock_result = Mock()
        mock_alternative = Mock()
        mock_alternative.transcript = "This is a test transcription"
        mock_result.alternatives = [mock_alternative]
        mock_response.results = [mock_result]
        mock_asr_service.offline_recognize.return_value = mock_response
        
        text, language = audio_processors.transcribe_audio(str(audio_file))
        
        assert text == "This is a test transcription"
        assert language == "en-US"
    
    @patch('helper_functions.audio_processors.RIVA_AVAILABLE', False)
    def test_transcribe_audio_riva_unavailable(self, tmp_path):
        """Test transcription when Riva is unavailable"""
        audio_file = tmp_path / "test_audio.wav"
        audio_file.write_bytes(b"fake audio data")
        
        text, language = audio_processors.transcribe_audio(str(audio_file))
        
        assert "unavailable" in text.lower()
        assert language == "en-US"
    
    @patch('helper_functions.audio_processors.RIVA_AVAILABLE', True)
    @patch('helper_functions.audio_processors.asr_service')
    def test_transcribe_audio_no_speech(self, mock_asr_service, tmp_path):
        """Test transcription with no speech detected"""
        audio_file = tmp_path / "test_audio.wav"
        audio_file.write_bytes(b"fake audio data")
        
        # Mock response with empty results
        mock_response = Mock()
        mock_response.results = []
        mock_asr_service.offline_recognize.return_value = mock_response
        
        text, language = audio_processors.transcribe_audio(str(audio_file))
        
        assert "No speech detected" in text or "empty" in text.lower()
    
    @patch('helper_functions.audio_processors.RIVA_AVAILABLE', True)
    @patch('helper_functions.audio_processors.asr_service')
    def test_transcribe_audio_error(self, mock_asr_service, tmp_path):
        """Test transcription with error"""
        audio_file = tmp_path / "test_audio.wav"
        audio_file.write_bytes(b"fake audio data")
        
        mock_asr_service.offline_recognize.side_effect = Exception("ASR error")
        
        text, language = audio_processors.transcribe_audio(str(audio_file))
        
        # Should handle error gracefully
        assert isinstance(text, str)
        assert isinstance(language, str)


class TestTextToSpeech:
    """Tests for text_to_speech() function"""
    
    @patch('helper_functions.audio_processors.RIVA_AVAILABLE', True)
    @patch('helper_functions.audio_processors.tts_service')
    @patch('helper_functions.audio_processors.sf.write')
    @patch('helper_functions.audio_processors.sd.play')
    @patch('helper_functions.audio_processors.sd.wait')
    def test_text_to_speech_success(self, mock_wait, mock_play, mock_write, 
                                    mock_tts_service, tmp_path):
        """Test successful text-to-speech"""
        output_path = tmp_path / "output.wav"
        
        # Mock TTS response
        mock_response = Mock()
        mock_response.audio = b"fake audio data"
        mock_tts_service.synthesize_online.return_value = [mock_response]
        
        result = audio_processors.text_to_speech(
            "Hello world", 
            str(output_path), 
            "en-US", 
            "GPT4OMINI"
        )
        
        assert result is True
        mock_tts_service.synthesize_online.assert_called_once()
    
    @patch('helper_functions.audio_processors.RIVA_AVAILABLE', False)
    def test_text_to_speech_riva_unavailable(self, tmp_path):
        """Test TTS when Riva is unavailable"""
        output_path = tmp_path / "output.wav"
        
        result = audio_processors.text_to_speech(
            "Hello world", 
            str(output_path), 
            "en-US", 
            "GPT4OMINI"
        )
        
        assert result is False
    
    @patch('helper_functions.audio_processors.RIVA_AVAILABLE', True)
    @patch('helper_functions.audio_processors.tts_service')
    def test_text_to_speech_voice_mapping(self, mock_tts_service, tmp_path):
        """Test voice mapping for different models"""
        output_path = tmp_path / "output.wav"
        
        mock_response = Mock()
        mock_response.audio = b"fake audio data"
        mock_tts_service.synthesize_online.return_value = [mock_response]
        
        # Test different model names
        for model_name in ["GEMINI", "GPT4", "COHERE", "BING+OPENAI", "MIXTRAL8x7B"]:
            with patch('helper_functions.audio_processors.sf.write'), \
                 patch('helper_functions.audio_processors.sd.play'), \
                 patch('helper_functions.audio_processors.sd.wait'):
                
                result = audio_processors.text_to_speech(
                    "Test", str(output_path), "en-US", model_name
                )
                assert result is True
    
    @patch('helper_functions.audio_processors.RIVA_AVAILABLE', True)
    @patch('helper_functions.audio_processors.tts_service')
    def test_text_to_speech_error(self, mock_tts_service, tmp_path):
        """Test TTS with error"""
        output_path = tmp_path / "output.wav"
        mock_tts_service.synthesize_online.side_effect = Exception("TTS error")
        
        result = audio_processors.text_to_speech(
            "Hello", str(output_path), "en-US", "GPT4OMINI"
        )
        
        assert result is False


class TestTextToSpeechNoSpeak:
    """Tests for text_to_speech_nospeak() function"""
    
    @patch('helper_functions.audio_processors.RIVA_AVAILABLE', True)
    @patch('helper_functions.audio_processors.tts_service')
    @patch('helper_functions.audio_processors.sf.write')
    def test_text_to_speech_nospeak_success(self, mock_write, mock_tts_service, tmp_path):
        """Test text-to-speech without playback"""
        output_path = tmp_path / "output.wav"
        
        mock_response = Mock()
        mock_response.audio = b"fake audio data"
        mock_tts_service.synthesize_online.return_value = [mock_response]
        
        result = audio_processors.text_to_speech_nospeak(
            "Hello world", 
            str(output_path)
        )
        
        assert result is True
        mock_write.assert_called_once()
    
    @patch('helper_functions.audio_processors.RIVA_AVAILABLE', False)
    def test_text_to_speech_nospeak_riva_unavailable(self, tmp_path):
        """Test TTS nospeak when Riva is unavailable"""
        output_path = tmp_path / "output.wav"
        
        result = audio_processors.text_to_speech_nospeak(
            "Hello world", 
            str(output_path)
        )
        
        assert result is False
    
    @patch('helper_functions.audio_processors.RIVA_AVAILABLE', True)
    @patch('helper_functions.audio_processors.tts_service')
    def test_text_to_speech_nospeak_different_languages(self, mock_tts_service, tmp_path):
        """Test TTS with different languages"""
        output_path = tmp_path / "output.wav"
        
        mock_response = Mock()
        mock_response.audio = b"fake audio data"
        mock_tts_service.synthesize_online.return_value = [mock_response]
        
        for language in ["en-US", "es-ES", "fr-FR"]:
            with patch('helper_functions.audio_processors.sf.write'):
                result = audio_processors.text_to_speech_nospeak(
                    "Test", str(output_path), language=language
                )
                assert result is True


class TestLocalTextToSpeech:
    """Tests for local_text_to_speech() function"""
    
    @patch('helper_functions.audio_processors.requests.post')
    @patch('helper_functions.audio_processors.sd.play')
    @patch('helper_functions.audio_processors.sd.wait')
    def test_local_text_to_speech_success(self, mock_wait, mock_play, 
                                         mock_post, tmp_path):
        """Test local TTS with server available"""
        output_path = tmp_path / "output.wav"
        
        mock_response = Mock()
        mock_response.content = b"fake audio data"
        mock_post.return_value = mock_response
        
        with patch('builtins.open', mock_open()):
            audio_processors.local_text_to_speech(
                "Hello world", 
                str(output_path), 
                "speaker1"
            )
        
        mock_post.assert_called_once()
    
    @patch('helper_functions.audio_processors.requests.post')
    def test_local_text_to_speech_server_unavailable(self, mock_post, tmp_path):
        """Test local TTS with server unavailable"""
        output_path = tmp_path / "output.wav"
        mock_post.side_effect = Exception("Connection error")
        
        # Should not raise exception
        try:
            audio_processors.local_text_to_speech(
                "Hello", str(output_path), "speaker1"
            )
        except Exception:
            pytest.fail("local_text_to_speech raised exception")


class TestTranslateText:
    """Tests for translate_text() function"""
    
    def test_translate_text_returns_original(self):
        """Test that translate_text currently returns original text"""
        text = "Hello world"
        result = audio_processors.translate_text(text, "es")
        assert result == text
    
    def test_translate_text_different_languages(self):
        """Test with different target languages"""
        text = "Test text"
        for lang in ["es", "fr", "de", "ja"]:
            result = audio_processors.translate_text(text, lang)
            assert result == text  # Currently returns original


class TestTranscribeAudioToText:
    """Tests for transcribe_audio_to_text() function"""
    
    @patch('helper_functions.audio_processors.transcribe_audio')
    def test_transcribe_audio_to_text_success(self, mock_transcribe):
        """Test successful audio-to-text transcription"""
        mock_transcribe.return_value = ("This is the transcribed text", "en-US")
        
        text, language = audio_processors.transcribe_audio_to_text("audio.wav")
        
        assert text == "This is the transcribed text"
        assert language == "en-US"
    
    @patch('helper_functions.audio_processors.transcribe_audio')
    def test_transcribe_audio_to_text_error(self, mock_transcribe):
        """Test transcription with error"""
        mock_transcribe.side_effect = Exception("Transcription error")
        
        text, language = audio_processors.transcribe_audio_to_text("audio.wav")
        
        # Should return error message
        assert "error" in text.lower() or "sorry" in text.lower()
        assert language == "en-US"
    
    @patch('helper_functions.audio_processors.transcribe_audio')
    def test_transcribe_audio_to_text_updates_conversation(self, mock_transcribe):
        """Test that conversation is updated"""
        mock_transcribe.return_value = ("User message", "en-US")
        
        # Reset conversation
        audio_processors.conversation = [{"role": "system", "content": "System prompt"}]
        
        text, language = audio_processors.transcribe_audio_to_text("audio.wav")
        
        # Check if user message was added to conversation
        assert len(audio_processors.conversation) == 2
        assert audio_processors.conversation[-1]["role"] == "user"


class TestGenerateResponse:
    """Tests for generate_response() function"""
    
    @patch('helper_functions.audio_processors.internet_connected_chatbot')
    def test_generate_response_success(self, mock_chatbot):
        """Test successful response generation"""
        mock_chatbot.return_value = "This is the assistant response"
        
        conversation = [{"role": "user", "content": "Hello"}]
        result = audio_processors.generate_response(
            "Hello", 
            conversation, 
            "GPT4OMINI", 
            1000, 
            0.7
        )
        
        assert result == "This is the assistant response"
        mock_chatbot.assert_called_once()
    
    @patch('helper_functions.audio_processors.internet_connected_chatbot')
    def test_generate_response_error(self, mock_chatbot):
        """Test response generation with error"""
        mock_chatbot.side_effect = Exception("Generation error")
        
        conversation = [{"role": "user", "content": "Hello"}]
        result = audio_processors.generate_response(
            "Hello", conversation, "GPT4OMINI", 1000, 0.7
        )
        
        # Should return error message
        assert "error" in result.lower() or "sorry" in result.lower()
    
    @patch('helper_functions.audio_processors.internet_connected_chatbot')
    def test_generate_response_different_models(self, mock_chatbot):
        """Test with different model names"""
        mock_chatbot.return_value = "Response"
        conversation = [{"role": "user", "content": "Test"}]
        
        for model in ["GPT4OMINI", "GEMINI", "COHERE", "GROQ"]:
            result = audio_processors.generate_response(
                "Test", conversation, model, 1000, 0.7
            )
            assert result == "Response"


class TestTranslateAndSpeak:
    """Tests for translate_and_speak() function"""
    
    @patch('helper_functions.audio_processors.translate_text')
    @patch('helper_functions.audio_processors.text_to_speech')
    def test_translate_and_speak_success(self, mock_tts, mock_translate):
        """Test successful translation and speech"""
        mock_translate.return_value = "Translated text"
        mock_tts.return_value = True
        
        # Should not raise exception
        try:
            audio_processors.translate_and_speak(
                "Assistant reply", 
                "en-US", 
                "output.wav", 
                "GPT4OMINI"
            )
        except Exception as e:
            pytest.fail(f"translate_and_speak raised exception: {e}")
    
    @patch('helper_functions.audio_processors.translate_text')
    @patch('helper_functions.audio_processors.text_to_speech')
    def test_translate_and_speak_error(self, mock_tts, mock_translate):
        """Test translation and speech with error"""
        mock_translate.side_effect = Exception("Translation error")
        
        # Should handle error gracefully
        try:
            audio_processors.translate_and_speak(
                "Reply", "en-US", "output.wav", "GPT4OMINI"
            )
        except Exception:
            pytest.fail("translate_and_speak should handle errors gracefully")
    
    @patch('helper_functions.audio_processors.translate_text')
    @patch('helper_functions.audio_processors.text_to_speech')
    def test_translate_and_speak_different_languages(self, mock_tts, mock_translate):
        """Test with different languages"""
        mock_translate.return_value = "Translated"
        mock_tts.return_value = True
        
        for language in ["en-US", "es-ES", "fr-FR", "de-DE"]:
            audio_processors.translate_and_speak(
                "Test", language, "output.wav", "GPT4OMINI"
            )
            mock_translate.assert_called_with("Test", language)


class TestVoiceMapping:
    """Tests for VOICE_MAP configuration"""
    
    def test_voice_map_exists(self):
        """Test that VOICE_MAP is defined"""
        assert hasattr(audio_processors, 'VOICE_MAP')
        assert isinstance(audio_processors.VOICE_MAP, dict)
    
    def test_voice_map_has_default(self):
        """Test that VOICE_MAP has a DEFAULT key"""
        assert "DEFAULT" in audio_processors.VOICE_MAP
    
    def test_voice_map_model_entries(self):
        """Test that VOICE_MAP has entries for common models"""
        expected_models = ["GEMINI", "GPT4", "GPT4OMINI", "COHERE"]
        for model in expected_models:
            assert model in audio_processors.VOICE_MAP


class TestRivaServiceInitialization:
    """Tests for NVIDIA Riva service initialization"""
    
    def test_riva_available_flag_exists(self):
        """Test that RIVA_AVAILABLE flag is defined"""
        assert hasattr(audio_processors, 'RIVA_AVAILABLE')
        assert isinstance(audio_processors.RIVA_AVAILABLE, bool)
    
    def test_asr_service_exists(self):
        """Test that asr_service is defined"""
        assert hasattr(audio_processors, 'asr_service')
    
    def test_tts_service_exists(self):
        """Test that tts_service is defined"""
        assert hasattr(audio_processors, 'tts_service')

