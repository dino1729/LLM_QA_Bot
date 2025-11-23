"""
Unit tests for helper_functions/nvidia_image_gen.py
Tests NVIDIA NIM image generation using Stable Diffusion 3
"""
import pytest
from unittest.mock import Mock, patch, MagicMock, mock_open
import sys
import os
import base64

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TestDebugPrint:
    """Test debug_print function"""

    @patch('helper_functions.nvidia_image_gen.DEBUG_LOG', True)
    def test_debug_print_enabled(self, capsys):
        """Test debug print when DEBUG_LOG is True"""
        from helper_functions.nvidia_image_gen import debug_print

        debug_print("Test Label", "Test Value")

        captured = capsys.readouterr()
        assert "Test Label" in captured.out
        assert "Test Value" in captured.out

    @patch('helper_functions.nvidia_image_gen.DEBUG_LOG', False)
    def test_debug_print_disabled(self, capsys):
        """Test debug print when DEBUG_LOG is False"""
        from helper_functions.nvidia_image_gen import debug_print

        debug_print("Test Label", "Test Value")

        captured = capsys.readouterr()
        assert captured.out == ""


class TestPromptEnhancerNvidia:
    """Test prompt_enhancer_nvidia function"""

    @patch('helper_functions.nvidia_image_gen.OpenAI')
    def test_prompt_enhancer_nvidia_success(self, mock_openai):
        """Test successful prompt enhancement"""
        from helper_functions.nvidia_image_gen import prompt_enhancer_nvidia

        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Enhanced: A beautiful sunset with vibrant colors"))]
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        result = prompt_enhancer_nvidia("a sunset")

        assert isinstance(result, str)
        assert "Enhanced:" in result
        mock_client.chat.completions.create.assert_called_once()

    @patch('helper_functions.nvidia_image_gen.OpenAI')
    def test_prompt_enhancer_nvidia_error(self, mock_openai):
        """Test prompt enhancement error handling"""
        from helper_functions.nvidia_image_gen import prompt_enhancer_nvidia

        mock_openai.side_effect = Exception("API error")

        result = prompt_enhancer_nvidia("a sunset")

        # Should return original prompt on error
        assert result == "a sunset"

    @patch('helper_functions.nvidia_image_gen.OpenAI')
    def test_prompt_enhancer_nvidia_empty_prompt(self, mock_openai):
        """Test with empty prompt"""
        from helper_functions.nvidia_image_gen import prompt_enhancer_nvidia

        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="A random creative scene"))]
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        result = prompt_enhancer_nvidia("")

        assert isinstance(result, str)

    @patch('helper_functions.nvidia_image_gen.OpenAI')
    def test_prompt_enhancer_nvidia_enhancement_validation(self, mock_openai):
        """Test enhancement validation"""
        from helper_functions.nvidia_image_gen import prompt_enhancer_nvidia

        mock_client = Mock()
        mock_response = Mock()
        enhanced = "A stunning photograph of a sunset over the ocean, with warm orange and pink hues"
        mock_response.choices = [Mock(message=Mock(content=enhanced))]
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        result = prompt_enhancer_nvidia("sunset")

        assert len(result) > len("sunset")


class TestGenerateSurprisePromptNvidia:
    """Test generate_surprise_prompt_nvidia function"""

    @patch('helper_functions.nvidia_image_gen.OpenAI')
    def test_generate_surprise_prompt_success(self, mock_openai):
        """Test successful surprise prompt generation"""
        from helper_functions.nvidia_image_gen import generate_surprise_prompt_nvidia

        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="A whimsical underwater city with bioluminescent creatures"))]
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        result = generate_surprise_prompt_nvidia()

        assert isinstance(result, str)
        assert len(result) > 0
        mock_client.chat.completions.create.assert_called_once()

    @patch('helper_functions.nvidia_image_gen.OpenAI')
    def test_generate_surprise_prompt_error(self, mock_openai):
        """Test surprise prompt error handling"""
        from helper_functions.nvidia_image_gen import generate_surprise_prompt_nvidia

        mock_openai.side_effect = Exception("API error")

        result = generate_surprise_prompt_nvidia()

        # Should return fallback prompt
        assert isinstance(result, str)
        assert "surprise" in result.lower() or "creative" in result.lower() or len(result) > 0

    @patch('helper_functions.nvidia_image_gen.OpenAI')
    def test_generate_surprise_prompt_temperature(self, mock_openai):
        """Test that temperature is set to 1.0 for creativity"""
        from helper_functions.nvidia_image_gen import generate_surprise_prompt_nvidia

        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Surprise prompt"))]
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        generate_surprise_prompt_nvidia()

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs['temperature'] == 1.0


class TestRunGenerateNvidia:
    """Test run_generate_nvidia function"""

    @patch('helper_functions.nvidia_image_gen.requests.post')
    @patch('helper_functions.nvidia_image_gen.prompt_enhancer_nvidia')
    @patch('builtins.open', new_callable=mock_open)
    def test_run_generate_nvidia_success(self, mock_file, mock_enhancer, mock_post):
        """Test successful image generation"""
        from helper_functions.nvidia_image_gen import run_generate_nvidia

        mock_enhancer.return_value = "Enhanced prompt"

        # Mock successful API response
        fake_image_data = base64.b64encode(b"fake_image_data").decode()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "artifacts": [{"base64": fake_image_data}]
        }
        mock_post.return_value = mock_response

        result = run_generate_nvidia("a sunset", "1024x1024")

        assert isinstance(result, str)
        assert result.endswith(".png")
        mock_post.assert_called_once()
        mock_file.assert_called_once()

    @patch('helper_functions.nvidia_image_gen.requests.post')
    @patch('helper_functions.nvidia_image_gen.generate_surprise_prompt_nvidia')
    def test_run_generate_nvidia_default_prompt(self, mock_surprise, mock_post):
        """Test with default prompt"""
        from helper_functions.nvidia_image_gen import run_generate_nvidia

        mock_surprise.return_value = "Surprise prompt"

        fake_image_data = base64.b64encode(b"fake_image_data").decode()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "artifacts": [{"base64": fake_image_data}]
        }
        mock_post.return_value = mock_response

        with patch('builtins.open', mock_open()):
            result = run_generate_nvidia(None, "1024x1024")

        mock_surprise.assert_called_once()
        assert isinstance(result, str)

    @patch('helper_functions.nvidia_image_gen.requests.post')
    def test_run_generate_nvidia_api_error(self, mock_post):
        """Test API error handling"""
        from helper_functions.nvidia_image_gen import run_generate_nvidia

        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal server error"
        mock_post.return_value = mock_response

        # Should handle error gracefully
        try:
            result = run_generate_nvidia("test", "1024x1024")
            assert result is None or isinstance(result, str)
        except Exception as e:
            # Exception is acceptable
            assert "500" in str(e) or "error" in str(e).lower()

    @patch('helper_functions.nvidia_image_gen.requests.post')
    @patch('helper_functions.nvidia_image_gen.prompt_enhancer_nvidia')
    @patch('builtins.open', new_callable=mock_open)
    def test_run_generate_nvidia_response_format_variations(self, mock_file, mock_enhancer, mock_post):
        """Test different response format variations"""
        from helper_functions.nvidia_image_gen import run_generate_nvidia

        mock_enhancer.return_value = "Enhanced"

        # Test different response formats
        fake_image_data = base64.b64encode(b"fake_image_data").decode()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "artifacts": [{"base64": fake_image_data, "finishReason": "SUCCESS"}]
        }
        mock_post.return_value = mock_response

        result = run_generate_nvidia("test", "1024x1024")

        assert isinstance(result, str)


class TestRunEditNvidia:
    """Test run_edit_nvidia function"""

    @patch('helper_functions.nvidia_image_gen.run_generate_nvidia')
    @patch('helper_functions.nvidia_image_gen.prompt_enhancer_nvidia')
    @patch('helper_functions.nvidia_image_gen.Image')
    def test_run_edit_nvidia_success(self, mock_image_class, mock_enhancer, mock_generate):
        """Test successful image editing"""
        from helper_functions.nvidia_image_gen import run_edit_nvidia

        mock_enhancer.return_value = "Enhanced edit prompt"
        mock_generate.return_value = "/path/to/edited_image.png"

        # Mock PIL Image
        mock_img = Mock()
        mock_image_class.open.return_value = mock_img

        result = run_edit_nvidia("/path/to/input.png", "make it blue", "1024x1024")

        assert result == "/path/to/edited_image.png"
        mock_enhancer.assert_called_once()
        mock_generate.assert_called_once()

    @patch('helper_functions.nvidia_image_gen.run_generate_nvidia')
    @patch('helper_functions.nvidia_image_gen.generate_surprise_prompt_nvidia')
    @patch('helper_functions.nvidia_image_gen.Image')
    def test_run_edit_nvidia_default_prompt(self, mock_image_class, mock_surprise, mock_generate):
        """Test editing with default prompt"""
        from helper_functions.nvidia_image_gen import run_edit_nvidia

        mock_surprise.return_value = "Surprise edit"
        mock_generate.return_value = "/path/to/edited.png"

        mock_img = Mock()
        mock_image_class.open.return_value = mock_img

        result = run_edit_nvidia("/path/to/input.png", None, "1024x1024")

        mock_surprise.assert_called_once()
        assert isinstance(result, str)

    @patch('helper_functions.nvidia_image_gen.prompt_enhancer_nvidia')
    def test_run_edit_nvidia_prompt_enhancement(self, mock_enhancer):
        """Test that prompt is enhanced for editing"""
        from helper_functions.nvidia_image_gen import run_edit_nvidia

        mock_enhancer.return_value = "Enhanced"

        with patch('helper_functions.nvidia_image_gen.run_generate_nvidia') as mock_gen:
            with patch('helper_functions.nvidia_image_gen.Image'):
                mock_gen.return_value = "/path/to/output.png"

                run_edit_nvidia("/path/to/input.png", "add colors", "1024x1024")

        # Verify prompt was enhanced
        mock_enhancer.assert_called_once()
        call_args = mock_enhancer.call_args[0][0]
        assert "add colors" in call_args.lower()

    @patch('helper_functions.nvidia_image_gen.Image')
    def test_run_edit_nvidia_error_handling(self, mock_image_class):
        """Test error handling in image editing"""
        from helper_functions.nvidia_image_gen import run_edit_nvidia

        mock_image_class.open.side_effect = Exception("Cannot open image")

        # Should handle error
        try:
            result = run_edit_nvidia("/invalid/path.png", "test", "1024x1024")
            assert result is None or isinstance(result, str)
        except Exception as e:
            assert "Cannot open image" in str(e) or "error" in str(e).lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
