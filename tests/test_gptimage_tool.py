"""
Unit tests for helper_functions/gptimage_tool.py
Tests unified image generation and editing tool with OpenAI
"""
import pytest
from unittest.mock import Mock, patch, MagicMock, mock_open
import sys
import os
import base64

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TestMask:
    """Test _mask utility function"""

    def test_mask_long_string(self):
        """Test masking long API keys"""
        from helper_functions.gptimage_tool import _mask

        result = _mask("sk-1234567890abcdef", show=4)

        assert result.startswith("sk-1")
        assert "*" in result
        assert len(result) == len("sk-1234567890abcdef")

    def test_mask_short_string(self):
        """Test masking short strings"""
        from helper_functions.gptimage_tool import _mask

        result = _mask("abc", show=4)

        # Short string handling
        assert isinstance(result, str)

    def test_mask_empty_string(self):
        """Test masking empty string"""
        from helper_functions.gptimage_tool import _mask

        result = _mask("", show=4)

        assert result == ""

    def test_mask_none(self):
        """Test masking None"""
        from helper_functions.gptimage_tool import _mask

        try:
            result = _mask(None, show=4)
            # Should handle None gracefully or raise TypeError
            assert result is None or isinstance(result, str)
        except (TypeError, AttributeError):
            # Expected for None input
            pass


class TestDebugPrint:
    """Test debug_print function"""

    @patch('helper_functions.gptimage_tool.DEBUG_LOG', True)
    def test_debug_print_enabled(self, capsys):
        """Test debug print when enabled"""
        from helper_functions.gptimage_tool import debug_print

        debug_print("Test", "Value")

        captured = capsys.readouterr()
        assert "Test" in captured.out

    @patch('helper_functions.gptimage_tool.DEBUG_LOG', False)
    def test_debug_print_disabled(self, capsys):
        """Test debug print when disabled"""
        from helper_functions.gptimage_tool import debug_print

        debug_print("Test", "Value")

        captured = capsys.readouterr()
        assert captured.out == ""

    @patch('helper_functions.gptimage_tool.DEBUG_LOG', True)
    def test_debug_print_with_dict(self, capsys):
        """Test debug print with dictionary (pretty print)"""
        from helper_functions.gptimage_tool import debug_print

        debug_print("Dict", {"key": "value"}, pretty=True)

        captured = capsys.readouterr()
        assert "Dict" in captured.out


class TestEnsurePng:
    """Test _ensure_png function"""

    @patch('helper_functions.gptimage_tool.Image')
    def test_ensure_png_already_png(self, mock_image):
        """Test with PNG file (return as-is)"""
        from helper_functions.gptimage_tool import _ensure_png

        result = _ensure_png("/path/to/image.png")

        assert result == "/path/to/image.png"
        mock_image.open.assert_not_called()

    @patch('helper_functions.gptimage_tool.Image')
    def test_ensure_png_jpg_conversion(self, mock_image):
        """Test JPG to PNG conversion"""
        from helper_functions.gptimage_tool import _ensure_png

        mock_img = Mock()
        mock_image.open.return_value = mock_img

        result = _ensure_png("/path/to/image.jpg")

        assert result.endswith(".png")
        mock_image.open.assert_called_once()
        mock_img.save.assert_called_once()

    @patch('helper_functions.gptimage_tool.Image')
    def test_ensure_png_other_formats(self, mock_image):
        """Test other format conversions"""
        from helper_functions.gptimage_tool import _ensure_png

        mock_img = Mock()
        mock_image.open.return_value = mock_img

        for ext in [".jpeg", ".bmp", ".gif"]:
            result = _ensure_png(f"/path/to/image{ext}")
            assert result.endswith(".png")


class TestPromptEnhancer:
    """Test prompt_enhancer function"""

    @patch('helper_functions.gptimage_tool.OpenAI')
    def test_prompt_enhancer_success(self, mock_openai):
        """Test successful prompt enhancement"""
        from helper_functions.gptimage_tool import prompt_enhancer

        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Enhanced prompt with cinematic details"))]
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        result = prompt_enhancer("a sunset", mock_client)

        assert isinstance(result, str)
        assert len(result) > len("a sunset")

    @patch('helper_functions.gptimage_tool.OpenAI')
    def test_prompt_enhancer_error_fallback(self, mock_openai):
        """Test fallback to original prompt on error"""
        from helper_functions.gptimage_tool import prompt_enhancer

        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("API error")

        result = prompt_enhancer("original prompt", mock_client)

        assert result == "original prompt"

    @patch('helper_functions.gptimage_tool.OpenAI')
    def test_prompt_enhancer_empty_prompt(self, mock_openai):
        """Test with empty prompt"""
        from helper_functions.gptimage_tool import prompt_enhancer

        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="A creative scene"))]
        mock_client.chat.completions.create.return_value = mock_response

        result = prompt_enhancer("", mock_client)

        assert isinstance(result, str)


class TestGenerateSurprisePrompt:
    """Test generate_surprise_prompt function"""

    @patch('helper_functions.gptimage_tool.OpenAI')
    def test_generate_surprise_prompt_success(self, mock_openai):
        """Test successful surprise prompt generation"""
        from helper_functions.gptimage_tool import generate_surprise_prompt

        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="A whimsical underwater city"))]
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        result = generate_surprise_prompt(mock_client)

        assert isinstance(result, str)
        assert len(result) > 0

    @patch('helper_functions.gptimage_tool.OpenAI')
    def test_generate_surprise_prompt_temperature(self, mock_openai):
        """Test that temperature is high for creativity"""
        from helper_functions.gptimage_tool import generate_surprise_prompt

        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Surprise"))]
        mock_client.chat.completions.create.return_value = mock_response

        generate_surprise_prompt(mock_client)

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs['temperature'] == 1.0


class TestRunGenerate:
    """Test run_generate function"""

    @patch('helper_functions.gptimage_tool.OpenAI')
    @patch('helper_functions.gptimage_tool.prompt_enhancer')
    @patch('builtins.open', new_callable=mock_open)
    def test_run_generate_success(self, mock_file, mock_enhancer, mock_openai):
        """Test successful image generation"""
        from helper_functions.gptimage_tool import run_generate

        mock_enhancer.return_value = "Enhanced prompt"

        fake_image_data = base64.b64encode(b"fake_image_data").decode()
        mock_client = Mock()
        mock_response = Mock()
        mock_response.data = [Mock(b64_json=fake_image_data)]
        mock_client.images.generate.return_value = mock_response
        mock_openai.return_value = mock_client

        result = run_generate("a sunset", "1024x1024")

        assert isinstance(result, str)
        assert result.endswith(".png")
        mock_file.assert_called_once()

    @patch('helper_functions.gptimage_tool.OpenAI')
    @patch('helper_functions.gptimage_tool.generate_surprise_prompt')
    @patch('builtins.open', new_callable=mock_open)
    def test_run_generate_default_prompt(self, mock_file, mock_surprise, mock_openai):
        """Test with default prompt"""
        from helper_functions.gptimage_tool import run_generate

        mock_surprise.return_value = "Surprise prompt"

        fake_image_data = base64.b64encode(b"fake_image").decode()
        mock_client = Mock()
        mock_response = Mock()
        mock_response.data = [Mock(b64_json=fake_image_data)]
        mock_client.images.generate.return_value = mock_response
        mock_openai.return_value = mock_client

        result = run_generate(None, "1024x1024")

        mock_surprise.assert_called_once()
        assert isinstance(result, str)

    @patch('helper_functions.gptimage_tool.OpenAI')
    def test_run_generate_error_handling(self, mock_openai):
        """Test error handling"""
        from helper_functions.gptimage_tool import run_generate

        mock_client = Mock()
        mock_client.images.generate.side_effect = Exception("API error")
        mock_openai.return_value = mock_client

        try:
            result = run_generate("test", "1024x1024")
            assert result is None or isinstance(result, str)
        except Exception as e:
            assert "error" in str(e).lower()


class TestRunEdit:
    """Test run_edit function"""

    @patch('helper_functions.gptimage_tool._ensure_png')
    @patch('helper_functions.gptimage_tool.prompt_enhancer')
    @patch('helper_functions.gptimage_tool.subprocess.run')
    @patch('builtins.open', new_callable=mock_open)
    def test_run_edit_success(self, mock_file, mock_subprocess, mock_enhancer, mock_ensure_png):
        """Test successful image editing"""
        from helper_functions.gptimage_tool import run_edit

        mock_ensure_png.return_value = "/path/to/image.png"
        mock_enhancer.return_value = "Enhanced edit prompt"

        # Mock subprocess (curl command)
        mock_result = Mock()
        fake_image_data = base64.b64encode(b"edited_image").decode()
        mock_result.stdout = f'{{"data": [{{"b64_json": "{fake_image_data}"}}]}}'
        mock_subprocess.return_value = mock_result

        result = run_edit("/path/to/input.png", "make it blue", "1024x1024")

        assert isinstance(result, str)
        mock_ensure_png.assert_called_once()

    @patch('helper_functions.gptimage_tool._ensure_png')
    @patch('helper_functions.gptimage_tool.generate_surprise_prompt')
    def test_run_edit_default_prompt(self, mock_surprise, mock_ensure_png):
        """Test editing with default prompt"""
        from helper_functions.gptimage_tool import run_edit

        mock_ensure_png.return_value = "/path/to/image.png"
        mock_surprise.return_value = "Surprise edit"

        with patch('helper_functions.gptimage_tool.subprocess.run') as mock_subprocess:
            fake_image = base64.b64encode(b"img").decode()
            mock_result = Mock()
            mock_result.stdout = f'{{"data": [{{"b64_json": "{fake_image}"}}]}}'
            mock_subprocess.return_value = mock_result

            with patch('builtins.open', mock_open()):
                result = run_edit("/path/to/input.png", None, "1024x1024")

        mock_surprise.assert_called_once()


class TestUnifiedInterfaces:
    """Test unified interface functions"""

    @patch('helper_functions.gptimage_tool.run_generate')
    @patch('helper_functions.gptimage_tool.run_generate_nvidia')
    def test_run_generate_unified_openai(self, mock_nvidia, mock_openai):
        """Test unified generate with OpenAI provider"""
        from helper_functions.gptimage_tool import run_generate_unified

        mock_openai.return_value = "/path/to/image.png"

        result = run_generate_unified("test prompt", "1024x1024", "openai")

        assert result == "/path/to/image.png"
        mock_openai.assert_called_once()
        mock_nvidia.assert_not_called()

    @patch('helper_functions.gptimage_tool.run_generate')
    @patch('helper_functions.gptimage_tool.run_generate_nvidia')
    def test_run_generate_unified_nvidia(self, mock_nvidia, mock_openai):
        """Test unified generate with NVIDIA provider"""
        from helper_functions.gptimage_tool import run_generate_unified

        mock_nvidia.return_value = "/path/to/image.png"

        result = run_generate_unified("test prompt", "1024x1024", "nvidia")

        assert result == "/path/to/image.png"
        mock_nvidia.assert_called_once()
        mock_openai.assert_not_called()

    @patch('helper_functions.gptimage_tool.run_edit')
    @patch('helper_functions.gptimage_tool.run_edit_nvidia')
    def test_run_edit_unified_openai(self, mock_nvidia, mock_openai):
        """Test unified edit with OpenAI provider"""
        from helper_functions.gptimage_tool import run_edit_unified

        mock_openai.return_value = "/path/to/edited.png"

        result = run_edit_unified("/input.png", "edit prompt", "1024x1024", "openai")

        assert result == "/path/to/edited.png"
        mock_openai.assert_called_once()
        mock_nvidia.assert_not_called()

    @patch('helper_functions.gptimage_tool.run_edit')
    @patch('helper_functions.gptimage_tool.run_edit_nvidia')
    def test_run_edit_unified_nvidia(self, mock_nvidia, mock_openai):
        """Test unified edit with NVIDIA provider"""
        from helper_functions.gptimage_tool import run_edit_unified

        mock_nvidia.return_value = "/path/to/edited.png"

        result = run_edit_unified("/input.png", "edit prompt", "1024x1024", "nvidia")

        assert result == "/path/to/edited.png"
        mock_nvidia.assert_called_once()
        mock_openai.assert_not_called()

    @patch('helper_functions.gptimage_tool.prompt_enhancer')
    @patch('helper_functions.gptimage_tool.prompt_enhancer_nvidia')
    def test_prompt_enhancer_unified_openai(self, mock_nvidia, mock_openai):
        """Test unified prompt enhancer with OpenAI"""
        from helper_functions.gptimage_tool import prompt_enhancer_unified

        mock_openai.return_value = "Enhanced by OpenAI"

        result = prompt_enhancer_unified("test", "openai")

        assert result == "Enhanced by OpenAI"
        mock_openai.assert_called_once()

    @patch('helper_functions.gptimage_tool.prompt_enhancer')
    @patch('helper_functions.gptimage_tool.prompt_enhancer_nvidia')
    def test_prompt_enhancer_unified_nvidia(self, mock_nvidia, mock_openai):
        """Test unified prompt enhancer with NVIDIA"""
        from helper_functions.gptimage_tool import prompt_enhancer_unified

        mock_nvidia.return_value = "Enhanced by NVIDIA"

        result = prompt_enhancer_unified("test", "nvidia")

        assert result == "Enhanced by NVIDIA"
        mock_nvidia.assert_called_once()

    @patch('helper_functions.gptimage_tool.generate_surprise_prompt')
    @patch('helper_functions.gptimage_tool.generate_surprise_prompt_nvidia')
    def test_generate_surprise_prompt_unified_openai(self, mock_nvidia, mock_openai):
        """Test unified surprise prompt with OpenAI"""
        from helper_functions.gptimage_tool import generate_surprise_prompt_unified

        mock_openai.return_value = "OpenAI surprise"

        result = generate_surprise_prompt_unified("openai")

        assert result == "OpenAI surprise"
        mock_openai.assert_called_once()

    @patch('helper_functions.gptimage_tool.generate_surprise_prompt')
    @patch('helper_functions.gptimage_tool.generate_surprise_prompt_nvidia')
    def test_generate_surprise_prompt_unified_nvidia(self, mock_nvidia, mock_openai):
        """Test unified surprise prompt with NVIDIA"""
        from helper_functions.gptimage_tool import generate_surprise_prompt_unified

        mock_nvidia.return_value = "NVIDIA surprise"

        result = generate_surprise_prompt_unified("nvidia")

        assert result == "NVIDIA surprise"
        mock_nvidia.assert_called_once()


class TestFunnyThread:
    """Test funny_thread functions"""

    @patch('helper_functions.gptimage_tool.threading.Thread')
    def test_spawn_funny_thread(self, mock_thread):
        """Test spawning funny thread"""
        from helper_functions.gptimage_tool import spawn_funny_thread

        mock_thread_instance = Mock()
        mock_thread.return_value = mock_thread_instance

        result = spawn_funny_thread("generate", "test prompt", None, None, None, None)

        # Should return a stop event
        assert hasattr(result, 'set') or result is not None
        mock_thread.assert_called_once()

    def test_pop_funny_messages_empty(self):
        """Test popping funny messages from empty queue"""
        from helper_functions.gptimage_tool import pop_funny_messages

        result = pop_funny_messages()

        assert isinstance(result, list)
        assert len(result) == 0

    @patch('helper_functions.gptimage_tool.funny_queue')
    def test_pop_funny_messages_with_messages(self, mock_queue):
        """Test popping funny messages"""
        from helper_functions.gptimage_tool import pop_funny_messages
        import queue

        # Mock queue with messages
        mock_queue.empty.side_effect = [False, False, True]
        mock_queue.get_nowait.side_effect = [
            ("Speaker1", "Message1"),
            ("Speaker2", "Message2")
        ]

        result = pop_funny_messages()

        assert len(result) == 2
        assert result[0] == ("Speaker1", "Message1")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
