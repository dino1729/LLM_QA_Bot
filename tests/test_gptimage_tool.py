import pytest
import os
import sys
import base64
from unittest.mock import Mock, patch, mock_open
from helper_functions import gptimage_tool

class TestMask:
    """Tests for _mask() function"""
    
    def test_mask_long_string(self):
        """Test masking long string"""
        from helper_functions.gptimage_tool import _mask
        
        result = _mask("1234567890", show=2)
        assert result.startswith("12")
        assert result.endswith("90")
        assert "*" in result

    def test_mask_short_string(self):
        """Test masking short string"""
        from helper_functions.gptimage_tool import _mask
        
        result = _mask("123", show=2)
        assert result == "***" # Logic: len <= show*2 -> return * * len

    def test_mask_empty_string(self):
        """Test masking empty string"""
        from helper_functions.gptimage_tool import _mask
        
        result = _mask("", show=4)
        
        assert result == "None"

    def test_mask_none(self):
        """Test masking None"""
        from helper_functions.gptimage_tool import _mask
        
        result = _mask(None, show=4)
        
        assert result == "None"

class TestDebugPrint:
    """Tests for debug_print() function"""
    
    @patch('builtins.print')
    def test_debug_print_enabled(self, mock_print):
        """Test debug_print when enabled"""
        with patch('helper_functions.gptimage_tool.DEBUG_LOG', True):
            gptimage_tool.debug_print("Test", "Value")
            mock_print.assert_called()

    @patch('builtins.print')
    def test_debug_print_disabled(self, mock_print):
        """Test debug_print when disabled"""
        with patch('helper_functions.gptimage_tool.DEBUG_LOG', False):
            gptimage_tool.debug_print("Test", "Value")
            mock_print.assert_not_called()

    @patch('builtins.print')
    def test_debug_print_with_dict(self, mock_print):
        """Test debug_print with dictionary"""
        with patch('helper_functions.gptimage_tool.DEBUG_LOG', True):
            gptimage_tool.debug_print("Test", {"key": "value"}, pretty=True)
            mock_print.assert_called()

class TestEnsurePng:
    """Tests for _ensure_png() function"""
    
    def test_ensure_png_already_png(self):
        """Test when file is already PNG"""
        from helper_functions.gptimage_tool import _ensure_png
        
        result = _ensure_png("image.png")
        assert result == "image.png"

    @patch('helper_functions.gptimage_tool.Image.open')
    @patch('helper_functions.gptimage_tool.tempfile.mktemp')
    def test_ensure_png_jpg_conversion(self, mock_mktemp, mock_open):
        """Test conversion of JPG to PNG"""
        from helper_functions.gptimage_tool import _ensure_png
        
        mock_mktemp.return_value = "temp.png"
        mock_img = Mock()
        mock_open.return_value = mock_img
        
        result = _ensure_png("image.jpg")
        
        assert result == "temp.png"
        mock_img.save.assert_called_with("temp.png", format="PNG")

    @patch('helper_functions.gptimage_tool.Image.open')
    @patch('helper_functions.gptimage_tool.tempfile.mktemp')
    def test_ensure_png_other_formats(self, mock_mktemp, mock_open):
        """Test conversion of other formats"""
        from helper_functions.gptimage_tool import _ensure_png
        
        mock_mktemp.return_value = "temp.png"
        mock_img = Mock()
        mock_open.return_value = mock_img
        
        result = _ensure_png("image.webp")
        
        assert result == "temp.png"
        mock_img.save.assert_called_with("temp.png", format="PNG")

class TestPromptEnhancer:
    """Tests for prompt_enhancer() function"""
    
    def test_prompt_enhancer_success(self):
        """Test successful enhancement"""
        from helper_functions.gptimage_tool import prompt_enhancer
        
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Enhanced prompt"
        mock_client.chat.completions.create.return_value = mock_response
        
        result = prompt_enhancer("Basic prompt", mock_client)
        assert result == "Enhanced prompt"

    def test_prompt_enhancer_error_fallback(self):
        """Test fallback on error"""
        from helper_functions.gptimage_tool import prompt_enhancer
        
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        
        result = prompt_enhancer("Basic prompt", mock_client)
        assert result == "Basic prompt"

    def test_prompt_enhancer_empty_prompt(self):
        """Test when enhancer returns empty string"""
        from helper_functions.gptimage_tool import prompt_enhancer
        
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = ""
        mock_client.chat.completions.create.return_value = mock_response
        
        result = prompt_enhancer("Basic prompt", mock_client)
        assert result == "Basic prompt"

class TestGenerateSurprisePrompt:
    """Tests for generate_surprise_prompt() function"""
    
    def test_generate_surprise_prompt_success(self):
        """Test successful generation"""
        from helper_functions.gptimage_tool import generate_surprise_prompt
        
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Surprise!"
        mock_client.chat.completions.create.return_value = mock_response
        
        result = generate_surprise_prompt(mock_client)
        assert result == "Surprise!"

    def test_generate_surprise_prompt_temperature(self):
        """Test that high temperature is used"""
        from helper_functions.gptimage_tool import generate_surprise_prompt
        
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Surprise!"
        mock_client.chat.completions.create.return_value = mock_response
        
        generate_surprise_prompt(mock_client)
        
        _, kwargs = mock_client.chat.completions.create.call_args
        assert kwargs["temperature"] == 1.0

class TestRunGenerate:
    """Tests for run_generate() function"""
    
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
        mock_client.images.generate.assert_called_once()

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
        
        # Should use default prompt from code, not surprise prompt logic (which was removed/changed in recent version logic maybe?)
        # Wait, implementation uses a hardcoded default string if prompt is None
        # `prompt = "A futuristic cityscape at sunset, synthwave style"`
        _, kwargs = mock_client.images.generate.call_args
        assert "futuristic cityscape" in kwargs["prompt"]

    @patch('helper_functions.gptimage_tool.OpenAI')
    def test_run_generate_error_handling(self, mock_openai):
        """Test error handling"""
        from helper_functions.gptimage_tool import run_generate
        
        mock_client = Mock()
        mock_client.images.generate.side_effect = Exception("API Error")
        mock_openai.return_value = mock_client
        
        with pytest.raises(Exception):
            run_generate("test", "1024x1024")

class TestRunEdit:
    """Tests for run_edit() function"""
    
    @patch('helper_functions.gptimage_tool.subprocess.run')
    @patch('helper_functions.gptimage_tool.spawn_funny_thread')
    @patch('builtins.open', new_callable=mock_open)
    def test_run_edit_success(self, mock_file, mock_thread, mock_run):
        """Test successful image edit"""
        from helper_functions.gptimage_tool import run_edit
        
        # Mock env vars
        with patch.dict(os.environ, {
            "AZURE_OPENAI_API_KEY": "test",
            "AZURE_OPENAI_ENDPOINT": "test",
            "AZURE_OPENAI_DEPLOYMENT": "test"
        }):
            mock_proc = Mock()
            fake_data = base64.b64encode(b"image").decode()
            mock_proc.stdout = json.dumps({"data": [{"b64_json": fake_data}]})
            mock_run.return_value = mock_proc
            
            result = run_edit("image.png", "Edit this", "1024x1024")
            
            assert isinstance(result, str)
            mock_run.assert_called_once()

    @patch('helper_functions.gptimage_tool.subprocess.run')
    @patch('helper_functions.gptimage_tool.spawn_funny_thread')
    @patch('builtins.open', new_callable=mock_open)
    def test_run_edit_default_prompt(self, mock_file, mock_thread, mock_run):
        """Test default prompt in edit"""
        from helper_functions.gptimage_tool import run_edit
        
        with patch.dict(os.environ, {
            "AZURE_OPENAI_API_KEY": "test",
            "AZURE_OPENAI_ENDPOINT": "test",
            "AZURE_OPENAI_DEPLOYMENT": "test"
        }):
            mock_proc = Mock()
            fake_data = base64.b64encode(b"image").decode()
            mock_proc.stdout = json.dumps({"data": [{"b64_json": fake_data}]})
            mock_run.return_value = mock_proc
            
            run_edit("image.png", None, "1024x1024")
            
            args, _ = mock_run.call_args
            cmd = args[0]
            # Check prompt in curl command
            prompt_arg = next(arg for arg in cmd if arg.startswith("-F") and "prompt=" in arg)
            assert "cyberpunk" in prompt_arg

class TestUnifiedInterfaces:
    """Tests for unified interface functions"""
    
    @patch('helper_functions.gptimage_tool.run_generate')
    @patch('helper_functions.nvidia_image_gen.run_generate_nvidia')
    def test_run_generate_unified_openai(self, mock_nvidia, mock_openai):
        """Test OpenAI dispatch"""
        from helper_functions.gptimage_tool import run_generate_unified
        
        run_generate_unified("prompt", "1024x1024", "openai")
        mock_openai.assert_called_once()
        mock_nvidia.assert_not_called()

    @patch('helper_functions.gptimage_tool.run_generate')
    @patch('helper_functions.nvidia_image_gen.run_generate_nvidia')
    def test_run_generate_unified_nvidia(self, mock_nvidia, mock_openai):
        """Test NVIDIA dispatch"""
        from helper_functions.gptimage_tool import run_generate_unified
        
        run_generate_unified("prompt", "1024x1024", "nvidia")
        mock_nvidia.assert_called_once()
        mock_openai.assert_not_called()

    @patch('helper_functions.gptimage_tool.run_edit')
    @patch('helper_functions.nvidia_image_gen.run_edit_nvidia')
    def test_run_edit_unified_openai(self, mock_nvidia, mock_openai):
        """Test OpenAI dispatch"""
        from helper_functions.gptimage_tool import run_edit_unified
        
        run_edit_unified("image.png", "prompt", "1024x1024", "openai")
        mock_openai.assert_called_once()
        mock_nvidia.assert_not_called()

    @patch('helper_functions.gptimage_tool.run_edit')
    @patch('helper_functions.nvidia_image_gen.run_edit_nvidia')
    def test_run_edit_unified_nvidia(self, mock_nvidia, mock_openai):
        """Test NVIDIA dispatch"""
        from helper_functions.gptimage_tool import run_edit_unified
        
        run_edit_unified("image.png", "prompt", "1024x1024", "nvidia")
        mock_nvidia.assert_called_once()
        mock_openai.assert_not_called()

    @patch('helper_functions.gptimage_tool.prompt_enhancer')
    @patch('helper_functions.nvidia_image_gen.prompt_enhancer_nvidia')
    @patch('helper_functions.gptimage_tool.OpenAI')
    def test_prompt_enhancer_unified_openai(self, mock_client, mock_nvidia, mock_openai):
        """Test OpenAI dispatch"""
        from helper_functions.gptimage_tool import prompt_enhancer_unified
        
        prompt_enhancer_unified("prompt", "openai")
        mock_openai.assert_called_once()
        mock_nvidia.assert_not_called()

    @patch('helper_functions.gptimage_tool.prompt_enhancer')
    @patch('helper_functions.nvidia_image_gen.prompt_enhancer_nvidia')
    def test_prompt_enhancer_unified_nvidia(self, mock_nvidia, mock_openai):
        """Test NVIDIA dispatch"""
        from helper_functions.gptimage_tool import prompt_enhancer_unified
        
        prompt_enhancer_unified("prompt", "nvidia")
        mock_nvidia.assert_called_once()
        mock_openai.assert_not_called()

    @patch('helper_functions.gptimage_tool.generate_surprise_prompt')
    @patch('helper_functions.nvidia_image_gen.generate_surprise_prompt_nvidia')
    @patch('helper_functions.gptimage_tool.OpenAI')
    def test_generate_surprise_prompt_unified_openai(self, mock_client, mock_nvidia, mock_openai):
        """Test OpenAI dispatch"""
        from helper_functions.gptimage_tool import generate_surprise_prompt_unified
        
        generate_surprise_prompt_unified("openai")
        mock_openai.assert_called_once()
        mock_nvidia.assert_not_called()

    @patch('helper_functions.gptimage_tool.generate_surprise_prompt')
    @patch('helper_functions.nvidia_image_gen.generate_surprise_prompt_nvidia')
    def test_generate_surprise_prompt_unified_nvidia(self, mock_nvidia, mock_openai):
        """Test NVIDIA dispatch"""
        from helper_functions.gptimage_tool import generate_surprise_prompt_unified
        
        generate_surprise_prompt_unified("nvidia")
        mock_nvidia.assert_called_once()
        mock_openai.assert_not_called()

class TestFunnyThread:
    """Tests for funny thread functionality"""
    
    @patch('helper_functions.gptimage_tool.threading.Thread')
    def test_spawn_funny_thread(self, mock_thread):
        """Test thread spawning"""
        from helper_functions.gptimage_tool import spawn_funny_thread
        
        spawn_funny_thread("generate", "prompt", client=Mock())
        mock_thread.assert_called_once()
        
    def test_pop_funny_messages_empty(self):
        """Test empty queue"""
        from helper_functions.gptimage_tool import pop_funny_messages, message_queue
        
        # clear queue
        while not message_queue.empty():
            message_queue.get()
            
        result = pop_funny_messages()
        assert result == []

    def test_pop_funny_messages_with_messages(self):
        """Test with messages"""
        from helper_functions.gptimage_tool import pop_funny_messages, message_queue
        
        message_queue.put(("User", "Msg"))
        
        result = pop_funny_messages()
        assert result == [("User", "Msg")]
