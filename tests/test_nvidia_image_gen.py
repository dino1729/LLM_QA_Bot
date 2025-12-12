import pytest
import os
import base64
from unittest.mock import Mock, patch, mock_open
from helper_functions import nvidia_image_gen

class TestDebugPrint:
    """Tests for debug_print function"""
    
    @patch('builtins.print')
    def test_debug_print_enabled(self, mock_print):
        """Test debug_print when enabled"""
        # Enable debug log
        with patch('helper_functions.nvidia_image_gen.DEBUG_LOG', True):
            nvidia_image_gen.debug_print("Label", "Value")
            mock_print.assert_called_with("[DEBUG NVIDIA] Label: Value")

    @patch('builtins.print')
    def test_debug_print_disabled(self, mock_print):
        """Test debug_print when disabled"""
        # Disable debug log
        with patch('helper_functions.nvidia_image_gen.DEBUG_LOG', False):
            nvidia_image_gen.debug_print("Label", "Value")
            mock_print.assert_not_called()

class TestPromptEnhancerNvidia:
    """Tests for prompt_enhancer_nvidia function"""
    
    @patch('helper_functions.nvidia_image_gen.OpenAI')
    def test_prompt_enhancer_nvidia_success(self, mock_openai):
        """Test successful prompt enhancement"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Enhanced Prompt"
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        result = nvidia_image_gen.prompt_enhancer_nvidia("Test Prompt")
        assert result == "Enhanced Prompt"

    @patch('helper_functions.nvidia_image_gen.OpenAI')
    def test_prompt_enhancer_nvidia_error(self, mock_openai):
        """Test error handling in prompt enhancement"""
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        mock_openai.return_value = mock_client
        
        result = nvidia_image_gen.prompt_enhancer_nvidia("Test Prompt")
        assert result == "Test Prompt" # Fallback to original

    @patch('helper_functions.nvidia_image_gen.OpenAI')
    def test_prompt_enhancer_nvidia_empty_prompt(self, mock_openai):
        """Test enhancement with empty prompt result"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = ""
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        result = nvidia_image_gen.prompt_enhancer_nvidia("Test Prompt")
        assert result == "Test Prompt"

    @patch('helper_functions.nvidia_image_gen.OpenAI')
    def test_prompt_enhancer_nvidia_enhancement_validation(self, mock_openai):
        """Test that enhancement is used only if different from original"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "test prompt" # Same (case insensitive)
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        result = nvidia_image_gen.prompt_enhancer_nvidia("Test Prompt")
        assert result == "Test Prompt"

class TestGenerateSurprisePromptNvidia:
    """Tests for generate_surprise_prompt_nvidia function"""
    
    @patch('helper_functions.nvidia_image_gen.OpenAI')
    def test_generate_surprise_prompt_success(self, mock_openai):
        """Test successful surprise prompt generation"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Surprise Prompt"
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        result = nvidia_image_gen.generate_surprise_prompt_nvidia()
        assert result == "Surprise Prompt"

    @patch('helper_functions.nvidia_image_gen.OpenAI')
    def test_generate_surprise_prompt_error(self, mock_openai):
        """Test error handling in surprise prompt generation"""
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        mock_openai.return_value = mock_client
        
        result = nvidia_image_gen.generate_surprise_prompt_nvidia()
        assert "clockwork octopus" in result # Default fallback

    @patch('helper_functions.nvidia_image_gen.OpenAI')
    def test_generate_surprise_prompt_temperature(self, mock_openai):
        """Test that high temperature is used"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Prompt"
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        nvidia_image_gen.generate_surprise_prompt_nvidia()
        
        # Verify temperature in payload
        _, kwargs = mock_client.chat.completions.create.call_args
        assert kwargs['temperature'] == 1.0

class TestRunGenerateNvidia:
    """Tests for run_generate_nvidia function"""
    
    @patch('helper_functions.nvidia_image_gen.requests.post')
    @patch('builtins.open', new_callable=mock_open)
    def test_run_generate_nvidia_success(self, mock_file, mock_post):
        """Test successful image generation"""
        fake_image_data = base64.b64encode(b"fake_image_data").decode()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"image": fake_image_data}
        mock_post.return_value = mock_response
        
        result = nvidia_image_gen.run_generate_nvidia("Test Prompt")
        
        assert isinstance(result, str)
        assert result.endswith(".png")
        mock_file.assert_called()
        mock_file().write.assert_called()

    @patch('helper_functions.nvidia_image_gen.requests.post')
    @patch('builtins.open', new_callable=mock_open)
    def test_run_generate_nvidia_default_prompt(self, mock_file, mock_post):
        """Test generation with default prompt"""
        fake_image_data = base64.b64encode(b"fake_image_data").decode()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"image": fake_image_data}
        mock_post.return_value = mock_response
        
        nvidia_image_gen.run_generate_nvidia()
        
        args, kwargs = mock_post.call_args
        payload = kwargs['json']
        assert "futuristic cityscape" in payload['prompt']

    @patch('helper_functions.nvidia_image_gen.requests.post')
    def test_run_generate_nvidia_api_error(self, mock_post):
        """Test API error handling"""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal server error"
        mock_post.return_value = mock_response
        
        try:
            nvidia_image_gen.run_generate_nvidia("test", "1024x1024")
        except Exception as e:
            assert "500" in str(e) or "Error" in str(e)

    @patch('helper_functions.nvidia_image_gen.requests.post')
    @patch('builtins.open', new_callable=mock_open)
    def test_run_generate_nvidia_response_format_variations(self, mock_file, mock_post):
        """Test different response format variations"""
        fake_image_data = base64.b64encode(b"fake_image_data").decode()
        
        # Format 1: 'image' key
        mock_response1 = Mock()
        mock_response1.status_code = 200
        mock_response1.json.return_value = {"image": fake_image_data}
        mock_post.return_value = mock_response1
        
        nvidia_image_gen.run_generate_nvidia("test")
        
        # Format 2: 'data' list
        mock_response2 = Mock()
        mock_response2.status_code = 200
        mock_response2.json.return_value = {"data": [{"b64_json": fake_image_data}]}
        mock_post.return_value = mock_response2
        
        nvidia_image_gen.run_generate_nvidia("test")
        
        # Format 3: 'artifacts' list (common in some NIMs)
        # Note: current implementation does NOT support artifacts list, it raises Exception.
        # So we should expect an exception here.
        mock_response3 = Mock()
        mock_response3.status_code = 200
        mock_response3.json.return_value = {"artifacts": [{"base64": fake_image_data}]}
        mock_post.return_value = mock_response3
        
        with pytest.raises(Exception) as excinfo:
            nvidia_image_gen.run_generate_nvidia("test")
        assert "Unexpected response format" in str(excinfo.value)

class TestRunEditNvidia:
    """Tests for run_edit_nvidia function"""
    
    @patch('helper_functions.nvidia_image_gen.requests.post')
    @patch('builtins.open', new_callable=mock_open, read_data=b"fake_image_bytes")
    def test_run_edit_nvidia_success(self, mock_file, mock_post):
        """Test successful image editing"""
        # Mock successful response
        fake_image_data = base64.b64encode(b"fake_edited_image").decode()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"image": fake_image_data}
        mock_post.return_value = mock_response
        
        result = nvidia_image_gen.run_edit_nvidia("input.png", "Edit Prompt")
        
        assert isinstance(result, str)
        assert result.endswith(".png")

    @patch('helper_functions.nvidia_image_gen.requests.post')
    @patch('builtins.open', new_callable=mock_open, read_data=b"fake_image_bytes")
    def test_run_edit_nvidia_default_prompt(self, mock_file, mock_post):
        """Test editing with default prompt"""
        fake_image_data = base64.b64encode(b"fake_edited_image").decode()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"image": fake_image_data}
        mock_post.return_value = mock_response
        
        nvidia_image_gen.run_edit_nvidia("input.png")
        
        args, kwargs = mock_post.call_args
        payload = kwargs['json']
        assert "cyberpunk" in payload['prompt']

    @patch('helper_functions.nvidia_image_gen.requests.post')
    @patch('builtins.open', new_callable=mock_open, read_data=b"fake_image_bytes")
    def test_run_edit_nvidia_error_handling(self, mock_file, mock_post):
        """Test error handling in edit"""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Error"
        mock_post.return_value = mock_response
        
        try:
            nvidia_image_gen.run_edit_nvidia("input.png", "Edit")
        except Exception as e:
            assert "500" in str(e) or "Error" in str(e)
