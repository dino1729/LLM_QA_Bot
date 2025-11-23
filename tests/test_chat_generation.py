"""
Unit tests for helper_functions/chat_generation.py
Tests for multi-provider LLM chat completion support
"""
import pytest
from unittest.mock import Mock, MagicMock, patch
from helper_functions import chat_generation


class TestGenerateChat:
    """Tests for generate_chat() function"""
    
    @patch('helper_functions.chat_generation.OpenAI')
    def test_generate_chat_dynamic_litellm(self, mock_openai_class):
        """Test dynamic LiteLLM model (e.g., LITELLM:deepseek-v3.1)"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "LiteLLM response"
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client
        
        conversation = [{"role": "user", "content": "Hello"}]
        result = chat_generation.generate_chat(
            "LITELLM:deepseek-v3.1", 
            conversation, 
            0.7, 
            1000
        )
        
        assert result == "LiteLLM response"
        mock_client.chat.completions.create.assert_called_once()
        # Verify the actual model name was passed (without LITELLM: prefix)
        call_args = mock_client.chat.completions.create.call_args
        assert call_args[1]["model"] == "deepseek-v3.1"
    
    @patch('helper_functions.chat_generation.OpenAI')
    def test_generate_chat_dynamic_ollama(self, mock_openai_class):
        """Test dynamic Ollama model (e.g., OLLAMA:llama3.2:3b)"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Ollama response"
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client
        
        conversation = [{"role": "user", "content": "Hello"}]
        result = chat_generation.generate_chat(
            "OLLAMA:llama3.2:3b", 
            conversation, 
            0.7, 
            1000
        )
        
        assert result == "Ollama response"
        call_args = mock_client.chat.completions.create.call_args
        assert call_args[1]["model"] == "llama3.2:3b"
    
    @patch('helper_functions.chat_generation.get_client')
    def test_generate_chat_litellm_fast(self, mock_get_client):
        """Test predefined LiteLLM FAST tier"""
        mock_client = Mock()
        mock_client.chat_completion.return_value = "Fast response"
        mock_get_client.return_value = mock_client
        
        conversation = [{"role": "user", "content": "Hello"}]
        result = chat_generation.generate_chat(
            "LITELLM_FAST", 
            conversation, 
            0.7, 
            1000
        )
        
        assert result == "Fast response"
        mock_get_client.assert_called_once_with(provider="litellm", model_tier="fast")
    
    @patch('helper_functions.chat_generation.get_client')
    def test_generate_chat_litellm_smart(self, mock_get_client):
        """Test predefined LiteLLM SMART tier"""
        mock_client = Mock()
        mock_client.chat_completion.return_value = "Smart response"
        mock_get_client.return_value = mock_client
        
        conversation = [{"role": "user", "content": "Hello"}]
        result = chat_generation.generate_chat(
            "LITELLM_SMART", 
            conversation, 
            0.7, 
            1000
        )
        
        assert result == "Smart response"
        mock_get_client.assert_called_once_with(provider="litellm", model_tier="smart")
    
    @patch('helper_functions.chat_generation.get_client')
    def test_generate_chat_litellm_strategic(self, mock_get_client):
        """Test predefined LiteLLM STRATEGIC tier"""
        mock_client = Mock()
        mock_client.chat_completion.return_value = "Strategic response"
        mock_get_client.return_value = mock_client
        
        conversation = [{"role": "user", "content": "Hello"}]
        result = chat_generation.generate_chat(
            "LITELLM_STRATEGIC", 
            conversation, 
            0.7, 
            1000
        )
        
        assert result == "Strategic response"
        mock_get_client.assert_called_once_with(provider="litellm", model_tier="strategic")
    
    @patch('helper_functions.chat_generation.get_client')
    def test_generate_chat_ollama_fast(self, mock_get_client):
        """Test predefined Ollama FAST tier"""
        mock_client = Mock()
        mock_client.chat_completion.return_value = "Ollama fast response"
        mock_get_client.return_value = mock_client
        
        conversation = [{"role": "user", "content": "Hello"}]
        result = chat_generation.generate_chat(
            "OLLAMA_FAST", 
            conversation, 
            0.7, 
            1000
        )
        
        assert result == "Ollama fast response"
        mock_get_client.assert_called_once_with(provider="ollama", model_tier="fast")
    
    @patch('helper_functions.chat_generation.get_client')
    def test_generate_chat_ollama_smart(self, mock_get_client):
        """Test predefined Ollama SMART tier"""
        mock_client = Mock()
        mock_client.chat_completion.return_value = "Ollama smart response"
        mock_get_client.return_value = mock_client
        
        conversation = [{"role": "user", "content": "Hello"}]
        result = chat_generation.generate_chat(
            "OLLAMA_SMART", 
            conversation, 
            0.7, 
            1000
        )
        
        assert result == "Ollama smart response"
        mock_get_client.assert_called_once_with(provider="ollama", model_tier="smart")
    
    @patch('helper_functions.chat_generation.get_client')
    def test_generate_chat_ollama_strategic(self, mock_get_client):
        """Test predefined Ollama STRATEGIC tier"""
        mock_client = Mock()
        mock_client.chat_completion.return_value = "Ollama strategic response"
        mock_get_client.return_value = mock_client
        
        conversation = [{"role": "user", "content": "Hello"}]
        result = chat_generation.generate_chat(
            "OLLAMA_STRATEGIC", 
            conversation, 
            0.7, 
            1000
        )
        
        assert result == "Ollama strategic response"
        mock_get_client.assert_called_once_with(provider="ollama", model_tier="strategic")
    
    @patch('helper_functions.chat_generation.cohere.Client')
    def test_generate_chat_cohere(self, mock_cohere_class):
        """Test Cohere provider"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.text = "Cohere response"
        mock_client.chat.return_value = mock_response
        mock_cohere_class.return_value = mock_client
        
        conversation = [{"role": "user", "content": "Hello"}]
        result = chat_generation.generate_chat(
            "COHERE", 
            conversation, 
            0.7, 
            1000
        )
        
        assert result == "Cohere response"
        mock_client.chat.assert_called_once()
    
    @patch('helper_functions.chat_generation.genai.configure')
    @patch('helper_functions.chat_generation.genai.GenerativeModel')
    def test_generate_chat_gemini(self, mock_model_class, mock_configure):
        """Test Gemini provider"""
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = "Gemini response"
        mock_model.generate_content.return_value = mock_response
        mock_model_class.return_value = mock_model
        
        conversation = [{"role": "user", "content": "Hello"}]
        result = chat_generation.generate_chat(
            "GEMINI", 
            conversation, 
            0.7, 
            1000
        )
        
        assert result == "Gemini response"
        mock_model.generate_content.assert_called_once()
    
    @patch('helper_functions.chat_generation.genai.configure')
    @patch('helper_functions.chat_generation.genai.GenerativeModel')
    def test_generate_chat_gemini_thinking(self, mock_model_class, mock_configure):
        """Test Gemini Thinking model"""
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = "Gemini thinking response"
        mock_model.generate_content.return_value = mock_response
        mock_model_class.return_value = mock_model
        
        conversation = [{"role": "user", "content": "Complex question"}]
        result = chat_generation.generate_chat(
            "GEMINI_THINKING", 
            conversation, 
            0.7, 
            1000
        )
        
        assert result == "Gemini thinking response"
    
    @patch('helper_functions.chat_generation.Groq')
    def test_generate_chat_groq(self, mock_groq_class):
        """Test Groq provider"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Groq response"
        mock_client.chat.completions.create.return_value = mock_response
        mock_groq_class.return_value = mock_client
        
        conversation = [{"role": "user", "content": "Hello"}]
        result = chat_generation.generate_chat(
            "GROQ", 
            conversation, 
            0.7, 
            1000
        )
        
        assert result == "Groq response"
    
    @patch('helper_functions.chat_generation.Groq')
    def test_generate_chat_groq_llama(self, mock_groq_class):
        """Test Groq with Llama model"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Groq Llama response"
        mock_client.chat.completions.create.return_value = mock_response
        mock_groq_class.return_value = mock_client
        
        conversation = [{"role": "user", "content": "Hello"}]
        result = chat_generation.generate_chat(
            "GROQ_LLAMA", 
            conversation, 
            0.7, 
            1000
        )
        
        assert result == "Groq Llama response"
        # Verify Llama model was used
        call_args = mock_client.chat.completions.create.call_args
        assert "llama" in call_args[1]["model"].lower()
    
    @patch('helper_functions.chat_generation.Groq')
    def test_generate_chat_groq_mixtral(self, mock_groq_class):
        """Test Groq with Mixtral model"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Groq Mixtral response"
        mock_client.chat.completions.create.return_value = mock_response
        mock_groq_class.return_value = mock_client
        
        conversation = [{"role": "user", "content": "Hello"}]
        result = chat_generation.generate_chat(
            "GROQ_MIXTRAL", 
            conversation, 
            0.7, 
            1000
        )
        
        assert result == "Groq Mixtral response"
        # Verify Mixtral model was used
        call_args = mock_client.chat.completions.create.call_args
        assert "mixtral" in call_args[1]["model"].lower()
    
    def test_generate_chat_invalid_model(self):
        """Test with invalid model name"""
        conversation = [{"role": "user", "content": "Hello"}]
        result = chat_generation.generate_chat(
            "INVALID_MODEL", 
            conversation, 
            0.7, 
            1000
        )
        
        assert "Invalid model name" in result
    
    @patch('helper_functions.chat_generation.get_client')
    def test_generate_chat_empty_conversation(self, mock_get_client):
        """Test with empty conversation"""
        mock_client = Mock()
        mock_client.chat_completion.return_value = "Response"
        mock_get_client.return_value = mock_client
        
        result = chat_generation.generate_chat(
            "LITELLM_FAST", 
            [], 
            0.7, 
            1000
        )
        
        assert result == "Response"
    
    @patch('helper_functions.chat_generation.get_client')
    def test_generate_chat_temperature_parameter(self, mock_get_client):
        """Test that temperature parameter is passed correctly"""
        mock_client = Mock()
        mock_client.chat_completion.return_value = "Response"
        mock_get_client.return_value = mock_client
        
        conversation = [{"role": "user", "content": "Hello"}]
        
        for temp in [0.0, 0.5, 1.0, 1.5]:
            chat_generation.generate_chat(
                "LITELLM_FAST", 
                conversation, 
                temp, 
                1000
            )
            
            call_args = mock_client.chat_completion.call_args
            assert call_args[0][1] == temp
    
    @patch('helper_functions.chat_generation.get_client')
    def test_generate_chat_max_tokens_parameter(self, mock_get_client):
        """Test that max_tokens parameter is passed correctly"""
        mock_client = Mock()
        mock_client.chat_completion.return_value = "Response"
        mock_get_client.return_value = mock_client
        
        conversation = [{"role": "user", "content": "Hello"}]
        
        for max_tokens in [100, 500, 1000, 2000]:
            chat_generation.generate_chat(
                "LITELLM_FAST", 
                conversation, 
                0.7, 
                max_tokens
            )
            
            call_args = mock_client.chat_completion.call_args
            assert call_args[0][2] == max_tokens
    
    @patch('helper_functions.chat_generation.OpenAI')
    def test_generate_chat_litellm_uses_correct_config(self, mock_openai_class):
        """Test that LiteLLM uses correct base_url and api_key from config"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Response"
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client
        
        conversation = [{"role": "user", "content": "Hello"}]
        chat_generation.generate_chat(
            "LITELLM:test-model", 
            conversation, 
            0.7, 
            1000
        )
        
        # Verify OpenAI client was initialized with config values
        mock_openai_class.assert_called_once()
        call_args = mock_openai_class.call_args
        assert "base_url" in call_args[1]
        assert "api_key" in call_args[1]
    
    @patch('helper_functions.chat_generation.OpenAI')
    def test_generate_chat_ollama_uses_correct_config(self, mock_openai_class):
        """Test that Ollama uses correct base_url"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Response"
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client
        
        conversation = [{"role": "user", "content": "Hello"}]
        chat_generation.generate_chat(
            "OLLAMA:test-model", 
            conversation, 
            0.7, 
            1000
        )
        
        # Verify OpenAI client was initialized with ollama config
        call_args = mock_openai_class.call_args
        assert call_args[1]["api_key"] == "ollama"
    
    @patch('helper_functions.chat_generation.get_client')
    def test_generate_chat_conversation_format(self, mock_get_client):
        """Test that conversation messages are passed correctly"""
        mock_client = Mock()
        mock_client.chat_completion.return_value = "Response"
        mock_get_client.return_value = mock_client
        
        conversation = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "How are you?"}
        ]
        
        chat_generation.generate_chat(
            "LITELLM_SMART", 
            conversation, 
            0.7, 
            1000
        )
        
        # Verify conversation was passed as-is
        call_args = mock_client.chat_completion.call_args
        assert call_args[0][0] == conversation


class TestAPIKeyConfiguration:
    """Tests for API key configuration"""
    
    def test_api_keys_loaded_from_config(self):
        """Test that API keys are loaded from config"""
        assert hasattr(chat_generation, 'cohere_api_key')
        assert hasattr(chat_generation, 'google_api_key')
        assert hasattr(chat_generation, 'groq_api_key')
    
    def test_model_names_loaded_from_config(self):
        """Test that model names are loaded from config"""
        assert hasattr(chat_generation, 'gemini_model_name')
        assert hasattr(chat_generation, 'gemini_thinkingmodel_name')


class TestEdgeCases:
    """Tests for edge cases and error handling"""
    
    @patch('helper_functions.chat_generation.get_client')
    def test_generate_chat_with_exception(self, mock_get_client):
        """Test handling of exceptions during chat generation"""
        mock_client = Mock()
        mock_client.chat_completion.side_effect = Exception("API error")
        mock_get_client.return_value = mock_client
        
        conversation = [{"role": "user", "content": "Hello"}]
        
        with pytest.raises(Exception):
            chat_generation.generate_chat(
                "LITELLM_FAST", 
                conversation, 
                0.7, 
                1000
            )
    
    @patch('helper_functions.chat_generation.OpenAI')
    def test_generate_chat_with_unicode_content(self, mock_openai_class):
        """Test handling of unicode content in conversation"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Response with unicode: café ñ"
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client
        
        conversation = [{"role": "user", "content": "Tell me about café culture"}]
        result = chat_generation.generate_chat(
            "LITELLM:test-model", 
            conversation, 
            0.7, 
            1000
        )
        
        assert "café" in result
        assert "ñ" in result

