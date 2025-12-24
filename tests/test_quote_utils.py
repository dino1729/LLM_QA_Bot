"""
Tests for quote_utils.py - Quote generation functionality

Note: These tests intentionally use specific provider values ("litellm", "ollama")
to test the provider-specific code paths. This is correct behavior for unit tests.
For integration tests that need actual config values, use fixtures from conftest.py.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from helper_functions import quote_utils
from config import config


class TestGenerateQuote:
    """Tests for generate_quote() function"""
    
    @patch('helper_functions.quote_utils.get_client')
    def test_generate_quote_with_quoted_text(self, mock_get_client):
        """Test quote generation with properly quoted response"""
        mock_client = Mock()
        mock_client.chat_completion.return_value = '"This is a great inspirational quote about life and success."'
        mock_get_client.return_value = mock_client
        
        result = quote_utils.generate_quote("Steve Jobs", "litellm")
        
        assert result == '"This is a great inspirational quote about life and success."'
        assert result.startswith('"')
        assert result.endswith('"')
    
    @patch('helper_functions.quote_utils.get_client')
    def test_generate_quote_with_multiple_quotes(self, mock_get_client):
        """Test quote generation when response has multiple quoted segments"""
        mock_client = Mock()
        mock_client.chat_completion.return_value = 'Here is a "short quote" and here is a "much longer inspirational quote about persistence and success in life"'
        mock_get_client.return_value = mock_client
        
        result = quote_utils.generate_quote("Albert Einstein", "litellm")
        
        # Should select the longest quote
        assert "much longer inspirational quote about persistence and success in life" in result
    
    @patch('helper_functions.quote_utils.get_client')
    def test_generate_quote_without_quotes(self, mock_get_client):
        """Test quote generation when response lacks quote marks"""
        mock_client = Mock()
        mock_client.chat_completion.return_value = "This is an inspirational quote without quote marks"
        mock_get_client.return_value = mock_client
        
        result = quote_utils.generate_quote("Mahatma Gandhi", "litellm")
        
        assert result.startswith('"')
        assert result.endswith('"')
        assert "inspirational quote" in result
    
    @patch('helper_functions.quote_utils.get_client')
    def test_generate_quote_with_meta_text(self, mock_get_client):
        """Test quote generation with meta-text that should be skipped"""
        mock_client = Mock()
        # Response with meta-text on first line, actual quote on second
        mock_client.chat_completion.return_value = '''Here is a famous quote from Gandhi:
Be the change you wish to see in the world'''
        mock_get_client.return_value = mock_client
        
        result = quote_utils.generate_quote("Mahatma Gandhi", "litellm")
        
        # Should skip the meta-text line and use the actual quote
        assert "Be the change" in result
        assert "Here is" not in result
    
    @patch('helper_functions.quote_utils.get_client')
    def test_generate_quote_empty_response(self, mock_get_client):
        """Test quote generation with empty LLM response"""
        mock_client = Mock()
        mock_client.chat_completion.return_value = ""
        mock_get_client.return_value = mock_client
        
        result = quote_utils.generate_quote("Steve Jobs", "litellm")
        
        # Should return fallback quote for Steve Jobs
        assert result == '"Stay hungry, stay foolish."'
    
    @patch('helper_functions.quote_utils.get_client')
    def test_generate_quote_whitespace_only_response(self, mock_get_client):
        """Test quote generation with whitespace-only response"""
        mock_client = Mock()
        mock_client.chat_completion.return_value = "   \n\n   \t   "
        mock_get_client.return_value = mock_client
        
        result = quote_utils.generate_quote("Albert Einstein", "litellm")
        
        # Should return fallback quote for Einstein
        assert result == '"Imagination is more important than knowledge."'
    
    @patch('helper_functions.quote_utils.get_client')
    def test_generate_quote_none_response(self, mock_get_client):
        """Test quote generation with None response"""
        mock_client = Mock()
        mock_client.chat_completion.return_value = None
        mock_get_client.return_value = mock_client
        
        result = quote_utils.generate_quote("Nelson Mandela", "litellm")
        
        # Should return fallback quote
        assert result == '"It always seems impossible until it is done."'
    
    @patch('helper_functions.quote_utils.get_client')
    def test_generate_quote_exception_handling(self, mock_get_client):
        """Test quote generation when LLM client raises exception"""
        mock_client = Mock()
        mock_client.chat_completion.side_effect = Exception("API error")
        mock_get_client.return_value = mock_client
        
        result = quote_utils.generate_quote("Winston Churchill", "litellm")
        
        # Should return fallback quote
        assert result == '"Success is not final, failure is not fatal: it is the courage to continue that counts."'
    
    @patch('helper_functions.quote_utils.get_client')
    def test_generate_quote_with_multiline_response(self, mock_get_client):
        """Test quote generation with multiline response"""
        mock_client = Mock()
        mock_client.chat_completion.return_value = '''
The only way to do great work is to love what you do.
'''
        mock_get_client.return_value = mock_client
        
        result = quote_utils.generate_quote("Steve Jobs", "litellm")
        
        assert "The only way to do great work" in result
    
    @patch('helper_functions.quote_utils.get_client')
    def test_generate_quote_with_short_response(self, mock_get_client):
        """Test quote generation with response too short to be a valid quote"""
        mock_client = Mock()
        mock_client.chat_completion.return_value = '"OK"'  # Too short
        mock_get_client.return_value = mock_client
        
        result = quote_utils.generate_quote("Steve Jobs", "litellm")
        
        # Should use fallback because quote is too short (< 15 chars in quotes)
        assert result == '"Stay hungry, stay foolish."'
    
    @patch('helper_functions.quote_utils.get_client')
    def test_generate_quote_skips_meta_keywords(self, mock_get_client):
        """Test that meta-text keywords are properly skipped"""
        mock_client = Mock()
        mock_client.chat_completion.return_value = '''The user asked for a quote
Here's a famous quote from Einstein
This is the actual inspirational quote'''
        mock_get_client.return_value = mock_client
        
        result = quote_utils.generate_quote("Albert Einstein", "litellm")
        
        # Should skip lines with "the user" and "here's" and use the actual quote
        assert "actual inspirational quote" in result
    
    @patch('helper_functions.quote_utils.get_client')
    def test_generate_quote_with_cleaned_fallback(self, mock_get_client):
        """Test using cleaned response as fallback when no good lines found"""
        mock_client = Mock()
        # Short response without meta-text, between 20-300 chars
        mock_client.chat_completion.return_value = "Do or do not. There is no try."
        mock_get_client.return_value = mock_client
        
        result = quote_utils.generate_quote("Yoda", "litellm")
        
        assert "Do or do not" in result
        assert result.startswith('"')
        assert result.endswith('"')
    
    @patch('helper_functions.quote_utils.get_client')
    def test_generate_quote_different_providers(self, mock_get_client):
        """Test quote generation with different LLM providers"""
        mock_client = Mock()
        mock_client.chat_completion.return_value = '"Success is not the key to happiness."'
        mock_get_client.return_value = mock_client
        
        for provider in ["litellm", "ollama"]:
            result = quote_utils.generate_quote("Buddha", provider)
            assert "Success is not the key to happiness" in result
    
    @patch('helper_functions.quote_utils.get_client')
    def test_generate_quote_calls_client_with_correct_params(self, mock_get_client):
        """Test that LLM client is called with correct parameters"""
        mock_client = Mock()
        mock_client.chat_completion.return_value = '"Test quote"'
        mock_get_client.return_value = mock_client
        
        quote_utils.generate_quote("Test Person", "litellm")
        
        # Verify client was created with correct provider and tier
        mock_get_client.assert_called_once_with(provider="litellm", model_tier="fast")
        
        # Verify chat_completion was called with correct params
        args, kwargs = mock_client.chat_completion.call_args
        assert kwargs["max_tokens"] == 250
        assert kwargs["temperature"] == 0.8
        assert "messages" in kwargs
    
    @patch('helper_functions.quote_utils.get_client')
    def test_generate_quote_system_prompt_includes_personality(self, mock_get_client):
        """Test that system prompt includes the requested personality"""
        mock_client = Mock()
        mock_client.chat_completion.return_value = '"Test quote"'
        mock_get_client.return_value = mock_client
        
        personality = "Marcus Aurelius"
        quote_utils.generate_quote(personality, "litellm")
        
        # Check that the personality appears in the conversation
        args, kwargs = mock_client.chat_completion.call_args
        messages = kwargs["messages"]
        system_message = messages[0]["content"]
        assert personality in system_message
    
    @patch('helper_functions.quote_utils.get_client')
    def test_generate_quote_rejects_user_keyword_in_cleaned(self, mock_get_client):
        """Test that cleaned response with 'user' keyword is rejected"""
        mock_client = Mock()
        # Response has "user" in it, should use fallback
        mock_client.chat_completion.return_value = "The user wants inspiration"
        mock_get_client.return_value = mock_client
        
        result = quote_utils.generate_quote("Default", "litellm")
        
        # Should use fallback because "user" appears in cleaned response
        assert result == '"The only way to do great work is to love what you do."'


class TestGetFallbackQuote:
    """Tests for _get_fallback_quote() function"""
    
    def test_fallback_quote_steve_jobs(self):
        """Test fallback quote for Steve Jobs"""
        result = quote_utils._get_fallback_quote("Steve Jobs")
        assert result == '"Stay hungry, stay foolish."'
    
    def test_fallback_quote_albert_einstein(self):
        """Test fallback quote for Albert Einstein"""
        result = quote_utils._get_fallback_quote("Albert Einstein")
        assert result == '"Imagination is more important than knowledge."'
    
    def test_fallback_quote_mahatma_gandhi(self):
        """Test fallback quote for Mahatma Gandhi"""
        result = quote_utils._get_fallback_quote("Mahatma Gandhi")
        assert result == '"Be the change you wish to see in the world."'
    
    def test_fallback_quote_martin_luther_king(self):
        """Test fallback quote for Martin Luther King Jr."""
        result = quote_utils._get_fallback_quote("Martin Luther King Jr.")
        assert result == '"Darkness cannot drive out darkness; only light can do that."'
    
    def test_fallback_quote_winston_churchill(self):
        """Test fallback quote for Winston Churchill"""
        result = quote_utils._get_fallback_quote("Winston Churchill")
        assert result == '"Success is not final, failure is not fatal: it is the courage to continue that counts."'
    
    def test_fallback_quote_nelson_mandela(self):
        """Test fallback quote for Nelson Mandela"""
        result = quote_utils._get_fallback_quote("Nelson Mandela")
        assert result == '"It always seems impossible until it is done."'
    
    def test_fallback_quote_unknown_personality(self):
        """Test fallback quote for unknown personality"""
        result = quote_utils._get_fallback_quote("Unknown Person")
        assert result == '"The only way to do great work is to love what you do."'
    
    def test_fallback_quote_default(self):
        """Test default fallback quote"""
        result = quote_utils._get_fallback_quote("Random Name")
        assert result == '"The only way to do great work is to love what you do."'

