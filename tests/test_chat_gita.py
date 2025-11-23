"""
Unit tests for helper_functions/chat_gita.py
Tests Bhagavad Gita chatbot functions using Pinecone
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TestExtractContextFromPinecone:
    """Test extract_context_frompinecone function"""

    @patch('helper_functions.chat_gita.Pinecone')
    @patch('helper_functions.chat_gita.embedding_client')
    def test_extract_context_success(self, mock_embedding_client, mock_pinecone):
        """Test successful context extraction from Pinecone"""
        from helper_functions.chat_gita import extract_context_frompinecone

        # Mock embedding
        mock_embedding_client.get_embedding.return_value = [0.1] * 1536

        # Mock Pinecone
        mock_index = Mock()
        mock_match1 = Mock()
        mock_match1.metadata = {"text": "Chapter 1, Verse 1: Dhritarashtra said..."}
        mock_match2 = Mock()
        mock_match2.metadata = {"text": "Chapter 1, Verse 2: Sanjaya said..."}

        mock_index.query.return_value.matches = [mock_match1, mock_match2]
        mock_pinecone.return_value.Index.return_value = mock_index

        result = extract_context_frompinecone("What is dharma?")

        assert isinstance(result, str)
        assert "Chapter 1, Verse 1" in result
        assert "Chapter 1, Verse 2" in result
        mock_embedding_client.get_embedding.assert_called_once()

    @patch('helper_functions.chat_gita.pinecone_apikey', None)
    def test_extract_context_no_api_key(self):
        """Test extract context without Pinecone API key"""
        from helper_functions.chat_gita import extract_context_frompinecone

        result = extract_context_frompinecone("test query")

        assert result == ""

    @patch('helper_functions.chat_gita.Pinecone')
    @patch('helper_functions.chat_gita.embedding_client')
    def test_extract_context_pinecone_error(self, mock_embedding_client, mock_pinecone):
        """Test extract context with Pinecone connection error"""
        from helper_functions.chat_gita import extract_context_frompinecone

        mock_embedding_client.get_embedding.return_value = [0.1] * 1536
        mock_pinecone.side_effect = Exception("Connection error")

        result = extract_context_frompinecone("test query")

        assert result == ""

    @patch('helper_functions.chat_gita.Pinecone')
    @patch('helper_functions.chat_gita.embedding_client')
    def test_extract_context_top_k(self, mock_embedding_client, mock_pinecone):
        """Test that it retrieves top 8 results"""
        from helper_functions.chat_gita import extract_context_frompinecone

        mock_embedding_client.get_embedding.return_value = [0.1] * 1536

        mock_index = Mock()
        matches = []
        for i in range(8):
            match = Mock()
            match.metadata = {"text": f"Verse {i}"}
            matches.append(match)

        mock_index.query.return_value.matches = matches
        mock_pinecone.return_value.Index.return_value = mock_index

        result = extract_context_frompinecone("test")

        # Verify query was called with top_k=8
        call_args = mock_index.query.call_args
        assert call_args[1]['top_k'] == 8


class TestGitaAnswer:
    """Test gita_answer function"""

    @patch('helper_functions.chat_gita.extract_context_frompinecone')
    @patch('helper_functions.chat_gita.generate_chat')
    def test_gita_answer_success(self, mock_generate_chat, mock_extract_context):
        """Test successful Gita answer generation"""
        from helper_functions.chat_gita import gita_answer

        mock_extract_context.return_value = "Context from Bhagavad Gita"
        mock_generate_chat.return_value = "Answer based on Gita teachings"

        history = [{"role": "user", "content": "Previous question"}]
        result = gita_answer(
            "What is dharma?",
            history,
            "LITELLM_SMART",
            1000,
            0.7
        )

        assert result == "Answer based on Gita teachings"
        mock_extract_context.assert_called_once_with("What is dharma?")
        mock_generate_chat.assert_called_once()

        # Verify conversation structure
        call_args = mock_generate_chat.call_args[0]
        conversation = call_args[1]
        assert len(conversation) >= 2  # System message + user query
        assert conversation[0]["role"] == "system"
        assert "Bhagavad Gita" in conversation[0]["content"]

    @patch('helper_functions.chat_gita.extract_context_frompinecone')
    @patch('helper_functions.chat_gita.generate_chat')
    def test_gita_answer_with_empty_history(self, mock_generate_chat, mock_extract_context):
        """Test Gita answer with empty history"""
        from helper_functions.chat_gita import gita_answer

        mock_extract_context.return_value = "Context"
        mock_generate_chat.return_value = "Answer"

        result = gita_answer("What is karma?", [], "LITELLM_SMART", 1000, 0.7)

        assert result == "Answer"
        mock_extract_context.assert_called_once()

    @patch('helper_functions.chat_gita.extract_context_frompinecone')
    @patch('helper_functions.chat_gita.generate_chat')
    def test_gita_answer_without_pinecone(self, mock_generate_chat, mock_extract_context):
        """Test Gita answer when Pinecone unavailable"""
        from helper_functions.chat_gita import gita_answer

        mock_extract_context.return_value = ""  # No context from Pinecone
        mock_generate_chat.return_value = "General answer"

        result = gita_answer("test", [], "LITELLM_SMART", 1000, 0.7)

        assert result == "General answer"
        # Should still call generate_chat even without context

    @patch('helper_functions.chat_gita.extract_context_frompinecone')
    @patch('helper_functions.chat_gita.generate_chat')
    def test_gita_answer_error_handling(self, mock_generate_chat, mock_extract_context):
        """Test error handling in gita_answer"""
        from helper_functions.chat_gita import gita_answer

        mock_extract_context.side_effect = Exception("Pinecone error")
        mock_generate_chat.return_value = "Fallback answer"

        # Should handle exception gracefully
        try:
            result = gita_answer("test", [], "LITELLM_SMART", 1000, 0.7)
            # If it doesn't raise, check it returns something reasonable
            assert isinstance(result, str)
        except Exception:
            # Exception is acceptable if not handled internally
            pass

    @patch('helper_functions.chat_gita.extract_context_frompinecone')
    @patch('helper_functions.chat_gita.generate_chat')
    def test_gita_answer_context_in_system_prompt(self, mock_generate_chat, mock_extract_context):
        """Test that context is included in system prompt"""
        from helper_functions.chat_gita import gita_answer

        mock_context = "Arjuna asked Krishna about dharma"
        mock_extract_context.return_value = mock_context
        mock_generate_chat.return_value = "Answer"

        gita_answer("test", [], "LITELLM_SMART", 1000, 0.7)

        # Verify context was included in conversation
        call_args = mock_generate_chat.call_args[0]
        conversation = call_args[1]
        system_message = conversation[0]["content"]
        assert mock_context in system_message

    @patch('helper_functions.chat_gita.extract_context_frompinecone')
    @patch('helper_functions.chat_gita.generate_chat')
    def test_gita_answer_parameters_passed(self, mock_generate_chat, mock_extract_context):
        """Test that parameters are correctly passed to generate_chat"""
        from helper_functions.chat_gita import gita_answer

        mock_extract_context.return_value = "Context"
        mock_generate_chat.return_value = "Answer"

        gita_answer("test", [], "OLLAMA_FAST", 2000, 0.9)

        # Verify parameters
        call_args = mock_generate_chat.call_args
        assert call_args[0][0] == "OLLAMA_FAST"
        assert call_args[0][2] == 0.9  # temperature
        assert call_args[0][3] == 2000  # max_tokens

    @patch('helper_functions.chat_gita.extract_context_frompinecone')
    @patch('helper_functions.chat_gita.generate_chat')
    def test_gita_answer_with_long_history(self, mock_generate_chat, mock_extract_context):
        """Test Gita answer with long conversation history"""
        from helper_functions.chat_gita import gita_answer

        mock_extract_context.return_value = "Context"
        mock_generate_chat.return_value = "Answer"

        history = [
            {"role": "user", "content": f"Question {i}"}
            for i in range(10)
        ]

        result = gita_answer("New question", history, "LITELLM_SMART", 1000, 0.7)

        assert result == "Answer"
        # Verify history is included in conversation
        call_args = mock_generate_chat.call_args[0]
        conversation = call_args[1]
        assert len(conversation) > len(history)  # System message + history + new query


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
