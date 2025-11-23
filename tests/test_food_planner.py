"""
Unit tests for helper_functions/food_planner.py
Tests restaurant recommendation functionality
"""
import pytest
from unittest.mock import Mock, patch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TestCravingSatisfier:
    """Test craving_satisfier function"""

    @patch('helper_functions.food_planner.generate_chat')
    def test_craving_satisfier_specific_food(self, mock_generate_chat):
        """Test with specific food craving"""
        from helper_functions.food_planner import craving_satisfier

        mock_generate_chat.return_value = "1. Restaurant A\n2. Restaurant B\n3. Restaurant C"

        result = craving_satisfier("New York", "pizza", "LITELLM_SMART")

        assert isinstance(result, str)
        assert "Restaurant" in result
        mock_generate_chat.assert_called_once()

        # Verify conversation structure
        call_args = mock_generate_chat.call_args[0]
        conversation = call_args[1]
        assert len(conversation) == 2  # System + user message
        assert "pizza" in conversation[1]["content"]
        assert "New York" in conversation[1]["content"]

    @patch('helper_functions.food_planner.generate_chat')
    def test_craving_satisfier_idk(self, mock_generate_chat):
        """Test with 'idk' for random restaurant selection"""
        from helper_functions.food_planner import craving_satisfier

        mock_generate_chat.return_value = "1. Thai Restaurant\n2. Italian Place"

        result = craving_satisfier("Los Angeles", "idk", "LITELLM_SMART")

        assert isinstance(result, str)
        mock_generate_chat.assert_called_once()

        # Verify it asks for any type of restaurant
        call_args = mock_generate_chat.call_args[0]
        conversation = call_args[1]
        user_message = conversation[1]["content"]
        assert "Los Angeles" in user_message
        # Should not contain "idk" but ask for general recommendations
        assert "idk" not in user_message.lower() or "any" in user_message.lower()

    @patch('helper_functions.food_planner.generate_chat')
    def test_craving_satisfier_default_model(self, mock_generate_chat):
        """Test with default model"""
        from helper_functions.food_planner import craving_satisfier

        mock_generate_chat.return_value = "Restaurant list"

        result = craving_satisfier("Boston", "seafood")

        # Verify default model is used
        call_args = mock_generate_chat.call_args[0]
        assert call_args[0] == "LITELLM_SMART"

    @patch('helper_functions.food_planner.generate_chat')
    def test_craving_satisfier_custom_model(self, mock_generate_chat):
        """Test with custom model"""
        from helper_functions.food_planner import craving_satisfier

        mock_generate_chat.return_value = "Restaurant list"

        result = craving_satisfier("Chicago", "burger", "OLLAMA_FAST")

        # Verify custom model is used
        call_args = mock_generate_chat.call_args[0]
        assert call_args[0] == "OLLAMA_FAST"

    @patch('helper_functions.food_planner.generate_chat')
    def test_craving_satisfier_system_prompt(self, mock_generate_chat):
        """Test system prompt includes dietary restrictions"""
        from helper_functions.food_planner import craving_satisfier

        mock_generate_chat.return_value = "Restaurant list"

        craving_satisfier("Miami", "sushi", "LITELLM_SMART")

        call_args = mock_generate_chat.call_args[0]
        conversation = call_args[1]
        system_message = conversation[0]["content"]

        # Verify dietary restrictions mentioned
        assert "beef" in system_message.lower() or "pork" in system_message.lower()

    @patch('helper_functions.food_planner.generate_chat')
    def test_craving_satisfier_eight_restaurants(self, mock_generate_chat):
        """Test that it requests 8 restaurants"""
        from helper_functions.food_planner import craving_satisfier

        mock_generate_chat.return_value = "8 restaurants"

        craving_satisfier("Seattle", "coffee", "LITELLM_SMART")

        call_args = mock_generate_chat.call_args[0]
        conversation = call_args[1]
        user_message = conversation[1]["content"]

        # Should request 8 restaurants
        assert "8" in user_message

    @patch('helper_functions.food_planner.generate_chat')
    def test_craving_satisfier_error_handling(self, mock_generate_chat):
        """Test error handling"""
        from helper_functions.food_planner import craving_satisfier

        mock_generate_chat.side_effect = Exception("API error")

        # Should handle exception gracefully or raise
        try:
            result = craving_satisfier("Denver", "mexican", "LITELLM_SMART")
            # If no exception raised, verify it returns something
            assert isinstance(result, str) or result is None
        except Exception as e:
            # Exception is acceptable
            assert "API error" in str(e)

    @patch('helper_functions.food_planner.generate_chat')
    def test_craving_satisfier_empty_city(self, mock_generate_chat):
        """Test with empty city"""
        from helper_functions.food_planner import craving_satisfier

        mock_generate_chat.return_value = "Restaurant list"

        result = craving_satisfier("", "pizza", "LITELLM_SMART")

        # Should still make the call
        mock_generate_chat.assert_called_once()

    @patch('helper_functions.food_planner.generate_chat')
    def test_craving_satisfier_empty_craving(self, mock_generate_chat):
        """Test with empty food craving"""
        from helper_functions.food_planner import craving_satisfier

        mock_generate_chat.return_value = "Restaurant list"

        result = craving_satisfier("Portland", "", "LITELLM_SMART")

        mock_generate_chat.assert_called_once()

    @patch('helper_functions.food_planner.generate_chat')
    def test_craving_satisfier_special_characters(self, mock_generate_chat):
        """Test with special characters in inputs"""
        from helper_functions.food_planner import craving_satisfier

        mock_generate_chat.return_value = "Restaurant list"

        result = craving_satisfier("São Paulo", "crêpes", "LITELLM_SMART")

        assert isinstance(result, str)
        mock_generate_chat.assert_called_once()

    @patch('helper_functions.food_planner.generate_chat')
    def test_craving_satisfier_long_city_name(self, mock_generate_chat):
        """Test with long city name"""
        from helper_functions.food_planner import craving_satisfier

        mock_generate_chat.return_value = "Restaurant list"

        long_city = "Llanfairpwllgwyngyllgogerychwyrndrobwllllantysiliogogogoch"
        result = craving_satisfier(long_city, "fish", "LITELLM_SMART")

        assert isinstance(result, str)
        mock_generate_chat.assert_called_once()

    @patch('helper_functions.food_planner.generate_chat')
    def test_craving_satisfier_temperature_and_tokens(self, mock_generate_chat):
        """Test that temperature and max_tokens are set correctly"""
        from helper_functions.food_planner import craving_satisfier

        mock_generate_chat.return_value = "Restaurant list"

        craving_satisfier("Austin", "BBQ", "LITELLM_SMART")

        call_args = mock_generate_chat.call_args[0]
        # Check temperature (should be reasonable, like 0.7 or similar)
        temperature = call_args[2]
        max_tokens = call_args[3]

        assert isinstance(temperature, (int, float))
        assert isinstance(max_tokens, int)
        assert 0 <= temperature <= 2
        assert max_tokens > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
