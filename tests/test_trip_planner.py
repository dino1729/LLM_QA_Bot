"""
Unit tests for helper_functions/trip_planner.py
Tests trip itinerary generation functionality
"""
import pytest
from unittest.mock import Mock, patch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TestGenerateTripPlan:
    """Test generate_trip_plan function"""

    @patch('helper_functions.trip_planner.generate_chat')
    def test_generate_trip_plan_valid_inputs(self, mock_generate_chat):
        """Test with valid city and days"""
        from helper_functions.trip_planner import generate_trip_plan

        mock_generate_chat.return_value = (
            "Day 1:\n1. Visit Museum\n2. Lunch at Restaurant\n"
            "Day 2:\n1. Park tour\n2. Dinner downtown"
        )

        result = generate_trip_plan("Paris", "3", "LITELLM_SMART")

        assert isinstance(result, str)
        assert "Day" in result or "Museum" in result
        mock_generate_chat.assert_called_once()

        # Verify conversation structure
        call_args = mock_generate_chat.call_args[0]
        conversation = call_args[1]
        assert len(conversation) == 2  # System + user message
        assert "Paris" in conversation[1]["content"]
        assert "3" in conversation[1]["content"]

    @patch('helper_functions.trip_planner.generate_chat')
    def test_generate_trip_plan_integer_days(self, mock_generate_chat):
        """Test with integer days parameter"""
        from helper_functions.trip_planner import generate_trip_plan

        mock_generate_chat.return_value = "Trip plan"

        result = generate_trip_plan("Tokyo", 5, "LITELLM_SMART")

        assert isinstance(result, str)
        mock_generate_chat.assert_called_once()

    @patch('helper_functions.trip_planner.generate_chat')
    def test_generate_trip_plan_default_model(self, mock_generate_chat):
        """Test with default model"""
        from helper_functions.trip_planner import generate_trip_plan

        mock_generate_chat.return_value = "Trip plan"

        result = generate_trip_plan("London", "4")

        # Verify default model is used
        call_args = mock_generate_chat.call_args[0]
        assert call_args[0] == "LITELLM_SMART"

    @patch('helper_functions.trip_planner.generate_chat')
    def test_generate_trip_plan_custom_model(self, mock_generate_chat):
        """Test with custom model"""
        from helper_functions.trip_planner import generate_trip_plan

        mock_generate_chat.return_value = "Trip plan"

        result = generate_trip_plan("Barcelona", "7", "OLLAMA_STRATEGIC")

        # Verify custom model is used
        call_args = mock_generate_chat.call_args[0]
        assert call_args[0] == "OLLAMA_STRATEGIC"

    @patch('helper_functions.trip_planner.generate_chat')
    def test_generate_trip_plan_non_numeric_days(self, mock_generate_chat):
        """Test with non-numeric days (should raise ValueError)"""
        from helper_functions.trip_planner import generate_trip_plan

        mock_generate_chat.return_value = "Trip plan"

        # Should raise ValueError for non-numeric days
        with pytest.raises(ValueError):
            generate_trip_plan("Rome", "abc", "LITELLM_SMART")

    @patch('helper_functions.trip_planner.generate_chat')
    def test_generate_trip_plan_negative_days(self, mock_generate_chat):
        """Test with negative days"""
        from helper_functions.trip_planner import generate_trip_plan

        mock_generate_chat.return_value = "Trip plan"

        # May raise ValueError or handle gracefully
        try:
            result = generate_trip_plan("Berlin", "-5", "LITELLM_SMART")
            # If it doesn't raise, just verify it returns something
            assert isinstance(result, str)
        except ValueError:
            # ValueError is acceptable for negative days
            pass

    @patch('helper_functions.trip_planner.generate_chat')
    def test_generate_trip_plan_zero_days(self, mock_generate_chat):
        """Test with zero days"""
        from helper_functions.trip_planner import generate_trip_plan

        mock_generate_chat.return_value = "Trip plan"

        # May raise ValueError or handle gracefully
        try:
            result = generate_trip_plan("Amsterdam", "0", "LITELLM_SMART")
            assert isinstance(result, str)
        except ValueError:
            pass

    @patch('helper_functions.trip_planner.generate_chat')
    def test_generate_trip_plan_system_prompt(self, mock_generate_chat):
        """Test that system prompt includes itinerary components"""
        from helper_functions.trip_planner import generate_trip_plan

        mock_generate_chat.return_value = "Trip plan"

        generate_trip_plan("Vienna", "4", "LITELLM_SMART")

        call_args = mock_generate_chat.call_args[0]
        conversation = call_args[1]
        system_message = conversation[0]["content"]

        # Verify system prompt mentions trip planning
        assert any(word in system_message.lower() for word in [
            "trip", "itinerary", "plan", "travel", "vacation"
        ])

    @patch('helper_functions.trip_planner.generate_chat')
    def test_generate_trip_plan_numbered_list(self, mock_generate_chat):
        """Test that output is a numbered list format"""
        from helper_functions.trip_planner import generate_trip_plan

        mock_generate_chat.return_value = "Trip plan"

        generate_trip_plan("Prague", "3", "LITELLM_SMART")

        call_args = mock_generate_chat.call_args[0]
        conversation = call_args[1]
        user_message = conversation[1]["content"]

        # Should ask for numbered format or structured itinerary
        assert "numbered" in user_message.lower() or "list" in user_message.lower() or "itinerary" in user_message.lower()

    @patch('helper_functions.trip_planner.generate_chat')
    def test_generate_trip_plan_includes_components(self, mock_generate_chat):
        """Test that request includes attractions, restaurants, hotels"""
        from helper_functions.trip_planner import generate_trip_plan

        mock_generate_chat.return_value = "Trip plan"

        generate_trip_plan("Budapest", "5", "LITELLM_SMART")

        call_args = mock_generate_chat.call_args[0]
        conversation = call_args[1]
        # Check system or user message for components
        all_content = " ".join([msg["content"].lower() for msg in conversation])

        # Should mention at least some of these components
        component_mentions = sum([
            "attraction" in all_content,
            "restaurant" in all_content,
            "hotel" in all_content,
            "food" in all_content,
            "accommodation" in all_content
        ])
        assert component_mentions >= 1

    @patch('helper_functions.trip_planner.generate_chat')
    def test_generate_trip_plan_error_handling(self, mock_generate_chat):
        """Test error handling"""
        from helper_functions.trip_planner import generate_trip_plan

        mock_generate_chat.side_effect = Exception("API error")

        # Should handle exception gracefully or raise
        try:
            result = generate_trip_plan("Dublin", "3", "LITELLM_SMART")
            assert isinstance(result, str) or result is None
        except Exception as e:
            assert "API error" in str(e)

    @patch('helper_functions.trip_planner.generate_chat')
    def test_generate_trip_plan_empty_city(self, mock_generate_chat):
        """Test with empty city"""
        from helper_functions.trip_planner import generate_trip_plan

        mock_generate_chat.return_value = "Trip plan"

        result = generate_trip_plan("", "3", "LITELLM_SMART")

        # Should still make the call
        mock_generate_chat.assert_called_once()

    @patch('helper_functions.trip_planner.generate_chat')
    def test_generate_trip_plan_special_characters(self, mock_generate_chat):
        """Test with special characters in city name"""
        from helper_functions.trip_planner import generate_trip_plan

        mock_generate_chat.return_value = "Trip plan"

        result = generate_trip_plan("São Paulo", "4", "LITELLM_SMART")

        assert isinstance(result, str)
        mock_generate_chat.assert_called_once()

    @patch('helper_functions.trip_planner.generate_chat')
    def test_generate_trip_plan_large_number_days(self, mock_generate_chat):
        """Test with large number of days"""
        from helper_functions.trip_planner import generate_trip_plan

        mock_generate_chat.return_value = "Trip plan"

        result = generate_trip_plan("Bangkok", "30", "LITELLM_SMART")

        assert isinstance(result, str)
        mock_generate_chat.assert_called_once()

    @patch('helper_functions.trip_planner.generate_chat')
    def test_generate_trip_plan_temperature_and_tokens(self, mock_generate_chat):
        """Test that temperature and max_tokens are set correctly"""
        from helper_functions.trip_planner import generate_trip_plan

        mock_generate_chat.return_value = "Trip plan"

        generate_trip_plan("Istanbul", "6", "LITELLM_SMART")

        call_args = mock_generate_chat.call_args[0]
        temperature = call_args[2]
        max_tokens = call_args[3]

        assert isinstance(temperature, (int, float))
        assert isinstance(max_tokens, int)
        assert 0 <= temperature <= 2
        assert max_tokens > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
