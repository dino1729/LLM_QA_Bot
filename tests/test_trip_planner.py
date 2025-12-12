import pytest
from unittest.mock import Mock, patch
from helper_functions import trip_planner

class TestGenerateTripPlan:
    """Tests for generate_trip_plan() function"""
    
    @patch('helper_functions.trip_planner.generate_chat')
    def test_generate_trip_plan_valid_inputs(self, mock_generate_chat):
        """Test with valid city and days"""
        mock_generate_chat.return_value = (
            "Day 1:\n1. Visit Museum\n2. Lunch at Restaurant\n"
            "Day 2:\n1. Park tour\n2. Dinner downtown"
        )
        
        result = trip_planner.generate_trip_plan("Paris", "3", "LITELLM_SMART")
        
        assert isinstance(result, str)
        assert "Day" in result or "Museum" in result
        mock_generate_chat.assert_called_once()
        
        # Verify call args (passed as kwargs)
        _, kwargs = mock_generate_chat.call_args
        assert "conversation" in kwargs
        conversation = kwargs["conversation"]
        assert len(conversation) > 1
        assert "Paris" in conversation[-1]["content"]
        assert "3" in conversation[-1]["content"]

    @patch('helper_functions.trip_planner.generate_chat')
    def test_generate_trip_plan_integer_days(self, mock_generate_chat):
        """Test with integer days input"""
        mock_generate_chat.return_value = "Trip plan"
        
        result = trip_planner.generate_trip_plan("Tokyo", 5)
        
        assert isinstance(result, str)
        mock_generate_chat.assert_called_once()

    @patch('helper_functions.trip_planner.generate_chat')
    def test_generate_trip_plan_default_model(self, mock_generate_chat):
        """Test with default model"""
        mock_generate_chat.return_value = "Trip plan"
        
        result = trip_planner.generate_trip_plan("London", "4")
        
        # Verify default model is used
        _, kwargs = mock_generate_chat.call_args
        assert kwargs["model_name"] == "LITELLM_SMART"

    @patch('helper_functions.trip_planner.generate_chat')
    def test_generate_trip_plan_custom_model(self, mock_generate_chat):
        """Test with custom model"""
        mock_generate_chat.return_value = "Trip plan"
        
        result = trip_planner.generate_trip_plan("Barcelona", "7", "OLLAMA_STRATEGIC")
        
        # Verify custom model is used
        _, kwargs = mock_generate_chat.call_args
        assert kwargs["model_name"] == "OLLAMA_STRATEGIC"

    @patch('helper_functions.trip_planner.generate_chat')
    def test_generate_trip_plan_non_numeric_days(self, mock_generate_chat):
        """Test with non-numeric days"""
        result = trip_planner.generate_trip_plan("Rome", "three")
        
        assert "Please enter a number for days" in result
        mock_generate_chat.assert_not_called()

    @patch('helper_functions.trip_planner.generate_chat')
    def test_generate_trip_plan_negative_days(self, mock_generate_chat):
        """Test with negative days"""
        # Logic allows negative days technically but creates a plan for it
        mock_generate_chat.return_value = "Trip plan"
        result = trip_planner.generate_trip_plan("Berlin", "-2")
        
        assert isinstance(result, str)
        mock_generate_chat.assert_called_once()

    @patch('helper_functions.trip_planner.generate_chat')
    def test_generate_trip_plan_zero_days(self, mock_generate_chat):
        """Test with zero days"""
        mock_generate_chat.return_value = "Trip plan"
        result = trip_planner.generate_trip_plan("Madrid", "0")
        
        assert isinstance(result, str)
        mock_generate_chat.assert_called_once()

    @patch('helper_functions.trip_planner.generate_chat')
    def test_generate_trip_plan_system_prompt(self, mock_generate_chat):
        """Test that system prompt includes itinerary components"""
        mock_generate_chat.return_value = "Trip plan"
        
        trip_planner.generate_trip_plan("Vienna", "4", "LITELLM_SMART")
        
        _, kwargs = mock_generate_chat.call_args
        conversation = kwargs["conversation"]
        assert conversation[0]["role"] == "system"
        assert "world class trip planner" in conversation[0]["content"]

    @patch('helper_functions.trip_planner.generate_chat')
    def test_generate_trip_plan_numbered_list(self, mock_generate_chat):
        """Test that output is a numbered list format"""
        mock_generate_chat.return_value = "Trip plan"
        
        trip_planner.generate_trip_plan("Prague", "3", "LITELLM_SMART")
        
        _, kwargs = mock_generate_chat.call_args
        conversation = kwargs["conversation"]
        user_content = conversation[-1]["content"]
        assert "numbered list" in user_content

    @patch('helper_functions.trip_planner.generate_chat')
    def test_generate_trip_plan_includes_components(self, mock_generate_chat):
        """Test that request includes attractions, restaurants, hotels"""
        mock_generate_chat.return_value = "Trip plan"
        
        trip_planner.generate_trip_plan("Budapest", "5", "LITELLM_SMART")
        
        _, kwargs = mock_generate_chat.call_args
        conversation = kwargs["conversation"]
        user_content = conversation[-1]["content"]
        assert "tourist attractions" in user_content
        assert "restaurants" in user_content
        assert "hotels" in user_content or "resorts" in user_content

    @patch('helper_functions.trip_planner.generate_chat')
    def test_generate_trip_plan_error_handling(self, mock_generate_chat):
        """Test error handling during generation"""
        mock_generate_chat.side_effect = Exception("API Error")
        
        result = trip_planner.generate_trip_plan("Dubai", "4")
        
        assert "Error generating trip plan" in result
        assert "API Error" in result

    @patch('helper_functions.trip_planner.generate_chat')
    def test_generate_trip_plan_empty_city(self, mock_generate_chat):
        """Test with empty city name"""
        mock_generate_chat.return_value = "Trip plan"
        
        result = trip_planner.generate_trip_plan("", "3")
        
        assert isinstance(result, str)
        _, kwargs = mock_generate_chat.call_args
        conversation = kwargs["conversation"]
        assert "" in conversation[-1]["content"] # Should just contain empty string in prompt

    @patch('helper_functions.trip_planner.generate_chat')
    def test_generate_trip_plan_special_characters(self, mock_generate_chat):
        """Test with special characters in city"""
        mock_generate_chat.return_value = "Trip plan"
        
        trip_planner.generate_trip_plan("New York & New Jersey", "5")
        
        _, kwargs = mock_generate_chat.call_args
        conversation = kwargs["conversation"]
        assert "New York & New Jersey" in conversation[-1]["content"]

    @patch('helper_functions.trip_planner.generate_chat')
    def test_generate_trip_plan_large_number_days(self, mock_generate_chat):
        """Test with large number of days"""
        mock_generate_chat.return_value = "Trip plan"
        
        trip_planner.generate_trip_plan("Sydney", "30")
        
        _, kwargs = mock_generate_chat.call_args
        conversation = kwargs["conversation"]
        assert "30" in conversation[-1]["content"]

    @patch('helper_functions.trip_planner.generate_chat')
    def test_generate_trip_plan_temperature_and_tokens(self, mock_generate_chat):
        """Test that temperature and max_tokens are set correctly"""
        mock_generate_chat.return_value = "Trip plan"
        
        trip_planner.generate_trip_plan("Istanbul", "6", "LITELLM_SMART")
        
        _, kwargs = mock_generate_chat.call_args
        assert kwargs["temperature"] == 0.3
        assert kwargs["max_tokens"] == 2048
