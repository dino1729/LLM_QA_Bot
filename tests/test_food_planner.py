import pytest
from unittest.mock import Mock, patch
from helper_functions import food_planner

class TestCravingSatisfier:
    """Tests for craving_satisfier() function"""
    
    @patch('helper_functions.food_planner.generate_chat')
    def test_craving_satisfier_specific_food(self, mock_generate_chat):
        """Test with specific food craving"""
        mock_generate_chat.return_value = "1. Restaurant A\n2. Restaurant B\n3. Restaurant C"
        
        result = food_planner.craving_satisfier("New York", "pizza", "LITELLM_SMART")
        
        assert isinstance(result, str)
        assert "Restaurant" in result
        mock_generate_chat.assert_called_once()
        
        # Verify conversation structure (passed as kwargs)
        _, kwargs = mock_generate_chat.call_args
        conversation = kwargs["conversation"]
        assert "New York" in conversation[-1]["content"]
        assert "pizza" in conversation[-1]["content"]

    @patch('helper_functions.food_planner.generate_chat')
    def test_craving_satisfier_idk(self, mock_generate_chat):
        """Test with 'idk' for random restaurant selection"""
        # It calls generate_chat TWICE: once for cuisine, once for restaurants
        mock_generate_chat.side_effect = [
            "Thai", # Cuisine recommendation
            "1. Thai Restaurant\n2. Spicy Place" # Restaurant list
        ]
        
        result = food_planner.craving_satisfier("Los Angeles", "idk", "LITELLM_SMART")
        
        assert isinstance(result, str)
        assert mock_generate_chat.call_count == 2
        
        # Verify first call (cuisine generation)
        first_call = mock_generate_chat.call_args_list[0]
        _, kwargs1 = first_call
        assert "random cuisine" in kwargs1["conversation"][-1]["content"]
        
        # Verify second call (restaurant list)
        second_call = mock_generate_chat.call_args_list[1]
        _, kwargs2 = second_call
        assert "Thai" in kwargs2["conversation"][-1]["content"]

    @patch('helper_functions.food_planner.generate_chat')
    def test_craving_satisfier_default_model(self, mock_generate_chat):
        """Test with default model"""
        mock_generate_chat.return_value = "Restaurant list"
        
        result = food_planner.craving_satisfier("Boston", "seafood")
        
        # Verify default model is used
        _, kwargs = mock_generate_chat.call_args
        assert kwargs["model_name"] == "LITELLM_SMART"

    @patch('helper_functions.food_planner.generate_chat')
    def test_craving_satisfier_custom_model(self, mock_generate_chat):
        """Test with custom model"""
        mock_generate_chat.return_value = "Restaurant list"
        
        result = food_planner.craving_satisfier("Chicago", "burger", "OLLAMA_FAST")
        
        # Verify custom model is used
        _, kwargs = mock_generate_chat.call_args
        assert kwargs["model_name"] == "OLLAMA_FAST"

    @patch('helper_functions.food_planner.generate_chat')
    def test_craving_satisfier_system_prompt(self, mock_generate_chat):
        """Test system prompt includes dietary restrictions"""
        mock_generate_chat.return_value = "Restaurant list"
        
        food_planner.craving_satisfier("Miami", "sushi", "LITELLM_SMART")
        
        _, kwargs = mock_generate_chat.call_args
        conversation = kwargs["conversation"]
        user_msg = conversation[-1]["content"]
        assert "neither beef nor pork" in user_msg

    @patch('helper_functions.food_planner.generate_chat')
    def test_craving_satisfier_eight_restaurants(self, mock_generate_chat):
        """Test that it requests 8 restaurants"""
        mock_generate_chat.return_value = "8 restaurants"
        
        food_planner.craving_satisfier("Seattle", "coffee", "LITELLM_SMART")
        
        _, kwargs = mock_generate_chat.call_args
        conversation = kwargs["conversation"]
        user_msg = conversation[-1]["content"]
        assert "8 restaurants" in user_msg

    @patch('helper_functions.food_planner.generate_chat')
    def test_craving_satisfier_error_handling(self, mock_generate_chat):
        """Test error handling"""
        mock_generate_chat.side_effect = Exception("API Error")
        
        result = food_planner.craving_satisfier("Austin", "tacos")
        
        assert "Error generating food plan" in result
        assert "API Error" in result

    @patch('helper_functions.food_planner.generate_chat')
    def test_craving_satisfier_empty_city(self, mock_generate_chat):
        """Test with empty city"""
        mock_generate_chat.return_value = "List"
        
        food_planner.craving_satisfier("", "food")
        
        _, kwargs = mock_generate_chat.call_args
        conversation = kwargs["conversation"]
        assert "" in conversation[-1]["content"] # Should contain empty city string

    @patch('helper_functions.food_planner.generate_chat')
    def test_craving_satisfier_empty_craving(self, mock_generate_chat):
        """Test with empty craving"""
        # Should treat like specific craving but empty string
        mock_generate_chat.return_value = "List"
        
        food_planner.craving_satisfier("City", "")
        
        _, kwargs = mock_generate_chat.call_args
        conversation = kwargs["conversation"]
        assert "" in conversation[-1]["content"]

    @patch('helper_functions.food_planner.generate_chat')
    def test_craving_satisfier_special_characters(self, mock_generate_chat):
        """Test with special characters"""
        mock_generate_chat.return_value = "List"
        
        food_planner.craving_satisfier("San Francisco", "dim sum & tea")
        
        _, kwargs = mock_generate_chat.call_args
        conversation = kwargs["conversation"]
        assert "dim sum & tea" in conversation[-1]["content"]

    @patch('helper_functions.food_planner.generate_chat')
    def test_craving_satisfier_long_city_name(self, mock_generate_chat):
        """Test with long city name"""
        mock_generate_chat.return_value = "List"
        
        long_city = "Llanfairpwllgwyngyllgogerychwyrndrobwllllantysiliogogogoch"
        food_planner.craving_satisfier(long_city, "food")
        
        _, kwargs = mock_generate_chat.call_args
        conversation = kwargs["conversation"]
        assert long_city in conversation[-1]["content"]

    @patch('helper_functions.food_planner.generate_chat')
    def test_craving_satisfier_temperature_and_tokens(self, mock_generate_chat):
        """Test that temperature and max_tokens are set correctly"""
        mock_generate_chat.return_value = "Restaurant list"
        
        food_planner.craving_satisfier("Austin", "BBQ", "LITELLM_SMART")
        
        _, kwargs = mock_generate_chat.call_args
        # Temperature seems to be 0.4 based on previous failures
        assert kwargs["temperature"] == 0.4
        assert kwargs["max_tokens"] == 2048
