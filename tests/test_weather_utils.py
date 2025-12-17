"""
Tests for weather_utils.py - Weather information retrieval
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from helper_functions import weather_utils


class TestGetWeather:
    """Tests for get_weather() function"""
    
    @patch('helper_functions.weather_utils.OWM')
    def test_get_weather_success(self, mock_owm):
        """Test successful weather retrieval"""
        # Setup mock chain: OWM -> weather_manager() -> weather_at_id() -> weather
        mock_weather = Mock()
        mock_weather.temperature.return_value = {"temp": 22.5}
        mock_weather.detailed_status = "clear sky"
        
        mock_observation = Mock()
        mock_observation.weather = mock_weather
        
        mock_manager = Mock()
        mock_manager.weather_at_id.return_value = mock_observation
        
        mock_owm_instance = Mock()
        mock_owm_instance.weather_manager.return_value = mock_manager
        mock_owm.return_value = mock_owm_instance
        
        temp, status = weather_utils.get_weather()
        
        assert temp == 22.5
        assert status == "clear sky"
        mock_manager.weather_at_id.assert_called_once_with(5743413)
        mock_weather.temperature.assert_called_once_with("celsius")
    
    @patch('helper_functions.weather_utils.OWM')
    def test_get_weather_freezing_temp(self, mock_owm):
        """Test weather retrieval with freezing temperature"""
        mock_weather = Mock()
        mock_weather.temperature.return_value = {"temp": -5.0}
        mock_weather.detailed_status = "light snow"
        
        mock_observation = Mock()
        mock_observation.weather = mock_weather
        
        mock_manager = Mock()
        mock_manager.weather_at_id.return_value = mock_observation
        
        mock_owm_instance = Mock()
        mock_owm_instance.weather_manager.return_value = mock_manager
        mock_owm.return_value = mock_owm_instance
        
        temp, status = weather_utils.get_weather()
        
        assert temp == -5.0
        assert status == "light snow"
    
    @patch('helper_functions.weather_utils.OWM')
    def test_get_weather_hot_temp(self, mock_owm):
        """Test weather retrieval with hot temperature"""
        mock_weather = Mock()
        mock_weather.temperature.return_value = {"temp": 38.0}
        mock_weather.detailed_status = "sunny"
        
        mock_observation = Mock()
        mock_observation.weather = mock_weather
        
        mock_manager = Mock()
        mock_manager.weather_at_id.return_value = mock_observation
        
        mock_owm_instance = Mock()
        mock_owm_instance.weather_manager.return_value = mock_manager
        mock_owm.return_value = mock_owm_instance
        
        temp, status = weather_utils.get_weather()
        
        assert temp == 38.0
        assert status == "sunny"
    
    @patch('helper_functions.weather_utils.OWM')
    def test_get_weather_api_error(self, mock_owm):
        """Test weather retrieval with API error"""
        mock_owm.side_effect = Exception("API connection failed")
        
        temp, status = weather_utils.get_weather()
        
        assert temp == "N/A"
        assert status == "Unknown"
    
    @patch('helper_functions.weather_utils.OWM')
    def test_get_weather_invalid_api_key(self, mock_owm):
        """Test weather retrieval with invalid API key"""
        mock_owm.side_effect = Exception("Invalid API key")
        
        temp, status = weather_utils.get_weather()
        
        assert temp == "N/A"
        assert status == "Unknown"
    
    @patch('helper_functions.weather_utils.OWM')
    def test_get_weather_network_timeout(self, mock_owm):
        """Test weather retrieval with network timeout"""
        mock_manager = Mock()
        mock_manager.weather_at_id.side_effect = Exception("Network timeout")
        
        mock_owm_instance = Mock()
        mock_owm_instance.weather_manager.return_value = mock_manager
        mock_owm.return_value = mock_owm_instance
        
        temp, status = weather_utils.get_weather()
        
        assert temp == "N/A"
        assert status == "Unknown"
    
    @patch('helper_functions.weather_utils.OWM')
    def test_get_weather_missing_temperature_data(self, mock_owm):
        """Test weather retrieval when temperature data is missing"""
        mock_weather = Mock()
        mock_weather.temperature.side_effect = KeyError("temp")
        
        mock_observation = Mock()
        mock_observation.weather = mock_weather
        
        mock_manager = Mock()
        mock_manager.weather_at_id.return_value = mock_observation
        
        mock_owm_instance = Mock()
        mock_owm_instance.weather_manager.return_value = mock_manager
        mock_owm.return_value = mock_owm_instance
        
        temp, status = weather_utils.get_weather()
        
        assert temp == "N/A"
        assert status == "Unknown"
    
    @patch('helper_functions.weather_utils.OWM')
    def test_get_weather_various_statuses(self, mock_owm):
        """Test weather retrieval with various weather statuses"""
        statuses = [
            ("cloudy", 15.0),
            ("partly cloudy", 20.0),
            ("overcast clouds", 18.5),
            ("light rain", 12.0),
            ("heavy rain", 10.0),
            ("thunderstorm", 16.0),
            ("fog", 8.0),
            ("mist", 9.5)
        ]
        
        for expected_status, expected_temp in statuses:
            mock_weather = Mock()
            mock_weather.temperature.return_value = {"temp": expected_temp}
            mock_weather.detailed_status = expected_status
            
            mock_observation = Mock()
            mock_observation.weather = mock_weather
            
            mock_manager = Mock()
            mock_manager.weather_at_id.return_value = mock_observation
            
            mock_owm_instance = Mock()
            mock_owm_instance.weather_manager.return_value = mock_manager
            mock_owm.return_value = mock_owm_instance
            
            temp, status = weather_utils.get_weather()
            
            assert temp == expected_temp
            assert status == expected_status
    
    @patch('helper_functions.weather_utils.OWM')
    @patch('helper_functions.weather_utils.config')
    def test_get_weather_uses_config_api_key(self, mock_config, mock_owm):
        """Test that get_weather uses API key from config"""
        mock_config.pyowm_api_key = "test_api_key_12345"
        
        mock_weather = Mock()
        mock_weather.temperature.return_value = {"temp": 20.0}
        mock_weather.detailed_status = "clear sky"
        
        mock_observation = Mock()
        mock_observation.weather = mock_weather
        
        mock_manager = Mock()
        mock_manager.weather_at_id.return_value = mock_observation
        
        mock_owm_instance = Mock()
        mock_owm_instance.weather_manager.return_value = mock_manager
        mock_owm.return_value = mock_owm_instance
        
        weather_utils.get_weather()
        
        mock_owm.assert_called_once_with("test_api_key_12345")
    
    @patch('helper_functions.weather_utils.OWM')
    def test_get_weather_correct_location_id(self, mock_owm):
        """Test that get_weather queries the correct location ID (North Plains, OR)"""
        mock_weather = Mock()
        mock_weather.temperature.return_value = {"temp": 18.0}
        mock_weather.detailed_status = "clear"
        
        mock_observation = Mock()
        mock_observation.weather = mock_weather
        
        mock_manager = Mock()
        mock_manager.weather_at_id.return_value = mock_observation
        
        mock_owm_instance = Mock()
        mock_owm_instance.weather_manager.return_value = mock_manager
        mock_owm.return_value = mock_owm_instance
        
        weather_utils.get_weather()
        
        # Verify the correct location ID for North Plains, OR is used
        mock_manager.weather_at_id.assert_called_once_with(5743413)

