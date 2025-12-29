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
    
    @patch('helper_functions.weather_utils.config')
    @patch('helper_functions.weather_utils.OWM')
    def test_get_weather_correct_location_id(self, mock_owm, mock_config):
        """Test that get_weather queries the configured location ID"""
        mock_config.pyowm_api_key = "test_key"
        mock_config.pyowm_city_id = 5743413  # North Plains, OR
        
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
        
        # Verify the configured location ID is used
        mock_manager.weather_at_id.assert_called_once_with(5743413)


class TestGetWeatherLocation:
    """Tests for get_weather_location() function"""
    
    @patch('helper_functions.weather_utils.config')
    def test_get_weather_location_returns_configured_name(self, mock_config):
        """Test that get_weather_location returns the configured city name"""
        mock_config.pyowm_city_name = "Portland, OR"
        
        result = weather_utils.get_weather_location()
        
        assert result == "Portland, OR"
    
    @patch('helper_functions.weather_utils.config')
    def test_get_weather_location_returns_unknown_when_not_configured(self, mock_config):
        """Test that get_weather_location returns 'Unknown' when not configured"""
        mock_config.pyowm_city_name = None
        
        result = weather_utils.get_weather_location()
        
        assert result == "Unknown"


class TestGetWeatherForecast:
    """Tests for get_weather_forecast() function - Free 5-day/3h API based forecast"""
    
    def _create_mock_3h_forecast(self, temp, status, ref_time, pop=0.0):
        """Helper to create a mock 3-hour forecast interval object"""
        from datetime import datetime
        mock_fc = Mock()
        mock_fc.temperature.return_value = {"temp": temp}
        mock_fc.detailed_status = status
        mock_fc.precipitation_probability = pop
        mock_fc.wind.return_value = {"speed": 5.0}
        mock_fc.reference_time.return_value = ref_time
        mock_fc.to_dict.return_value = {"precipitation_probability": pop}
        return mock_fc
    
    @patch('helper_functions.weather_utils.datetime')
    @patch('helper_functions.weather_utils.OWM')
    def test_get_weather_forecast_success(self, mock_owm, mock_datetime):
        """Test successful weather forecast retrieval with today and tomorrow data"""
        from datetime import datetime, timedelta
        
        # Mock current date
        today = datetime(2025, 12, 28)
        tomorrow = today + timedelta(days=1)
        mock_datetime.now.return_value = today
        mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)
        
        # Create mock 3-hour forecasts for today and tomorrow
        today_forecasts = [
            self._create_mock_3h_forecast(8.0, "partly cloudy", datetime(2025, 12, 28, 9, 0)),
            self._create_mock_3h_forecast(15.0, "partly cloudy", datetime(2025, 12, 28, 12, 0)),
            self._create_mock_3h_forecast(12.0, "partly cloudy", datetime(2025, 12, 28, 15, 0)),
        ]
        tomorrow_forecasts = [
            self._create_mock_3h_forecast(6.0, "light rain", datetime(2025, 12, 29, 9, 0), 0.8),
            self._create_mock_3h_forecast(12.0, "light rain", datetime(2025, 12, 29, 12, 0), 0.8),
        ]
        
        mock_forecast_obj = Mock()
        mock_forecast_obj.forecast.weathers = today_forecasts + tomorrow_forecasts
        
        mock_owm_instance = Mock()
        mock_owm_instance.weather_manager.return_value.forecast_at_id.return_value = mock_forecast_obj
        mock_owm.return_value = mock_owm_instance
        
        result = weather_utils.get_weather_forecast()
        
        # Should return a dict with outlook string and structured data
        assert isinstance(result, dict)
        assert "outlook" in result
        assert "today" in result
        assert "tomorrow" in result
        
        # Outlook should mention both days
        assert "Today" in result["outlook"] or "today" in result["outlook"].lower()
        assert "Tomorrow" in result["outlook"] or "tomorrow" in result["outlook"].lower()
        
        # Today data should have expected structure (high=15, low=8)
        assert result["today"]["high"] == 15.0
        assert result["today"]["low"] == 8.0
        assert result["today"]["status"] == "partly cloudy"
        
        # Tomorrow data should have expected structure (high=12, low=6)
        assert result["tomorrow"]["high"] == 12.0
        assert result["tomorrow"]["low"] == 6.0
    
    @patch('helper_functions.weather_utils.datetime')
    @patch('helper_functions.weather_utils.OWM')
    def test_get_weather_forecast_includes_precip_probability(self, mock_owm, mock_datetime):
        """Test that forecast includes precipitation probability when significant"""
        from datetime import datetime, timedelta
        
        today = datetime(2025, 12, 28)
        mock_datetime.now.return_value = today
        mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)
        
        # Create forecasts with high precip for today
        forecasts = [
            self._create_mock_3h_forecast(10.0, "rain", datetime(2025, 12, 28, 9, 0), 0.75),
            self._create_mock_3h_forecast(18.0, "rain", datetime(2025, 12, 28, 12, 0), 0.75),
            self._create_mock_3h_forecast(12.0, "cloudy", datetime(2025, 12, 29, 12, 0), 0.1),
        ]
        
        mock_forecast_obj = Mock()
        mock_forecast_obj.forecast.weathers = forecasts
        
        mock_owm_instance = Mock()
        mock_owm_instance.weather_manager.return_value.forecast_at_id.return_value = mock_forecast_obj
        mock_owm.return_value = mock_owm_instance
        
        result = weather_utils.get_weather_forecast()
        
        # High precip probability should be mentioned in outlook
        assert "75%" in result["outlook"] or "rain" in result["outlook"].lower()
        assert result["today"]["precip_probability"] == 0.75
    
    @patch('helper_functions.weather_utils.datetime')
    @patch('helper_functions.weather_utils.OWM')
    def test_get_weather_forecast_no_alerts_in_free_api(self, mock_owm, mock_datetime):
        """Test that free API returns empty alerts (alerts require One Call)"""
        from datetime import datetime
        
        today = datetime(2025, 12, 28)
        mock_datetime.now.return_value = today
        mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)
        
        forecasts = [
            self._create_mock_3h_forecast(5.0, "snow", datetime(2025, 12, 28, 12, 0), 0.9),
        ]
        
        mock_forecast_obj = Mock()
        mock_forecast_obj.forecast.weathers = forecasts
        
        mock_owm_instance = Mock()
        mock_owm_instance.weather_manager.return_value.forecast_at_id.return_value = mock_forecast_obj
        mock_owm.return_value = mock_owm_instance
        
        result = weather_utils.get_weather_forecast()
        
        # Free API doesn't include alerts
        assert "alerts" in result
        assert result["alerts"] == []
    
    @patch('helper_functions.weather_utils.OWM')
    def test_get_weather_forecast_api_error_returns_fallback(self, mock_owm):
        """Test that API errors return graceful fallback (empty outlook)"""
        mock_owm.side_effect = Exception("API not available")
        
        result = weather_utils.get_weather_forecast()
        
        # Should return empty/fallback structure, not crash
        assert isinstance(result, dict)
        assert result["outlook"] == ""
        assert result["today"] is None
        assert result["tomorrow"] is None
        assert result["alerts"] == []
    
    @patch('helper_functions.weather_utils.OWM')
    def test_get_weather_forecast_permission_error(self, mock_owm):
        """Test graceful degradation on permission errors"""
        mock_owm_instance = Mock()
        mock_owm_instance.weather_manager.return_value.forecast_at_id.side_effect = Exception(
            "Unauthorized"
        )
        mock_owm.return_value = mock_owm_instance
        
        result = weather_utils.get_weather_forecast()
        
        # Should return empty/fallback structure
        assert isinstance(result, dict)
        assert result["outlook"] == ""
        assert result["today"] is None
        assert result["tomorrow"] is None
    
    @patch('helper_functions.weather_utils.datetime')
    @patch('helper_functions.weather_utils.OWM')
    def test_get_weather_forecast_empty_forecast_list(self, mock_owm, mock_datetime):
        """Test handling of empty forecast list"""
        from datetime import datetime
        
        mock_datetime.now.return_value = datetime(2025, 12, 28)
        mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)
        
        mock_forecast_obj = Mock()
        mock_forecast_obj.forecast.weathers = []
        
        mock_owm_instance = Mock()
        mock_owm_instance.weather_manager.return_value.forecast_at_id.return_value = mock_forecast_obj
        mock_owm.return_value = mock_owm_instance
        
        result = weather_utils.get_weather_forecast()
        
        # Should handle gracefully
        assert isinstance(result, dict)
        assert result["today"] is None
        assert result["tomorrow"] is None
    
    @patch('helper_functions.weather_utils.datetime')
    @patch('helper_functions.weather_utils.OWM')
    def test_get_weather_forecast_only_today_available(self, mock_owm, mock_datetime):
        """Test when only today's forecast is available (no tomorrow)"""
        from datetime import datetime
        
        today = datetime(2025, 12, 28)
        mock_datetime.now.return_value = today
        mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)
        
        forecasts = [
            self._create_mock_3h_forecast(10.0, "sunny", datetime(2025, 12, 28, 9, 0)),
            self._create_mock_3h_forecast(20.0, "sunny", datetime(2025, 12, 28, 15, 0)),
        ]
        
        mock_forecast_obj = Mock()
        mock_forecast_obj.forecast.weathers = forecasts
        
        mock_owm_instance = Mock()
        mock_owm_instance.weather_manager.return_value.forecast_at_id.return_value = mock_forecast_obj
        mock_owm.return_value = mock_owm_instance
        
        result = weather_utils.get_weather_forecast()
        
        # Today should be present, tomorrow should be None
        assert result["today"] is not None
        assert result["today"]["high"] == 20.0
        assert result["today"]["low"] == 10.0
        assert result["tomorrow"] is None
    
    @patch('helper_functions.weather_utils.datetime')
    @patch('helper_functions.weather_utils.OWM')
    @patch('helper_functions.weather_utils.config')
    def test_get_weather_forecast_uses_correct_city_id(self, mock_config, mock_owm, mock_datetime):
        """Test that forecast uses the configured city ID"""
        from datetime import datetime
        
        mock_config.pyowm_api_key = "test_key"
        mock_config.pyowm_city_id = 5743413  # North Plains, OR
        mock_datetime.now.return_value = datetime(2025, 12, 28)
        mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)
        
        forecasts = [
            self._create_mock_3h_forecast(10.0, "clear", datetime(2025, 12, 28, 12, 0)),
        ]
        
        mock_forecast_obj = Mock()
        mock_forecast_obj.forecast.weathers = forecasts
        
        mock_weather_mgr = Mock()
        mock_weather_mgr.forecast_at_id.return_value = mock_forecast_obj
        
        mock_owm_instance = Mock()
        mock_owm_instance.weather_manager.return_value = mock_weather_mgr
        mock_owm.return_value = mock_owm_instance
        
        weather_utils.get_weather_forecast()
        
        # Verify forecast_at_id was called with the configured city ID
        mock_weather_mgr.forecast_at_id.assert_called_once_with(5743413, '3h')
