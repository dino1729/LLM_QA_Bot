import logging
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from pyowm import OWM

from config import config

logger = logging.getLogger(__name__)


def get_weather():
    """Fetch current weather for the configured location."""
    try:
        owm = OWM(config.pyowm_api_key)
        mgr = owm.weather_manager()
        weather = mgr.weather_at_id(config.pyowm_city_id).weather
        temp = weather.temperature("celsius")["temp"]
        status = weather.detailed_status
        return temp, status
    except Exception as e:
        logger.error("Error fetching weather: %s", e)
        return "N/A", "Unknown"


def get_weather_location() -> str:
    """Return the configured weather location name."""
    return config.pyowm_city_name or "Unknown"


def _aggregate_daily_from_3h_forecasts(forecasts: List, target_date: datetime.date) -> Optional[Dict[str, Any]]:
    """
    Aggregate 3-hour forecast intervals into daily high/low/status/precip.
    
    Args:
        forecasts: List of PyOWM Weather objects from 5-day/3h forecast
        target_date: The date to aggregate for
        
    Returns:
        Dict with high, low, status, precip_probability, wind or None if no data
    """
    temps = []
    statuses = []
    precip_probs = []
    wind_speeds = []
    
    for fc in forecasts:
        try:
            fc_time = fc.reference_time('date')
            if fc_time.date() == target_date:
                temp_data = fc.temperature('celsius')
                temps.append(temp_data.get('temp', temp_data.get('day')))
                statuses.append(fc.detailed_status)
                # precipitation_probability is available in 5-day forecast as 'pop'
                pop = getattr(fc, 'precipitation_probability', None)
                if pop is None:
                    # Try alternate access method
                    pop = fc.to_dict().get('precipitation_probability', 0.0)
                precip_probs.append(pop if pop else 0.0)
                wind_data = fc.wind()
                if wind_data:
                    wind_speeds.append(wind_data.get('speed', 0))
        except Exception as e:
            logger.debug("Error processing forecast interval: %s", e)
            continue
    
    if not temps:
        return None
    
    # Find most common status (mode) for the day
    status_counts = defaultdict(int)
    for s in statuses:
        status_counts[s] += 1
    dominant_status = max(status_counts, key=status_counts.get) if status_counts else "unknown"
    
    return {
        "high": max(temps),
        "low": min(temps),
        "status": dominant_status,
        "precip_probability": max(precip_probs) if precip_probs else 0.0,
        "wind": {"speed": sum(wind_speeds) / len(wind_speeds) if wind_speeds else 0},
    }


def _format_outlook(
    today: Optional[Dict[str, Any]],
    tomorrow: Optional[Dict[str, Any]],
    alerts: List[Dict[str, str]],
) -> str:
    """Build a compact, spoken-friendly weather outlook string."""
    parts = []

    if today:
        today_str = f"Today: {today['status']}, high {today['high']:.0f}C, low {today['low']:.0f}C"
        if today.get("precip_probability", 0) >= 0.3:
            today_str += f" ({int(today['precip_probability'] * 100)}% chance of precipitation)"
        parts.append(today_str)

    if tomorrow:
        tom_str = f"Tomorrow: {tomorrow['status']}, high {tomorrow['high']:.0f}C, low {tomorrow['low']:.0f}C"
        if tomorrow.get("precip_probability", 0) >= 0.3:
            tom_str += f" ({int(tomorrow['precip_probability'] * 100)}% precipitation risk)"
        parts.append(tom_str)

    if alerts:
        alert_events = [a["event"] for a in alerts[:2]]  # Limit to 2 alerts
        parts.append("Weather alert: " + "; ".join(alert_events))

    return ". ".join(parts)


def get_weather_forecast() -> Dict[str, Any]:
    """
    Fetch weather forecast for configured location using PyOWM free 5-day/3-hour API.
    Aggregates 3-hour intervals into daily summaries for today and tomorrow.
    
    Returns:
        Dict with keys:
            - outlook: str - compact spoken-friendly forecast summary
            - today: dict or None - today's forecast data (high, low, status, precip_probability)
            - tomorrow: dict or None - tomorrow's forecast data
            - alerts: list - always empty for free API (alerts require One Call)
    """
    fallback = {
        "outlook": "",
        "today": None,
        "tomorrow": None,
        "alerts": [],
    }

    try:
        owm = OWM(config.pyowm_api_key)
        mgr = owm.weather_manager()
        
        # Use free 5-day/3-hour forecast API (no subscription required)
        forecast = mgr.forecast_at_id(config.pyowm_city_id, '3h')
        forecasts = forecast.forecast.weathers
        
        today = datetime.now().date()
        tomorrow = today + timedelta(days=1)
        
        today_data = _aggregate_daily_from_3h_forecasts(forecasts, today)
        tomorrow_data = _aggregate_daily_from_3h_forecasts(forecasts, tomorrow)
        
        # Free API doesn't include alerts - that's One Call only
        alerts = []
        
        outlook = _format_outlook(today_data, tomorrow_data, alerts)
        
        return {
            "outlook": outlook,
            "today": today_data,
            "tomorrow": tomorrow_data,
            "alerts": alerts,
        }
        
    except Exception as e:
        logger.warning("Failed to fetch weather forecast (5-day/3h): %s", e)
        return fallback
