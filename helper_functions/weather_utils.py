import logging
from pyowm import OWM

from config import config

logger = logging.getLogger(__name__)


def get_weather():
    try:
        owm = OWM(config.pyowm_api_key)
        mgr = owm.weather_manager()
        weather = mgr.weather_at_id(5743413).weather  # North Plains, OR
        temp = weather.temperature("celsius")["temp"]
        status = weather.detailed_status
        return temp, status
    except Exception as e:
        logger.error("Error fetching weather: %s", e)
        return "N/A", "Unknown"
