import json
import os
import logging
from datetime import datetime
from pathlib import Path
import threading

# Configure logging
logger = logging.getLogger(__name__)

# Constants
CACHE_DIR = Path("web_search_cache")

# Global variables for session management
_session_file = None
_file_lock = threading.Lock()

def _get_session_file():
    """Get or create the session file path for the current process."""
    global _session_file
    if _session_file is None:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pid = os.getpid()
        _session_file = CACHE_DIR / f"debug_log_{timestamp}_{pid}.json"
        
        # Initialize with empty list
        with _file_lock:
            if not _session_file.exists():
                with open(_session_file, 'w', encoding='utf-8') as f:
                    json.dump([], f)
                    
    return _session_file

def log_debug_data(event_type: str, data: dict):
    """
    Log debug data to the session JSON file.
    Appends the new entry to the list in the JSON file in a thread-safe manner.
    
    Args:
        event_type: String identifier for the event (e.g., "scrape", "search")
        data: Dictionary containing the data to log
    """
    session_file = _get_session_file()
    
    entry = {
        "timestamp": datetime.now().isoformat(),
        "type": event_type,
        "data": data
    }
    
    try:
        with _file_lock:
            # Read-Modify-Write cycle
            # NOTE: For very large logs/high frequency, this is inefficient.
            # But for web search debugging (dozens of requests), it's acceptable and ensures valid JSON.
            current_data = []
            if session_file.exists():
                try:
                    with open(session_file, 'r', encoding='utf-8') as f:
                        file_content = f.read()
                        if file_content.strip():
                            current_data = json.loads(file_content)
                except json.JSONDecodeError:
                    logger.warning(f"Corrupt JSON in {session_file}, starting fresh list.")
                    current_data = []
            
            current_data.append(entry)
            
            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(current_data, f, indent=2, ensure_ascii=False)
                
    except Exception as e:
        logger.error(f"Failed to log debug data to {session_file}: {e}")
