"""
Tests for news_cache.py - News caching functionality
"""
import pytest
import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, mock_open
from helper_functions import news_cache


@pytest.fixture
def temp_cache_dir(tmp_path):
    """Create a temporary directory for cache testing"""
    cache_dir = tmp_path / "test_cache"
    cache_dir.mkdir(exist_ok=True)
    return cache_dir


@pytest.fixture
def sample_news_data():
    """Sample news data for testing"""
    return {
        "technology": "Latest tech news content",
        "financial": "Financial market updates",
        "india": "India news summary"
    }


class TestCachePath:
    """Tests for _cache_path() function"""
    
    def test_cache_path_default(self):
        """Test default cache path"""
        path = news_cache._cache_path()
        assert path.name == "news_cache.json"
        assert "newsletter_research_data" in str(path)
    
    def test_cache_path_custom_dir(self, temp_cache_dir):
        """Test cache path with custom directory"""
        path = news_cache._cache_path(temp_cache_dir)
        assert path.name == "news_cache.json"
        assert str(temp_cache_dir) in str(path)


class TestSaveNewsCache:
    """Tests for save_news_cache() function"""
    
    def test_save_news_cache_success(self, temp_cache_dir, sample_news_data):
        """Test successful cache save"""
        result = news_cache.save_news_cache(sample_news_data, temp_cache_dir)
        
        assert result is True
        cache_file = temp_cache_dir / "news_cache.json"
        assert cache_file.exists()
        
        # Verify content
        with open(cache_file, "r") as f:
            saved_data = json.load(f)
        
        assert "timestamp" in saved_data
        assert "date_iso" in saved_data
        assert "news" in saved_data
        assert saved_data["news"] == sample_news_data
    
    def test_save_news_cache_creates_directory(self, tmp_path, sample_news_data):
        """Test that cache save creates directory if it doesn't exist"""
        non_existent_dir = tmp_path / "nested" / "cache" / "dir"
        
        result = news_cache.save_news_cache(sample_news_data, non_existent_dir)
        
        assert result is True
        assert non_existent_dir.exists()
        assert (non_existent_dir / "news_cache.json").exists()
    
    def test_save_news_cache_with_unicode(self, temp_cache_dir):
        """Test cache save with Unicode content"""
        unicode_data = {
            "news": "测试新闻 🎉 Émojis and spëcial chars",
            "tech": "Künstliche Intelligenz"
        }
        
        result = news_cache.save_news_cache(unicode_data, temp_cache_dir)
        
        assert result is True
        cache_file = temp_cache_dir / "news_cache.json"
        with open(cache_file, "r", encoding="utf-8") as f:
            saved_data = json.load(f)
        
        assert saved_data["news"]["news"] == unicode_data["news"]
    
    def test_save_news_cache_empty_data(self, temp_cache_dir):
        """Test cache save with empty data"""
        empty_data = {}
        
        result = news_cache.save_news_cache(empty_data, temp_cache_dir)
        
        assert result is True
        cache_file = temp_cache_dir / "news_cache.json"
        with open(cache_file, "r") as f:
            saved_data = json.load(f)
        
        assert saved_data["news"] == {}
    
    def test_save_news_cache_overwrites_existing(self, temp_cache_dir, sample_news_data):
        """Test that save overwrites existing cache"""
        # Save first time
        news_cache.save_news_cache({"old": "data"}, temp_cache_dir)
        
        # Save again with new data
        result = news_cache.save_news_cache(sample_news_data, temp_cache_dir)
        
        assert result is True
        cache_file = temp_cache_dir / "news_cache.json"
        with open(cache_file, "r") as f:
            saved_data = json.load(f)
        
        assert saved_data["news"] == sample_news_data
        assert "old" not in saved_data["news"]
    
    @patch('helper_functions.news_cache.open', side_effect=PermissionError("Access denied"))
    def test_save_news_cache_permission_error(self, mock_open, temp_cache_dir, sample_news_data):
        """Test cache save with permission error"""
        result = news_cache.save_news_cache(sample_news_data, temp_cache_dir)
        
        assert result is False
    
    @patch('helper_functions.news_cache.json.dump', side_effect=Exception("JSON error"))
    def test_save_news_cache_json_error(self, mock_dump, temp_cache_dir, sample_news_data):
        """Test cache save with JSON serialization error"""
        result = news_cache.save_news_cache(sample_news_data, temp_cache_dir)
        
        assert result is False


class TestLoadNewsCache:
    """Tests for load_news_cache() function"""
    
    def test_load_news_cache_success(self, temp_cache_dir, sample_news_data):
        """Test successful cache load"""
        # Save cache first
        news_cache.save_news_cache(sample_news_data, temp_cache_dir)
        
        # Load cache
        loaded_data = news_cache.load_news_cache(temp_cache_dir)
        
        assert loaded_data is not None
        assert loaded_data == sample_news_data
    
    def test_load_news_cache_not_exists(self, temp_cache_dir):
        """Test loading non-existent cache"""
        loaded_data = news_cache.load_news_cache(temp_cache_dir)
        
        assert loaded_data is None
    
    def test_load_news_cache_expired(self, temp_cache_dir, sample_news_data):
        """Test loading expired cache"""
        # Save cache
        news_cache.save_news_cache(sample_news_data, temp_cache_dir)
        
        # Manually modify timestamp to be old
        cache_file = temp_cache_dir / "news_cache.json"
        with open(cache_file, "r") as f:
            cache_data = json.load(f)
        
        old_time = datetime.now() - timedelta(hours=10)
        cache_data["timestamp"] = old_time.isoformat()
        
        with open(cache_file, "w") as f:
            json.dump(cache_data, f)
        
        # Try to load with 6-hour max age
        loaded_data = news_cache.load_news_cache(temp_cache_dir, max_age_hours=6)
        
        assert loaded_data is None
    
    def test_load_news_cache_within_max_age(self, temp_cache_dir, sample_news_data):
        """Test loading cache within max age"""
        # Save cache
        news_cache.save_news_cache(sample_news_data, temp_cache_dir)
        
        # Load with generous max age
        loaded_data = news_cache.load_news_cache(temp_cache_dir, max_age_hours=24)
        
        assert loaded_data is not None
        assert loaded_data == sample_news_data
    
    def test_load_news_cache_different_day(self, temp_cache_dir, sample_news_data):
        """Test loading cache from a different day"""
        # Save cache
        news_cache.save_news_cache(sample_news_data, temp_cache_dir)
        
        # Manually modify date_iso to be yesterday
        cache_file = temp_cache_dir / "news_cache.json"
        with open(cache_file, "r") as f:
            cache_data = json.load(f)
        
        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        cache_data["date_iso"] = yesterday
        
        with open(cache_file, "w") as f:
            json.dump(cache_data, f)
        
        # Try to load
        loaded_data = news_cache.load_news_cache(temp_cache_dir)
        
        assert loaded_data is None
    
    def test_load_news_cache_corrupted_json(self, temp_cache_dir):
        """Test loading corrupted cache file"""
        cache_file = temp_cache_dir / "news_cache.json"
        
        # Write corrupted JSON
        with open(cache_file, "w") as f:
            f.write("{ invalid json content")
        
        loaded_data = news_cache.load_news_cache(temp_cache_dir)
        
        assert loaded_data is None
    
    def test_load_news_cache_missing_timestamp(self, temp_cache_dir):
        """Test loading cache with missing timestamp"""
        cache_file = temp_cache_dir / "news_cache.json"
        
        # Write cache without timestamp
        with open(cache_file, "w") as f:
            json.dump({"news": {"test": "data"}}, f)
        
        loaded_data = news_cache.load_news_cache(temp_cache_dir)
        
        assert loaded_data is None
    
    def test_load_news_cache_custom_max_age(self, temp_cache_dir, sample_news_data):
        """Test loading cache with custom max age"""
        # Save cache
        news_cache.save_news_cache(sample_news_data, temp_cache_dir)
        
        # Load with very short max age (should still pass since cache is fresh)
        loaded_data = news_cache.load_news_cache(temp_cache_dir, max_age_hours=1)
        
        assert loaded_data is not None


class TestIsCacheValid:
    """Tests for is_cache_valid() function"""
    
    def test_is_cache_valid_true(self, temp_cache_dir, sample_news_data):
        """Test cache validity check when cache is valid"""
        news_cache.save_news_cache(sample_news_data, temp_cache_dir)
        
        is_valid = news_cache.is_cache_valid(temp_cache_dir)
        
        assert is_valid is True
    
    def test_is_cache_valid_false_not_exists(self, temp_cache_dir):
        """Test cache validity when cache doesn't exist"""
        is_valid = news_cache.is_cache_valid(temp_cache_dir)
        
        assert is_valid is False
    
    def test_is_cache_valid_false_expired(self, temp_cache_dir, sample_news_data):
        """Test cache validity when cache is expired"""
        # Save and modify to be expired
        news_cache.save_news_cache(sample_news_data, temp_cache_dir)
        
        cache_file = temp_cache_dir / "news_cache.json"
        with open(cache_file, "r") as f:
            cache_data = json.load(f)
        
        old_time = datetime.now() - timedelta(hours=10)
        cache_data["timestamp"] = old_time.isoformat()
        
        with open(cache_file, "w") as f:
            json.dump(cache_data, f)
        
        is_valid = news_cache.is_cache_valid(temp_cache_dir, max_age_hours=6)
        
        assert is_valid is False


class TestGetCacheInfo:
    """Tests for get_cache_info() function"""
    
    def test_get_cache_info_exists(self, temp_cache_dir, sample_news_data):
        """Test getting cache info when cache exists"""
        news_cache.save_news_cache(sample_news_data, temp_cache_dir)
        
        info = news_cache.get_cache_info(temp_cache_dir)
        
        assert info is not None
        assert info["exists"] is True
        assert "timestamp" in info
        assert "date_iso" in info
        assert "age_hours" in info
        assert info["has_tech"] is True
        assert info["has_financial"] is True
        assert info["has_india"] is True
        assert "cache_file" in info
        assert info["is_valid"] is True
    
    def test_get_cache_info_not_exists(self, temp_cache_dir):
        """Test getting cache info when cache doesn't exist"""
        info = news_cache.get_cache_info(temp_cache_dir)
        
        assert info is None
    
    def test_get_cache_info_no_topics(self, temp_cache_dir):
        """Test cache info with empty news data"""
        news_cache.save_news_cache({}, temp_cache_dir)
        
        info = news_cache.get_cache_info(temp_cache_dir)
        
        assert info is not None
        assert info["has_tech"] is False
        assert info["has_financial"] is False
        assert info["has_india"] is False
    
    def test_get_cache_info_expired(self, temp_cache_dir, sample_news_data):
        """Test cache info for expired cache"""
        # Save and modify to be expired
        news_cache.save_news_cache(sample_news_data, temp_cache_dir)
        
        cache_file = temp_cache_dir / "news_cache.json"
        with open(cache_file, "r") as f:
            cache_data = json.load(f)
        
        old_time = datetime.now() - timedelta(hours=10)
        cache_data["timestamp"] = old_time.isoformat()
        
        with open(cache_file, "w") as f:
            json.dump(cache_data, f)
        
        info = news_cache.get_cache_info(temp_cache_dir, max_age_hours=6)
        
        assert info is not None
        assert info["is_valid"] is False
        assert info["age_hours"] > 6
    
    def test_get_cache_info_corrupted(self, temp_cache_dir):
        """Test cache info with corrupted file"""
        cache_file = temp_cache_dir / "news_cache.json"
        
        # Write corrupted JSON
        with open(cache_file, "w") as f:
            f.write("{ invalid")
        
        info = news_cache.get_cache_info(temp_cache_dir)
        
        assert info is not None
        assert info["exists"] is True
        assert "error" in info
    
    def test_get_cache_info_age_calculation(self, temp_cache_dir, sample_news_data):
        """Test that age is calculated correctly"""
        news_cache.save_news_cache(sample_news_data, temp_cache_dir)
        
        info = news_cache.get_cache_info(temp_cache_dir)
        
        assert info is not None
        # Age should be very small (just created)
        assert info["age_hours"] < 0.1
        assert isinstance(info["age_hours"], float)

