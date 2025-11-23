# LLM QA Bot - Unit Tests

This directory contains comprehensive unit tests for the LLM QA Bot project, targeting 90%+ code coverage.

## 📊 Current Status

- **Overall Coverage**: 80%
- **Total Tests**: 161
- **Passing Tests**: 107 (66.5%)
- **Sections Covered**: 1-4 (Analyzers, Audio, Chat Generation, Internet Chat)

## 🗂️ Test Files

| File | Coverage | Tests | Description |
|------|----------|-------|-------------|
| `test_analyzers.py` | 92% | 60+ | Content analysis (files, videos, articles, media) |
| `test_audio_processors.py` | 67% | 27 | NVIDIA Riva audio transcription & TTS |
| `test_chat_generation.py` | 86% | 25+ | Multi-provider LLM chat completion |
| `test_chat_generation_with_internet.py` | 74% | 45+ | Internet-connected chatbot with Firecrawl |

## 🚀 Quick Start

### Setup
```bash
cd /home/dino/myprojects/LLM_QA_Bot
source venv/bin/activate
```

### Run All Tests
```bash
pytest tests/ -v
```

### Run Specific Test File
```bash
pytest tests/test_analyzers.py -v
pytest tests/test_audio_processors.py -v
pytest tests/test_chat_generation.py -v
pytest tests/test_chat_generation_with_internet.py -v
```

### Run Specific Test Class
```bash
pytest tests/test_analyzers.py::TestClearAllFiles -v
```

### Run Specific Test Function
```bash
pytest tests/test_analyzers.py::TestClearAllFiles::test_clearallfiles_empty_folder -v
```

## 📈 Coverage Reports

### Terminal Coverage Report
```bash
pytest tests/ --cov=helper_functions --cov-report=term-missing
```

### HTML Coverage Report
```bash
pytest tests/ --cov=helper_functions --cov-report=html
# View at: htmlcov/index.html
```

### Coverage for Specific Module
```bash
pytest tests/ --cov=helper_functions.analyzers --cov-report=term
pytest tests/ --cov=helper_functions.chat_generation --cov-report=term
```

## 🧪 Test Options

### Stop on First Failure
```bash
pytest tests/ -x
```

### Show Print Statements
```bash
pytest tests/ -v -s
```

### Run Tests in Parallel (faster)
```bash
pip install pytest-xdist
pytest tests/ -n auto
```

### Run Only Failed Tests (from last run)
```bash
pytest tests/ --lf
```

### Run Failed Tests First
```bash
pytest tests/ --ff
```

### Quiet Mode (less verbose)
```bash
pytest tests/ -q
```

### Show Slowest Tests
```bash
pytest tests/ --durations=10
```

## 🔧 Test Structure

### conftest.py
Contains shared fixtures:
- `temp_upload_folder` - Temporary upload directory
- `temp_summary_folder` - Temporary summary directory
- `temp_vector_folder` - Temporary vector index directory
- `temp_bing_folder` - Temporary Bing search directory
- `mock_openai_client` - Mock OpenAI client
- `mock_unified_llm_client` - Mock UnifiedLLMClient
- `mock_riva_asr_service` - Mock NVIDIA Riva ASR
- `mock_riva_tts_service` - Mock NVIDIA Riva TTS
- And more...

### Test Organization
Tests are organized by:
1. **Module** (one test file per helper function file)
2. **Function** (test classes group tests for each function)
3. **Scenario** (test methods cover different scenarios)

Example:
```python
class TestClearAllFiles:
    """Tests for clearallfiles() function"""
    
    def test_clearallfiles_empty_folder(self):
        """Test clearing an empty folder"""
        # Test implementation
    
    def test_clearallfiles_with_files(self):
        """Test clearing a folder with files"""
        # Test implementation
```

## 📝 Writing New Tests

### Template
```python
import pytest
from unittest.mock import Mock, patch
from helper_functions import module_name

class TestFunctionName:
    """Tests for function_name() function"""
    
    def test_function_success(self):
        """Test successful execution"""
        result = module_name.function_name()
        assert result == expected_value
    
    @patch('helper_functions.module_name.dependency')
    def test_function_with_mock(self, mock_dependency):
        """Test with mocked dependency"""
        mock_dependency.return_value = "mocked value"
        result = module_name.function_name()
        assert "mocked value" in result
    
    def test_function_error(self):
        """Test error handling"""
        with pytest.raises(ValueError):
            module_name.function_name(invalid_input)
```

## 🐛 Debugging Tests

### Run with Python Debugger
```bash
pytest tests/test_analyzers.py::TestClearAllFiles::test_clearallfiles_empty_folder --pdb
```

### Show Local Variables on Failure
```bash
pytest tests/ -v --showlocals
```

### Full Traceback
```bash
pytest tests/ -v --tb=long
```

### No Traceback
```bash
pytest tests/ -v --tb=no
```

## 📚 Dependencies

Tests require:
- `pytest==9.0.1` - Testing framework
- `pytest-mock==3.15.1` - Mocking support
- `pytest-asyncio==1.3.0` - Async test support
- `pytest-cov==7.0.0` - Coverage reporting
- `responses==0.25.8` - HTTP mocking
- `coverage==7.12.0` - Coverage measurement

Install all test dependencies:
```bash
pip install pytest pytest-mock pytest-asyncio pytest-cov responses
```

## 🎯 Coverage Goals

| Module | Current | Target | Status |
|--------|---------|--------|--------|
| analyzers.py | 92% | 95% | ✅ Near target |
| audio_processors.py | 67% | 85% | ⚠️ Needs work |
| chat_generation.py | 86% | 95% | ✅ Near target |
| chat_generation_with_internet.py | 74% | 90% | ⚠️ Needs work |

## 🔍 Test Coverage Details

View detailed coverage in `TEST_SUMMARY.md` for:
- Coverage breakdown by module
- Failed test analysis
- Common failure patterns
- Next steps to reach 90%+ coverage

## 🤝 Contributing

When adding new tests:
1. Follow the existing test structure
2. Use descriptive test names
3. Add docstrings to explain what you're testing
4. Mock external dependencies
5. Test both success and error cases
6. Check code coverage after adding tests

## 📞 Support

For issues or questions about tests:
1. Check `TEST_SUMMARY.md` for common issues
2. Review existing tests for examples
3. Check fixture definitions in `conftest.py`

---

**Last Updated**: 2025-11-20  
**Test Framework**: pytest 9.0.1  
**Python Version**: 3.11.14

