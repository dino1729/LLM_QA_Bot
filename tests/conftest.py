"""
Pytest configuration and common fixtures for all tests
"""
import os
import sys
import pytest
import tempfile
import shutil
from unittest.mock import Mock, MagicMock, patch

# Add the parent directory to sys.path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


@pytest.fixture
def temp_upload_folder():
    """Create a temporary upload folder for testing"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup after test
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def temp_summary_folder():
    """Create a temporary summary folder for testing"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def temp_vector_folder():
    """Create a temporary vector folder for testing"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def temp_bing_folder():
    """Create a temporary bing folder for testing"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_file():
    """Create a mock file object"""
    file = Mock()
    file.name = "test.pdf"
    return file


@pytest.fixture
def mock_files_valid():
    """Create multiple valid mock file objects"""
    files = []
    for ext in ["pdf", "txt", "docx", "png", "jpg", "jpeg", "mp3"]:
        file = Mock()
        file.name = f"test.{ext}"
        files.append(file)
    return files


@pytest.fixture
def mock_files_invalid():
    """Create mock file objects with invalid extensions"""
    files = []
    for ext in ["exe", "bat", "sh"]:
        file = Mock()
        file.name = f"test.{ext}"
        files.append(file)
    return files


@pytest.fixture
def mock_openai_client():
    """Create a mock OpenAI client"""
    client = Mock()
    response = Mock()
    response.choices = [Mock()]
    response.choices[0].message.content = "Test response"
    client.chat.completions.create.return_value = response
    return client


@pytest.fixture
def mock_unified_llm_client():
    """Create a mock UnifiedLLMClient"""
    client = Mock()
    client.chat_completion.return_value = "Test response"
    client.get_embedding.return_value = [0.1] * 1536
    
    # Mock LlamaIndex objects
    mock_llm = Mock()
    mock_embedding = Mock()
    client.get_llamaindex_llm.return_value = mock_llm
    client.get_llamaindex_embedding.return_value = mock_embedding
    
    return client


@pytest.fixture
def sample_conversation():
    """Create a sample conversation for testing"""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ]


@pytest.fixture
def mock_youtube_video():
    """Create a mock YouTube object"""
    video = Mock()
    video.title = "Test Video"
    video.streams.filter.return_value.first.return_value.download.return_value = "test.mp4"
    return video


@pytest.fixture
def mock_article():
    """Create a mock Article object"""
    article = Mock()
    article.title = "Test Article"
    article.text = "This is a test article with more than 75 words. " * 10
    article.download = Mock()
    article.parse = Mock()
    return article


@pytest.fixture
def mock_riva_asr_service():
    """Create a mock NVIDIA Riva ASR service"""
    service = Mock()
    response = Mock()
    response.results = [Mock()]
    response.results[0].alternatives = [Mock()]
    response.results[0].alternatives[0].transcript = "Test transcription"
    service.offline_recognize.return_value = response
    return service


@pytest.fixture
def mock_riva_tts_service():
    """Create a mock NVIDIA Riva TTS service"""
    service = Mock()
    response = Mock()
    response.audio = b"fake_audio_data"
    service.synthesize_online.return_value = [response]
    return service


@pytest.fixture
def mock_cohere_client():
    """Create a mock Cohere client"""
    client = Mock()
    response = Mock()
    response.text = "Cohere response"
    client.chat.return_value = response
    return client


@pytest.fixture
def mock_gemini_model():
    """Create a mock Gemini model"""
    model = Mock()
    response = Mock()
    response.text = "Gemini response"
    model.generate_content.return_value = response
    return model


@pytest.fixture
def mock_groq_client():
    """Create a mock Groq client"""
    client = Mock()
    response = Mock()
    response.choices = [Mock()]
    response.choices[0].message.content = "Groq response"
    client.chat.completions.create.return_value = response
    return client


@pytest.fixture
def mock_pinecone_index():
    """Create a mock Pinecone index"""
    index = Mock()
    match = Mock()
    match.metadata = {"text": "Sample text from Bhagavad Gita"}
    index.query.return_value.matches = [match] * 8
    return index


@pytest.fixture
def mock_supabase_client():
    """Create a mock Supabase client"""
    client = Mock()
    return client


@pytest.fixture(autouse=True)
def reset_settings():
    """Reset LlamaIndex Settings after each test"""
    yield
    # Reset Settings to default after each test
    from llama_index.core import Settings
    Settings.llm = None
    Settings.embed_model = None

