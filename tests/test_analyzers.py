"""
Unit tests for helper_functions/analyzers.py
Tests for content analysis functions for files, videos, articles, and media
"""
import os
import pytest
from unittest.mock import Mock, MagicMock, patch, mock_open
from helper_functions import analyzers


class TestClearAllFiles:
    """Tests for clearallfiles() function"""
    
    def test_clearallfiles_empty_folder(self, temp_upload_folder, monkeypatch):
        """Test clearing an empty folder"""
        monkeypatch.setattr(analyzers, 'UPLOAD_FOLDER', temp_upload_folder)
        analyzers.clearallfiles()
        assert len(os.listdir(temp_upload_folder)) == 0
    
    def test_clearallfiles_with_files(self, temp_upload_folder, monkeypatch):
        """Test clearing a folder with files"""
        monkeypatch.setattr(analyzers, 'UPLOAD_FOLDER', temp_upload_folder)
        # Create test files
        for i in range(3):
            with open(os.path.join(temp_upload_folder, f"test{i}.txt"), "w") as f:
                f.write("test content")
        
        assert len(os.listdir(temp_upload_folder)) == 3
        analyzers.clearallfiles()
        assert len(os.listdir(temp_upload_folder)) == 0
    
    def test_clearallfiles_with_subdirectories(self, temp_upload_folder, monkeypatch):
        """Test clearing folder with subdirectories"""
        monkeypatch.setattr(analyzers, 'UPLOAD_FOLDER', temp_upload_folder)
        # Create subdirectory with files
        subdir = os.path.join(temp_upload_folder, "subdir")
        os.makedirs(subdir)
        with open(os.path.join(subdir, "test.txt"), "w") as f:
            f.write("test content")
        
        analyzers.clearallfiles()
        # Files should be removed, but empty subdirectory may remain
        for root, dirs, files in os.walk(temp_upload_folder):
            assert len(files) == 0


class TestFileFormatValidityCheck:
    """Tests for fileformatvaliditycheck() function"""
    
    def test_valid_single_file(self):
        """Test with a single valid file"""
        file = Mock()
        file.name = "test.pdf"
        assert analyzers.fileformatvaliditycheck([file]) is True
    
    def test_valid_multiple_files(self, mock_files_valid):
        """Test with multiple valid files"""
        assert analyzers.fileformatvaliditycheck(mock_files_valid) is True
    
    def test_invalid_file_format(self):
        """Test with an invalid file format"""
        file = Mock()
        file.name = "test.exe"
        assert analyzers.fileformatvaliditycheck([file]) is False
    
    def test_mixed_valid_invalid_files(self):
        """Test with mixed valid and invalid files"""
        files = []
        file1 = Mock()
        file1.name = "test.pdf"
        file2 = Mock()
        file2.name = "test.exe"
        files.extend([file1, file2])
        assert analyzers.fileformatvaliditycheck(files) is False
    
    def test_empty_file_list(self):
        """Test with empty file list"""
        assert analyzers.fileformatvaliditycheck([]) is True
    
    def test_case_insensitive_extension(self):
        """Test that file extension check is case insensitive"""
        file = Mock()
        file.name = "test.PDF"
        assert analyzers.fileformatvaliditycheck([file]) is True


class TestBuildIndex:
    """Tests for build_index() function"""
    
    @patch('helper_functions.analyzers.SimpleDirectoryReader')
    @patch('helper_functions.analyzers.VectorStoreIndex')
    @patch('helper_functions.analyzers.SummaryIndex')
    def test_build_index_with_documents(self, mock_summary_index, mock_vector_index, 
                                       mock_reader, temp_upload_folder, 
                                       temp_vector_folder, temp_summary_folder, monkeypatch):
        """Test building index with documents"""
        monkeypatch.setattr(analyzers, 'UPLOAD_FOLDER', temp_upload_folder)
        monkeypatch.setattr(analyzers, 'VECTOR_FOLDER', temp_vector_folder)
        monkeypatch.setattr(analyzers, 'SUMMARY_FOLDER', temp_summary_folder)
        
        # Mock document loading
        mock_docs = [Mock(), Mock()]
        mock_reader.return_value.load_data.return_value = mock_docs
        
        # Mock index creation
        mock_q_index = Mock()
        mock_s_index = Mock()
        mock_vector_index.from_documents.return_value = mock_q_index
        mock_summary_index.from_documents.return_value = mock_s_index
        
        analyzers.build_index()
        
        # Verify indexes were created and persisted
        mock_vector_index.from_documents.assert_called_once_with(mock_docs)
        mock_summary_index.from_documents.assert_called_once_with(mock_docs)
        mock_q_index.storage_context.persist.assert_called_once()
        mock_s_index.storage_context.persist.assert_called_once()
    
    @patch('helper_functions.analyzers.SimpleDirectoryReader')
    def test_build_index_with_empty_folder(self, mock_reader, temp_upload_folder, monkeypatch):
        """Test building index with empty folder"""
        monkeypatch.setattr(analyzers, 'UPLOAD_FOLDER', temp_upload_folder)
        mock_reader.return_value.load_data.return_value = []
        
        # Should not raise exception
        try:
            analyzers.build_index()
        except Exception as e:
            pytest.fail(f"build_index raised exception: {e}")


class TestSummaryGenerator:
    """Tests for summary_generator() function"""
    
    @patch('helper_functions.analyzers.ask_fromfullcontext')
    def test_summary_generator_success(self, mock_ask):
        """Test successful summary generation"""
        mock_ask.return_value = "This is a generated summary."
        result = analyzers.summary_generator()
        assert result == "This is a generated summary."
        mock_ask.assert_called_once()
    
    @patch('helper_functions.analyzers.ask_fromfullcontext')
    def test_summary_generator_exception(self, mock_ask):
        """Test summary generation with exception"""
        mock_ask.side_effect = Exception("Test error")
        result = analyzers.summary_generator()
        assert result == "Summary not available"


class TestExampleGenerator:
    """Tests for example_generator() function"""
    
    @patch('helper_functions.analyzers.ask_fromfullcontext')
    def test_example_generator_success(self, mock_ask):
        """Test successful example generation"""
        mock_ask.return_value = '["Question 1?", "Question 2?", "Question 3?"]'
        result = analyzers.example_generator()
        assert isinstance(result, list)
        assert len(result) == 3
    
    @patch('helper_functions.analyzers.ask_fromfullcontext')
    def test_example_generator_with_code_block(self, mock_ask):
        """Test example generation with code block wrapper"""
        mock_ask.return_value = '```json\n["Q1?", "Q2?"]\n```'
        result = analyzers.example_generator()
        assert isinstance(result, list)
        assert len(result) == 2
    
    @patch('helper_functions.analyzers.ask_fromfullcontext')
    def test_example_generator_malformed_response(self, mock_ask):
        """Test example generation with malformed response"""
        mock_ask.return_value = "Not a valid list"
        result = analyzers.example_generator()
        # Should return default examples
        assert isinstance(result, list)
        assert len(result) > 0
    
    @patch('helper_functions.analyzers.ask_fromfullcontext')
    def test_example_generator_exception(self, mock_ask):
        """Test example generation with exception"""
        mock_ask.side_effect = Exception("Test error")
        result = analyzers.example_generator()
        # Should return default examples
        assert isinstance(result, list)


class TestAskFromFullContext:
    """Tests for ask_fromfullcontext() function"""
    
    @patch('helper_functions.analyzers.load_index_from_storage')
    @patch('helper_functions.analyzers.StorageContext')
    def test_ask_fromfullcontext_valid_question(self, mock_storage_context, 
                                                mock_load_index, temp_summary_folder, monkeypatch):
        """Test asking a valid question"""
        monkeypatch.setattr(analyzers, 'SUMMARY_FOLDER', temp_summary_folder)
        
        # Mock index and query engine
        mock_index = Mock()
        mock_query_engine = Mock()
        mock_response = Mock()
        mock_response.response = "This is the answer"
        
        mock_load_index.return_value = mock_index
        mock_index.as_query_engine.return_value = mock_query_engine
        mock_query_engine.query.return_value = mock_response
        
        result = analyzers.ask_fromfullcontext("Test question", Mock())
        assert result == "This is the answer"


class TestExtractVideoId:
    """Tests for extract_video_id() function"""
    
    def test_extract_video_id_standard_url(self):
        """Test extracting video ID from standard YouTube URL"""
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        result = analyzers.extract_video_id(url)
        assert result == "dQw4w9WgXcQ"
    
    def test_extract_video_id_short_url(self):
        """Test extracting video ID from youtu.be URL"""
        url = "https://youtu.be/dQw4w9WgXcQ"
        result = analyzers.extract_video_id(url)
        assert result == "dQw4w9WgXcQ"
    
    def test_extract_video_id_with_parameters(self):
        """Test extracting video ID with additional parameters"""
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=10s"
        result = analyzers.extract_video_id(url)
        assert result == "dQw4w9WgXcQ"
    
    def test_extract_video_id_invalid_url(self):
        """Test with invalid URL format"""
        url = "https://www.example.com/video"
        result = analyzers.extract_video_id(url)
        assert result == url  # Returns original URL if no match


class TestGetVideoTitle:
    """Tests for get_video_title() function"""
    
    @patch('helper_functions.analyzers.YouTube')
    def test_get_video_title_success(self, mock_youtube):
        """Test getting video title successfully"""
        mock_yt = Mock()
        mock_yt.title = "Test Video Title"
        mock_youtube.return_value = mock_yt
        
        result = analyzers.get_video_title("https://youtube.com/watch?v=test", "test")
        assert result == "Test Video Title"
    
    @patch('helper_functions.analyzers.YouTube')
    def test_get_video_title_error(self, mock_youtube):
        """Test getting video title with error"""
        mock_youtube.side_effect = Exception("Video unavailable")
        
        result = analyzers.get_video_title("https://youtube.com/watch?v=test", "test_id")
        assert result == "test_id"


class TestTranscriptExtractor:
    """Tests for transcript_extractor() function"""
    
    @patch('helper_functions.analyzers.YouTubeTranscriptApi')
    def test_transcript_extractor_success(self, mock_api, temp_upload_folder, monkeypatch):
        """Test successful transcript extraction"""
        monkeypatch.setattr(analyzers, 'UPLOAD_FOLDER', temp_upload_folder)
        
        mock_transcript = [
            {"text": "Hello", "start": 0.0},
            {"text": "World", "start": 1.0}
        ]
        mock_api.get_transcript.return_value = mock_transcript
        
        result = analyzers.transcript_extractor("test_video_id")
        assert result is True
        
        # Check if file was created
        transcript_file = os.path.join(temp_upload_folder, "test_video_id.txt")
        assert os.path.exists(transcript_file)
    
    @patch('helper_functions.analyzers.YouTubeTranscriptApi')
    def test_transcript_extractor_no_transcript(self, mock_api):
        """Test when transcript is not available"""
        mock_api.get_transcript.side_effect = Exception("No transcript available")
        
        result = analyzers.transcript_extractor("test_video_id")
        assert result is False


class TestVideoDownloader:
    """Tests for video_downloader() function"""
    
    @patch('helper_functions.analyzers.YouTube')
    def test_video_downloader_success(self, mock_youtube, temp_upload_folder, monkeypatch):
        """Test successful video download"""
        monkeypatch.setattr(analyzers, 'UPLOAD_FOLDER', temp_upload_folder)
        
        mock_yt = Mock()
        mock_stream = Mock()
        mock_yt.streams.filter.return_value.first.return_value = mock_stream
        mock_youtube.return_value = mock_yt
        
        analyzers.video_downloader("https://youtube.com/watch?v=test")
        mock_stream.download.assert_called_once_with(temp_upload_folder)
    
    @patch('helper_functions.analyzers.YouTube')
    def test_video_downloader_error(self, mock_youtube):
        """Test video download with error"""
        mock_youtube.side_effect = Exception("Video unavailable")
        
        # Should not raise exception
        try:
            analyzers.video_downloader("https://youtube.com/watch?v=test")
        except Exception:
            pytest.fail("video_downloader raised exception")


class TestProcessVideo:
    """Tests for process_video() function"""
    
    @patch('helper_functions.analyzers.transcript_extractor')
    @patch('helper_functions.analyzers.extract_video_id')
    @patch('helper_functions.analyzers.build_index')
    @patch('helper_functions.analyzers.summary_generator')
    @patch('helper_functions.analyzers.example_generator')
    def test_process_video_with_transcript(self, mock_example, mock_summary, mock_build,
                                          mock_extract_id, mock_transcript):
        """Test processing video with available transcript"""
        mock_extract_id.return_value = "video_id"
        mock_transcript.return_value = True
        mock_summary.return_value = "Summary"
        mock_example.return_value = ["Q1?", "Q2?"]
        
        result = analyzers.process_video("https://youtube.com/watch?v=test", False, False)
        
        assert result[0] == "Successfully processed video with transcript"
        assert result[1] == "Summary"
        assert result[2] == ["Q1?", "Q2?"]
    
    @patch('helper_functions.analyzers.transcript_extractor')
    @patch('helper_functions.analyzers.video_downloader')
    @patch('helper_functions.analyzers.extract_video_id')
    @patch('helper_functions.analyzers.build_index')
    @patch('helper_functions.analyzers.summary_generator')
    @patch('helper_functions.analyzers.example_generator')
    def test_process_video_without_transcript_nonlite(self, mock_example, mock_summary, 
                                                      mock_build, mock_extract_id, 
                                                      mock_download, mock_transcript):
        """Test processing video without transcript in non-lite mode"""
        mock_extract_id.return_value = "video_id"
        mock_transcript.return_value = False
        mock_summary.return_value = "Summary"
        mock_example.return_value = ["Q1?", "Q2?"]
        
        result = analyzers.process_video("https://youtube.com/watch?v=test", False, False)
        
        mock_download.assert_called_once()
        assert "video downloaded" in result[0]
    
    @patch('helper_functions.analyzers.transcript_extractor')
    @patch('helper_functions.analyzers.extract_video_id')
    def test_process_video_without_transcript_lite(self, mock_extract_id, mock_transcript):
        """Test processing video without transcript in lite mode"""
        mock_extract_id.return_value = "video_id"
        mock_transcript.return_value = False
        
        result = analyzers.process_video("https://youtube.com/watch?v=test", False, True)
        
        assert "No transcript available" in result[0]


class TestAnalyzeYTVideo:
    """Tests for analyze_ytvideo() function"""
    
    @patch('helper_functions.analyzers.process_video')
    @patch('helper_functions.analyzers.get_video_title')
    @patch('helper_functions.analyzers.extract_video_id')
    @patch('helper_functions.analyzers.clearallfiles')
    def test_analyze_ytvideo_valid_url(self, mock_clear, mock_extract, mock_title, mock_process):
        """Test analyzing a valid YouTube URL"""
        mock_extract.return_value = "video_id"
        mock_title.return_value = "Test Video"
        mock_process.return_value = ("Success", "Summary", ["Q1?"], "Test Video")
        
        result = analyzers.analyze_ytvideo("https://youtube.com/watch?v=test", False)
        
        assert result["message"] == "Success"
        assert result["summary"] == "Summary"
        assert result["video_title"] == "Test Video"
        mock_clear.assert_called_once()
    
    def test_analyze_ytvideo_invalid_url(self):
        """Test analyzing an invalid URL"""
        result = analyzers.analyze_ytvideo("", False)
        assert "not a valid YouTube URL" in result["message"]
    
    def test_analyze_ytvideo_non_youtube_url(self):
        """Test analyzing a non-YouTube URL"""
        result = analyzers.analyze_ytvideo("https://example.com", False)
        assert "not a valid YouTube URL" in result["message"]


class TestSaveToDisk:
    """Tests for savetodisk() function"""
    
    def test_savetodisk_single_file(self, temp_upload_folder, monkeypatch):
        """Test saving a single file to disk"""
        monkeypatch.setattr(analyzers, 'UPLOAD_FOLDER', temp_upload_folder)
        
        # Create a mock file
        mock_file = Mock()
        test_file_path = os.path.join(temp_upload_folder, "source.txt")
        with open(test_file_path, "w") as f:
            f.write("test content")
        mock_file.name = test_file_path
        
        result = analyzers.savetodisk([mock_file])
        assert len(result) == 1
        assert "source.txt" in result[0]
    
    def test_savetodisk_multiple_files(self, temp_upload_folder, monkeypatch):
        """Test saving multiple files to disk"""
        monkeypatch.setattr(analyzers, 'UPLOAD_FOLDER', temp_upload_folder)
        
        files = []
        for i in range(3):
            mock_file = Mock()
            test_file_path = os.path.join(temp_upload_folder, f"source{i}.txt")
            with open(test_file_path, "w") as f:
                f.write(f"test content {i}")
            mock_file.name = test_file_path
            files.append(mock_file)
        
        result = analyzers.savetodisk(files)
        assert len(result) == 3


class TestProcessFiles:
    """Tests for process_files() function"""
    
    @patch('helper_functions.analyzers.fileformatvaliditycheck')
    @patch('helper_functions.analyzers.savetodisk')
    @patch('helper_functions.analyzers.build_index')
    @patch('helper_functions.analyzers.summary_generator')
    @patch('helper_functions.analyzers.example_generator')
    def test_process_files_valid(self, mock_example, mock_summary, mock_build, 
                                 mock_save, mock_check):
        """Test processing valid files"""
        mock_check.return_value = True
        mock_save.return_value = ["file1.pdf"]
        mock_summary.return_value = "Summary"
        mock_example.return_value = ["Q1?", "Q2?"]
        
        files = [Mock()]
        result = analyzers.process_files(files, False)
        
        assert "Successfully processed" in result[0]
        assert result[1] == "Summary"
        assert result[2] == ["Q1?", "Q2?"]
    
    @patch('helper_functions.analyzers.fileformatvaliditycheck')
    def test_process_files_invalid(self, mock_check):
        """Test processing invalid files"""
        mock_check.return_value = False
        
        files = [Mock()]
        result = analyzers.process_files(files, False)
        
        assert "Invalid file format" in result[0]


class TestAnalyzeFile:
    """Tests for analyze_file() function"""
    
    @patch('helper_functions.analyzers.process_files')
    @patch('helper_functions.analyzers.clearallfiles')
    def test_analyze_file_with_files(self, mock_clear, mock_process):
        """Test analyzing files"""
        mock_process.return_value = ("Success", "Summary", ["Q1?"], "Files")
        
        files = [Mock()]
        result = analyzers.analyze_file(files, False)
        
        assert result["message"] == "Success"
        assert result["summary"] == "Summary"
        mock_clear.assert_called_once()
    
    def test_analyze_file_without_files(self):
        """Test analyzing without files"""
        result = analyzers.analyze_file([], False)
        assert "No files" in result["message"]


class TestDownloadAndParseArticle:
    """Tests for download_and_parse_article() function"""
    
    @patch('helper_functions.analyzers.Article')
    def test_download_and_parse_article_success(self, mock_article_class):
        """Test successful article download and parse"""
        mock_article = Mock()
        mock_article.text = "This is a test article with more than 75 words. " * 10
        mock_article_class.return_value = mock_article
        
        result = analyzers.download_and_parse_article("https://example.com/article")
        assert result == mock_article
    
    @patch('helper_functions.analyzers.Article')
    def test_download_and_parse_article_short(self, mock_article_class):
        """Test with short article (< 75 words)"""
        mock_article = Mock()
        mock_article.text = "Short article"
        mock_article_class.return_value = mock_article
        
        result = analyzers.download_and_parse_article("https://example.com/article")
        assert result is None
    
    @patch('helper_functions.analyzers.Article')
    def test_download_and_parse_article_error(self, mock_article_class):
        """Test with download error"""
        mock_article = Mock()
        mock_article.download.side_effect = Exception("Download error")
        mock_article_class.return_value = mock_article
        
        result = analyzers.download_and_parse_article("https://example.com/article")
        assert result is None


class TestAlternativeArticleDownload:
    """Tests for alternative_article_download() function"""
    
    @patch('helper_functions.analyzers.requests.get')
    def test_alternative_article_download_success(self, mock_get):
        """Test successful alternative article download"""
        mock_response = Mock()
        mock_response.text = "<html><body><p>Article content</p></body></html>"
        mock_get.return_value = mock_response
        
        result = analyzers.alternative_article_download("https://example.com/article")
        assert result is not None
        assert "Article content" in result
    
    @patch('helper_functions.analyzers.requests.get')
    def test_alternative_article_download_timeout(self, mock_get):
        """Test with timeout"""
        mock_get.side_effect = Exception("Timeout")
        
        result = analyzers.alternative_article_download("https://example.com/article")
        assert result is None


class TestSaveArticleText:
    """Tests for save_article_text() function"""
    
    def test_save_article_text(self, temp_upload_folder, monkeypatch):
        """Test saving article text"""
        monkeypatch.setattr(analyzers, 'UPLOAD_FOLDER', temp_upload_folder)
        
        text = "This is article text with unicode: café ñ"
        analyzers.save_article_text(text)
        
        article_file = os.path.join(temp_upload_folder, "article.txt")
        assert os.path.exists(article_file)
        
        with open(article_file, "r", encoding="utf-8") as f:
            content = f.read()
            assert text in content


class TestProcessArticle:
    """Tests for process_article() function"""
    
    @patch('helper_functions.analyzers.download_and_parse_article')
    @patch('helper_functions.analyzers.save_article_text')
    @patch('helper_functions.analyzers.build_index')
    @patch('helper_functions.analyzers.summary_generator')
    @patch('helper_functions.analyzers.example_generator')
    def test_process_article_success(self, mock_example, mock_summary, mock_build,
                                     mock_save, mock_download):
        """Test successful article processing"""
        mock_article = Mock()
        mock_article.title = "Test Article"
        mock_article.text = "Article text"
        mock_download.return_value = mock_article
        mock_summary.return_value = "Summary"
        mock_example.return_value = ["Q1?", "Q2?"]
        
        result = analyzers.process_article("https://example.com/article", False)
        
        assert "Successfully processed" in result[0]
        assert result[1] == "Summary"
        assert result[3] == "Test Article"
    
    @patch('helper_functions.analyzers.download_and_parse_article')
    @patch('helper_functions.analyzers.alternative_article_download')
    @patch('helper_functions.analyzers.save_article_text')
    @patch('helper_functions.analyzers.build_index')
    @patch('helper_functions.analyzers.summary_generator')
    @patch('helper_functions.analyzers.example_generator')
    def test_process_article_fallback(self, mock_example, mock_summary, mock_build,
                                      mock_save, mock_alt_download, mock_download):
        """Test article processing with fallback method"""
        mock_download.return_value = None
        mock_alt_download.return_value = "Alternative article text"
        mock_summary.return_value = "Summary"
        mock_example.return_value = ["Q1?"]
        
        result = analyzers.process_article("https://example.com/article", False)
        
        assert "Successfully processed" in result[0]
    
    @patch('helper_functions.analyzers.download_and_parse_article')
    @patch('helper_functions.analyzers.alternative_article_download')
    def test_process_article_failed(self, mock_alt_download, mock_download):
        """Test failed article processing"""
        mock_download.return_value = None
        mock_alt_download.return_value = None
        
        result = analyzers.process_article("https://example.com/article", False)
        
        assert "Failed to download" in result[0]


class TestAnalyzeArticle:
    """Tests for analyze_article() function"""
    
    @patch('helper_functions.analyzers.process_article')
    @patch('helper_functions.analyzers.clearallfiles')
    def test_analyze_article_valid_url(self, mock_clear, mock_process):
        """Test analyzing a valid article URL"""
        mock_process.return_value = ("Success", "Summary", ["Q1?"], "Article Title")
        
        result = analyzers.analyze_article("https://example.com/article", False)
        
        assert result["message"] == "Success"
        assert result["article_title"] == "Article Title"
        mock_clear.assert_called_once()
    
    def test_analyze_article_empty_url(self):
        """Test analyzing with empty URL"""
        result = analyzers.analyze_article("", False)
        assert "not a valid URL" in result["message"]


class TestExtractMediaUrlFromOvercast:
    """Tests for extract_media_url_from_overcast() function"""
    
    @patch('helper_functions.analyzers.requests.get')
    def test_extract_media_url_success(self, mock_get):
        """Test successful media URL extraction"""
        mock_response = Mock()
        mock_response.text = '<html><audio src="https://example.com/audio.mp3"></audio></html>'
        mock_get.return_value = mock_response
        
        result = analyzers.extract_media_url_from_overcast("https://overcast.fm/+test")
        assert result == "https://example.com/audio.mp3"
    
    @patch('helper_functions.analyzers.requests.get')
    def test_extract_media_url_no_audio(self, mock_get):
        """Test with no audio tag"""
        mock_response = Mock()
        mock_response.text = '<html><body>No audio here</body></html>'
        mock_get.return_value = mock_response
        
        result = analyzers.extract_media_url_from_overcast("https://overcast.fm/+test")
        assert result is None


class TestDownloadMediaFile:
    """Tests for download_media_file() function"""
    
    @patch('helper_functions.analyzers.wget.download')
    def test_download_media_file_success(self, mock_wget, temp_upload_folder, monkeypatch):
        """Test successful media file download"""
        monkeypatch.setattr(analyzers, 'UPLOAD_FOLDER', temp_upload_folder)
        mock_wget.return_value = os.path.join(temp_upload_folder, "audio.mp3")
        
        result = analyzers.download_media_file("https://example.com/audio.mp3")
        assert "audio.mp3" in result


class TestRenameAndExtractAudio:
    """Tests for rename_and_extract_audio() function"""
    
    @patch('helper_functions.analyzers.ffmpeg_extract_audio')
    def test_rename_and_extract_audio_file(self, mock_ffmpeg, temp_upload_folder, monkeypatch):
        """Test renaming and extracting audio from audio file"""
        monkeypatch.setattr(analyzers, 'UPLOAD_FOLDER', temp_upload_folder)
        
        # Create a test audio file
        test_file = os.path.join(temp_upload_folder, "test.m4a")
        with open(test_file, "w") as f:
            f.write("fake audio data")
        
        analyzers.rename_and_extract_audio("test.m4a")
        
        # Check if renamed to audio.m4a
        assert os.path.exists(os.path.join(temp_upload_folder, "audio.m4a"))
    
    @patch('helper_functions.analyzers.ffmpeg_extract_audio')
    def test_rename_and_extract_video_file(self, mock_ffmpeg, temp_upload_folder, monkeypatch):
        """Test extracting audio from video file"""
        monkeypatch.setattr(analyzers, 'UPLOAD_FOLDER', temp_upload_folder)
        
        # Create a test video file
        test_file = os.path.join(temp_upload_folder, "test.mp4")
        with open(test_file, "w") as f:
            f.write("fake video data")
        
        analyzers.rename_and_extract_audio("test.mp4")
        
        mock_ffmpeg.assert_called_once()


class TestTranscribeAudioWithWhisper:
    """Tests for transcribe_audio_with_whisper() function"""
    
    @patch('helper_functions.analyzers.whisper.load_model')
    def test_transcribe_audio_with_whisper(self, mock_whisper, temp_upload_folder, monkeypatch):
        """Test transcribing audio with Whisper"""
        monkeypatch.setattr(analyzers, 'UPLOAD_FOLDER', temp_upload_folder)
        
        # Create a fake audio file
        audio_file = os.path.join(temp_upload_folder, "audio.mp3")
        with open(audio_file, "w") as f:
            f.write("fake audio")
        
        mock_model = Mock()
        mock_result = {"text": "Transcribed text"}
        mock_model.transcribe.return_value = mock_result
        mock_whisper.return_value = mock_model
        
        result = analyzers.transcribe_audio_with_whisper()
        assert result == mock_result


class TestSaveTranscript:
    """Tests for save_transcript() function"""
    
    def test_save_transcript(self, temp_upload_folder, monkeypatch):
        """Test saving transcript"""
        monkeypatch.setattr(analyzers, 'UPLOAD_FOLDER', temp_upload_folder)
        
        text = "This is a transcript with unicode: café"
        analyzers.save_transcript(text)
        
        transcript_file = os.path.join(temp_upload_folder, "transcript.txt")
        assert os.path.exists(transcript_file)


class TestProcessMedia:
    """Tests for process_media() function"""
    
    @patch('helper_functions.analyzers.extract_media_url_from_overcast')
    @patch('helper_functions.analyzers.download_media_file')
    @patch('helper_functions.analyzers.rename_and_extract_audio')
    @patch('helper_functions.analyzers.transcribe_audio_with_whisper')
    @patch('helper_functions.analyzers.save_transcript')
    @patch('helper_functions.analyzers.build_index')
    @patch('helper_functions.analyzers.summary_generator')
    @patch('helper_functions.analyzers.example_generator')
    def test_process_media_direct_url(self, mock_example, mock_summary, mock_build,
                                     mock_save_transcript, mock_transcribe, 
                                     mock_rename, mock_download, mock_extract):
        """Test processing media from direct URL"""
        mock_extract.return_value = None  # Not an overcast URL
        mock_download.return_value = "audio.mp3"
        mock_transcribe.return_value = {"text": "Transcription"}
        mock_summary.return_value = "Summary"
        mock_example.return_value = ["Q1?"]
        
        result = analyzers.process_media("https://example.com/audio.mp3", False)
        
        assert "Successfully processed" in result[0]
    
    @patch('helper_functions.analyzers.extract_media_url_from_overcast')
    @patch('helper_functions.analyzers.download_media_file')
    def test_process_media_download_failed(self, mock_download, mock_extract):
        """Test processing media with download failure"""
        mock_extract.return_value = None
        mock_download.side_effect = Exception("Download failed")
        
        result = analyzers.process_media("https://example.com/audio.mp3", False)
        
        assert "Failed to download" in result[0]


class TestAnalyzeMedia:
    """Tests for analyze_media() function"""
    
    @patch('helper_functions.analyzers.process_media')
    @patch('helper_functions.analyzers.clearallfiles')
    def test_analyze_media_valid_url(self, mock_clear, mock_process):
        """Test analyzing a valid media URL"""
        mock_process.return_value = ("Success", "Summary", ["Q1?"], "Media Title")
        
        result = analyzers.analyze_media("https://example.com/audio.mp3", False)
        
        assert result["message"] == "Success"
        mock_clear.assert_called_once()
    
    def test_analyze_media_empty_url(self):
        """Test analyzing with empty URL"""
        result = analyzers.analyze_media("", False)
        assert "not a valid URL" in result["message"]

