import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path

# Import the module to test and config
import year_progress_and_news_reporter_litellm as reporter
from config import config
from helper_functions.tts_chatterbox import split_text_for_chatterbox


class TestSplitTextForChatterbox(unittest.TestCase):
    """Unit tests for the shared sentence splitting function."""
    
    def test_empty_text(self):
        """Empty or whitespace-only text returns empty list."""
        self.assertEqual(split_text_for_chatterbox(""), [])
        self.assertEqual(split_text_for_chatterbox("   "), [])
        self.assertEqual(split_text_for_chatterbox("\n\t"), [])
    
    def test_single_short_sentence(self):
        """Single short sentence is returned as-is."""
        text = "Hello world."
        result = split_text_for_chatterbox(text)
        self.assertEqual(result, ["Hello world."])
    
    def test_multiple_sentences_split_on_punctuation(self):
        """Multiple sentences are split at sentence boundaries."""
        text = "First sentence. Second sentence! Third sentence?"
        result = split_text_for_chatterbox(text)
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0], "First sentence.")
        self.assertEqual(result[1], "Second sentence!")
        self.assertEqual(result[2], "Third sentence?")
    
    def test_long_sentence_splits_at_comma_semicolon(self):
        """Long sentences (>max_chars) are split at comma/semicolon boundaries."""
        # Create a sentence longer than 50 chars with commas
        long_sentence = "This is a very long sentence that exceeds the limit, so it should be split at the comma, right here."
        result = split_text_for_chatterbox(long_sentence, max_chars=50)
        self.assertGreater(len(result), 1)
        # Each chunk should be <= 50 chars (or close if word boundaries prevent exact split)
        for chunk in result:
            self.assertLessEqual(len(chunk), 60)  # Allow some tolerance for word boundaries
    
    def test_very_long_word_falls_back_to_word_split(self):
        """When commas don't help, splits at word boundaries."""
        # A sentence with no commas that exceeds the limit
        words = " ".join(["word"] * 20)  # 99 chars
        result = split_text_for_chatterbox(words, max_chars=30)
        self.assertGreater(len(result), 1)
        for chunk in result:
            self.assertLessEqual(len(chunk), 35)  # Some tolerance
    
    def test_preserves_punctuation(self):
        """Sentence-ending punctuation is preserved."""
        text = "Question? Exclamation! Statement."
        result = split_text_for_chatterbox(text)
        self.assertTrue(result[0].endswith("?"))
        self.assertTrue(result[1].endswith("!"))
        self.assertTrue(result[2].endswith("."))
    
    def test_default_max_chars(self):
        """Default max_chars is 300."""
        # Create a sentence just under 300 chars
        short = "A" * 290 + "."
        result = split_text_for_chatterbox(short)
        self.assertEqual(len(result), 1)
        
        # Create a sentence over 300 chars with a comma
        long = "A" * 200 + ", " + "B" * 150 + "."
        result = split_text_for_chatterbox(long)
        self.assertGreater(len(result), 1)


class TestChatterboxPersonaSeparation(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """Get config values for test assertions"""
        # These must be configured in config.yml - tests will fail if not set
        cls.progress_voice = config.newsletter_progress_voice
        cls.news_voice = config.newsletter_news_voice
        cls.podcast_provider = config.podcast_voice_a_provider
        cls.podcast_model = config.podcast_voice_a_model_name
        
        # Validate required config
        if not cls.progress_voice:
            raise ValueError("Test requires 'newsletter_progress_voice' to be set in config.yml")
        if not cls.news_voice:
            raise ValueError("Test requires 'newsletter_news_voice' to be set in config.yml")
    
    @patch('year_progress_and_news_reporter_litellm.generate_gpt_response')
    @patch('year_progress_and_news_reporter_litellm.generate_gpt_response_voicebot')
    @patch('year_progress_and_news_reporter_litellm.chatterbox_text_to_speech')
    @patch('year_progress_and_news_reporter_litellm.get_client')
    def test_persona_and_voice_assignment(self, mock_get_client, mock_tts, mock_voicebot_gen, mock_progress_gen):
        """
        Verify that configured voices are used for Year Progress and News.
        Uses voices from config.yml: newsletter_progress_voice and newsletter_news_voice
        """
        # Setup mocks
        mock_progress_gen.return_value = f"{self.progress_voice} Progress Script"
        mock_voicebot_gen.return_value = f"{self.news_voice} News Script"
        mock_tts.return_value = True
        
        # Create dummy args using config values
        args = MagicMock()
        args.chatterbox_tts = True
        args.full_cache = False
        args.skip_audio = False
        args.skip_news_audio = False
        args.progress_voice = self.progress_voice
        args.news_voice = self.news_voice
        args.podcast_voice_a_provider = self.podcast_provider
        args.podcast_voice_a_model_name = self.podcast_model
        args.podcast_voice_b_provider = self.podcast_provider
        args.podcast_voice_b_model_name = self.podcast_model
        args.test = False
        args.skip_email = True
        args.use_cache = True
        args.refresh_cache = False
        args.cache_info = False
        args.list_voices = False
        args.list_chatterbox_voices = False
        args.html_only = False
        args.review = False
        args.podcast = False
        
        # Mock weather and other data dependencies to avoid network calls
        with patch('year_progress_and_news_reporter_litellm.get_weather', return_value=(20, "Sunny")), \
             patch('year_progress_and_news_reporter_litellm.load_news_cache', return_value={"technology": "tech", "financial": "fin", "india": "ind"}), \
             patch('year_progress_and_news_reporter_litellm.generate_newsletter_sections', return_value={"tech": [], "financial": [], "india": []}), \
             patch('year_progress_and_news_reporter_litellm.build_daily_bundle', return_value={
                 "meta": {"date_iso": "2025-12-23", "date_formatted": "December 23, 2025", "day_of_week": "Tuesday"}, 
                 "progress": {
                     "quote": {"text": "q", "author": "a"}, 
                     "lesson": {"topic": "t", "key_insight": "ki", "historical": "h", "application": "app"},
                     "time": {"days_completed": 357, "weeks_completed": 51.0, "days_left": 8, "weeks_left": 1.1, "percent_complete": 97.8, "percent_days_left": 2.2, "year": 2025, "total_days_in_year": 365},
                     "quarter": {"current_quarter": 4, "days_completed_in_quarter": 60, "days_left_in_quarter": 31, "percent_complete": 65.9, "days_in_quarter": 91},
                     "weather": {"temp_c": 20, "status": "Sunny"}
                 }, 
                 "news": {
                     "raw_sources": {},
                     "newsletter": {
                         "sections": {"tech": [], "financial": [], "india": []}, 
                         "voicebot_script": f"{self.news_voice} News Script",
                         "podcast_transcript": []
                     }
                 }
             }), \
             patch('year_progress_and_news_reporter_litellm.write_bundle_json', return_value=Path("dummy.json")), \
             patch('year_progress_and_news_reporter_litellm.save_to_output_dir', return_value=Path("dummy.html")), \
             patch('year_progress_and_news_reporter_litellm.send_email'), \
             patch('year_progress_and_news_reporter_litellm.parse_arguments', return_value=args):
            
            # Run main
            reporter.main()
            
            # ASSERTIONS
            
            # 1. Verify Transcript Generation Personas
            # Check progress generation called with configured voice
            progress_call_args = mock_progress_gen.call_args
            self.assertEqual(progress_call_args.kwargs['voice_persona'], self.progress_voice, 
                             f"Year Progress transcript should be generated with {self.progress_voice} persona (from config)")
            
            # Check news generation called with configured voice
            news_call_args = mock_voicebot_gen.call_args
            self.assertEqual(news_call_args.kwargs['voice_persona'], self.news_voice,
                             f"News briefing transcript should be generated with {self.news_voice} persona (from config)")
            
            # 2. Verify TTS Voice Assignment
            # TTS should be called twice
            self.assertEqual(mock_tts.call_count, 2)
            
            # First call for Year Progress
            progress_tts_call = mock_tts.call_args_list[0]
            self.assertEqual(progress_tts_call.kwargs['voice_name'], self.progress_voice,
                             f"Year Progress audio should use {self.progress_voice} voice file (from config)")
            
            # Second call for News
            news_tts_call = mock_tts.call_args_list[1]
            self.assertEqual(news_tts_call.kwargs['voice_name'], self.news_voice,
                             f"News briefing audio should use {self.news_voice} voice file (from config)")

if __name__ == '__main__':
    unittest.main()

