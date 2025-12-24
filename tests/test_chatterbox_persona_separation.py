import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path

# Import the module to test
import year_progress_and_news_reporter_litellm as reporter

class TestChatterboxPersonaSeparation(unittest.TestCase):
    
    @patch('year_progress_and_news_reporter_litellm.generate_gpt_response')
    @patch('year_progress_and_news_reporter_litellm.generate_gpt_response_voicebot')
    @patch('year_progress_and_news_reporter_litellm.chatterbox_text_to_speech')
    @patch('year_progress_and_news_reporter_litellm.get_client')
    def test_persona_and_voice_assignment(self, mock_get_client, mock_tts, mock_voicebot_gen, mock_progress_gen):
        """
        Verify that Jensen Huang is used for Year Progress and Morgan Freeman for News.
        """
        # Setup mocks
        mock_progress_gen.return_value = "Jensen Progress Script"
        mock_voicebot_gen.return_value = "Morgan News Script"
        mock_tts.return_value = True
        
        # Create dummy args
        args = MagicMock()
        args.chatterbox_tts = True
        args.full_cache = False
        args.skip_audio = False
        args.skip_news_audio = False
        args.progress_voice = "jensen_huang"
        args.news_voice = "morgan_freeman"
        args.podcast_voice_a_provider = "litellm"
        args.podcast_voice_a_model_name = "deepseek-v3.2"
        args.podcast_voice_b_provider = "litellm"
        args.podcast_voice_b_model_name = "deepseek-v3.2"
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
                         "voicebot_script": "Morgan News Script",
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
            # Check progress generation called with Jensen
            progress_call_args = mock_progress_gen.call_args
            self.assertEqual(progress_call_args.kwargs['voice_persona'], "jensen_huang", 
                             "Year Progress transcript should be generated with Jensen Huang persona")
            
            # Check news generation called with Morgan
            news_call_args = mock_voicebot_gen.call_args
            self.assertEqual(news_call_args.kwargs['voice_persona'], "morgan_freeman",
                             "News briefing transcript should be generated with Morgan Freeman persona")
            
            # 2. Verify TTS Voice Assignment
            # TTS should be called twice
            self.assertEqual(mock_tts.call_count, 2)
            
            # First call for Year Progress
            progress_tts_call = mock_tts.call_args_list[0]
            self.assertEqual(progress_tts_call.kwargs['voice_name'], "jensen_huang",
                             "Year Progress audio should use jensen_huang voice file")
            
            # Second call for News
            news_tts_call = mock_tts.call_args_list[1]
            self.assertEqual(news_tts_call.kwargs['voice_name'], "morgan_freeman",
                             "News briefing audio should use morgan_freeman voice file")

if __name__ == '__main__':
    unittest.main()

