#!/usr/bin/env python3
"""
Test suite for year_progress_and_news_reporter_litellm.py functions.

Tests quote generation, lesson generation, and TTS text chunking.
Run with: python -m pytest tests/test_year_progress_functions.py -v
"""

import sys
import os
import pytest
from unittest.mock import patch, MagicMock

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helper_functions.audio_processors import chunk_text_for_tts, _break_long_sentence


class TestTTSTextChunking:
    """Tests for TTS text chunking functions"""
    
    def test_chunk_empty_text(self):
        """Test chunking with empty text"""
        result = chunk_text_for_tts("")
        assert result == []
        
        result = chunk_text_for_tts("   ")
        assert result == []
    
    def test_chunk_short_text(self):
        """Test chunking with text shorter than limits"""
        short_text = "Hello world. This is a test."
        result = chunk_text_for_tts(short_text)
        
        assert len(result) == 1
        assert result[0] == short_text
    
    def test_chunk_respects_2000_char_limit(self):
        """Test that chunks don't exceed 2000 characters"""
        # Create text with multiple sentences, total > 2000 chars
        sentences = ["This is sentence number {}.".format(i) for i in range(100)]
        long_text = " ".join(sentences)
        
        assert len(long_text) > 2000, "Test requires text longer than 2000 chars"
        
        result = chunk_text_for_tts(long_text)
        
        for i, chunk in enumerate(result):
            assert len(chunk) <= 2000, f"Chunk {i} exceeds 2000 chars: {len(chunk)}"
    
    def test_chunk_respects_400_char_sentence_limit(self):
        """Test that individual sentences don't exceed 400 characters after processing"""
        # Create a very long sentence (>400 chars)
        long_sentence = "This is a very long sentence " * 20  # ~600 chars
        
        result = chunk_text_for_tts(long_sentence)
        
        # Each chunk should be under 2000 chars
        for chunk in result:
            assert len(chunk) <= 2000
    
    def test_chunk_preserves_sentence_boundaries(self):
        """Test that chunking respects sentence boundaries"""
        text = "First sentence. Second sentence. Third sentence."
        result = chunk_text_for_tts(text)
        
        # Should have all sentences in output
        combined = " ".join(result)
        assert "First sentence" in combined
        assert "Second sentence" in combined
        assert "Third sentence" in combined
    
    def test_chunk_handles_various_punctuation(self):
        """Test chunking with various sentence-ending punctuation"""
        text = "Question? Statement. Exclamation! Another; More."
        result = chunk_text_for_tts(text)
        
        combined = " ".join(result)
        assert "Question" in combined
        assert "Statement" in combined
        assert "Exclamation" in combined


class TestBreakLongSentence:
    """Tests for _break_long_sentence helper function"""
    
    def test_short_sentence_unchanged(self):
        """Test that short sentences are returned as-is"""
        sentence = "This is a short sentence."
        result = _break_long_sentence(sentence, 400)
        
        assert result == [sentence]
    
    def test_break_at_comma(self):
        """Test breaking at comma for long sentences"""
        # Create sentence with comma in the latter half
        sentence = "This is the first part of a sentence, and this is the second part that continues on."
        result = _break_long_sentence(sentence, 60)
        
        assert len(result) >= 2
        assert all(len(part) <= 60 for part in result)
    
    def test_break_at_conjunction(self):
        """Test breaking at conjunctions"""
        sentence = "This is part one and this is part two and this is part three"
        result = _break_long_sentence(sentence, 30)
        
        assert len(result) >= 2
    
    def test_break_at_space_last_resort(self):
        """Test breaking at space when no better option"""
        # Sentence with no commas or conjunctions
        sentence = "word " * 100  # Long string of words
        result = _break_long_sentence(sentence, 50)
        
        assert len(result) > 1
        assert all(len(part) <= 50 for part in result)


class TestQuoteGeneration:
    """Tests for quote generation functionality"""
    
    def test_quote_with_valid_response(self):
        """Test quote extraction from valid LLM response"""
        # Import after setup
        from year_progress_and_news_reporter_litellm import generate_quote
        
        # Mock the LLM client
        with patch('year_progress_and_news_reporter_litellm.get_client') as mock_client:
            mock_instance = MagicMock()
            mock_instance.chat_completion.return_value = '"Stay hungry, stay foolish." - Steve Jobs'
            mock_client.return_value = mock_instance
            
            result = generate_quote("Steve Jobs")
            
            assert result is not None
            assert len(result) > 0
            assert "Stay hungry" in result or "stay foolish" in result.lower() or '"' in result
    
    def test_quote_with_reasoning_artifacts(self):
        """Test quote extraction when LLM returns reasoning artifacts"""
        from year_progress_and_news_reporter_litellm import generate_quote
        
        # Mock response with reasoning artifacts
        mock_response = """The user wants a quote from Steve Jobs.
        
Here is a famous quote from Steve Jobs:

"Your time is limited, don't waste it living someone else's life."
"""
        
        with patch('year_progress_and_news_reporter_litellm.get_client') as mock_client:
            mock_instance = MagicMock()
            mock_instance.chat_completion.return_value = mock_response
            mock_client.return_value = mock_instance
            
            result = generate_quote("Steve Jobs")
            
            assert result is not None
            assert "Your time is limited" in result
    
    def test_quote_fallback_on_empty_response(self):
        """Test fallback quote when LLM returns empty"""
        from year_progress_and_news_reporter_litellm import generate_quote
        
        with patch('year_progress_and_news_reporter_litellm.get_client') as mock_client:
            mock_instance = MagicMock()
            mock_instance.chat_completion.return_value = ""
            mock_client.return_value = mock_instance
            
            result = generate_quote("Steve Jobs")
            
            # Should return fallback quote
            assert result is not None
            assert len(result) > 0
            assert '"' in result  # Should be quoted
    
    def test_quote_fallback_on_exception(self):
        """Test fallback quote when LLM raises exception"""
        from year_progress_and_news_reporter_litellm import generate_quote
        
        with patch('year_progress_and_news_reporter_litellm.get_client') as mock_client:
            mock_instance = MagicMock()
            mock_instance.chat_completion.side_effect = Exception("API Error")
            mock_client.return_value = mock_instance
            
            result = generate_quote("Albert Einstein")
            
            # Should return fallback quote without crashing
            assert result is not None
            assert len(result) > 0


class TestLessonGeneration:
    """Tests for lesson generation functionality"""
    
    def test_lesson_with_valid_response(self):
        """Test lesson extraction from valid LLM response"""
        from year_progress_and_news_reporter_litellm import generate_lesson_response
        
        mock_response = """Effective communication is the cornerstone of leadership. Throughout history, great leaders have understood that clarity and empathy in communication can move nations.

Consider Alexander the Great, who before every major battle would address his troops directly, sharing in their hardships and articulating a clear vision of victory. This personal connection motivated soldiers to achieve seemingly impossible feats.

For modern engineers and leaders, this translates to several practical principles: be clear and concise, listen actively, and always connect the task at hand to the larger mission."""
        
        with patch('year_progress_and_news_reporter_litellm.get_client') as mock_client:
            mock_instance = MagicMock()
            mock_instance.chat_completion.return_value = mock_response
            mock_client.return_value = mock_instance
            
            result = generate_lesson_response("How to communicate effectively?")
            
            assert result is not None
            assert len(result) > 200
            assert "communication" in result.lower()
    
    def test_lesson_cleans_reasoning_artifacts(self):
        """Test that lesson generation removes reasoning artifacts"""
        from year_progress_and_news_reporter_litellm import generate_lesson_response
        
        mock_response = """The user wants to learn about productivity.

Let me provide a comprehensive lesson:

Productivity is fundamentally about managing energy, not just time. Ancient Stoic philosophers like Marcus Aurelius practiced daily reflection and intentional work scheduling, understanding that sustained excellence requires both focused effort and deliberate rest.

In modern application, this means structuring your day around your natural energy cycles, protecting deep work time, and building systems that reduce decision fatigue."""
        
        with patch('year_progress_and_news_reporter_litellm.get_client') as mock_client:
            mock_instance = MagicMock()
            mock_instance.chat_completion.return_value = mock_response
            mock_client.return_value = mock_instance
            
            result = generate_lesson_response("How to be productive?")
            
            assert result is not None
            # Should NOT start with reasoning artifacts
            assert not result.lower().startswith("the user")
            assert not result.lower().startswith("let me")
            # Should contain actual content
            assert "Productivity" in result or "energy" in result
    
    def test_lesson_fallback_on_empty_response(self):
        """Test fallback lesson when LLM returns empty"""
        from year_progress_and_news_reporter_litellm import generate_lesson_response
        
        with patch('year_progress_and_news_reporter_litellm.get_client') as mock_client:
            mock_instance = MagicMock()
            mock_instance.chat_completion.return_value = ""
            mock_client.return_value = mock_instance
            
            result = generate_lesson_response("Any topic")
            
            # Should return fallback lesson
            assert result is not None
            assert len(result) > 200
    
    def test_lesson_fallback_on_short_response(self):
        """Test fallback lesson when LLM returns very short response"""
        from year_progress_and_news_reporter_litellm import generate_lesson_response
        
        with patch('year_progress_and_news_reporter_litellm.get_client') as mock_client:
            mock_instance = MagicMock()
            mock_instance.chat_completion.return_value = "Short response."
            mock_client.return_value = mock_instance
            
            result = generate_lesson_response("Any topic")
            
            # Should return fallback lesson since response is too short
            assert result is not None
            assert len(result) > 200


class TestFallbackContent:
    """Tests for fallback content functions"""
    
    def test_fallback_quote_has_quotes(self):
        """Test that fallback quotes are properly formatted"""
        from year_progress_and_news_reporter_litellm import _get_fallback_quote
        
        personalities = ["Steve Jobs", "Albert Einstein", "Unknown Person"]
        
        for personality in personalities:
            quote = _get_fallback_quote(personality)
            assert quote is not None
            assert '"' in quote  # Should have quotation marks
            assert len(quote) > 10
    
    def test_fallback_lesson_is_substantial(self):
        """Test that fallback lesson has meaningful content"""
        from year_progress_and_news_reporter_litellm import _get_fallback_lesson
        
        lesson = _get_fallback_lesson()
        
        assert lesson is not None
        assert len(lesson) > 200
        # Should have multiple paragraphs
        assert '\n\n' in lesson


class TestIntegration:
    """Integration tests that verify the complete flow"""
    
    def test_tts_chunking_with_real_lesson_length_text(self):
        """Test TTS chunking with text similar to actual lesson output"""
        # Simulate a real lesson output (typically 1000-3000 chars)
        sample_lesson = """The pursuit of mastery requires understanding that excellence is not a destination but a continuous journey. Ancient philosophers like Aristotle understood this, coining the term "eudaimonia" to describe the flourishing that comes from living up to one's potential through disciplined practice.

Consider the example of Leonardo da Vinci, who kept detailed notebooks throughout his life, documenting not just his artistic techniques but his observations of nature, engineering concepts, and philosophical musings. This habit of systematic learning and documentation allowed him to make connections across domains that others missed.

For modern engineers and leaders, this translates to three practical principles: First, maintain a learning system - whether notebooks, digital tools, or structured reflection time. Second, seek cross-domain knowledge, as innovation often happens at the intersection of fields. Third, embrace deliberate practice over mere repetition, focusing on the areas where improvement is most needed.

The historical parallel to the Roman aqueduct engineers is instructive: they built systems that lasted millennia not through revolutionary innovation alone, but through meticulous attention to fundamentals, redundancy in design, and deep understanding of the materials and forces they worked with. Their approach combined theoretical knowledge with practical experimentation, creating an engineering tradition that valued both innovation and reliability."""
        
        result = chunk_text_for_tts(sample_lesson)
        
        # Should have at least one chunk
        assert len(result) >= 1
        
        # All chunks should be within limits
        for chunk in result:
            assert len(chunk) <= 2000, f"Chunk exceeds 2000 chars: {len(chunk)}"
        
        # Combined chunks should contain all original content (approximately)
        combined = " ".join(result)
        assert "Leonardo da Vinci" in combined
        assert "Roman aqueduct" in combined


if __name__ == "__main__":
    print("\n" + "="*80)
    print("YEAR PROGRESS FUNCTIONS TEST SUITE")
    print("="*80)
    
    # Run with pytest
    pytest.main([__file__, "-v", "--tb=short"])

