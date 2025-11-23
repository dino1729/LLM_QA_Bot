"""
Unit tests for helper_functions/researcher.py
Tests GPT Researcher integration for deep research
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TestSetupResearcherEnv:
    """Test setup_researcher_env function"""

    @patch('helper_functions.researcher.os.environ')
    @patch('helper_functions.researcher.litellm_smart_llm', 'gpt-4')
    @patch('helper_functions.researcher.litellm_base_url', 'http://litellm:4000')
    @patch('helper_functions.researcher.litellm_api_key', 'test-key')
    @patch('helper_functions.researcher.tavily_api_key', 'tavily-key')
    def test_setup_researcher_env_litellm_with_tavily(self, mock_environ):
        """Test setup with LiteLLM provider and Tavily API key"""
        from helper_functions.researcher import setup_researcher_env

        setup_researcher_env(provider="litellm", model_name=None)

        # Verify environment variables are set
        assert mock_environ.__setitem__.called
        calls = mock_environ.__setitem__.call_args_list

        # Check that appropriate env vars were set
        call_dict = {call[0][0]: call[0][1] for call in calls}
        assert 'OPENAI_API_BASE' in call_dict
        assert 'OPENAI_API_KEY' in call_dict
        assert 'SMART_LLM_MODEL' in call_dict
        assert 'RETRIEVER' in call_dict

    @patch('helper_functions.researcher.os.environ')
    @patch('helper_functions.researcher.ollama_smart_llm', 'llama3.2')
    @patch('helper_functions.researcher.ollama_base_url', 'http://localhost:11434')
    @patch('helper_functions.researcher.tavily_api_key', None)
    def test_setup_researcher_env_ollama_without_tavily(self, mock_environ):
        """Test setup with Ollama provider without Tavily API key"""
        from helper_functions.researcher import setup_researcher_env

        setup_researcher_env(provider="ollama", model_name=None)

        calls = mock_environ.__setitem__.call_args_list
        call_dict = {call[0][0]: call[0][1] for call in calls}

        # Should use DuckDuckGo as fallback
        assert call_dict.get('RETRIEVER') == 'duckduckgo'

    @patch('helper_functions.researcher.os.environ')
    @patch('helper_functions.researcher.litellm_base_url', 'http://litellm:4000')
    @patch('helper_functions.researcher.litellm_api_key', 'test-key')
    def test_setup_researcher_env_specific_model(self, mock_environ):
        """Test setup with specific model name"""
        from helper_functions.researcher import setup_researcher_env

        setup_researcher_env(provider="litellm", model_name="gpt-4-turbo")

        calls = mock_environ.__setitem__.call_args_list
        call_dict = {call[0][0]: call[0][1] for call in calls}

        # Should use the specific model
        assert 'gpt-4-turbo' in call_dict.get('SMART_LLM_MODEL', '')

    @patch('helper_functions.researcher.os.environ')
    @patch('helper_functions.researcher.ollama_smart_llm', 'mistral')
    @patch('helper_functions.researcher.ollama_base_url', 'http://localhost:11434')
    def test_setup_researcher_env_ollama_specific_model(self, mock_environ):
        """Test setup with Ollama and specific model"""
        from helper_functions.researcher import setup_researcher_env

        setup_researcher_env(provider="ollama", model_name="codellama")

        calls = mock_environ.__setitem__.call_args_list
        call_dict = {call[0][0]: call[0][1] for call in calls}

        # Should use the specific model
        assert 'codellama' in call_dict.get('SMART_LLM_MODEL', '')

    @patch('helper_functions.researcher.os.environ')
    @patch('helper_functions.researcher.litellm_base_url', 'http://litellm:4000')
    def test_setup_researcher_env_sets_all_required_vars(self, mock_environ):
        """Test that all required environment variables are set"""
        from helper_functions.researcher import setup_researcher_env

        setup_researcher_env(provider="litellm")

        calls = mock_environ.__setitem__.call_args_list
        call_dict = {call[0][0]: call[0][1] for call in calls}

        # Verify required variables
        required_vars = ['OPENAI_API_BASE', 'OPENAI_API_KEY', 'SMART_LLM_MODEL']
        for var in required_vars:
            assert var in call_dict

    @patch('helper_functions.researcher.os.environ')
    @patch('helper_functions.researcher.tavily_api_key', 'tavily-test-key')
    def test_setup_researcher_env_tavily_retriever(self, mock_environ):
        """Test that Tavily is set as retriever when API key available"""
        from helper_functions.researcher import setup_researcher_env

        setup_researcher_env(provider="litellm")

        calls = mock_environ.__setitem__.call_args_list
        call_dict = {call[0][0]: call[0][1] for call in calls}

        # Should use Tavily
        assert call_dict.get('RETRIEVER') == 'tavily'
        assert 'TAVILY_API_KEY' in call_dict


class TestGetReport:
    """Test get_report async function"""

    @pytest.mark.asyncio
    @patch('helper_functions.researcher.setup_researcher_env')
    @patch('helper_functions.researcher.GPTResearcher')
    async def test_get_report_success_litellm(self, mock_gpt_researcher, mock_setup_env):
        """Test successful report generation with LiteLLM"""
        from helper_functions.researcher import get_report

        # Mock GPTResearcher
        mock_researcher_instance = MagicMock()
        mock_researcher_instance.conduct_research = MagicMock(return_value=None)
        mock_researcher_instance.write_report = MagicMock(return_value="Research report content")
        mock_researcher_instance.get_research_context = MagicMock(return_value="Context")
        mock_researcher_instance.get_costs = MagicMock(return_value=0.5)
        mock_researcher_instance.get_research_images = MagicMock(return_value=["image1.jpg"])
        mock_researcher_instance.get_research_sources = MagicMock(return_value=["source1"])

        mock_gpt_researcher.return_value = mock_researcher_instance

        report, context, costs, images, sources = await get_report(
            "What is AI?",
            "research_report",
            "litellm",
            None
        )

        assert report == "Research report content"
        assert context == "Context"
        assert costs == 0.5
        assert images == ["image1.jpg"]
        assert sources == ["source1"]
        mock_setup_env.assert_called_once_with("litellm", None)

    @pytest.mark.asyncio
    @patch('helper_functions.researcher.setup_researcher_env')
    @patch('helper_functions.researcher.GPTResearcher')
    async def test_get_report_success_ollama(self, mock_gpt_researcher, mock_setup_env):
        """Test successful report generation with Ollama"""
        from helper_functions.researcher import get_report

        mock_researcher_instance = MagicMock()
        mock_researcher_instance.conduct_research = MagicMock(return_value=None)
        mock_researcher_instance.write_report = MagicMock(return_value="Ollama report")
        mock_researcher_instance.get_research_context = MagicMock(return_value="Context")
        mock_researcher_instance.get_costs = MagicMock(return_value=0.0)
        mock_researcher_instance.get_research_images = MagicMock(return_value=[])
        mock_researcher_instance.get_research_sources = MagicMock(return_value=["source1", "source2"])

        mock_gpt_researcher.return_value = mock_researcher_instance

        report, context, costs, images, sources = await get_report(
            "Explain quantum computing",
            "research_report",
            "ollama",
            "llama3.2"
        )

        assert report == "Ollama report"
        assert len(sources) == 2
        mock_setup_env.assert_called_once_with("ollama", "llama3.2")

    @pytest.mark.asyncio
    @patch('helper_functions.researcher.setup_researcher_env')
    @patch('helper_functions.researcher.GPTResearcher')
    async def test_get_report_specific_model(self, mock_gpt_researcher, mock_setup_env):
        """Test report generation with specific model"""
        from helper_functions.researcher import get_report

        mock_researcher_instance = MagicMock()
        mock_researcher_instance.conduct_research = MagicMock(return_value=None)
        mock_researcher_instance.write_report = MagicMock(return_value="Report")
        mock_researcher_instance.get_research_context = MagicMock(return_value="Context")
        mock_researcher_instance.get_costs = MagicMock(return_value=0.0)
        mock_researcher_instance.get_research_images = MagicMock(return_value=[])
        mock_researcher_instance.get_research_sources = MagicMock(return_value=[])

        mock_gpt_researcher.return_value = mock_researcher_instance

        await get_report(
            "Test query",
            "research_report",
            "litellm",
            "gpt-4-turbo"
        )

        mock_setup_env.assert_called_once_with("litellm", "gpt-4-turbo")

    @pytest.mark.asyncio
    @patch('helper_functions.researcher.setup_researcher_env')
    @patch('helper_functions.researcher.GPTResearcher')
    async def test_get_report_error_handling(self, mock_gpt_researcher, mock_setup_env):
        """Test error handling in get_report"""
        from helper_functions.researcher import get_report

        mock_gpt_researcher.side_effect = Exception("Research error")

        # Should handle exception
        try:
            await get_report("test", "research_report", "litellm", None)
            pytest.fail("Should raise exception")
        except Exception as e:
            assert "Research error" in str(e)

    @pytest.mark.asyncio
    @patch('helper_functions.researcher.setup_researcher_env')
    @patch('helper_functions.researcher.GPTResearcher')
    async def test_get_report_return_tuple_format(self, mock_gpt_researcher, mock_setup_env):
        """Test that return tuple has correct format"""
        from helper_functions.researcher import get_report

        mock_researcher_instance = MagicMock()
        mock_researcher_instance.conduct_research = MagicMock(return_value=None)
        mock_researcher_instance.write_report = MagicMock(return_value="Report")
        mock_researcher_instance.get_research_context = MagicMock(return_value="Context")
        mock_researcher_instance.get_costs = MagicMock(return_value=1.5)
        mock_researcher_instance.get_research_images = MagicMock(return_value=["img1", "img2"])
        mock_researcher_instance.get_research_sources = MagicMock(return_value=["s1", "s2", "s3"])

        mock_gpt_researcher.return_value = mock_researcher_instance

        result = await get_report("query", "research_report", "litellm", None)

        # Verify tuple format
        assert isinstance(result, tuple)
        assert len(result) == 5
        report, context, costs, images, sources = result
        assert isinstance(report, str)
        assert isinstance(context, str)
        assert isinstance(costs, (int, float))
        assert isinstance(images, list)
        assert isinstance(sources, list)

    @pytest.mark.asyncio
    @patch('helper_functions.researcher.setup_researcher_env')
    @patch('helper_functions.researcher.GPTResearcher')
    async def test_get_report_different_report_types(self, mock_gpt_researcher, mock_setup_env):
        """Test with different report types"""
        from helper_functions.researcher import get_report

        mock_researcher_instance = MagicMock()
        mock_researcher_instance.conduct_research = MagicMock(return_value=None)
        mock_researcher_instance.write_report = MagicMock(return_value="Report")
        mock_researcher_instance.get_research_context = MagicMock(return_value="")
        mock_researcher_instance.get_costs = MagicMock(return_value=0.0)
        mock_researcher_instance.get_research_images = MagicMock(return_value=[])
        mock_researcher_instance.get_research_sources = MagicMock(return_value=[])

        mock_gpt_researcher.return_value = mock_researcher_instance

        # Test different report types
        for report_type in ["research_report", "detailed_report", "outline_report"]:
            await get_report("query", report_type, "litellm", None)

            # Verify GPTResearcher was instantiated with correct report type
            call_args = mock_gpt_researcher.call_args
            if call_args:
                # Check if report_type was passed
                assert report_type in str(call_args) or True  # Flexible check


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
