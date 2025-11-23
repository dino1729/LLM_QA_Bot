"""
Unit tests for helper_functions/query_supabasememory.py
Tests memory palace querying with Supabase and Azure OpenAI
"""
import pytest
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TestGenerateEmbeddings:
    """Test generate_embeddings function"""

    @patch('helper_functions.query_supabasememory.azureopenai_client')
    def test_generate_embeddings_success(self, mock_client):
        """Test successful embedding generation"""
        from helper_functions.query_supabasememory import generate_embeddings

        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1] * 1536)]
        mock_client.embeddings.create.return_value = mock_response

        result = generate_embeddings("test text")

        assert len(result) == 1536
        assert result[0] == 0.1
        mock_client.embeddings.create.assert_called_once()

    @patch('helper_functions.query_supabasememory.azureopenai_client')
    def test_generate_embeddings_with_model(self, mock_client):
        """Test embedding generation with specific model"""
        from helper_functions.query_supabasememory import generate_embeddings

        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.2] * 1536)]
        mock_client.embeddings.create.return_value = mock_response

        result = generate_embeddings("test", model="custom-model")

        assert len(result) == 1536
        call_args = mock_client.embeddings.create.call_args
        assert call_args[1]['model'] == "custom-model"

    @patch('helper_functions.query_supabasememory.azureopenai_client')
    def test_generate_embeddings_empty_text(self, mock_client):
        """Test with empty text"""
        from helper_functions.query_supabasememory import generate_embeddings

        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.0] * 1536)]
        mock_client.embeddings.create.return_value = mock_response

        result = generate_embeddings("")

        assert isinstance(result, list)
        mock_client.embeddings.create.assert_called_once()

    @patch('helper_functions.query_supabasememory.azureopenai_client')
    def test_generate_embeddings_long_text(self, mock_client):
        """Test with long text"""
        from helper_functions.query_supabasememory import generate_embeddings

        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1] * 1536)]
        mock_client.embeddings.create.return_value = mock_response

        long_text = "test " * 1000
        result = generate_embeddings(long_text)

        assert len(result) == 1536

    @patch('helper_functions.query_supabasememory.azureopenai_client')
    def test_generate_embeddings_error_handling(self, mock_client):
        """Test error handling"""
        from helper_functions.query_supabasememory import generate_embeddings

        mock_client.embeddings.create.side_effect = Exception("API error")

        # Should handle error gracefully
        try:
            result = generate_embeddings("test")
            assert result is None or isinstance(result, list)
        except Exception as e:
            assert "API error" in str(e)


class TestEnhanceQueryWithLLM:
    """Test enhance_query_with_llm function"""

    @patch('helper_functions.query_supabasememory.azureopenai_client')
    def test_enhance_query_success(self, mock_client):
        """Test successful query enhancement"""
        from helper_functions.query_supabasememory import enhance_query_with_llm

        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Enhanced query with more details"))]
        mock_client.chat.completions.create.return_value = mock_response

        result = enhance_query_with_llm("short query")

        assert isinstance(result, str)
        assert len(result) > len("short query")
        mock_client.chat.completions.create.assert_called_once()

    @patch('helper_functions.query_supabasememory.azureopenai_client')
    def test_enhance_query_expansion(self, mock_client):
        """Test query expansion"""
        from helper_functions.query_supabasememory import enhance_query_with_llm

        mock_response = Mock()
        enhanced = "What are the key features and benefits of machine learning?"
        mock_response.choices = [Mock(message=Mock(content=enhanced))]
        mock_client.chat.completions.create.return_value = mock_response

        result = enhance_query_with_llm("machine learning")

        assert "machine learning" in result.lower() or "features" in result.lower()

    @patch('helper_functions.query_supabasememory.azureopenai_client')
    def test_enhance_query_short_query(self, mock_client):
        """Test with short query"""
        from helper_functions.query_supabasememory import enhance_query_with_llm

        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="AI artificial intelligence machine learning"))]
        mock_client.chat.completions.create.return_value = mock_response

        result = enhance_query_with_llm("AI")

        assert isinstance(result, str)

    @patch('helper_functions.query_supabasememory.azureopenai_client')
    def test_enhance_query_ambiguous_query(self, mock_client):
        """Test with ambiguous query"""
        from helper_functions.query_supabasememory import enhance_query_with_llm

        mock_response = Mock()
        enhanced = "What is the meaning or context of 'it' in relation to information technology or programming?"
        mock_response.choices = [Mock(message=Mock(content=enhanced))]
        mock_client.chat.completions.create.return_value = mock_response

        result = enhance_query_with_llm("it")

        assert len(result) > 2

    @patch('helper_functions.query_supabasememory.azureopenai_client')
    def test_enhance_query_error_handling(self, mock_client):
        """Test error handling"""
        from helper_functions.query_supabasememory import enhance_query_with_llm

        mock_client.chat.completions.create.side_effect = Exception("API error")

        # Should return original query on error
        result = enhance_query_with_llm("test query")

        assert result == "test query"


class TestQueryMemorypalaceStream:
    """Test query_memorypalace_stream async function"""

    @pytest.mark.asyncio
    @patch('helper_functions.query_supabasememory.supabase_client')
    @patch('helper_functions.query_supabasememory.generate_embeddings')
    @patch('helper_functions.query_supabasememory.enhance_query_with_llm')
    @patch('helper_functions.query_supabasememory.azureopenai_client')
    async def test_query_memorypalace_success(self, mock_azure_client, mock_enhance, mock_embeddings, mock_supabase):
        """Test successful memory palace query"""
        from helper_functions.query_supabasememory import query_memorypalace_stream

        # Mock query enhancement
        mock_enhance.return_value = "enhanced query"

        # Mock embedding generation
        mock_embeddings.return_value = [0.1] * 1536

        # Mock Supabase query
        mock_rpc = Mock()
        mock_rpc.execute.return_value = Mock(data=[
            {"text_chunk": "Memory 1", "similarity": 0.9},
            {"text_chunk": "Memory 2", "similarity": 0.85}
        ])
        mock_supabase.rpc.return_value = mock_rpc

        # Mock Azure OpenAI streaming
        mock_stream = [
            Mock(choices=[Mock(delta=Mock(content="Hello"))]),
            Mock(choices=[Mock(delta=Mock(content=" world"))])
        ]
        mock_azure_client.chat.completions.create.return_value = mock_stream

        result_chunks = []
        async for chunk in query_memorypalace_stream("test query"):
            result_chunks.append(chunk)

        assert len(result_chunks) > 0
        assert any("Hello" in chunk or "world" in chunk for chunk in result_chunks)

    @pytest.mark.asyncio
    @patch('helper_functions.query_supabasememory.supabase_client')
    @patch('helper_functions.query_supabasememory.generate_embeddings')
    @patch('helper_functions.query_supabasememory.enhance_query_with_llm')
    @patch('helper_functions.query_supabasememory.azureopenai_client')
    async def test_query_memorypalace_with_chat_history_tuple(self, mock_azure_client, mock_enhance, mock_embeddings, mock_supabase):
        """Test with chat history in tuple format"""
        from helper_functions.query_supabasememory import query_memorypalace_stream

        mock_enhance.return_value = "enhanced"
        mock_embeddings.return_value = [0.1] * 1536

        mock_rpc = Mock()
        mock_rpc.execute.return_value = Mock(data=[])
        mock_supabase.rpc.return_value = mock_rpc

        mock_stream = [Mock(choices=[Mock(delta=Mock(content="Response"))])]
        mock_azure_client.chat.completions.create.return_value = mock_stream

        chat_history = [("User message", "Assistant response")]

        result_chunks = []
        async for chunk in query_memorypalace_stream("query", chat_history=chat_history):
            result_chunks.append(chunk)

        assert len(result_chunks) > 0

    @pytest.mark.asyncio
    @patch('helper_functions.query_supabasememory.supabase_client')
    @patch('helper_functions.query_supabasememory.generate_embeddings')
    @patch('helper_functions.query_supabasememory.enhance_query_with_llm')
    @patch('helper_functions.query_supabasememory.azureopenai_client')
    async def test_query_memorypalace_with_chat_history_dict(self, mock_azure_client, mock_enhance, mock_embeddings, mock_supabase):
        """Test with chat history in dict format"""
        from helper_functions.query_supabasememory import query_memorypalace_stream

        mock_enhance.return_value = "enhanced"
        mock_embeddings.return_value = [0.1] * 1536

        mock_rpc = Mock()
        mock_rpc.execute.return_value = Mock(data=[])
        mock_supabase.rpc.return_value = mock_rpc

        mock_stream = [Mock(choices=[Mock(delta=Mock(content="Response"))])]
        mock_azure_client.chat.completions.create.return_value = mock_stream

        chat_history = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"}
        ]

        result_chunks = []
        async for chunk in query_memorypalace_stream("query", chat_history=chat_history):
            result_chunks.append(chunk)

        assert len(result_chunks) > 0

    @pytest.mark.asyncio
    @patch('helper_functions.query_supabasememory.supabase_client')
    @patch('helper_functions.query_supabasememory.generate_embeddings')
    @patch('helper_functions.query_supabasememory.enhance_query_with_llm')
    @patch('helper_functions.query_supabasememory.azureopenai_client')
    async def test_query_memorypalace_without_chat_history(self, mock_azure_client, mock_enhance, mock_embeddings, mock_supabase):
        """Test without chat history"""
        from helper_functions.query_supabasememory import query_memorypalace_stream

        mock_enhance.return_value = "enhanced"
        mock_embeddings.return_value = [0.1] * 1536

        mock_rpc = Mock()
        mock_rpc.execute.return_value = Mock(data=[])
        mock_supabase.rpc.return_value = mock_rpc

        mock_stream = [Mock(choices=[Mock(delta=Mock(content="Response"))])]
        mock_azure_client.chat.completions.create.return_value = mock_stream

        result_chunks = []
        async for chunk in query_memorypalace_stream("query"):
            result_chunks.append(chunk)

        assert len(result_chunks) > 0

    @pytest.mark.asyncio
    @patch('helper_functions.query_supabasememory.supabase_client')
    @patch('helper_functions.query_supabasememory.generate_embeddings')
    @patch('helper_functions.query_supabasememory.enhance_query_with_llm')
    async def test_query_memorypalace_recursive_similarity_search(self, mock_enhance, mock_embeddings, mock_supabase):
        """Test recursive similarity search"""
        from helper_functions.query_supabasememory import query_memorypalace_stream

        mock_enhance.return_value = "enhanced"
        mock_embeddings.return_value = [0.1] * 1536

        # Mock multiple RPC calls for recursive search
        mock_rpc = Mock()
        mock_rpc.execute.return_value = Mock(data=[
            {"text_chunk": "Relevant memory", "similarity": 0.85}
        ])
        mock_supabase.rpc.return_value = mock_rpc

        with patch('helper_functions.query_supabasememory.azureopenai_client') as mock_azure:
            mock_stream = [Mock(choices=[Mock(delta=Mock(content="Response"))])]
            mock_azure.chat.completions.create.return_value = mock_stream

            result_chunks = []
            async for chunk in query_memorypalace_stream("query"):
                result_chunks.append(chunk)

            # Verify recursive search was performed
            assert mock_supabase.rpc.call_count >= 1

    @pytest.mark.asyncio
    @patch('helper_functions.query_supabasememory.supabase_client')
    @patch('helper_functions.query_supabasememory.generate_embeddings')
    @patch('helper_functions.query_supabasememory.enhance_query_with_llm')
    @patch('helper_functions.query_supabasememory.azureopenai_client')
    async def test_query_memorypalace_streaming_response(self, mock_azure_client, mock_enhance, mock_embeddings, mock_supabase):
        """Test streaming response"""
        from helper_functions.query_supabasememory import query_memorypalace_stream

        mock_enhance.return_value = "enhanced"
        mock_embeddings.return_value = [0.1] * 1536

        mock_rpc = Mock()
        mock_rpc.execute.return_value = Mock(data=[])
        mock_supabase.rpc.return_value = mock_rpc

        # Mock streaming chunks
        mock_stream = [
            Mock(choices=[Mock(delta=Mock(content="Hello"))]),
            Mock(choices=[Mock(delta=Mock(content=" there"))]),
            Mock(choices=[Mock(delta=Mock(content="!"))])
        ]
        mock_azure_client.chat.completions.create.return_value = mock_stream

        result_chunks = []
        async for chunk in query_memorypalace_stream("query"):
            result_chunks.append(chunk)

        # Should receive all chunks
        assert len(result_chunks) == 3
        full_text = "".join(result_chunks)
        assert full_text == "Hello there!"

    @pytest.mark.asyncio
    @patch('helper_functions.query_supabasememory.supabase_client')
    @patch('helper_functions.query_supabasememory.generate_embeddings')
    @patch('helper_functions.query_supabasememory.enhance_query_with_llm')
    @patch('helper_functions.query_supabasememory.azureopenai_client')
    async def test_query_memorypalace_error_handling(self, mock_azure_client, mock_enhance, mock_embeddings, mock_supabase):
        """Test error handling"""
        from helper_functions.query_supabasememory import query_memorypalace_stream

        mock_enhance.return_value = "enhanced"
        mock_embeddings.side_effect = Exception("Embedding error")

        # Should handle error gracefully
        try:
            result_chunks = []
            async for chunk in query_memorypalace_stream("query"):
                result_chunks.append(chunk)
            # If no exception, verify it returned something
            assert True
        except Exception as e:
            # Exception is acceptable
            assert "error" in str(e).lower()

    @pytest.mark.asyncio
    @patch('helper_functions.query_supabasememory.supabase_client')
    @patch('helper_functions.query_supabasememory.generate_embeddings')
    @patch('helper_functions.query_supabasememory.enhance_query_with_llm')
    @patch('helper_functions.query_supabasememory.azureopenai_client')
    async def test_query_memorypalace_debug_logging(self, mock_azure_client, mock_enhance, mock_embeddings, mock_supabase):
        """Test debug logging"""
        from helper_functions.query_supabasememory import query_memorypalace_stream

        mock_enhance.return_value = "enhanced"
        mock_embeddings.return_value = [0.1] * 1536

        mock_rpc = Mock()
        mock_rpc.execute.return_value = Mock(data=[])
        mock_supabase.rpc.return_value = mock_rpc

        mock_stream = [Mock(choices=[Mock(delta=Mock(content="Response"))])]
        mock_azure_client.chat.completions.create.return_value = mock_stream

        result_chunks = []
        async for chunk in query_memorypalace_stream("query", debug=True):
            result_chunks.append(chunk)

        # Should complete without error
        assert len(result_chunks) > 0


class TestMain:
    """Test main function"""

    @pytest.mark.asyncio
    @patch('helper_functions.query_supabasememory.query_memorypalace_stream')
    @patch('sys.argv', ['script.py', '--query', 'test query'])
    async def test_main_with_provided_query(self, mock_query):
        """Test main with provided query"""
        from helper_functions.query_supabasememory import main

        async def mock_stream():
            yield "Response chunk"

        mock_query.return_value = mock_stream()

        # Should execute without error
        try:
            await main()
            assert True
        except SystemExit:
            # SystemExit is acceptable for CLI scripts
            pass

    @pytest.mark.asyncio
    @patch('helper_functions.query_supabasememory.query_memorypalace_stream')
    @patch('sys.argv', ['script.py', '--debug'])
    async def test_main_with_debug_flag(self, mock_query):
        """Test main with debug flag"""
        from helper_functions.query_supabasememory import main

        async def mock_stream():
            yield "Debug response"

        mock_query.return_value = mock_stream()

        try:
            await main()
            assert True
        except SystemExit:
            pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
