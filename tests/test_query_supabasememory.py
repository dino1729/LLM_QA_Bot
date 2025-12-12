import pytest
from unittest.mock import Mock, patch
from helper_functions import query_supabasememory

class TestMain:
    """Tests for main execution"""
    
    @pytest.mark.asyncio
    @patch('helper_functions.query_supabasememory.query_memorypalace_stream')
    @patch('sys.argv', ['script.py', '--query', 'test query'])
    @patch('helper_functions.query_supabasememory.asyncio.run')
    async def test_main_with_provided_query(self, mock_run, mock_query):
        """Test main with provided query"""
        from helper_functions.query_supabasememory import main
        
        # When main calls run_query(), it's inside main() closure.
        # We need to ensure main() runs without error.
        # We can't easily execute the inner coroutine unless we capture it from mock_run.call_args
        
        main()
        mock_run.assert_called_once()

    @pytest.mark.asyncio
    @patch('helper_functions.query_supabasememory.query_memorypalace_stream')
    @patch('sys.argv', ['script.py', '--debug'])
    @patch('helper_functions.query_supabasememory.asyncio.run')
    async def test_main_with_debug_flag(self, mock_run, mock_query):
        """Test main with debug flag"""
        from helper_functions.query_supabasememory import main
        
        main()
        mock_run.assert_called_once()
