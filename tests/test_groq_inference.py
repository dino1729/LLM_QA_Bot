"""
Unit and integration tests for direct Groq API inference
Tests all three Groq models configured in config.yml
"""
import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import config
from groq import Groq


class TestGroqInferenceConfig:
    """Test Groq configuration from config.yml"""

    def test_groq_api_key_exists(self):
        """Test that Groq API key is configured"""
        assert hasattr(config, 'groq_api_key'), "Groq API key not found in config"
        assert config.groq_api_key is not None, "Groq API key is None"
        assert config.groq_api_key != "", "Groq API key is empty"

    def test_groq_model_names_exist(self):
        """Test that all Groq model names are configured"""
        model_attrs = [
            'groq_model_name',
            'groq_llama_model_name',
            'groq_qwen_model_name'
        ]

        for attr in model_attrs:
            assert hasattr(config, attr), f"Config missing {attr}"
            assert getattr(config, attr) is not None, f"{attr} is None"
            assert getattr(config, attr) != "", f"{attr} is empty"


@pytest.mark.integration
class TestGroqInference:
    """Integration tests for direct Groq API inference"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up Groq client before each test"""
        self.client = Groq(api_key=config.groq_api_key)
        self.test_prompt = "What is the capital of France? Answer in one word."

    def test_groq_client_initialization(self):
        """Test Groq client initializes correctly"""
        assert self.client is not None
        assert hasattr(self.client, 'chat')

    def test_groq_primary_model_inference(self):
        """Test inference with primary Groq model (moonshotai/kimi-k2-instruct-0905)"""
        model_name = config.groq_model_name

        completion = self.client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": self.test_prompt
                }
            ],
            temperature=0.6,
            max_completion_tokens=4096,
            top_p=0.95,
            stream=False,
            stop=None
        )

        assert completion is not None
        assert hasattr(completion, 'choices')
        assert len(completion.choices) > 0
        assert hasattr(completion.choices[0], 'message')
        assert hasattr(completion.choices[0].message, 'content')
        
        response_text = completion.choices[0].message.content
        assert response_text is not None
        assert len(response_text) > 0
        
        print(f"\n[{model_name}] Response: {response_text}")

    def test_groq_llama_model_inference(self):
        """Test inference with Groq Llama model (llama-3.3-70b-versatile)"""
        model_name = config.groq_llama_model_name

        completion = self.client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": self.test_prompt
                }
            ],
            temperature=0.6,
            max_completion_tokens=4096,
            top_p=0.95,
            stream=False,
            stop=None
        )

        assert completion is not None
        assert hasattr(completion, 'choices')
        assert len(completion.choices) > 0
        assert hasattr(completion.choices[0], 'message')
        assert hasattr(completion.choices[0].message, 'content')
        
        response_text = completion.choices[0].message.content
        assert response_text is not None
        assert len(response_text) > 0
        
        print(f"\n[{model_name}] Response: {response_text}")

    def test_groq_qwen_model_inference(self):
        """Test inference with Groq Qwen model (qwen/qwen3-32b)"""
        model_name = config.groq_qwen_model_name

        completion = self.client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": self.test_prompt
                }
            ],
            temperature=0.6,
            max_completion_tokens=4096,
            top_p=0.95,
            stream=False,
            stop=None
        )

        assert completion is not None
        assert hasattr(completion, 'choices')
        assert len(completion.choices) > 0
        assert hasattr(completion.choices[0], 'message')
        assert hasattr(completion.choices[0].message, 'content')
        
        response_text = completion.choices[0].message.content
        assert response_text is not None
        assert len(response_text) > 0
        
        print(f"\n[{model_name}] Response: {response_text}")


@pytest.mark.integration
class TestGroqStreamingInference:
    """Test streaming responses from Groq API"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up Groq client before each test"""
        self.client = Groq(api_key=config.groq_api_key)
        self.test_prompt = "Count from 1 to 5, one number per line."

    def test_groq_primary_model_streaming(self):
        """Test streaming inference with primary Groq model"""
        model_name = config.groq_model_name

        completion = self.client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": self.test_prompt
                }
            ],
            temperature=0.6,
            max_completion_tokens=4096,
            top_p=0.95,
            stream=True,
            stop=None
        )

        chunks = []
        for chunk in completion:
            if hasattr(chunk, 'choices') and len(chunk.choices) > 0:
                if hasattr(chunk.choices[0], 'delta'):
                    content = chunk.choices[0].delta.content
                    if content:
                        chunks.append(content)

        full_response = "".join(chunks)
        assert len(full_response) > 0, "No response received from streaming"
        print(f"\n[{model_name}] Streaming response: {full_response}")

    def test_groq_llama_model_streaming(self):
        """Test streaming inference with Groq Llama model"""
        model_name = config.groq_llama_model_name

        completion = self.client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": self.test_prompt
                }
            ],
            temperature=0.6,
            max_completion_tokens=4096,
            top_p=0.95,
            stream=True,
            stop=None
        )

        chunks = []
        for chunk in completion:
            if hasattr(chunk, 'choices') and len(chunk.choices) > 0:
                if hasattr(chunk.choices[0], 'delta'):
                    content = chunk.choices[0].delta.content
                    if content:
                        chunks.append(content)

        full_response = "".join(chunks)
        assert len(full_response) > 0, "No response received from streaming"
        print(f"\n[{model_name}] Streaming response: {full_response}")

    def test_groq_qwen_model_streaming(self):
        """Test streaming inference with Groq Qwen model"""
        model_name = config.groq_qwen_model_name

        completion = self.client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": self.test_prompt
                }
            ],
            temperature=0.6,
            max_completion_tokens=4096,
            top_p=0.95,
            stream=True,
            stop=None
        )

        chunks = []
        for chunk in completion:
            if hasattr(chunk, 'choices') and len(chunk.choices) > 0:
                if hasattr(chunk.choices[0], 'delta'):
                    content = chunk.choices[0].delta.content
                    if content:
                        chunks.append(content)

        full_response = "".join(chunks)
        assert len(full_response) > 0, "No response received from streaming"
        print(f"\n[{model_name}] Streaming response: {full_response}")


@pytest.mark.integration
class TestGroqMultiTurnConversation:
    """Test multi-turn conversations with Groq models"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up Groq client before each test"""
        self.client = Groq(api_key=config.groq_api_key)

    def test_multi_turn_primary_model(self):
        """Test multi-turn conversation with primary Groq model"""
        model_name = config.groq_model_name

        messages = [
            {"role": "user", "content": "What is 2 + 2?"},
        ]

        # First turn
        completion1 = self.client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.6,
            max_completion_tokens=4096,
            stream=False
        )

        response1 = completion1.choices[0].message.content
        assert response1 is not None
        messages.append({"role": "assistant", "content": response1})

        # Second turn
        messages.append({"role": "user", "content": "Now multiply that by 3."})
        
        completion2 = self.client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.6,
            max_completion_tokens=4096,
            stream=False
        )

        response2 = completion2.choices[0].message.content
        assert response2 is not None
        assert len(response2) > 0
        
        print(f"\n[{model_name}] Multi-turn conversation:")
        print(f"  Turn 1: {response1}")
        print(f"  Turn 2: {response2}")


@pytest.mark.integration
class TestGroqErrorHandling:
    """Test error handling for Groq API"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up Groq client before each test"""
        self.client = Groq(api_key=config.groq_api_key)

    def test_invalid_model_name(self):
        """Test that invalid model name raises appropriate error"""
        with pytest.raises(Exception):
            self.client.chat.completions.create(
                model="invalid-model-name-xyz",
                messages=[{"role": "user", "content": "Test"}],
                temperature=0.6,
                max_completion_tokens=100,
                stream=False
            )

    def test_empty_message(self):
        """Test handling of empty message"""
        model_name = config.groq_model_name
        
        # This should either work or raise a clear error
        try:
            completion = self.client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": ""}],
                temperature=0.6,
                max_completion_tokens=100,
                stream=False
            )
            # If it works, check response
            assert completion is not None
        except Exception as e:
            # If it fails, that's also acceptable behavior
            assert isinstance(e, Exception)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

