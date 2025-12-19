#!/usr/bin/env python3
"""
Test script for dynamic model listing functionality
Tests the new list_available_models function with LiteLLM and Ollama
"""

from helper_functions.llm_client import list_available_models, get_client

def test_litellm_models():
    """Test fetching models from LiteLLM"""
    print("=" * 60)
    print("Testing LiteLLM Model Listing")
    print("=" * 60)
    
    try:
        models = list_available_models("litellm")
        print(f"✓ Successfully fetched {len(models)} models from LiteLLM")
        print("\nAvailable models:")
        for i, model in enumerate(models[:10], 1):  # Show first 10
            print(f"  {i}. {model}")
        if len(models) > 10:
            print(f"  ... and {len(models) - 10} more")
        return True
    except Exception as e:
        print(f"✗ Error fetching LiteLLM models: {str(e)}")
        return False

def test_ollama_models():
    """Test fetching models from Ollama"""
    print("\n" + "=" * 60)
    print("Testing Ollama Model Listing")
    print("=" * 60)
    
    try:
        models = list_available_models("ollama")
        print(f"✓ Successfully fetched {len(models)} models from Ollama")
        print("\nAvailable models:")
        for i, model in enumerate(models[:10], 1):  # Show first 10
            print(f"  {i}. {model}")
        if len(models) > 10:
            print(f"  ... and {len(models) - 10} more")
        return True
    except Exception as e:
        print(f"✗ Error fetching Ollama models: {str(e)}")
        return False

def test_client_creation():
    """Test creating clients with specific model names"""
    print("\n" + "=" * 60)
    print("Testing Client Creation with Specific Models")
    print("=" * 60)
    
    # Test with LiteLLM
    try:
        models = list_available_models("litellm")
        if models:
            test_model = models[0]
            print(f"\nCreating LiteLLM client with model: {test_model}")
            client = get_client(provider="litellm", model_name=test_model)
            print(f"✓ Successfully created client with model: {client.model}")
    except Exception as e:
        print(f"✗ Error creating LiteLLM client: {str(e)}")
    
    # Test with Ollama
    try:
        models = list_available_models("ollama")
        if models:
            test_model = models[0]
            print(f"\nCreating Ollama client with model: {test_model}")
            client = get_client(provider="ollama", model_name=test_model)
            print(f"✓ Successfully created client with model: {client.model}")
    except Exception as e:
        print(f"✗ Error creating Ollama client: {str(e)}")

def main():
    """Run all tests"""
    print("\n" + "🚀 Dynamic Model Selection - Test Suite")
    print("=" * 60)
    
    litellm_ok = test_litellm_models()
    ollama_ok = test_ollama_models()
    test_client_creation()
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"LiteLLM: {'✓ PASS' if litellm_ok else '✗ FAIL'}")
    print(f"Ollama:  {'✓ PASS' if ollama_ok else '✗ FAIL'}")
    
    if litellm_ok or ollama_ok:
        print("\n✓ At least one provider is working correctly!")
        print("\nNext steps:")
        print("1. Start the application: python gradio_ui_full.py")
        print("2. Open http://localhost:7860 in your browser")
        print("3. Test the dynamic model dropdowns in any tab")
    else:
        print("\n✗ Both providers failed. Please check:")
        print("1. LiteLLM proxy is running and accessible")
        print("2. Ollama is running and accessible")
        print("3. config/config.yml has correct URLs and API keys")

if __name__ == "__main__":
    main()

