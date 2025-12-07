"""
Test script for Audio Processors (TTS)
Tests the NVIDIA Riva integration
"""
import os
import sys
import time
from helper_functions.audio_processors import text_to_speech_nospeak

def test_tts():
    print("=" * 80)
    print("Testing NVIDIA Riva Text-to-Speech")
    print("=" * 80)
    
    # Test parameters
    text = "This is a test of the NVIDIA Riva text to speech system. If you can hear this, the system is working correctly."
    output_file = "test_tts_output.wav"
    model_name = "RIVA_ARIA_VOICE" # Maps to Magpie-Multilingual.EN-US.Aria
    
    print(f"Text: {text}")
    print(f"Output File: {output_file}")
    print(f"Model Name: {model_name}")
    print("-" * 80)
    
    try:
        # Run TTS
        start_time = time.time()
        success = text_to_speech_nospeak(text, output_file, model_name=model_name)
        duration = time.time() - start_time
        
        if success:
            print(f"\n✓ TTS Generation Successful (took {duration:.2f}s)")
            
            # Check if file exists and has content
            if os.path.exists(output_file):
                size = os.path.getsize(output_file)
                print(f"✓ Output file created: {output_file} ({size} bytes)")
                if size > 1000:
                    print("✓ File size looks reasonable")
                else:
                    print("✗ File seems too small")
            else:
                print("✗ Output file not found")
                
        else:
            print("\n✗ TTS Generation Failed")
            
    except Exception as e:
        print(f"\n✗ Error during test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_tts()
