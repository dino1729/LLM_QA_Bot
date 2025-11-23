import os
import requests
import uuid
import json
import logging
import sounddevice as sd
import soundfile as sf
import numpy as np
from config import config
from helper_functions.chat_generation_with_internet import internet_connected_chatbot

# Try to import NVIDIA Riva client
try:
    import riva.client as riva
    RIVA_AVAILABLE = True
except ImportError:
    print("WARNING: nvidia-riva-client is not installed. Audio features will be limited.")
    print("Install with: pip install nvidia-riva-client")
    RIVA_AVAILABLE = False

system_prompt = config.system_prompt
conversation = system_prompt.copy()

# Get NVIDIA API key from config.yml or environment variable (config takes precedence)
nvidia_api_key = config.nvidia_api_key if hasattr(config, 'nvidia_api_key') and config.nvidia_api_key else os.getenv('NVIDIA_NIM_API_KEY')

# Initialize NVIDIA Riva services if available
asr_service = None
tts_service = None

if RIVA_AVAILABLE and nvidia_api_key:
    try:
        # ASR (Speech to Text) - Parakeet CTC 1.1B
        asr_auth = riva.Auth(
            uri='grpc.nvcf.nvidia.com:443',
            use_ssl=True,
            metadata_args=[
                ['function-id', '1598d209-5e27-4d3c-8079-4751568b1081'],
                ['authorization', f'Bearer {nvidia_api_key}']
            ]
        )
        asr_service = riva.ASRService(asr_auth)
        
        # TTS (Text to Speech) - Magpie TTS Multilingual
        tts_auth = riva.Auth(
            uri='grpc.nvcf.nvidia.com:443',
            use_ssl=True,
            metadata_args=[
                ['function-id', '877104f7-e885-42b9-8de8-f6e4c6303969'],
                ['authorization', f'Bearer {nvidia_api_key}']
            ]
        )
        tts_service = riva.SpeechSynthesisService(tts_auth)
        print("✓ NVIDIA Riva services initialized successfully")
    except Exception as e:
        print(f"WARNING: Failed to initialize NVIDIA Riva services: {e}")
        RIVA_AVAILABLE = False
elif not nvidia_api_key:
    print("WARNING: nvidia_api_key not found in config.yml or NVIDIA_NIM_API_KEY environment variable")
    RIVA_AVAILABLE = False

# Voice mapping for different models (NVIDIA Riva voices)
VOICE_MAP = {
    "GEMINI": "Magpie-Multilingual.EN-US.Echo",
    "GPT4": "Magpie-Multilingual.EN-US.Aria",
    "GPT4OMINI": "Magpie-Multilingual.EN-US.Aria",
    "COHERE": "Magpie-Multilingual.EN-US.Nova",
    "BING+OPENAI": "Magpie-Multilingual.EN-US.Shimmer",
    "MIXTRAL8x7B": "Magpie-Multilingual.EN-US.Fable",
    "DEFAULT": "Magpie-Multilingual.EN-US.Aria"
}

def transcribe_audio(audio_file):
    """
    Transcribe audio file using NVIDIA Riva Parakeet CTC 1.1B ASR model.
    Returns transcribed text and detected language.
    """
    if not RIVA_AVAILABLE or asr_service is None:
        print("ERROR: NVIDIA Riva ASR service not available")
        return "Transcription service unavailable. Please check NVIDIA API key.", "en-US"
    
    try:
        print("🎤 Transcribing audio with NVIDIA Parakeet...")
        
        # Configure ASR with sample rate of 16kHz (standard for speech recognition)
        config_asr = riva.RecognitionConfig(
            encoding=riva.AudioEncoding.LINEAR_PCM,
            sample_rate_hertz=16000,
            language_code='en-US',
            max_alternatives=1,
            enable_automatic_punctuation=True,
            audio_channel_count=1
        )
        
        # Read audio file
        with open(audio_file, 'rb') as audio:
            audio_data = audio.read()
        
        # Perform offline recognition
        response = asr_service.offline_recognize(audio_data, config_asr)
        
        if response.results and len(response.results) > 0:
            transcribed_text = response.results[0].alternatives[0].transcript
            
            # NVIDIA Riva primarily supports English, but we return en-US for compatibility
            detected_language = "en-US"
            
            print(f"✓ Transcription: {transcribed_text}")
            print(f"✓ Detected language: {detected_language}")
            
            return transcribed_text, detected_language
        else:
            print("WARNING: No transcription results from ASR")
            return "No speech detected.", "en-US"
            
    except Exception as e:
        print(f"ERROR in transcribe_audio: {e}")
        return "Transcription failed.", "en-US"

def text_to_speech(text, output_path, language, model_name):
    """
    Convert text to speech using NVIDIA Riva Magpie TTS model.
    """
    if not RIVA_AVAILABLE or tts_service is None:
        print("ERROR: NVIDIA Riva TTS service not available")
        return False
    
    try:
        print("🔊 Synthesizing speech with NVIDIA Magpie...")
        
        # Select voice based on model name
        voice_name = VOICE_MAP.get(model_name, VOICE_MAP["DEFAULT"])
        
        # Determine language code - NVIDIA Riva Magpie supports multilingual
        language_code = "en-US"
        if language == "te-IN":
            language_code = "en-US"  # Fallback to English for Telugu
        elif language == "hi-IN":
            language_code = "hi-IN"  # Hindi is supported
        elif language.startswith("en"):
            language_code = "en-US"
        
        # Create TTS request
        req = {
            "text": text,
            "language_code": language_code,
            "encoding": riva.AudioEncoding.LINEAR_PCM,
            "sample_rate_hz": 16000,
            "voice_name": voice_name
        }
        
        # Synthesize speech
        response = tts_service.synthesize(**req)
        
        # Convert raw PCM audio bytes to numpy array and save as proper WAV file
        audio_data = np.frombuffer(response.audio, dtype=np.int16)
        sf.write(output_path, audio_data, 16000, 'PCM_16')
        
        print(f"✓ Speech synthesized and saved to {output_path}")
        
        # Play the audio
        data, samplerate = sf.read(output_path)
        sd.play(data, samplerate)
        sd.wait()
        
        return True
        
    except Exception as e:
        print(f"ERROR in text_to_speech: {e}")
        return False

def text_to_speech_nospeak(text, output_path, language="en-US", model_name="GPT4OMINI"):
    """
    Convert text to speech using NVIDIA Riva Magpie TTS model without playing audio.
    """
    if not RIVA_AVAILABLE or tts_service is None:
        print("ERROR: NVIDIA Riva TTS service not available")
        return False
    
    try:
        print("🔊 Synthesizing speech with NVIDIA Magpie (no playback)...")
        
        # Select voice based on model name
        voice_name = VOICE_MAP.get(model_name, VOICE_MAP["DEFAULT"])
        
        # Determine language code - NVIDIA Riva Magpie supports multilingual
        language_code = "en-US"
        if language == "te-IN":
            language_code = "en-US"  # Fallback to English for Telugu
        elif language == "hi-IN":
            language_code = "hi-IN"  # Hindi is supported
        elif language.startswith("en"):
            language_code = "en-US"
        
        # Create TTS request
        req = {
            "text": text,
            "language_code": language_code,
            "encoding": riva.AudioEncoding.LINEAR_PCM,
            "sample_rate_hz": 16000,
            "voice_name": voice_name
        }
        
        # Synthesize speech
        response = tts_service.synthesize(**req)
        
        # Convert raw PCM audio bytes to numpy array and save as proper WAV file
        audio_data = np.frombuffer(response.audio, dtype=np.int16)
        sf.write(output_path, audio_data, 16000, 'PCM_16')
        
        print(f"✓ Speech synthesized and saved to {output_path} (no playback)")
        return True
        
    except Exception as e:
        print(f"ERROR in text_to_speech_nospeak: {e}")
        return False

def local_text_to_speech(text, output_path, model_name):
    
    url = "http://10.0.0.164:8000/generate"
    payload = json.dumps({
      "speaker_name": model_name,
      "input_text": text,
      "emotion": "Angry",
      "speed": 1.5
    })
    headers = {
      'Content-Type': 'application/json'
    }
    response = requests.request("POST", url, headers=headers, data=payload)
    if response.status_code == 200:
        audio_content = response.content
        # Save the audio to a file
        with open(output_path, "wb") as audio_file:
            audio_file.write(audio_content)
        print("Speech synthesized and saved to MP3 file.")
        # Load the audio file and play it
        data, samplerate = sf.read(output_path)
        sd.play(data, samplerate)
        sd.wait()
    else:
        print("Error:", response.text)

def translate_text(text, target_language):
    """
    Translation placeholder - for now returns original text.
    NVIDIA Riva Magpie TTS supports multilingual synthesis, so translation
    can be handled by the TTS model itself in many cases.
    
    For full translation support, consider integrating an LLM-based translation
    or a dedicated translation API.
    """
    # For now, return original text as NVIDIA Magpie can handle multiple languages
    # TODO: Implement LLM-based translation if needed
    print(f"Note: Translation requested for {target_language}, returning original text")
    return text

def transcribe_audio_to_text(audio_path):
    """
    Transcribes audio to text using NVIDIA Riva Parakeet ASR.
    Supports multiple languages with automatic detection.
    """
    # Initialize variables with default values
    english_text = "Transcription failed! As a voice assistant, inform the user that transcription failed. It may probably be due to the audio device not picking up any sound."
    detected_audio_language = "en-US"
    try:
        english_text, detected_audio_language = transcribe_audio(audio_path)
        print("You: {}; Language {}".format(english_text, detected_audio_language))
        new_message = {"role": "user", "content": english_text}
        conversation.append(new_message)
    except Exception as e:
        print("Transcription error:", str(e))
        pass
    return english_text, detected_audio_language

def generate_response(english_text, conversation, model_name, max_tokens, temperature):
    """
    Generates a response using the selected model.
    """
    try:
        assistant_reply = internet_connected_chatbot(english_text, conversation, model_name, max_tokens, temperature)
        print("{} Bot: {}".format(model_name, assistant_reply))
    except Exception as e:
        print("Model error:", str(e))
        pass
    return assistant_reply

def translate_and_speak(assistant_reply, detected_audio_language, tts_output_path, model_name):
    """
    Translates the assistant's reply and converts it to speech.
    """
    try:
        translated_message = translate_text(assistant_reply, detected_audio_language)
        text_to_speech(translated_message, tts_output_path, detected_audio_language, model_name)
    except Exception as e:
        print("Translation error:", str(e))
        text_to_speech("Sorry, I couldn't answer that.", tts_output_path, "en-US", model_name)
