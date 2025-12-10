import os
import re
import requests
import uuid
import json
import logging
import time
import signal
from contextlib import contextmanager
import sounddevice as sd
import soundfile as sf
import numpy as np
from config import config
from helper_functions.chat_generation_with_internet import internet_connected_chatbot

# Configure logging
logger = logging.getLogger(__name__)

# NVIDIA Magpie TTS limits
# Note: The TTS service internally expands text, so we use conservative limits
# Actual limit is 2000, but we use 1200 to leave buffer for internal expansion
MAX_CHARS_PER_REQUEST = 1200
MAX_CHARS_PER_SENTENCE = 350

# Try to import NVIDIA Riva client
try:
    import riva.client as riva
    import grpc
    RIVA_AVAILABLE = True
except ImportError:
    print("WARNING: nvidia-riva-client is not installed. Audio features will be limited.")
    print("Install with: pip install nvidia-riva-client")
    RIVA_AVAILABLE = False
    riva = None
    grpc = None

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
    "RIVA_ARIA_VOICE": "Magpie-Multilingual.EN-US.Aria",
    "BING+OPENAI": "Magpie-Multilingual.EN-US.Shimmer",
    "MIXTRAL8x7B": "Magpie-Multilingual.EN-US.Fable",
    # LiteLLM/Ollama models will fall back to DEFAULT
    "LITELLM_SMART": "Magpie-Multilingual.EN-US.Aria",
    "LITELLM_STRATEGIC": "Magpie-Multilingual.EN-US.Echo",
    "OLLAMA_SMART": "Magpie-Multilingual.EN-US.Aria",
    "OLLAMA_STRATEGIC": "Magpie-Multilingual.EN-US.Echo",
    "DEFAULT": "Magpie-Multilingual.EN-US.Aria"
}


class TimeoutException(Exception):
    """Custom exception for timeout"""
    pass


@contextmanager
def time_limit(seconds):
    """
    Context manager for setting a timeout on a block of code.
    Note: Only works on Unix-based systems.
    """
    def signal_handler(signum, frame):
        raise TimeoutException("Operation timed out")
    
    # Set the signal handler
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)  # Disable the alarm

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

def chunk_text_for_tts(text, max_sentence_length=MAX_CHARS_PER_SENTENCE, max_chunk_length=MAX_CHARS_PER_REQUEST):
    """
    Split text into chunks suitable for NVIDIA Magpie TTS.
    
    NVIDIA Magpie TTS has two limits:
    - Max 2000 characters total per request
    - Max 400 characters per sentence
    
    This function:
    1. Splits text into sentences
    2. Breaks long sentences at punctuation/word boundaries
    3. Groups sentences into chunks up to max_chunk_length
    
    Returns:
        List of text chunks, each suitable for a single TTS request
    """
    if not text or len(text.strip()) == 0:
        logger.warning("Empty text provided for TTS chunking")
        return []
    
    logger.debug(f"Chunking text of {len(text)} characters for TTS")
    
    # Split into sentences using regex
    # Handles: periods, question marks, exclamation points, semicolons
    sentence_pattern = r'(?<=[.!?;])\s+'
    sentences = re.split(sentence_pattern, text.strip())
    
    logger.debug(f"Split into {len(sentences)} initial sentences")
    
    # Process each sentence to ensure it's under the limit
    processed_sentences = []
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        if len(sentence) <= max_sentence_length:
            processed_sentences.append(sentence)
        else:
            # Break long sentences at commas or other natural break points
            sub_sentences = _break_long_sentence(sentence, max_sentence_length)
            processed_sentences.extend(sub_sentences)
    
    logger.debug(f"After processing: {len(processed_sentences)} sentences")
    
    # Group sentences into chunks under max_chunk_length
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in processed_sentences:
        sentence_len = len(sentence)
        # Only add space if this isn't the first sentence in the chunk
        if current_chunk:
            sentence_len += 1  # +1 for space between sentences
        
        if current_length + sentence_len <= max_chunk_length:
            current_chunk.append(sentence)
            current_length += sentence_len
        else:
            # Save current chunk and start new one
            if current_chunk:
                chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_length = len(sentence)
    
    # Don't forget the last chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    logger.info(f"Created {len(chunks)} TTS chunks from {len(text)} chars of text")
    for i, chunk in enumerate(chunks):
        logger.debug(f"Chunk {i+1}: {len(chunk)} chars")
    
    return chunks


def _break_long_sentence(sentence, max_length):
    """
    Break a long sentence into smaller parts at natural break points.
    
    Tries to break at:
    1. Commas
    2. Conjunctions (and, or, but)
    3. Word boundaries (last resort)
    """
    if len(sentence) <= max_length:
        return [sentence]
    
    parts = []
    remaining = sentence
    
    while len(remaining) > max_length:
        # Find the best break point within max_length
        break_point = max_length
        
        # Try to break at comma
        comma_pos = remaining[:max_length].rfind(',')
        if comma_pos > max_length // 2:  # Only if comma is in latter half
            break_point = comma_pos + 1
        else:
            # Try to break at conjunction
            for conj in [' and ', ' or ', ' but ', ' which ', ' that ']:
                conj_pos = remaining[:max_length].rfind(conj)
                if conj_pos > max_length // 2:
                    break_point = conj_pos
                    break
            else:
                # Break at last space
                space_pos = remaining[:max_length].rfind(' ')
                if space_pos > 0:
                    break_point = space_pos
        
        parts.append(remaining[:break_point].strip())
        remaining = remaining[break_point:].strip()
    
    if remaining:
        parts.append(remaining)
    
    logger.debug(f"Broke long sentence ({len(sentence)} chars) into {len(parts)} parts")
    return parts


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
        
        # Retry logic for TTS synthesis with timeout
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                # Synthesize speech with timeout (30 seconds)
                with time_limit(30):
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
                
            except TimeoutException:
                print(f"⚠️ TTS timeout, attempt {attempt+1}/{max_retries}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    print(f"ERROR: TTS synthesis timed out after {max_retries} attempts")
                    return False
            except grpc.RpcError as e:
                print(f"⚠️ gRPC error, attempt {attempt+1}/{max_retries}: {e.code()}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    print(f"ERROR: TTS synthesis failed after {max_retries} attempts")
                    return False
            except Exception as e:
                print(f"⚠️ Error in TTS, attempt {attempt+1}/{max_retries}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    print(f"ERROR in text_to_speech: {e}")
                    return False
        
        return False
        
    except Exception as e:
        print(f"ERROR in text_to_speech: {e}")
        return False

def text_to_speech_nospeak(text, output_path, language="en-US", model_name="GPT4OMINI"):
    """
    Convert text to speech using NVIDIA Riva Magpie TTS model without playing audio.
    
    Handles long text by:
    1. Chunking text into segments respecting TTS limits (2000 chars total, 400 per sentence)
    2. Synthesizing each chunk separately
    3. Concatenating audio output
    """
    if not RIVA_AVAILABLE or tts_service is None:
        logger.error("NVIDIA Riva TTS service not available")
        print("ERROR: NVIDIA Riva TTS service not available")
        return False
    
    try:
        logger.info(f"🔊 Synthesizing speech for {len(text)} characters of text")
        print("🔊 Synthesizing speech with NVIDIA Magpie (no playback)...")
        
        # Select voice based on model name
        voice_name = VOICE_MAP.get(model_name, VOICE_MAP["DEFAULT"])
        logger.debug(f"Using voice: {voice_name}")
        
        # Determine language code - NVIDIA Riva Magpie supports multilingual
        language_code = "en-US"
        if language == "te-IN":
            language_code = "en-US"  # Fallback to English for Telugu
        elif language == "hi-IN":
            language_code = "hi-IN"  # Hindi is supported
        elif language.startswith("en"):
            language_code = "en-US"
        
        # Chunk the text to respect TTS limits
        chunks = chunk_text_for_tts(text)
        
        if not chunks:
            logger.warning("No text chunks to synthesize")
            return False
        
        logger.info(f"Processing {len(chunks)} text chunks for TTS")
        
        # Synthesize each chunk and collect audio data
        all_audio_data = []
        
        for i, chunk in enumerate(chunks):
            logger.debug(f"Synthesizing chunk {i+1}/{len(chunks)} ({len(chunk)} chars)")
            
            # Create TTS request for this chunk
            req = {
                "text": chunk,
                "language_code": language_code,
                "encoding": riva.AudioEncoding.LINEAR_PCM,
                "sample_rate_hz": 16000,
                "voice_name": voice_name
            }
            
            # Retry logic for TTS synthesis with timeout
            max_retries = 3
            retry_delay = 2
            
            for attempt in range(max_retries):
                try:
                    # Synthesize speech for this chunk with timeout (30 seconds)
                    with time_limit(30):
                        response = tts_service.synthesize(**req)
                    
                    # Convert raw PCM audio bytes to numpy array
                    chunk_audio = np.frombuffer(response.audio, dtype=np.int16)
                    all_audio_data.append(chunk_audio)
                    
                    logger.debug(f"Chunk {i+1} synthesized: {len(chunk_audio)} samples")
                    break  # Success, exit retry loop
                    
                except TimeoutException:
                    logger.warning(f"TTS timeout on chunk {i+1}, attempt {attempt+1}/{max_retries}")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                    else:
                        logger.error(f"Chunk {i+1} timed out after {max_retries} attempts")
                        # Generate silent audio for failed chunk to maintain continuity
                        chunk_duration = len(chunk) * 0.08  # Rough estimate: 0.08s per char
                        silent_audio = np.zeros(int(16000 * chunk_duration), dtype=np.int16)
                        all_audio_data.append(silent_audio)
                        logger.warning(f"Inserted silence for failed chunk {i+1}")
                        break
                except grpc.RpcError as e:
                    logger.warning(f"gRPC error on chunk {i+1}, attempt {attempt+1}/{max_retries}: {e.code()} - {e.details()}")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                    else:
                        # Final attempt failed, skip this chunk or fail
                        logger.error(f"Failed to synthesize chunk {i+1} after {max_retries} attempts")
                        # Generate silent audio for failed chunk to maintain continuity
                        chunk_duration = len(chunk) * 0.08  # Rough estimate: 0.08s per char
                        silent_audio = np.zeros(int(16000 * chunk_duration), dtype=np.int16)
                        all_audio_data.append(silent_audio)
                        logger.warning(f"Inserted silence for failed chunk {i+1}")
                        break
                except Exception as e:
                    logger.warning(f"Unexpected error on chunk {i+1}, attempt {attempt+1}/{max_retries}: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                    else:
                        logger.error(f"Failed to synthesize chunk {i+1} after {max_retries} attempts")
                        chunk_duration = len(chunk) * 0.08
                        silent_audio = np.zeros(int(16000 * chunk_duration), dtype=np.int16)
                        all_audio_data.append(silent_audio)
                        logger.warning(f"Inserted silence for failed chunk {i+1}")
                        break
        
        # Concatenate all audio chunks
        if len(all_audio_data) > 1:
            # Add small silence (0.3 seconds) between chunks for natural pauses
            silence = np.zeros(int(16000 * 0.3), dtype=np.int16)
            combined_audio = []
            for i, audio in enumerate(all_audio_data):
                combined_audio.append(audio)
                if i < len(all_audio_data) - 1:
                    combined_audio.append(silence)
            final_audio = np.concatenate(combined_audio)
        else:
            final_audio = all_audio_data[0]
        
        # Save the combined audio as WAV format
        # Note: Even if output_path ends in .mp3, we save as WAV (most players handle this)
        # For proper MP3 encoding, additional libraries like pydub would be needed
        actual_output = output_path.replace('.mp3', '.wav') if output_path.endswith('.mp3') else output_path
        sf.write(actual_output, final_audio, 16000, subtype='PCM_16')
        
        duration_seconds = len(final_audio) / 16000
        logger.info(f"✓ Speech synthesized: {duration_seconds:.1f}s audio saved to {actual_output}")
        print(f"✓ Speech synthesized and saved to {actual_output} (no playback)")
        return True
        
    except Exception as e:
        logger.error(f"ERROR in text_to_speech_nospeak: {e}", exc_info=True)
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
