import azure.cognitiveservices.speech as speechsdk
import requests
import uuid
import json
import sounddevice as sd
import soundfile as sf
from config import config
from helper_functions.chat_generation_with_internet import internet_connected_chatbot

azurespeechkey = config.azurespeechkey
azurespeechregion = config.azurespeechregion
azuretexttranslatorkey = config.azuretexttranslatorkey
system_prompt = config.system_prompt
conversation = system_prompt.copy()

def transcribe_audio(audio_file):
    
    # Create an instance of a speech config with your subscription key and region
    # Currently the v2 endpoint is required. In a future SDK release you won't need to set it. 
    endpoint_string = "wss://{}.stt.speech.microsoft.com/speech/universal/v2".format(azurespeechregion)
    #speech_config = speechsdk.translation.SpeechTranslationConfig(subscription=azurespeechkey, endpoint=endpoint_string)
    audio_config = speechsdk.audio.AudioConfig(filename=audio_file)
    # set up translation parameters: source language and target languages
    # Currently the v2 endpoint is required. In a future SDK release you won't need to set it. 
    #endpoint_string = "wss://{}.stt.speech.microsoft.com/speech/universal/v2".format(service_region)
    translation_config = speechsdk.translation.SpeechTranslationConfig(
        subscription=azurespeechkey,
        endpoint=endpoint_string,
        speech_recognition_language='en-US',
        target_languages=('en','hi','te'))
    #audio_config = speechsdk.audio.AudioConfig(filename=weatherfilename)
    # Specify the AutoDetectSourceLanguageConfig, which defines the number of possible languages
    auto_detect_source_language_config = speechsdk.languageconfig.AutoDetectSourceLanguageConfig(languages=["en-US", "hi-IN", "te-IN"])
    # Creates a translation recognizer using and audio file as input.
    recognizer = speechsdk.translation.TranslationRecognizer(
        translation_config=translation_config, 
        audio_config=audio_config,
        auto_detect_source_language_config=auto_detect_source_language_config)
    result = recognizer.recognize_once()

    translated_result = format(result.translations['en'])
    detectedSrcLang = format(result.properties[speechsdk.PropertyId.SpeechServiceConnection_AutoDetectSourceLanguageResult])

    return translated_result, detectedSrcLang

def text_to_speech(text, output_path, language, model_name):
    
    speech_config = speechsdk.SpeechConfig(subscription=azurespeechkey, region=azurespeechregion)
    # Set the voice based on the language
    if language == "te-IN":
        # speech_config.speech_synthesis_voice_name = "te-IN-ShrutiNeural"
        speech_config.speech_synthesis_voice_name = "en-US-EmmaMultilingualNeural"
    elif language == "hi-IN":
        speech_config.speech_synthesis_voice_name = "hi-IN-SwaraNeural"
    else:
        # Use a default voice if the language is not specified or unsupported
        default_voice = "en-US-AriaNeural"
        if model_name == "GEMINI":
            speech_config.speech_synthesis_voice_name = "en-US-AnaNeural"
        elif model_name == "GPT4":
            speech_config.speech_synthesis_voice_name = "en-US-BlueNeural"
        elif model_name == "GPT4OMINI":
            speech_config.speech_synthesis_voice_name = "en-US-EmmaMultilingualNeural"
        elif model_name == "COHERE":
            speech_config.speech_synthesis_voice_name = "en-US-SaraNeural"
        elif model_name == "BING+OPENAI":
            speech_config.speech_synthesis_voice_name = "en-US-JennyNeural"
        elif model_name == "MIXTRAL8x7B":
            speech_config.speech_synthesis_voice_name = "en-US-AmberNeural"
        else:
            speech_config.speech_synthesis_voice_name = default_voice
    # Use the default speaker as audio output and start playing the audio
    speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config)
    #speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=None)
    result = speech_synthesizer.speak_text_async(text).get()
    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        # Get the audio data from the result object
        audio_data = result.audio_data  
        # Save the audio data as a WAV file
        with open(output_path, "wb") as audio_file:
            audio_file.write(audio_data)
            print("Speech synthesized and saved to WAV file.")

def text_to_speech_nospeak(text, output_path, language="en-US", model_name="GPT4OMINI"):
    
    speech_config = speechsdk.SpeechConfig(subscription=azurespeechkey, region=azurespeechregion)
    # Set the voice based on the language
    if language == "te-IN":
        # speech_config.speech_synthesis_voice_name = "te-IN-ShrutiNeural"
        speech_config.speech_synthesis_voice_name = "en-US-EmmaMultilingualNeural"
    elif language == "hi-IN":
        speech_config.speech_synthesis_voice_name = "hi-IN-SwaraNeural"
    else:
        # Use a default voice if the language is not specified or unsupported
        default_voice = "en-US-AriaNeural"
        if model_name == "GEMINI":
            speech_config.speech_synthesis_voice_name = "en-US-AnaNeural"
        elif model_name == "GPT4":
            speech_config.speech_synthesis_voice_name = "en-US-BlueNeural"
        elif model_name == "GPT4OMINI":
            speech_config.speech_synthesis_voice_name = "en-US-EmmaMultilingualNeural"
        elif model_name == "COHERE":
            speech_config.speech_synthesis_voice_name = "en-US-SaraNeural"
        elif model_name == "BING+OPENAI":
            speech_config.speech_synthesis_voice_name = "en-US-JennyNeural"
        elif model_name == "MIXTRAL8x7B":
            speech_config.speech_synthesis_voice_name = "en-US-AmberNeural"
        else:
            speech_config.speech_synthesis_voice_name = default_voice
    # Use the default speaker as audio output and start playing the audio
    # speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config)
    speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=None)
    result = speech_synthesizer.speak_text_async(text).get()
    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        # Get the audio data from the result object
        audio_data = result.audio_data  
        # Save the audio data as a WAV file
        with open(output_path, "wb") as audio_file:
            audio_file.write(audio_data)
            print("Speech synthesized and saved to WAV file.")

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
    
    # Add your key and endpoint
    key = azuretexttranslatorkey
    endpoint = "https://api.cognitive.microsofttranslator.com"
    # location, also known as region.
    location = azurespeechregion
    path = '/translate'
    constructed_url = endpoint + path
    params = {
        'api-version': '3.0',
        'from': 'en',
        'to': [target_language]
    }
    headers = {
        'Ocp-Apim-Subscription-Key': key,
        # location required if you're using a multi-service or regional (not global) resource.
        'Ocp-Apim-Subscription-Region': location,
        'Content-type': 'application/json',
        'X-ClientTraceId': str(uuid.uuid4())
    }
    body = [{
        'text': text
    }]
    request = requests.post(constructed_url, params=params, headers=headers, json=body)
    response = request.json()
    return response[0]['translations'][0]['text']

def transcribe_audio_to_text(audio_path):
    """
    Transcribes Telugu/Hindi audio to English text using Azure Speech Recognition.
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
