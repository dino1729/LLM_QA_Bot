import base64
import os
import time
import sys  # Import sys
import threading
from contextlib import ExitStack             # NEW
from openai import AzureOpenAI
import traceback  # NEW – for nicer error logging
import json       # NEW - for pretty printing debug info
import subprocess # NEW - for running curl

# ---------------- DEBUG helpers ----------------
def _mask(secret: str, show: int = 4) -> str:
    """Return a masked representation of a secret so it can be logged safely."""
    if not secret:
        return "None"
    if len(secret) <= show * 2:
        return "*" * len(secret)
    return f"{secret[:show]}{'*' * (len(secret) - show*2)}{secret[-show:]}"

def debug_print(label, value, pretty=False):
    """Print a debug message with consistent formatting."""
    if pretty and isinstance(value, dict):
        print(f"[DEBUG] {label}:")
        print(json.dumps(value, indent=2))
    else:
        print(f"[DEBUG] {label}: {value}")
# -----------------------------------------------

debug_print("Script started", time.strftime("%Y-%m-%d %H:%M:%S"))
debug_print("Python version", sys.version)
debug_print("Command line args", sys.argv)

azure_api_key   = os.getenv("AZURE_OPENAI_API_KEY")   or os.getenv("OPENAI_API_KEY")
azure_endpoint  = os.getenv("AZURE_OPENAI_ENDPOINT")  or os.getenv("OPENAI_API_BASE")
api_version     = os.getenv("AZURE_OPENAI_API_VERSION", "2025-04-01-preview")
azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT") # NEW: Get deployment name

# Print environment variables
debug_print("AZURE_OPENAI_API_KEY", _mask(azure_api_key))
debug_print("AZURE_OPENAI_ENDPOINT", azure_endpoint)
debug_print("AZURE_OPENAI_API_VERSION", api_version)
debug_print("AZURE_OPENAI_DEPLOYMENT", azure_deployment) # NEW: Print deployment name
debug_print("Environment variables", {k: v for k, v in os.environ.items() if "OPENAI" in k.upper()}, pretty=True)

# Initialize client with debug information
debug_print("Initializing Azure OpenAI client", "...")
client = AzureOpenAI(
    api_key=azure_api_key,
    azure_endpoint=azure_endpoint,
    api_version=api_version
)
debug_print("Client initialized", "success")

start_time_total = time.time() # Record total start time
debug_print("Total timer started", start_time_total)

# Concise prompt
prompt = """
Turn the subject into a Simpsons character
"""

# ------------------ NEW: parse image paths ------------------
if len(sys.argv) < 2:
    print("Usage: python gptimageedit.py <image_to_edit.png>") # MODIFIED: Expect only one image
    sys.exit(1)

# --- MODIFIED: Take only the first image ---
image_path = sys.argv[1]
if not os.path.isfile(image_path):
    print(f"Error: '{image_path}' is not a valid file.")
    sys.exit(1)
# ------------------------------------------------------------

# Flag to signal the background thread to stop
stop_funny_messages = threading.Event()

def print_funny_messages():
    # Initialize conversation history with the system prompt and initial user prompt
    conversation_history = [
        {"role": "system", "content": "You are simulating a funny and exaggerated conversation between Donald Trump and Narendra Modi about image editing. Trump speaks in his characteristic style with simple words, superlatives, and boastful statements. Modi speaks formally with references to Indian culture and diplomatic language. Keep responses relatively short and entertaining."},
        {"role": "user", "content": f"The system is processing this image editing prompt: '{prompt}'. Start a conversation where Trump and Modi comment on this process."}
    ]
    
    # Track which leader speaks next
    next_speaker = "Trump"  # Start with Trump

    while not stop_funny_messages.is_set():
        try:
            # Ensure history doesn't grow too large
            if len(conversation_history) > 9:
                 conversation_history = [conversation_history[0]] + conversation_history[-8:]

            # debug_print(f"[FunnyThread] Sending history for {next_speaker}:", conversation_history, pretty=True)

            # Determine the persona for the next response
            if next_speaker == "Trump":
                persona_prompt = "Respond as Donald Trump commenting on the image editing process in his distinctive style - use simple words, exaggerations, 'tremendous', 'huge', etc."
                speaker_name = "🇺🇸 Trump"
                temp = 0.7
                next_speaker = "Modi"  # Switch for next turn
            else:  # Modi's turn
                persona_prompt = "Respond as Narendra Modi commenting on the image editing process with references to Indian culture, technology achievement, and his diplomatic style with phrases like 'my friends' and occasional Hindi phrases."
                speaker_name = "🇮🇳 Modi"
                temp = 0.6
                next_speaker = "Trump"  # Switch for next turn

            # Add the instruction for the current turn
            turn_messages = conversation_history + [{"role": "system", "content": persona_prompt}]

            chat_response = client.chat.completions.create(
                model="gpt-4o",
                messages=turn_messages,
                max_tokens=120,  # Slightly longer responses for more character
                temperature=temp
            )
            
            # Extract the response
            response_message = chat_response.choices[0].message.content.strip()
            print(f"{speaker_name}: {response_message}\n")

            # Append the response to history
            conversation_history.append({"role": "assistant", "content": response_message})

            # Wait for some time or until stopped
            stop_funny_messages.wait(12)  # Slightly shorter wait time for better conversation flow

        except Exception as e:
            print(f"[FunnyThread] error: {e}")
            traceback.print_exc()
            stop_funny_messages.set()
            break

print("Starting image editing...")                         # CHANGED

# Start the background thread for funny messages
funny_thread = threading.Thread(target=print_funny_messages)
funny_thread.start()

try:
    print("Editing image using curl... (this may take a while)") # CHANGED
    start_time_gen = time.time()

    if not azure_deployment: # NEW: Check if deployment name is set
        print("Error: AZURE_OPENAI_DEPLOYMENT environment variable not set.")
        sys.exit(1)
    if not azure_api_key:
        print("Error: AZURE_OPENAI_API_KEY or OPENAI_API_KEY environment variable not set.")
        sys.exit(1)
    if not azure_endpoint:
        print("Error: AZURE_OPENAI_ENDPOINT or OPENAI_API_BASE environment variable not set.")
        sys.exit(1)

    # Construct the API URL
    api_url = f"{azure_endpoint}/openai/deployments/{azure_deployment}/images/edits?api-version={api_version}"
    debug_print("API URL", api_url)

    # Construct the curl command
    curl_command = [
        'curl', '-X', 'POST', api_url,
        '-H', f'api-key: {azure_api_key}',
        '-F', f'image=@{image_path}',
        # Note: Mask is omitted as it wasn't used in the original Python code
        '-F', f'prompt={prompt}'
    ]
    # Mask the API key in the debug output
    masked_curl_command = list(curl_command) # Create a copy to modify
    for i, part in enumerate(masked_curl_command):
        if part.startswith('api-key:'):
            masked_curl_command[i] = f'api-key: {_mask(azure_api_key)}'
            break
    debug_print("Curl command", " ".join(masked_curl_command)) # Log the masked command

    # Execute the curl command
    process = subprocess.run(curl_command, capture_output=True, text=True, check=False) # Use check=False to handle errors manually

    # Check for curl errors
    if process.returncode != 0:
        print(f"Error executing curl: {process.stderr}")
        sys.exit(1)

    # Parse the JSON response
    try:
        response_json = json.loads(process.stdout)
        debug_print("API Response JSON", response_json, pretty=True)
    except json.JSONDecodeError:
        print(f"Error decoding JSON response from curl: {process.stdout}")
        sys.exit(1)

    # Check for API errors in the response
    if 'error' in response_json:
        print(f"API Error: {response_json['error']}")
        sys.exit(1)
    if not response_json.get('data') or not response_json['data'][0].get('b64_json'):
        print(f"Unexpected API response format: {response_json}")
        sys.exit(1)

    # Extract the base64 encoded image data
    b64_image_data = response_json['data'][0]['b64_json']

    end_time_gen = time.time()  # Record generation end time
    elapsed_time_gen = end_time_gen - start_time_gen
    print(f"Image editing command completed in {elapsed_time_gen:.2f} seconds.") # CHANGED

    print("Decoding image data...")
    start_time_decode = time.time() # Record decoding start time
    # image_bytes = base64.b64decode(img.data[0].b64_json) # OLD
    image_bytes = base64.b64decode(b64_image_data) # NEW
    end_time_decode = time.time() # Record decoding end time
    elapsed_time_decode = end_time_decode - start_time_decode
    print(f"Image decoding completed in {elapsed_time_decode:.4f} seconds.")

    print("Saving image to output.png...")
    start_time_save = time.time() # Record saving start time
    with open("output.png", "wb") as f:
        f.write(image_bytes)
    end_time_save = time.time() # Record saving end time
    elapsed_time_save = end_time_save - start_time_save
    print(f"Image saved successfully in {elapsed_time_save:.4f} seconds.")

    end_time_total = time.time() # Record total end time
    elapsed_time_total = end_time_total - start_time_total
    print(f"\nTotal time elapsed: {elapsed_time_total:.2f} seconds.")

except KeyboardInterrupt:
    print("\nInterrupted by user. Exiting gracefully...")
    stop_funny_messages.set()
    # Optionally: cleanup or partial results here
    sys.exit(130)  # 128 + SIGINT

except Exception as err:                              # NEW
    print("ERROR:", err)
    traceback.print_exc()
    sys.exit(1)

finally:                                              # NEW – always stop thread
    stop_funny_messages.set()
    # If the thread is already dead join() returns immediately
    funny_thread.join(timeout=5)
