import base64
import os
import time
import sys # Import sys
import threading
from openai import OpenAI

api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_API_BASE")

client = OpenAI(api_key=api_key, base_url=base_url)

start_time_total = time.time() # Record total start time

# Concise prompt
prompt = """
Create a single page comic or graphic novel covering an entire story of Chelsea FC who lost its way and goes on an adventure, relentlessly, to win UEFA CL in the end. The entire story, along with dialogues, must fit within one page of 6 – 8 panels. You can create the characters and graphics based on any theme of your choice.
"""

# Flag to signal the background thread to stop
stop_funny_messages = threading.Event()

def print_funny_messages():
    while not stop_funny_messages.is_set():
        chat_response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are Rick Sanchez that makes funny comments about image prompts aand I'm Morty"},
                {"role": "user", "content": f"Make a funny comment about this image prompt: '{prompt}'"}
            ],
            max_tokens=60,
            temperature=0.8
        )
        funny_message = chat_response.choices[0].message.content.strip()
        print(f"{funny_message}\n")
        # Wait for 10 seconds or until stopped
        stop_funny_messages.wait(10)

print("Starting image generation...")

# Start the background thread for funny messages
funny_thread = threading.Thread(target=print_funny_messages)
funny_thread.start()

start_time_gen = time.time()  # Record generation start time

print("Generating image... (this may take a while)")

img = client.images.generate(
    model="gpt-image-1",
    prompt=prompt,
    n=1,
    size="1024x1536"
)

# Signal the background thread to stop and wait for it to finish
stop_funny_messages.set()
funny_thread.join()

end_time_gen = time.time()  # Record generation end time
elapsed_time_gen = end_time_gen - start_time_gen
print(f"Image generation completed in {elapsed_time_gen:.2f} seconds.")

print("Decoding image data...")
start_time_decode = time.time() # Record decoding start time
image_bytes = base64.b64decode(img.data[0].b64_json)
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
