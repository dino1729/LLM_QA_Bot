import base64
from mimetypes import guess_type
import os
from openai import AzureOpenAI
import dotenv
from io import BytesIO
import io

def encode_imagebuffer(image_buffer: BytesIO) -> bytes:

    mime_type, _ = guess_type(image_buffer.name)
    if mime_type is None:
        mime_type = 'application/octet-stream' # Default MIME type if none is found

    base64_encoded_data = base64.b64encode(image_buffer.read()).decode("utf-8")

    return f"data:{mime_type};base64,{base64_encoded_data}"

def local_image_to_url(image_path):
    # Guess the MIME type of the image based on the file extension
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = 'application/octet-stream'  # Default MIME type if none is found

    # Read and encode the image file
    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode("utf-8")

    # Construct the data URL
    return f"data:{mime_type};base64,{base64_encoded_data}"

def generate_prompt_messages_frombuffer(message, dialog_messages, image_buffer=None):
    
    system_prompt = "You are a truth seeking assistant."
    messages = []

    if image_buffer is None:
        messages.append({"role": "system", "content": system_prompt})
        # Text-based interaction
        for dialog_message in dialog_messages:
            messages.append({"role": "user", "content": dialog_message["user"]})
            messages.append({"role": "assistant", "content": dialog_message["bot"]})
        messages.append({"role": "user", "content": message})
    else:
        # Reset Buffer
        image_buffer.seek(0)
        messages.append({
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": system_prompt
                }
            ]
        })
        # Iterate over dialog messages and append them
        for dialog_message in dialog_messages:
            if "user" in dialog_message:
                messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": dialog_message["user"]
                        }
                    ]
                })
            if "bot" in dialog_message:
                messages.append({
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": dialog_message["bot"]
                        }
                    ]
                })
        # Add the current user message
        user_message_content = [{
            "type": "text",
            "text": message
        }]
        encoded_image = encode_imagebuffer(image_buffer)
        user_message_content.append({
            "type": "image_url",
            "image_url": {
                "url": encoded_image
            }
        })
        messages.append({
            "role": "user",
            "content": user_message_content
        })

    return messages

def generate_prompt_messages_fromlocal(message, dialog_messages, image_path=None):
    
    system_prompt = "You are a helpful and super-intelligent voice assistant, that accurately answers user queries. Be accurate, helpful, concise, and clear."
    messages = [{"role": "system", "content": system_prompt}]

    if image_path is None:
        # Text-based interaction
        for dialog_message in dialog_messages:
            messages.append({"role": "user", "content": dialog_message["user"]})
            messages.append({"role": "assistant", "content": dialog_message["bot"]})
        messages.append({"role": "user", "content": message})
    else:
        # Image-based interaction
        for dialog_message in dialog_messages:
            messages.append({
                "role": "user",
                "content": {
                    "type": "text",
                    "text": dialog_message["user"]
                }
            })
            messages.append({"role": "assistant", "content": dialog_message["bot"]})
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": message},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": local_image_to_url(image_path)
                    }
                }
            ]
        })

    return messages

dotenv.load_dotenv()
# image_path = "./dino.jpeg"
# image_path = "./superbowl.jpg"
image_path = "./solvay.jpg"
# image_path = None
# image_buffer = None

if image_path is not None:
    image_buffer = io.BytesIO()
    # Load the image into memory buffer
    with open(image_path, "rb") as image_file:
        image_buffer = BytesIO(image_file.read())
    image_buffer.name = os.path.basename(image_path)
    image_buffer.seek(0)

azure_api_key = os.getenv("AZURE_API_KEY")
azure_api_base = os.getenv("AZURE_API_BASE")
azure_chatapi_version = os.getenv("AZURE_CHATAPI_VERSION")

OPENAI_COMPLETION_OPTIONS = {
    "temperature": 0.5,
    "max_tokens": 512,
}

client = AzureOpenAI(
    api_key=azure_api_key,
    azure_endpoint=azure_api_base,
    api_version=azure_chatapi_version,
)

while True:
    user_query = input("Enter your query: ")
    dialog_messages = []
    dialog_messages.append({"user": "How are you?", "bot": "I'm doing great, thank you for asking!"})

    conversation_buffer = generate_prompt_messages_frombuffer(user_query, dialog_messages, image_buffer=image_buffer)
    conversation_local = generate_prompt_messages_fromlocal(user_query, dialog_messages, image_path=image_path)

    # Save the prompts to text files
    with open("conversation_buffer.txt", "w") as file:
        file.write(str(conversation_buffer))

    response = client.chat.completions.create(
        model="gpt-4",
        messages=conversation_buffer,
        **OPENAI_COMPLETION_OPTIONS
    )
    assistant_reply = response.choices[0].message.content
    print("Bot: ", assistant_reply)
    
