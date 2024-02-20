import whisper

model = whisper.load_model("base")
result = model.transcribe("finalproject_clip.mp3")
# print(result["text"])
# Save the result to a file
with open("transcribed_finalproject.txt", "w") as f:
    f.write(result["text"])

print("Transcription complete!")
