import sys
import os
from moviepy import VideoFileClip
import whisper
import openai
from dotenv import load_dotenv
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# Load environment variables from .env file
load_dotenv()
# Get the directory of the current script
current_script_directory = os.path.dirname(os.path.abspath(__file__))
# Construct the absolute path to the font file (assuming it's in the same directory)
monaco_font_path = os.path.join(current_script_directory, 'MONACO.TTF')
reportlab_monaco_registered = False # Initialize
if os.path.exists(monaco_font_path):
    pdfmetrics.registerFont(TTFont('Monaco', monaco_font_path))
    print("Monaco font registered successfully for ReportLab.")
    reportlab_monaco_registered = True
else:
    print(f"Warning: MONACO.TTF not found at {monaco_font_path}. ReportLab will use default fonts.")
    # reportlab_monaco_registered remains False

# --- Remove all global ReportLab style definitions ---

def extract_audio_from_video(video_path, audio_output_path):
    video = VideoFileClip(video_path)
    # Save as WAV (mono, 16kHz, 16-bit PCM) for compatibility with speech APIs
    wav_output_path = os.path.splitext(audio_output_path)[0] + ".wav"
    video.audio.write_audiofile(
        wav_output_path,
        codec='pcm_s16le',
        fps=16000,
        nbytes=2,
        ffmpeg_params=["-ac", "1"]  # mono
    )
    return wav_output_path

def format_transcript_with_llm(transcript):
    """
    Uses OpenAI GPT to organize and format the transcript in a more readable way.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    
    # Create client with the new OpenAI format
    client = openai.OpenAI(
        api_key=api_key,
        base_url=base_url if base_url else None
    )

    model_name = os.getenv("FAST_LLM_MODELNAME")

    print(f"Using model: {model_name}")

    # Updated prompt with new instructions
    prompt = (
        "You are a transcript analyst. I will give you an unedited transcript. Your tasks:\n\n"
        "Segment the Transcript\n"
        "Read the entire transcript.\n"
        "Divide it into 5–10 logical sections based on topic shifts or major sub‑topics.\n"
        "For Each Section\n"
        "a. Heading\n"
        "Assign a clear, descriptive heading (3–7 words).\n"
        "b. Verbatim Transcript\n"
        "Copy the exact transcript text that belongs to this section, preserving all words and punctuation.\n"
        "c. Summary\n"
        "Write a 2–4 sentence summary capturing the key points and purpose of that section.\n"
        "Quiz Questions\n"
        "After all sections, create 15 multiple‑choice questions covering the main concepts.\n"
        "For each question:\n"
        "• Provide four answer options labeled A, B, C, D.\n"
        "• Clearly indicate the correct answer.\n"
        "Ensure questions test comprehension, not just recall of trivial facts.\n"
        "Output exactly in this format:\n\n"
        "Format everything in clean Markdown using headings (##, ###), numbered lists, and bullet lists for readability.\n\n"
        "Section 1: <Heading>\n"
        "Transcript:\n"
        "“<Verbatim transcript for section 1>”\n"
        "Summary:\n"
        "<2–4 sentences>\n\n"
        "Section 2: <Heading>\n"
        "Transcript:\n"
        "“<Verbatim transcript for section 2>”\n"
        "Summary:\n"
        "<2–4 sentences>\n\n"
        "…\n\n"
        "Quiz Questions:\n\n"
        "1. <Question text>\n"
        "A. <Option A>\n"
        "B. <Option B>\n"
        "C. <Option C>\n"
        "D. <Option D>\n"
        "Answer: <A/B/C/D>\n"
        "…\n"
        "15. <Question text>\n"
        "A. …\n"
        "B. …\n"
        "C. …\n"
        "D. …\n"
        "Answer: <A/B/C/D>\n\n"
        f"Here is the raw transcript:\n\n{transcript}\n\nFormatted output:"
    )
    
    # Use the new client-based API format
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            # Updated system message to align with the new prompt persona
            {"role": "system", "content": "You are a transcript analyst tasked with segmenting, summarizing, and creating quiz questions from transcripts according to a specific format."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=8192,
    )
    
    return response.choices[0].message.content.strip()

def save_as_pdf(markdown_content, output_path):
    """
    Simplified PDF generator: saves the markdown as plain paragraphs using Apple SF Pro Display, Monaco, or Helvetica.
    """
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.pdfbase import pdfmetrics
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.styles import ParagraphStyle

    # Try to register Apple SF Pro Display, Monaco, or fallback to Helvetica
    font_candidates = [
        ('SFProDisplay', '/System/Library/Fonts/SFNSDisplay.ttf'),  # macOS system font
        ('Monaco', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'MONACO.TTF')),
    ]
    font_name = 'Helvetica'
    for name, path in font_candidates:
        if os.path.exists(path):
            try:
                pdfmetrics.registerFont(TTFont(name, path))
                font_name = name
                print(f"Using font: {name} ({path})")
                break
            except Exception as e:
                print(f"Could not register font {name}: {e}")

    doc = SimpleDocTemplate(output_path, pagesize=letter,
                            leftMargin=1*inch, rightMargin=1*inch,
                            topMargin=1*inch, bottomMargin=1*inch)
    style = ParagraphStyle(
        name='Custom',
        fontName=font_name,
        fontSize=11,
        leading=14,
        spaceAfter=8,
    )
    story = []
    for para in markdown_content.split('\n\n'):
        if para.strip():
            story.append(Paragraph(para.replace('\n', '<br/>'), style))
            story.append(Spacer(1, 8))
    doc.build(story)
    print(f"PDF saved to {output_path} with font: {font_name}")

def process_video(video_path):
    base_name = os.path.splitext(video_path)[0]
    audio_output_path = base_name + ".wav" # Use .wav directly as it's the target format
    transcript_output_path = base_name + "_transcript.txt"
    formatted_md_output_path = base_name + "_formatted_transcript.md" # Added MD path
    formatted_pdf_output_path = base_name + "_formatted_transcript.pdf"

    # --- Audio Extraction ---
    if os.path.exists(audio_output_path):
        print(f"Audio file already exists: {audio_output_path}. Skipping extraction.")
        wav_path = audio_output_path
    else:
        print(f"Extracting audio from {video_path}...")
        try:
            wav_path = extract_audio_from_video(video_path, audio_output_path)
            print(f"Audio extracted to {wav_path}")
        except Exception as e:
            print(f"Error extracting audio: {e}")
            return # Stop processing this file if audio extraction fails

    # --- Transcription ---
    transcript = None
    if os.path.exists(transcript_output_path):
        print(f"Raw transcript file already exists: {transcript_output_path}. Skipping transcription.")
        try:
            with open(transcript_output_path, "r", encoding="utf-8") as f:
                transcript = f.read()
            if not transcript:
                print(f"Warning: Existing transcript file {transcript_output_path} is empty. Re-transcribing.")
                os.remove(transcript_output_path) # Remove empty file to allow re-transcription
            else:
                 print("Loaded transcript from existing file.")
        except Exception as e:
            print(f"Error reading existing transcript file {transcript_output_path}: {e}. Re-transcribing.")
            # Attempt re-transcription if reading fails

    if transcript is None: # Proceed with transcription if file didn't exist, was empty, or reading failed
        print("Transcribing audio with Whisper...")
        try:
            model = whisper.load_model("base")  # or "small", "medium", etc.
            result = model.transcribe(wav_path)
            transcript = result["text"]
            print("Saving raw transcript...")
            with open(transcript_output_path, "w", encoding="utf-8") as f:
                f.write(transcript)
            print(f"Raw transcript saved to {transcript_output_path}")
        except Exception as e:
            print(f"Error during transcription: {e}")
            return # Stop processing if transcription fails

    # Ensure we have a transcript before proceeding
    if not transcript:
        print("Failed to obtain transcript. Skipping formatting.")
        return

    # --- Formatting and Saving ---
    formatted_transcript_md = None # Initialize variable

    # Check if formatted MD exists to potentially skip LLM call
    if os.path.exists(formatted_md_output_path):
        print(f"Formatted MD file already exists: {formatted_md_output_path}. Attempting to load content.")
        try:
            with open(formatted_md_output_path, "r", encoding="utf-8") as f:
                formatted_transcript_md = f.read()
            if not formatted_transcript_md:
                 print(f"Warning: Existing formatted MD file {formatted_md_output_path} is empty. Will call LLM.")
            else:
                 print("Loaded formatted transcript from existing MD file. Skipping LLM call.")
        except Exception as e:
            print(f"Error reading existing formatted MD file {formatted_md_output_path}: {e}. Will call LLM.")
            formatted_transcript_md = None # Ensure LLM call happens if read fails

    # Call LLM only if formatted content wasn't loaded from file
    if formatted_transcript_md is None:
        try:
            print("Formatting transcript with LLM...")
            formatted_transcript_md = format_transcript_with_llm(transcript)
            # Save the newly generated MD
            print(f"Saving formatted transcript as MD to {formatted_md_output_path}...")
            with open(formatted_md_output_path, "w", encoding="utf-8") as f:
                f.write(formatted_transcript_md)
            print(f"Formatted transcript MD saved to {formatted_md_output_path}")
        except Exception as e:
             print(f"Error formatting transcript with LLM or saving MD: {e}")
             # If LLM fails, we can't proceed to PDF generation
             return

    # Ensure we have formatted content before trying to save PDF
    if not formatted_transcript_md:
        print("Failed to obtain formatted transcript. Skipping PDF generation.")
        return

    # Save PDF (only if it doesn't exist)
    if os.path.exists(formatted_pdf_output_path):
         print(f"Formatted PDF file already exists: {formatted_pdf_output_path}. Skipping PDF save.")
    else:
        try:
            print(f"Saving formatted transcript as PDF to {formatted_pdf_output_path}...")
            save_as_pdf(formatted_transcript_md, formatted_pdf_output_path)
        except Exception as e:
            print(f"Error saving formatted transcript as PDF: {e}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python video_to_audio_to_text.py <folder_path_containing_mp4_or_mp3_videos>")
        sys.exit(1)
    folder_path = sys.argv[1]
    if not os.path.isdir(folder_path):
        print(f"Folder not found: {folder_path}")
        sys.exit(1)
    video_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.mp4', '.mp3'))]
    if not video_files:
        print("No .mp4 or .mp3 files found in the folder.")
        sys.exit(0)
    for video_file in video_files:
        video_path = os.path.join(folder_path, video_file)
        try:
            process_video(video_path)
        except Exception as e:
            print(f"Error processing {video_file}: {e}")

if __name__ == "__main__":
    main()
