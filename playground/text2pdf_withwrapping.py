import glob
import os
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

def clean_text(text):
    """
    Cleans the text by replacing common problematic whitespace characters.
    """
    text = text.replace('\u00A0', ' ')  # Replace non-breaking spaces with regular spaces
    text = text.replace('\t', '    ')   # Optionally, replace tabs with spaces
    return text

def merge_text_to_pdf_from_directory(directory_path, output_pdf):
    """
    Extracts text from all .txt files in the specified directory,
    stores it in memory, and then writes it to a single PDF.
    Each file's content is preceded by a break with the file name.
    """
    txt_files = glob.glob(f"{directory_path}/*.txt")  # Ensure correct file extension
    all_text = []

    # Read and store all text content in memory, with file name breaks
    for text_file in txt_files:
        file_name = os.path.basename(text_file)
        with open(text_file, 'r', encoding='utf-8') as file:
            text_content = file.read()
            cleaned_text = clean_text(text_content)
            # Format the file name break and prepend it to the text content
            file_break = f"\n\n{'*' * 10} {file_name} {'*' * 10}\n\n"
            all_text.append(file_break + cleaned_text)

    # Create a single PDF with all text content
    c = canvas.Canvas(output_pdf, pagesize=letter)
    top_margin = 750
    bottom_margin = 40
    left_margin = 40
    right_margin = 575  # Assuming letter size (612x792) and a right margin of 37 (612-575)
    text_object = c.beginText(40, 750)  # Start position; adjust as needed
    text_object.setFont("Helvetica", 10)
    
    for text_content in all_text:
        lines = text_content.split('\n')
        for line in lines:
            # Split the line into words and rebuild it to fit the page width
            words = line.split()
            line = ''
            for word in words:
                # Check if adding the next word exceeds the line width
                if c.stringWidth(line + word, "Helvetica", 10) + left_margin < right_margin:  # Adjusted for right margin
                    line += word + ' '
                else:
                    # Draw the current line and start a new one
                    text_object.textLine(line)
                    line = word + ' '
                # Check if we need to move to the next page due to reaching the bottom margin
                if text_object.getY() < bottom_margin + 10:  # Adjusted to check against bottom margin
                    c.drawText(text_object)
                    c.showPage()  # Create a new page
                    text_object = c.beginText(left_margin, top_margin)  # Reset text object for the new page
                    text_object.setFont("Helvetica", 10)
            # Draw the last line of the paragraph
            text_object.textLine(line)
            
            # Move to the next page if the text reaches the bottom margin
            if text_object.getY() < 40:  # Bottom margin
                c.drawText(text_object)
                c.showPage()  # Create a new page
                text_object = c.beginText(40, 750)  # Reset text object for the new page
                text_object.setFont("Helvetica", 10)
    
    c.drawText(text_object)
    c.save()

input_directory = os.getcwd()
output_pdf_file = 'combined_output.pdf'
merge_text_to_pdf_from_directory(input_directory, output_pdf_file)
print(f"Combined text files from {input_directory} into {output_pdf_file}")
