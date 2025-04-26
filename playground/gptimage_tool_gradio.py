import gradio as gr
import gptimage_tool as tool
import traceback
import os
from openai import OpenAI

# Create one client instance to reuse
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE")
)

# ---- Wrappers for Gradio ----
# Add size parameter
def ui_generate(prompt: str, size: str):
    img_path = tool.run_generate(prompt, size=size) # Pass size to tool function
    # REMOVED final poll
    # REMOVED history and polling flag from return
    return img_path

# Add size and enhanced_prompt parameters
def ui_edit(img_path: str, prompt: str, enhanced_prompt: str, size: str):
    edited_img_path = img_path # Default to original path in case of error
    final_prompt = enhanced_prompt if enhanced_prompt and enhanced_prompt.strip() else prompt
    try:
        # Pass size and the chosen prompt to tool function
        edited_img_path = tool.run_edit(img_path, final_prompt, size=size)
    except Exception as e:
        print(f"Error during image edit: {e}") # Log error to console
        traceback.print_exc() # Print full traceback for debugging
        # REMOVED history update on error
        # REMOVED polling flag from return
        # Return original image path on error
        return img_path

    # REMOVED final poll
    # REMOVED history and polling flag from return
    return edited_img_path

# Renamed function and removed enhance parameter/logic
# Add size and enhanced_prompt parameters
def ui_generate_simple(prompt: str, enhanced_prompt: str, size: str):
    # Use enhanced prompt if available, otherwise use original prompt
    final_prompt = enhanced_prompt if enhanced_prompt and enhanced_prompt.strip() else prompt
    # Generate directly using the chosen prompt and size.
    img_path = tool.run_generate(final_prompt, size=size) # Pass final_prompt and size
    return img_path

def clear_generate():
	# Clear the Generate tab fields: prompt, enhanced prompt display, generated image, and size dropdown.
	# Return default size
	return "", "", None, "1024x1024" # Enhanced prompt is cleared by returning ""

def clear_edit():
	# Clear the Edit tab fields: image file, prompt (reset to default), enhanced prompt, edited image, and size dropdown.
	# Return default size
	return None, "Turn the subject into a cyberpunk character with neon lights", "", None, "1024x1024" # Enhanced prompt is cleared by returning ""

with gr.Blocks() as demo:
	gr.Markdown("# GPT Image Tool (Generate & Edit)")

	with gr.Tab("Generate"):
		# Moved Clear button definition here, to the top
		clear_gen_btn = gr.Button("Clear")
		with gr.Row():
			gen_prompt = gr.Textbox(
				label="Prompt",
				value="A futuristic cityscape at sunset, synthwave style",
				scale=4 # Give more space to prompt
			)
			# NEW Size dropdown
			gen_size = gr.Dropdown(
				label="Size",
				choices=["1024x1024", "1024x1536", "1536x1024"], # Updated choices
				value="1024x1024",
				scale=1 # Adjust scale relative to prompt
			)
		# NEW Row for Enhance and Surprise buttons
		with gr.Row():
			enhance_btn = gr.Button("Enhance Prompt")
			# NEW Surprise Me button
			surprise_gen_btn = gr.Button("🎁 Surprise Me!")
		# Removed Clear button definition from here
		# Make enhanced prompt interactive
		gen_enhanced_prompt = gr.Textbox(label="Enhanced Prompt (Editable)", interactive=True)

		with gr.Row():
			gen_ex1_btn = gr.Button("🌆 Synthwave City")
			gen_ex2_btn = gr.Button("🐶 Dog Astronaut")
			gen_ex3_btn = gr.Button("🍔 Giant Burger")
			gen_ex4_btn = gr.Button("🎨 Watercolor Forest")
		gen_btn = gr.Button("Generate")
		# Removed height=512 parameter
		gen_out = gr.Image(label="Generated Image", show_download_button=True)

		# Removed the Enhance It button from here; retain Clear button row.
		# Removed the Clear button row definition from here

		# Updated click handler: call backend enhancer directly
		enhance_btn.click(
			lambda p: tool.prompt_enhancer(p, client),
			inputs=[gen_prompt],
			outputs=[gen_enhanced_prompt] # Output to the enhanced prompt box
		)
		# NEW Click handler for Surprise Me button
		surprise_gen_btn.click(
			lambda: tool.generate_surprise_prompt(client),
			inputs=None,
			outputs=[gen_prompt] # Update the main prompt box
		)
		# Moved clear button click handler here
		clear_gen_btn.click(
			clear_generate,
			inputs=None,
			# Add gen_size to outputs, gen_enhanced_prompt is already covered
			outputs=[gen_prompt, gen_enhanced_prompt, gen_out, gen_size]
		)
		# Click handlers for generate example buttons (unchanged)
		gen_ex1_btn.click(lambda: "A futuristic cityscape at sunset, synthwave style", None, gen_prompt)
		gen_ex2_btn.click(lambda: "A golden retriever wearing a space helmet, digital art", None, gen_prompt)
		gen_ex3_btn.click(lambda: "A giant cheeseburger resting on a mountaintop", None, gen_prompt)
		gen_ex4_btn.click(lambda: "A dense forest painted in watercolor style", None, gen_prompt)

		# Updated generate button click handler
		gen_btn.click(
			ui_generate_simple, # Use the simplified generation function
			# Add gen_enhanced_prompt and gen_size to inputs
			inputs=[gen_prompt, gen_enhanced_prompt, gen_size],
			outputs=[gen_out] # Output only the generated image
		)
		# Removed clear button click handler from here

	with gr.Tab("Edit"):
		# NEW Clear button for the Edit tab, moved to the top.
		clear_edit_btn = gr.Button("Clear")
		edit_img = gr.Image(type="filepath", label="Image to Edit", height=512)
		with gr.Row(): # NEW Row for prompt and size dropdown
			edit_prompt = gr.Textbox(
				label="Edit Prompt",
				value="Turn the subject into a cyberpunk character with neon lights",
				scale=4 # Give more space to prompt
			)
			# NEW Size dropdown for Edit tab
			edit_size = gr.Dropdown(
				label="Size",
				choices=["1024x1024", "1024x1792", "1792x1024"], # Updated choices
				value="1024x1024",
				scale=1 # Adjust scale relative to prompt
			)
		# NEW Row for Enhance and Surprise buttons for Edit tab
		with gr.Row():
			edit_enhance_btn = gr.Button("Enhance Prompt")
			# NEW Surprise Me button for Edit tab
			surprise_edit_btn = gr.Button("🎁 Surprise Me!")
		# NEW Enhanced prompt preview for Edit tab, make it interactive
		edit_enhanced_prompt = gr.Textbox(label="Enhanced Prompt (Editable)", interactive=True)

		with gr.Row():
			ghibli_btn = gr.Button("🎨 Ghibli Style")
			simp_btn = gr.Button("📺 Simpsons")
			sp_btn = gr.Button("☃️ South Park")
			comic_btn = gr.Button("💥 Comic Style")

		edit_btn = gr.Button("Edit")
		# Removed height=512 parameter
		edit_out = gr.Image(label="Edited Image", show_download_button=True)

		# Removed Clear button definition from here.

		# NEW Click handler for the Edit tab's Enhance button
		edit_enhance_btn.click(
			lambda p: tool.prompt_enhancer(p, client),
			inputs=[edit_prompt],
			outputs=[edit_enhanced_prompt] # Output to the enhanced prompt box
		)
		# NEW Click handler for Edit tab's Surprise Me button
		surprise_edit_btn.click(
			lambda: tool.generate_surprise_prompt(client),
			inputs=None,
			outputs=[edit_prompt] # Update the edit prompt box
		)

		# Click handlers for edit example buttons (unchanged)
		ghibli_btn.click(lambda: "Convert the picture into Ghibli style animation", None, edit_prompt)
		simp_btn.click(lambda: "Turn the subjects into a Simpsons character", None, edit_prompt)
		sp_btn.click(lambda: "Turn the subjects into a South Park character", None, edit_prompt)
		comic_btn.click(lambda: "Convert the picture into a comic book style drawing with compelling futuristic story and cool dialogues", None, edit_prompt)

		edit_btn.click(
			ui_edit,
			# Add edit_enhanced_prompt and edit_size to inputs
			inputs=[edit_img, edit_prompt, edit_enhanced_prompt, edit_size],
			outputs=[edit_out]
		)
		# Updated Clear button click handler outputs for Edit tab, moved earlier
		clear_edit_btn.click(
			clear_edit,
			inputs=None,
			# Add edit_size to outputs, edit_enhanced_prompt is already covered
			outputs=[edit_img, edit_prompt, edit_enhanced_prompt, edit_out, edit_size]
		)

# Launch the Gradio app
if __name__ == "__main__":
	demo.queue().launch()
