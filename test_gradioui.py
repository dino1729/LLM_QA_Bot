import gradio as gr
from helper_functions.query_supabasememory import query_memorypalace_stream

# Define custom CSS to make the chat interface more flexible
custom_css = """
#memory_palace_chat {
    flex-grow: 1;
    display: flex;
    flex-direction: column;
    height: 100%;
}
"""

with gr.Blocks(fill_height=True, css=custom_css) as demo:
    with gr.Tab(label="Memory Palace"):
        with gr.Column(elem_id="memory_palace_chat"):  # Use a Column to apply the CSS
            memory_palace_chat = gr.ChatInterface(
                title="Memory Palace Chat",
                description="Ask a question to generate a summary or lesson learned based on the search results from the memory palace.",
                fn=query_memorypalace_stream,
                submit_btn="Ask",
                examples=["Example question 1", "Example question 2"],
                fill_height=True
            )

demo.launch()
