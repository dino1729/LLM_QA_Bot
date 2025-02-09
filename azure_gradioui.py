import os
import gradio as gr
import logging
import sys
from llama_index.core import StorageContext, load_index_from_storage, get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import PromptTemplate
from helper_functions.chat_gita import gita_answer
from helper_functions.chat_generation_with_internet import internet_connected_chatbot
from helper_functions.trip_planner import generate_trip_plan
from helper_functions.food_planner import craving_satisfier
from helper_functions.analyzers import analyze_article, analyze_ytvideo, analyze_media, analyze_file, upload_data_to_supabase, clearallfiles
from config import config
from helper_functions.query_supabasememory import query_memorypalace_stream

def upload_file(files, memorize):

    global example_queries, summary, query_component

    analysis = analyze_file(files, memorize)
    message = analysis["message"]
    summary = analysis["summary"]
    example_queries = analysis["example_queries"]
    file_title = analysis["file_title"]
    file_memoryupload_status = analysis["file_memoryupload_status"]

    return message, gr.Dataset(components=[query_component], samples=example_queries), summary, file_title, file_memoryupload_status

def download_ytvideo(url, memorize):

    global example_queries, summary, video_title, query_component

    analysis = analyze_ytvideo(url, memorize)
    message = analysis["message"]
    summary = analysis["summary"]
    example_queries = analysis["example_queries"]
    video_title = analysis["video_title"]
    video_memoryupload_status = analysis["video_memoryupload_status"]

    return message, gr.Dataset(components=[query_component], samples=example_queries), summary, video_title, video_memoryupload_status

def download_art(url, memorize):

    global example_queries, summary, article_title, query_component

    analysis = analyze_article(url, memorize)
    message = analysis["message"]
    summary = analysis["summary"]
    example_queries = analysis["example_queries"]
    article_title = analysis["article_title"]
    article_memoryupload_status = analysis["article_memoryupload_status"]

    return message, gr.Dataset(components=[query_component], samples=example_queries), summary, article_title, article_memoryupload_status

def download_media(url, memorize):

    global example_queries, summary, media_title, query_component

    analysis = analyze_media(url, memorize)
    message = analysis["message"]
    summary = analysis["summary"]
    example_queries = analysis["example_queries"]
    media_title = analysis["media_title"]
    media_memoryupload_status = analysis["media_memoryupload_status"]

    return message, gr.Dataset(components=[query_component], samples=example_queries), summary, media_title, media_memoryupload_status

def ask(question, history):
    
    history = history or []
    answer = ask_query(question)

    return answer

def ask_query(question):

    storage_context = StorageContext.from_defaults(persist_dir=VECTOR_FOLDER)
    vector_index = load_index_from_storage(storage_context, index_id="vector_index")
    retriever = VectorIndexRetriever(
        index=vector_index,
        similarity_top_k=10,
    )
    response_synthesizer = get_response_synthesizer(
        text_qa_template=qa_template,
    )
    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
    )
    response = query_engine.query(question)
    answer = response.response

    return answer

def ask_fromfullcontext(question, fullcontext_template):
    
    storage_context = StorageContext.from_defaults(persist_dir=SUMMARY_FOLDER)
    summary_index = load_index_from_storage(storage_context, index_id="summary_index")
    retriever = summary_index.as_retriever(
        retriever_mode="default",
    )
    response_synthesizer = get_response_synthesizer(
        response_mode="tree_summarize",
        summary_template=fullcontext_template,
    )
    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
    )
    response = query_engine.query(question)
    answer = response.response
    
    return answer

def load_example(example_id):
    
    global example_queries
    return example_queries[example_id][0]

def cleartext(query, output):
    # Function to clear text
    return ["", ""]

def clearfield(field):
    # Function to clear text
    return [""]

def clearhistory(field1, field2, field3):
    # Function to clear history
    return ["", "", ""]

logging.basicConfig(stream=sys.stdout, level=logging.CRITICAL)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

sum_template = config.sum_template
eg_template = config.eg_template
ques_template = config.ques_template

example_qs = []
summary = "No Summary available yet"
example_queries = config.example_queries
example_memorypalacequeries = config.example_memorypalacequeries
example_internetqueries = config.example_internetqueries
example_bhagawatgeetaqueries = config.example_bhagawatgeetaqueries
summary_template = PromptTemplate(sum_template)
example_template = PromptTemplate(eg_template)
qa_template = PromptTemplate(ques_template)

UPLOAD_FOLDER = config.UPLOAD_FOLDER
SUMMARY_FOLDER = config.SUMMARY_FOLDER
VECTOR_FOLDER = config.VECTOR_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(SUMMARY_FOLDER):
    os.makedirs(SUMMARY_FOLDER)
if not os.path.exists(VECTOR_FOLDER):
    os.makedirs(VECTOR_FOLDER)

# theme = gr.Theme.from_hub("sudeepshouche/minimalist")
# with gr.Blocks(theme=theme) as llmapp:
with gr.Blocks(fill_height=True) as llmapp:
    gr.Markdown(
        """
        <h1><center><b>LLM Bot: Your AI-Powered Knowledge Companion</center></h1>
        """
    )
    gr.Markdown(
        """
        <center>
        <br>
        Dive into the world of AI with LLM Bot, your go-to assistant for exploring, learning, and storing knowledge. <br>
        Activate the "Memorize" feature to seamlessly integrate insights into your digital memory palace. <br>
        Powered by the cutting-edge GPT-4 model for accurate and insightful responses. <br>
        </center>
        """
    )
    with gr.Tab(label="LLM APP"):
        with gr.Row():
            with gr.Column():
                memorize = gr.Checkbox(label="I want this information stored in my memory palace!")
                reset_button = gr.Button(value="Reset Database. Deletes all local data")
        with gr.Row():
            with gr.Column():
                with gr.Tab(label="Video Analyzer"):
                    with gr.Row():
                        with gr.Column(scale=0):
                            yturl = gr.Textbox(placeholder="Input must be a Youtube URL", label="Enter Youtube URL")
                            video_output = gr.Textbox(label="Video Processing Status")
                            video_button = gr.Button(value="Process")
                        with gr.Column(scale=0):
                            video_title = gr.Textbox(placeholder="Video title to be generated here", label="Video Title")
                            video_memoryupload_status = gr.Textbox(label="Video Memory Upload Status")
                            video_memoryupload_button = gr.Button(value="Upload to Memory")
                        vsummary_output = gr.Textbox(placeholder="Summary will be generated here", label="Key takeaways", show_copy_button=True)
                with gr.Tab(label="Article Analyzer"):
                    with gr.Row():
                        with gr.Column(scale=0):
                            arturl = gr.Textbox(placeholder="Input must be a URL", label="Enter Article URL")
                            article_output = gr.Textbox(label="Article Processing Status")
                            article_button = gr.Button(value="Process")
                        with gr.Column(scale=0):
                            article_title = gr.Textbox(placeholder="Article title to be generated here", label="Article Title")
                            article_memoryupload_status = gr.Textbox(label="Article Memory Upload Status")
                            article_memoryupload_button = gr.Button(value="Upload to Memory")
                        asummary_output = gr.Textbox(placeholder="Summary will be generated here", label="Key takeaways", show_copy_button=True)
                with gr.Tab(label="File Analyzer"):
                    with gr.Row():
                        with gr.Column(scale=0):
                            files = gr.Files(label="Supported types: pdf, txt, docx, png, jpg, jpeg, mp3")
                            file_output = gr.Textbox(label="File Processing Status")
                            file_button = gr.Button(value="Process")
                        with gr.Column(scale=0):
                            file_title = gr.Textbox(placeholder="File title to be generated here", label="File Title")
                            file_memoryupload_status = gr.Textbox(label="File Memory Upload Status")
                            file_memoryupload_button = gr.Button(value="Upload to Memory")
                        fsummary_output = gr.Textbox(placeholder="Summary will be generated here", label="Key takeaways", show_copy_button=True)
                with gr.Tab(label="Media URL Analyzer"):
                    with gr.Row():
                        with gr.Column(scale=0):
                            mediaurl = gr.Textbox(placeholder="Input must be a URL", label="Enter Media URL")
                            media_output = gr.Textbox(label="Media Processing Status")
                            media_button = gr.Button(value="Process")
                        with gr.Column(scale=0):
                            media_title = gr.Textbox(placeholder="Media title to be generated here", label="Media Title")
                            media_memoryupload_status = gr.Textbox(label="Media Memory Upload Status")
                            media_memoryupload_button = gr.Button(value="Upload to Memory")
                        msummary_output = gr.Textbox(placeholder="Summary will be generated here", label="Key takeaways", show_copy_button=True)
        with gr.Row():
            with gr.Column(scale=8):
                chatui = gr.ChatInterface(
                    ask,
                    submit_btn="Ask",
                    fill_height=True,
                    type="messages"  # Updated to use 'messages' format
                )
                query_component = chatui.textbox
            with gr.Column(scale=2):
                examples = gr.Dataset(label="Questions", samples=example_queries, components=[query_component], type="index")      
    with gr.Tab(label="Memory Palace"):
        memory_palace_chat = gr.ChatInterface(
            title="Memory Palace Chat",
            description="Ask a question to generate a summary or lesson learned based on the search results from the memory palace.",
            fn=query_memorypalace_stream,
            submit_btn="Ask",
            examples=example_memorypalacequeries,
            fill_height=True,
            type="messages"  # Updated to use 'messages' format
        )  
    with gr.Tab(label="AI Assistant"):
        gr.ChatInterface(
            internet_connected_chatbot,
            additional_inputs=[
                gr.Radio(label="Model", choices=["GROQ", "GPT4", "GEMINI_THINKING", "COHERE"], value="GROQ"),
                gr.Slider(10, 8680, value=4840, label = "Max Output Tokens"),
                gr.Slider(0.1, 0.9, value=0.5, label = "Temperature"),
            ],
            examples=example_internetqueries,
            submit_btn="Ask",
            fill_height=True,
            type="messages"  # Updated to use 'messages' format
        )
    with gr.Tab(label="Fun"):
        with gr.Tab(label="City Planner"):
            with gr.Row():
                city_name = gr.Textbox(placeholder="Enter the name of the city", label="City Name")
                number_of_days = gr.Textbox(placeholder="Enter the number of days", label="Number of Days")
                city_button = gr.Button(value="Plan", scale=0)
            with gr.Row():
                city_output = gr.Textbox(label="Trip Plan", show_copy_button=True)
                clear_trip_button = gr.Button(value="Clear", scale=0)
        with gr.Tab(label="Bhagawad Gita"):
            gitachat = gr.ChatInterface(
                gita_answer,
                additional_inputs=[
                    gr.Radio(label="Model", choices=["COHERE", "GEMINI", "GPT4", "GPT35TURBO", "MIXTRAL8x7B"], value="GPT35TURBO"),
                    gr.Slider(10, 1680, value=840, label = "Max Output Tokens"),
                    gr.Slider(0.1, 0.9, value=0.5, label = "Temperature"),
                ],
                examples=example_bhagawatgeetaqueries,
                submit_btn="Ask",
                fill_height=True,
                type="messages"  # Updated to use 'messages' format
            )
            gita_question = gitachat.textbox
        with gr.Tab(label="Cravings Generator"):
            with gr.Row():
                craving_city_name = gr.Textbox(placeholder="Enter the name of the city", label="City Name")
                craving_cuisine = gr.Textbox(placeholder="What are you craving for? Enter idk if you don't know what you want to eat", label="Food")
                craving_button = gr.Button(value="Cook", scale=0)
            with gr.Row():
                craving_output = gr.Textbox(label="Food Places", show_copy_button=True)
                clear_craving_button = gr.Button(value="Clear", scale=0)
    
    memory_palace_question = memory_palace_chat.textbox

    video_button.click(download_ytvideo, inputs=[yturl, memorize], outputs=[video_output, examples, vsummary_output, video_title, video_memoryupload_status], show_progress=True)
    video_memoryupload_button.click(upload_data_to_supabase, inputs=[video_title, yturl], outputs=[video_memoryupload_status], show_progress=True)

    article_button.click(download_art, inputs=[arturl, memorize], outputs=[article_output, examples, asummary_output, article_title, article_memoryupload_status], show_progress=True)
    article_memoryupload_button.click(upload_data_to_supabase, inputs=[article_title, arturl], outputs=[article_memoryupload_status], show_progress=True)

    file_button.click(upload_file, inputs=[files, memorize], outputs=[file_output, examples, fsummary_output, file_title, file_memoryupload_status], show_progress=True)
    file_memoryupload_button.click(upload_data_to_supabase, inputs=[file_title, memorize], outputs=[file_memoryupload_status], show_progress=True)

    media_button.click(download_media, inputs=[mediaurl, memorize], outputs=[media_output, examples, msummary_output, media_title, media_memoryupload_status], show_progress=True)
    media_memoryupload_button.click(upload_data_to_supabase, inputs=[media_title, mediaurl], outputs=[media_memoryupload_status], show_progress=True)

    city_button.click(generate_trip_plan, inputs=[city_name, number_of_days], outputs=[city_output], show_progress=True)
    craving_button.click(craving_satisfier, inputs=[craving_city_name, craving_cuisine], outputs=[craving_output], show_progress=True)
    clear_trip_button.click(clearhistory, inputs=[city_name, number_of_days, city_output], outputs=[city_name, number_of_days, city_output])
    clear_craving_button.click(clearhistory, inputs=[craving_city_name, craving_cuisine, craving_output], outputs=[craving_city_name, craving_cuisine, craving_output])

    examples.click(load_example, inputs=[examples], outputs=[query_component])

    reset_button.click(clearallfiles)
    reset_button.click(lambda: ["", "", "", "", "", "", "", "", "", "", "", "",  "", "", "", ""], inputs=[], outputs=[video_output, vsummary_output, video_title, video_memoryupload_status, article_output, asummary_output, article_title, article_memoryupload_status, file_output, fsummary_output, file_title, file_memoryupload_status, media_output, msummary_output, media_title, media_memoryupload_status])
    

if __name__ == '__main__':
    llmapp.launch(server_name='0.0.0.0', server_port=7860)
