import os
import gradio as gr
import logging
import sys
from llama_index import (
    StorageContext,
    load_index_from_storage,
    get_response_synthesizer,
)
from llama_index.retrievers import VectorIndexRetriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.prompts import PromptTemplate
from helper_functions.chat_gita import gita_answer
from helper_functions.chat_generation_with_internet import internet_connected_chatbot
from helper_functions.trip_planner import generate_trip_plan
from helper_functions.food_planner import craving_satisfier
from helper_functions.analyzers import analyze_article, analyze_ytvideo, analyze_media, analyze_file
from config import config

def upload_file(files, memorize):

    global example_queries, summary, query_component

    analysis = analyze_file(files, memorize)
    message = analysis["message"]
    summary = analysis["summary"]
    example_queries = analysis["example_queries"]

    return message, gr.Dataset(components=[query_component], samples=example_queries), summary

def download_ytvideo(url, memorize):

    global example_queries, summary, query_component

    analysis = analyze_ytvideo(url, memorize)
    message = analysis["message"]
    summary = analysis["summary"]
    example_queries = analysis["example_queries"]

    return message, gr.Dataset(components=[query_component], samples=example_queries), summary

def download_art(url, memorize):

    global example_queries, summary, query_component

    analysis = analyze_article(url, memorize)
    message = analysis["message"]
    summary = analysis["summary"]
    example_queries = analysis["example_queries"]

    return message, gr.Dataset(components=[query_component], samples=example_queries), summary

def download_media(url, memorize):

    global example_queries, summary, query_component

    analysis = analyze_media(url, memorize)
    message = analysis["message"]
    summary = analysis["summary"]
    example_queries = analysis["example_queries"]

    return message, gr.Dataset(components=[query_component], samples=example_queries), summary

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

theme = gr.Theme.from_hub("sudeepshouche/minimalist")

with gr.Blocks(theme=theme) as llmapp:
    gr.Markdown(
        """
        <h1><center><b>LLM Bot</center></h1>
        """
    )
    gr.Markdown(
        """
        <center>
        <br>
        This app uses the Transformer magic to answer all your questions! <br>
        Check the "Memorize" Box if you want to add the information to your memory palace! <br>
        Using the default gpt-4 model! <br>
        </center>
        """
    )
    with gr.Tab(label="LLM APP"):
        with gr.Column():
            with gr.Row():
                memorize = gr.Checkbox(label="I want this information stored in my memory palace!")
            with gr.Row():
                with gr.Tab(label="Video Analyzer"):
                    yturl = gr.Textbox(placeholder="Input must be a Youtube URL", label="Enter Youtube URL")
                    with gr.Row():
                        download_output = gr.Textbox(label="Video download Status")
                        download_button = gr.Button(value="Download", scale=0)
                with gr.Tab(label="Article Analyzer"):
                    arturl = gr.Textbox(placeholder="Input must be a URL", label="Enter Article URL")
                    with gr.Row():
                        adownload_output = gr.Textbox(label="Article download Status")
                        adownload_button = gr.Button(value="Download", scale=0)
                with gr.Tab(label="File Analyzer"):
                    files = gr.Files(label="Supported types: pdf, txt, docx, png, jpg, jpeg, mp3")
                    with gr.Row():
                        upload_output = gr.Textbox(label="Upload Status")
                        upload_button = gr.Button(value="Upload", scale=0)
                with gr.Tab(label="Media URL Analyzer"):
                    mediaurl = gr.Textbox(placeholder="Input must be a URL", label="Enter Media URL")
                    with gr.Row():
                        mdownload_output = gr.Textbox(label="Media download Status")
                        mdownload_button = gr.Button(value="Download", scale=0)
            with gr.Row():
                with gr.Column():
                    summary_output = gr.Textbox(placeholder="Summary will be generated here", label="Key takeaways")
                    chatui = gr.ChatInterface(
                        ask,
                        submit_btn="Ask",
                        retry_btn=None,
                        undo_btn=None,
                    )
                    query_component = chatui.textbox
                    examples = gr.Dataset(label="Questions", samples=example_queries, components=[query_component], type="index")
    with gr.Tab(label="AI Assistant"):
        gr.ChatInterface(
            internet_connected_chatbot,
            additional_inputs=[
                gr.Radio(label="Model", choices=["COHERE", "GEMINI", "GPT4", "GPT35TURBO", "MIXTRAL8x7B"], value="GEMINI"),
                gr.Slider(10, 1680, value=840, label = "Max Output Tokens"),
                gr.Slider(0.1, 0.9, value=0.5, label = "Temperature"),
            ],
            examples=[["Latest news summary"], ["Explain special theory of relativity"], ["Latest Chelsea FC news"], ["Latest news from India"],["What's the latest GDP per capita of India?"], ["What is the current weather in North Plains?"], ["What is the latest on room temperature superconductors?"]],
            submit_btn="Ask",
            retry_btn=None,
            undo_btn=None,
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
                examples=[["What is the meaning of life?"], ["What is the purpose of life?"], ["What is the meaning of death?"], ["What is the purpose of death?"], ["What is the meaning of existence?"], ["What is the purpose of existence?"], ["What is the meaning of the universe?"], ["What is the purpose of the universe?"], ["What is the meaning of the world?"], ["What is the purpose of the world?"]],
                submit_btn="Ask",
                retry_btn=None,
                undo_btn=None,
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

    upload_button.click(upload_file, inputs=[files, memorize], outputs=[upload_output, examples, summary_output], show_progress=True)
    download_button.click(download_ytvideo, inputs=[yturl, memorize], outputs=[download_output, examples, summary_output], show_progress=True)
    adownload_button.click(download_art, inputs=[arturl, memorize], outputs=[adownload_output, examples, summary_output], show_progress=True)
    mdownload_button.click(download_media, inputs=[mediaurl, memorize], outputs=[mdownload_output, examples, summary_output], show_progress=True)
    city_button.click(generate_trip_plan, inputs=[city_name, number_of_days], outputs=[city_output], show_progress=True)
    craving_button.click(craving_satisfier, inputs=[craving_city_name, craving_cuisine], outputs=[craving_output], show_progress=True)
    clear_trip_button.click(clearhistory, inputs=[city_name, number_of_days, city_output], outputs=[city_name, number_of_days, city_output])
    clear_craving_button.click(clearhistory, inputs=[craving_city_name, craving_cuisine, craving_output], outputs=[craving_city_name, craving_cuisine, craving_output])
    examples.click(load_example, inputs=[examples], outputs=[query_component])

if __name__ == '__main__':
    llmapp.launch(server_name='0.0.0.0', server_port=7860)
