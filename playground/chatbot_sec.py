from dotenv import load_dotenv
import os

load_dotenv()

# Get secret keys from environment variables
openai_api_key = os.getenv('OPENAI_API_KEY')
openai_api_base = os.getenv('OPENAI_BASE_URL')

# Check if the secret keys are retrieved successfully
if openai_api_key is None:
    raise ValueError("OPENAI_API_KEY not found in environment variables. Please set it or add it to a .env file.")

if openai_api_base is None:
    # If OPENAI_API_BASE is not set, use the default value.
    openai_api_base = "https://api.openai.com/v1"
    print("OPENAI_API_BASE not found in environment variables, using default: https://api.openai.com/v1")
else:
    print(f"Using OPENAI_API_BASE: {openai_api_base}")

# Set environment variables (optional if already loaded, but good practice for consistency)
os.environ["OPENAI_API_KEY"] = openai_api_key
os.environ["OPENAI_API_BASE"] = openai_api_base

# The rest of your code using openai_api_key and openai_api_base...
import asyncio # Import asyncio

from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

# global defaults
Settings.llm = OpenAI(model="o4-mini", api_base=openai_api_base)
Settings.embed_model = OpenAIEmbedding(model_name="text-embedding-3-large", api_base=openai_api_base)
Settings.chunk_size = 512
Settings.chunk_overlap = 64

"""### Ingest Data

Let's first download the raw 10-k files, from 2019-2022.
"""

import os
import requests
import zipfile
import io

# Define paths
base_data_dir = "data"
target_data_dir = os.path.join(base_data_dir, "UBER")
zip_url = "https://www.dropbox.com/s/948jr9cfs7fgj99/UBER.zip?dl=1"
zip_path = os.path.join(base_data_dir, "UBER.zip")

# Check if the target data directory already exists
if os.path.exists(target_data_dir):
    print(f"Target data directory '{target_data_dir}' already exists. Skipping download and extraction.")
else:
    print(f"Target data directory '{target_data_dir}' not found. Proceeding with download and extraction.")
    # Create base data directory if it doesn't exist
    os.makedirs(base_data_dir, exist_ok=True)
    print(f"Directory '{base_data_dir}' created or already exists.")

    # Download the zip file
    # Check if the zip file already exists to avoid re-downloading (optional, but good practice)
    if not os.path.exists(zip_path):
        print(f"Downloading {zip_url} to {zip_path}...")
        try:
            response = requests.get(zip_url, stream=True)
            response.raise_for_status()  # Raise an exception for bad status codes
            with open(zip_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("Download complete.")
        except requests.exceptions.RequestException as e:
            print(f"Error downloading file: {e}")
            # Handle error appropriately, maybe exit or raise exception
            exit() # Or raise e
    else:
        print(f"Zip file '{zip_path}' already exists. Proceeding to extraction.")


    # Unzip the file
    unzip_dir = base_data_dir # Extract directly into the base data dir
    print(f"Unzipping {zip_path} to {unzip_dir}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(unzip_dir)
        print("Unzipping complete.")
        # Optional: Clean up the zip file after extraction
        # os.remove(zip_path)
        # print(f"Removed {zip_path}.")
    except zipfile.BadZipFile:
        print(f"Error: {zip_path} is not a valid zip file or is corrupted.")
        # Handle error appropriately
        exit() # Or raise exception
    except Exception as e:
        print(f"An error occurred during unzipping: {e}")
        # Handle error appropriately
        exit() # Or raise e

"""To parse the HTML files into formatted text, we use the [Unstructured](https://github.com/Unstructured-IO/unstructured) library. Thanks to [LlamaHub](https://llamahub.ai/), we can directly integrate with Unstructured, allowing conversion of any text into a Document format that LlamaIndex can ingest.

First we install the necessary packages:

Then we can use the `UnstructuredReader` to parse the HTML files into a list of `Document` objects.
"""

from llama_index.readers.file import UnstructuredReader
from pathlib import Path

years = [2022, 2021, 2020, 2019]

loader = UnstructuredReader()
doc_set = {}
all_docs = []
for year in years:
    year_docs = loader.load_data(
        file=Path(f"./data/UBER/UBER_{year}.html"), split_documents=False
    )
    # insert year metadata into each year
    for d in year_docs:
        d.metadata = {"year": year}
    doc_set[year] = year_docs
    all_docs.extend(year_docs)

"""### Setting up Vector Indices for each year

We first setup a vector index for each year. Each vector index allows us
to ask questions about the 10-K filing of a given year.

We build each index and save it to disk if it doesn't exist, otherwise we load it.
"""

# initialize simple vector indices
# NOTE: don't run this cell if the indices are already loaded!
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
import os # Ensure os is imported if not already

index_set = {}
for year in years:
    persist_dir = f"./storage/{year}"
    if os.path.exists(persist_dir):
        # Load the index if it already exists
        print(f"Loading index for {year} from {persist_dir}...")
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        cur_index = load_index_from_storage(storage_context)
        print(f"Index for {year} loaded.")
    else:
        # Create and persist the index if it doesn't exist
        print(f"Creating index for {year} and persisting to {persist_dir}...")
        storage_context = StorageContext.from_defaults()
        cur_index = VectorStoreIndex.from_documents(
            doc_set[year],
            storage_context=storage_context,
        )
        storage_context.persist(persist_dir=persist_dir)
        print(f"Index for {year} created and persisted.")
    index_set[year] = cur_index

"""### Setting up a Sub Question Query Engine to Synthesize Answers Across 10-K Filings

Since we have access to documents of 4 years, we may not only want to ask questions regarding the 10-K document of a given year, but ask questions that require analysis over all 10-K filings.

To address this, we can use a [Sub Question Query Engine](https://gpt-index.readthedocs.io/en/stable/examples/query_engine/sub_question_query_engine.html). It decomposes a query into subqueries, each answered by an individual vector index, and synthesizes the results to answer the overall query.

LlamaIndex provides some wrappers around indices (and query engines) so that they can be used by query engines and agents. First we define a `QueryEngineTool` for each vector index.
Each tool has a name and a description; these are what the LLM agent sees to decide which tool to choose.
"""

from llama_index.core.tools import QueryEngineTool

individual_query_engine_tools = [
    QueryEngineTool.from_defaults(
        query_engine=index_set[year].as_query_engine(),
        name=f"vector_index_{year}",
        description=(
            "useful for when you want to answer queries about the"
            f" {year} SEC 10-K for Uber"
        ),
    )
    for year in years
]

"""Now we can create the Sub Question Query Engine, which will allow us to synthesize answers across the 10-K filings. We pass in the `individual_query_engine_tools` we defined above."""

from llama_index.core.query_engine import SubQuestionQueryEngine

query_engine = SubQuestionQueryEngine.from_defaults(
    query_engine_tools=individual_query_engine_tools,
)

"""### Setting up the Chatbot Agent

We use a LlamaIndex Data Agent to setup the outer chatbot agent, which has access to a set of Tools. Specifically, we will use an OpenAIAgent, that takes advantage of OpenAI API function calling. We want to use the separate Tools we defined previously for each index (corresponding to a given year), as well as a tool for the sub question query engine we defined above.

First we define a `QueryEngineTool` for the sub question query engine:
"""

query_engine_tool = QueryEngineTool.from_defaults(
    query_engine=query_engine,
    name="sub_question_query_engine",
    description=(
        "useful for when you want to answer queries that require analyzing"
        " multiple SEC 10-K documents for Uber"
    ),
)

"""Then, we combine the Tools we defined above into a single list of tools for the agent:"""

tools = individual_query_engine_tools + [query_engine_tool]

"""Finally, we call `FunctionAgent` to create the agent, passing in the list of tools we defined above."""

from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI

agent = FunctionAgent(tools=tools, llm=OpenAI(model="gpt-4o"))
async def main():
    """### Testing the Agent

    We can now test the agent with various queries.

    If we test it with a simple "hello" query, the agent does not use any Tools.
    """

    from llama_index.core.workflow import Context
    from llama_index.core.agent.workflow import FunctionAgent
    from llama_index.llms.openai import OpenAI

    # Define tools within the async function scope or ensure they are accessible
    # Assuming 'tools' is defined globally or passed as an argument
    agent = FunctionAgent(tools=tools, llm=OpenAI(model="gpt-4o"))


    # Setup the context for this specific interaction
    ctx = Context(agent)

    print("Testing agent with 'hi, i am bob'...")
    response = await agent.run("hi, i am bob", ctx=ctx)
    print(str(response))

    """If we test it with a query regarding the 10-k of a given year, the agent will use
    the relevant vector index Tool.
    """
    print("\nTesting agent with 'What were some of the biggest risk factors in 2020 for Uber?'...")
    response = await agent.run(
        "What were some of the biggest risk factors in 2020 for Uber?", ctx=ctx
    )
    print(str(response))

    """Finally, if we test it with a query to compare/contrast risk factors across years, the agent will use the Sub Question Query Engine Tool."""

    cross_query_str = (
        "Compare/contrast the risk factors described in the Uber 10-K across"
        " years. Give answer in bullet points."
    )
    print(f"\nTesting agent with '{cross_query_str}'...")
    response = await agent.run(cross_query_str, ctx=ctx)
    print(str(response))

    """### Setting up the Chatbot Loop

    Now that we have the chatbot setup, it only takes a few more steps to setup a basic interactive loop to chat with our SEC-augmented chatbot!
    """
    print("\nStarting interactive chatbot loop (type 'exit' to quit)...")
    # Re-initialize agent and context for the loop if needed, or reuse the existing ones
    # agent = FunctionAgent(tools=tools, llm=OpenAI(model="gpt-4o")) # Already initialized above
    # ctx = Context(agent) # Already initialized above

    while True:
        try:
            text_input = input("User: ")
            if text_input.lower() == "exit":
                break
            response = await agent.run(text_input, ctx=ctx)
            print(f"Agent: {response}")
        except EOFError: # Handle case where input stream ends unexpectedly
            break
        except KeyboardInterrupt: # Allow graceful exit with Ctrl+C
            print("\nExiting...")
            break

# Run the async main function
if __name__ == "__main__":
    # Ensure all necessary setup (like loading data, creating indices, defining tools)
    # happens before calling main() or is handled within main().
    # The current structure assumes 'tools' is defined in the global scope before main() is called.
    asyncio.run(main())


# Example usage after running the script:
# User: What were some of the legal proceedings against Uber in 2022?
# User: What were some of the legal proceedings against Uber in 2022?
