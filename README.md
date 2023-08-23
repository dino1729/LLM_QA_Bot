# LLM_QA_Bot
A bot that answers questions to the content uploaded!

Install all the required packages mentioned in the requirements.txt file.

Upload the files or the article/YoutTube video you want analyzed, and let LLM magic take over!

There's also a memory palace feature now to help you remember the content you've analyzed with this app. "Memorize" checkbox feature has been added to let the user choose whether they want to save the content to their memory palace or not.

# Chat with LLMs
Added support to chat with Azure OpenAI, Bing search for latest news update, Google palm, Cohere and local AI models for local inferencing

# Holy Book chatbot
Pinecone database added with the Bhagawad Gita embeddings for getting gyan from the holy book!

# Random food cravings generator
Added a random food cravings generator to help you decide what to eat when you're hungry!

# Trip Planner
Added a trip planner to help you plan your next trip!

# Setup
Create a .env file with the following variables:

```bash
AZURE_API_BASE
AZURE_CHATAPI_VERSION
AZURE_API_VERSION
AZURE_API_KEY
LLAMA2_API_BASE
PUBLIC_SUPABASE_URL
SUPABASE_SERVICE_ROLE_KEY
BING_API_KEY
BING_ENDPOINT
GOOGLE_PALM_API_KEY
COHERE_API_KEY
PINECONE_API_KEY
PINECONE_ENVIRONMENT
```

# Instruction set up Docker container:

```bash
docker build -t llmqabot .
docker run --restart always -p 7860:7860 --name llmqabot llmqabot
```
