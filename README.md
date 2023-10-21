# LLM_QA_Bot

LLM_QA_Bot is a powerful and versatile bot that can answer questions related to various types of content. Whether you have uploaded files, articles, YouTube videos, audio files or even want to chat with AI models, LLM_QA_Bot has got you covered! 

There's also a memory palace feature now to help you remember the content you've analyzed with this app. "Memorize" checkbox feature has been added to let the user choose whether they want to save the content to their memory palace or not.

# Features

# Content Analysis
Upload your files or provide links to articles/YouTube videos, and let LLM_QA_Bot work its magic! It will analyze the content and provide you with detailed insights and information.

[LLM UI](screenshots/ui.png)

[LLM Article Analysis](screenshots/articleanalysis.png)

# Memory Palace
LLM_QA_Bot now comes with a memory palace feature. You can choose to save the analyzed content to your memory palace for easy access and reference later.

# Chat with LLMs
LLM_QA_Bot has integrated support for various AI models and services. You can chat with Azure OpenAI, perform Bing searches for the latest news updates, utilize Google palm, Cohere, and even run local AI models for local inferencing.

[LLM Chat](screenshots/palm.png)

[LLM Latest News](screenshots/news.png)

# Holy Book Chatbot
If you seek wisdom from the holy book, LLM_QA_Bot has you covered. It now includes a Pinecone database with Bhagavad Gita embeddings. You can access the database to get gyan (knowledge) from the holy book.

[LLM Gita](screenshots/gita.png)

# Random Food Cravings Generator
Feeling hungry but can't decide what to eat? LLM_QA_Bot can help! It includes a random food cravings generator that suggests delicious food options to satisfy your cravings.

[LLM Cravings generator](screenshots/cravings.png)

# Trip Planner
Planning a trip? LLM_QA_Bot is here to assist you. It now includes a trip planner feature that can help you plan your next adventure.

[LLM Trip Planner](screenshots/cityplanner.png)

# Setup
To use LLM_QA_Bot, follow these steps:

Install all the required packages mentioned in the requirements.txt file.
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
OPENWEATHER_API_KEY
```

# Instruction set up Docker container:

```bash
docker build -t llmqabot .
docker run --restart always -p 7860:7860 --name llmqabot llmqabot
```
