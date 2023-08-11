# LLM_QA_Bot
A bot that answers questions to the content uploaded!

Install all the required packages mentioned in the requirements.txt file.

Upload the files or the article/YoutTube video you want analyzed, and let LLM magic take over!

Create a .env file with the following variables:

```bash
AZUREOPENAIAPIKEY
AZUREOPENAIAPIENDPOINT
AZUREOPENAIAPITYPE
AZUREOPENAIAPIVERSION
AZURECHATAPIVERSION
PUBLIC_SUPABASE_URL
SUPABASE_SERVICE_ROLE_KEY
```

Instruction set up Docker container:

```bash
docker build -t llmqabot .
docker run --restart always -p 7860:7860 --name llmqabot llmqabot
```
