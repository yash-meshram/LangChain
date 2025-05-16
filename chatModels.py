import os
from dotenv import load_dotenv

load_dotenv("../.env")

groq_api_key = os.getenv("GROQ_API_KEY")

from langchain.chat_models import init_chat_model
# init_chat_model = used to initialize chat-base language model

model = init_chat_model(
    model = "llama3-8b-8192",
    model_provider = "groq"
)

model.invoke("What is decision Tree?")