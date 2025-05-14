from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langserve import add_routes    #ex.: one rout for interacting with openAi and another rout for interactive with ollama
import uvicorn
from dotenv import load_dotenv
import os

load_dotenv("../.env")

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
# Langsmith tracking
os.environ["LANGCHAIN_TRACING_V2"] = "true"

app = FastAPI(
    title = "Langchain Server",
    version = "1.0",
    description = "A simple API server"
)

# adding routes
add_routes(
    app,
    ChatGroq(),
    path = "/openai"
)

# Groq LLM
llm1 = ChatGroq(
    model = "llama-3.1-8b-instant",
    temperature = 0.5
)

llm2 = ChatGroq(
    model = "llama-3.1-8b-instant",
    temperature = 0.5
)

# prompt
prompt = ChatPromptTemplate