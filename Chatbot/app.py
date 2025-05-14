from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st 
import os 
from dotenv import load_dotenv

load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
# Langsmith tracking
os.environ["LANGCHAIN_TRACING_V2"] = "true"

# Prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "you are the helpfult assistant. Please response to the user queries."),
        ("user", "Question: {question}")
    ]
)

# sreamlit framework
st.title("Langchain demo with Groq API")

input_text = st.text_input(label = "Search the topic you want")

# Groq LLM
llm = ChatGroq(
    model = "llama-3.1-8b-instant",
    temperature = 0.5
)

output_parser = StrOutputParser()

chain = prompt | llm | output_parser

if input_text:
    response = chain.invoke({"question": input_text})
    st.write(response)