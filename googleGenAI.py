from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
import os
from dotenv import load_dotenv

load_dotenv('.env')

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# Simple text invocation
result = llm.invoke("Sing a ballad of LangChain.")
print(result.content)

# Multimodal invocation with gemini-pro-vision
message = HumanMessage(
    content=[
        {
            "type": "text",
            "text": "What's in this image?",
        },
        {
            "type": "image_url", 
            "image_url": "https://imgs.search.brave.com/st1pASm_ZpLkk_1MFyUaj0Cgj_pubOfpvvTZFFL3LXM/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly9wbTEu/YW1pbm9hcHBzLmNv/bS82NDYzLzQxZDcz/Yjg1ZTgzYzE4NGM4/OTAyYTYxZWQ5OTUy/ZDgzZDQ2M2U1YjFf/aHEuanBn"
        },
    ]
)
result = llm.invoke([message])
print(result.content)