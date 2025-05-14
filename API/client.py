from langchain.llms import HuggingFaceHub

llm = HuggingFaceHub(repo_id="meta-llama/Meta-Llama-3-8B-Instruct", model_kwargs={"temperature": 0.8})
output = llm("Write a short story about a robot who wants to be a painter.")
print(output)

from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os
load_dotenv("../.env")
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")

client = InferenceClient(model="meta-llama/Meta-Llama-3-8B-Instruct")
response = client.text_generation("Write a short story about a robot who wants to be a painter.", 
                                  max_new_tokens=200, 
                                  temperature=0.8)
print(response)
