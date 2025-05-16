from dotenv import load_dotenv
import os

load_dotenv('../.env')


os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
# Langsmith tracking
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "default"
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

from langchain.chat_models import init_chat_model

model = init_chat_model(
    model = "gemini-2.0-flash",
    model_provider = "google_genai"
)

response = model.invoke("WHat is transformers in AI")
print(response.content)


from langchain_core.prompts import PromptTemplate
prompt = PromptTemplate.from_template("What is {topic}")
chain = prompt | model
response = chain.invoke(
    input = {'topic': "LLM"}
)
print(response.content)


from langchain_core.messages import HumanMessage, SystemMessage
messages = [
    SystemMessage("Explain the followig topic"),
    HumanMessage("Attention in LLM")
]

response = model.invoke(messages)
print(response.content)

response = model.invoke("decision tree")
print(response.content)

response = model.invoke(
    [
        {
            "role": "user",
            "content": "Linear regression"
        }
    ]
)
print(response.content)

response = model.invoke([HumanMessage("Adam optimizer")])
print(response.content)

# printing token
messages = [
    SystemMessage("Explain the followig topic in 10 words"),
    HumanMessage("Machine Learning")
]
for token in model.stream(messages):
    print(token.content, end = "|")

for token in model.stream(messages):
    print(token.content, end = "\n")
    


# ****************
# Prompt Template
# ****************
from langchain_core.prompts import ChatPromptTemplate

system_message = "Explain the following concept in {_number} words"

prompt_template = ChatPromptTemplate.from_messages([
    ('system', system_message),
    ('user', '{topic}')
])

prompt = prompt_template.invoke({
    '_number': 30,
    'topic': 'Embedding'
})
print(prompt)
print(prompt.to_messages())

response = model.invoke(prompt)
print(response.content)

chain = prompt_template | model
response = chain.invoke(
    input = {
        '_number': 30, 
        'topic': 'embedding'
    }
)
print(response.content)