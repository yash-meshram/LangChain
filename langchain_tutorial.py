import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain, SequentialChain

# loading api_key
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# defining LLM
llm = ChatGroq(
    model = "gemma2-9b-it",
    temperature = 0.6,
    api_key = api_key
)

# passing the prompt to LLM and getting reponse back
response = llm.invoke("I want to start a new Indian restaurant. Suggest a one fancy name, NO PERAMBLE.")
print(response.content)

# defining Prompt
name_prompt = PromptTemplate(
    input_variables = ['cuisine'],
    template = "I want to start a new {cuisine} restaurant. Suggest a one fancy name, NO PERAMBLE."
)
name_prompt.format(cuisine = "Indian")

# define chain - passing the prompt to LLM
name_chain = LLMChain(
    llm = llm,
    prompt = name_prompt
)

# running the chain
restaurant_name = name_chain.run("Indian")
print(restaurant_name)

# Will pass the response from name_chain to another chain and get the menu itesm
# This is called "SimpleSequentialChain"
# input --> name_chain --> name (output) --> menu_chain --> menu (output)

# creating menu_chain
menu_prompt = PromptTemplate(
    input_variables = ['restaurant_name'],
    template = "Suggest some menu item for {restaurant_name} restaurant. Return it in comma seperated values. NO PERAMBLE."
)

# creating menu_chain
menu_chain = LLMChain(
    llm = llm,
    prompt = menu_prompt
)

# Createing Simple Sequential Chain
chain = SimpleSequentialChain(
    chains = [name_chain, menu_chain]
)

# getting response
response = chain.run("Indian")
print(response)

# we didnt get restaurant name in the response from above. we only got menu
# for multiple output or multiple input we use SequentialChain

# name chain
name_prompt = PromptTemplate(
    input_variables = ['cuisine'],
    template = "I want to start a new {cuisine} restaurant. Suggest a one fancy name, NO PERAMBLE."
)
name_chain = LLMChain(
    llm = llm,
    prompt = name_prompt,
    output_key = "restaurant_name"
)

# menu chain
menu_prompt = PromptTemplate(
    input_variables = ['restaurant_name'],
    template = "Suggest some menu item for {restaurant_name} restaurant. Return it in comma seperated values. NO PERAMBLE."
)
menu_chain = LLMChain(
    llm = llm,
    prompt = menu_prompt,
    output_key = "menu_items"
)

# Creating SequentialChain
chain = SequentialChain(
    chains = [name_chain, menu_chain],
    input_variables = ["cuisine"],
    output_variables = ["restaurant_name", "menu_items"]
)

# running the chain
response = chain(["Indian"])
response
response = chain({"cuisine": "Indian"})
response


# LLM contains 2 things:
# 1. Knowledge (which can be limited to let say 2021)
# 2. Reasoning 

# If you ask ChatGPT - what is the best chepest flight from Chennai to Delhi, today?
# He will not know it - knowledge limited to 2021
# we will give pulgin and then he will use Reasoning and search in pulgin, and give you a answer
# LLM = Reasoning + Knowledge

# ************
# * Agents
# ************
from langchain.agents import AgentType, initialize_agent, load_tools

# defining tool
tools = load_tools(
    ['wikipedia', 'llm-math'],
    llm = llm
)

# initializing agent
agent = initialize_agent(
    tools,
    llm,
    agent = AgentType.ZERO_SHOT_REACT_DESCRIPTION
)

# running the agent
agent.invoke("When was Elon Musk born? What is hig current age?")
