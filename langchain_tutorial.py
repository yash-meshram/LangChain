import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain, SequentialChain

# loading api_key
load_dotenv(dotenv_path = "./.env")
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
# He will not know it - knowledge limited to 2021 (or so)
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
    agent = AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose = True
)

# running the agent
agent.invoke("When was Elon Musk born? What is his current age?")


# using SerpApi as tool now
# serpapi = google search api
tools = load_tools(
    tool_names = ['serpapi', 'llm-math'],
    llm = llm
)
agent = initialize_agent(
    tools = tools,
    llm = llm,
    agent = AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose = True
)
agent.invoke("What was the GDP of India in 2022 plus 5?")



# *************************
#* Memory
# *************************
# by default LLMs are state-less (non memory)
chain = LLMChain(llm = llm, prompt = name_prompt)

chain.run("Indian").strip()

chain.run("Mexicon").strip()

type(chain.memory)

# I want to remember the chat
# will attached the memory
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory()

chain = LLMChain(
    llm = llm,
    prompt = name_prompt,
    memory = memory             # have to add the memory in the chain
)

chain.run("Indian").strip()
chain.run("Mexicon").strip()

chain.memory
print(chain.memory.buffer)

# i want to create a converstaion chain and not use LLMChain
# by default we have a convertation chain
from langchain.chains import ConversationChain

convo = ConversationChain(
    llm = llm
)

print(convo.prompt)
print(convo.prompt.template)

convo.run("Who won the first cricket world cup?")
convo.run("what is 5+78?")
convo.run("In which country the match had been played?")

convo.memory
print(convo.memory.buffer)

# Disadvantage = 
# it will keep on accumulating
# when call LLM  - all log will go there - and LLM charge per token (1 word = 1 token approx.)
# this increase the cost
# to handle that we can set how many past messages it should use
from langchain.memory import ConversationBufferWindowMemory

# here we set k = 2 --> LLM will remember last 2 conversation only.
memory = ConversationBufferWindowMemory(k = 2)

convo = ConversationChain(
    llm = llm,
    memory = memory                 # explicitly adding memory here - previously we had not added it (by default it is there). Now we had given our memory.
)

convo.run("Who won the first cricket world cup?")
convo.run("what is 5+78?")
print(convo.memory.buffer)
convo.run("what is sin(45) + tan(45)?")
print(convo.memory.buffer)
convo.run("In which country the match had been played?")
print(convo.memory.buffer)
