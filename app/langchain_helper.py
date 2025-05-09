from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
from langchain.chains import SequentialChain

class Chain:
    def __init__(self):
        load_dotenv(dotenv_path = "../.env")
        self.llm = ChatGroq(
            model = "gemma2-9b-it",
            temperature = 0.4,
            api_key = os.getenv("GROQ_API_KEY")
        )

    def generate_restaurant_name_and_menu(self, cuisine):
        # name chain
        name_prompt = PromptTemplate(
            input_variables = ['cuisine'],
            template = "I want to start a new {cuisine} restaurant. Suggest a one fancy name, NO PERAMBLE."
        )
        name_chain = LLMChain(
            llm = self.llm,
            prompt = name_prompt,
            output_key = "restaurant_name"
        )

        # menu chain
        menu_prompt = PromptTemplate(
            input_variables = ['restaurant_name'],
            template = "Suggest some menu item for {restaurant_name} restaurant. Return it in comma seperated values. NO PERAMBLE."
        )
        menu_chain = LLMChain(
            llm = self.llm,
            prompt = menu_prompt,
            output_key = "menu_items"
        )

        # Creating SequentialChain
        chain = SequentialChain(
            chains = [name_chain, menu_chain],
            input_variables = ["cuisine"],
            output_variables = ["restaurant_name", "menu_items"]
        )
        
        return chain({"cuisine": cuisine})