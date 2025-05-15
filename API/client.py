import requests
import streamlit as st 
import re

# creating an froen-end interacting with an api

def get_chatgroq_response(input_text):
    response = requests.post(
        "http://localhost:8000/essay/invoke",
        json = {'input': {'topic': input_text}}
    )
    return response.json()['output']['content']

def get_deepseek_response(input_text):
    response = requests.post(
        "http://localhost:8000/poem/invoke",
        json = {'input': {'topic': input_text}}
    )
    response = response.json()['output']['content']
    response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
    return response


st.title('Langchain langserve')
input_test1 = st.text_input("Write an essay on")
input_test2 = st.text_input("Write an poem on")

if input_test1:
    st.write("Essay:\n\n",get_chatgroq_response(input_test1))
    
if input_test2:
    st.write("Poem:\n",get_deepseek_response(input_test2))
    
