import streamlit as st
from langchain_helper import Chain

st.title("Restaurent Name and Menu Generator")

cuisine = st.sidebar.selectbox(
    "Pick a Cuisine",
    (None, "Indian", "Italian", "Japanese", "Mexicon", "Chinese", "Arabic", "American")
)

chain = Chain()

if cuisine:
    response = chain.generate_restaurant_name_and_menu(cuisine)

    st.header(response['restaurant_name'].strip())
    
    menu_items = response['menu_items'].strip().split(",")
    st.write("**Menu Items:**")
    for item in menu_items:
        st.write("-", item)
