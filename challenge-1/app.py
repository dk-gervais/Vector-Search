import streamlit as st
from langchain_community.llms import OpenAI

# Import dotenv, a module that provides a way to read environment variable files,
# and load the dotenv (.env) file that provides a few variables we need
from dotenv import load_dotenv
load_dotenv(override=True)

st.title("🦜🔗 Quickstart App")

def generate_response(input_text):
    llm = OpenAI(temperature=0.7)
    st.info(llm(input_text))


with st.form("my_form"):
    text = st.text_area(
        "Enter text:",
        "How is beyond meat doing as a company?",
    )
    submitted = st.form_submit_button("Submit")
    if submitted:
        generate_response(text)