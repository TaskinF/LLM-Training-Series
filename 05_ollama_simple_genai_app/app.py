import os
from dotenv import load_dotenv

from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st

load_dotenv()

# Langsmith Tracking
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_PROJECT"]=os.getenv("LANGCHAIN_PROJECT")

# Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a senior ai engineer. You are expert at the GenAI and LLM. Please respond to the question asked. Please give brief answers "),
        ("user", "Question:{question}")
    ]
)

# Streamlit Framework
st.title("Langchain Demo With Gemma 3")
input_text = st.text_input("Ask the question on your mind :)")

# Ollama Gemma 3 model
llm = Ollama(model="gemma3")
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

if input_text:
    st.write(chain.invoke({"question": input_text}))

