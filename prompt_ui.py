from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
import streamlit as st
from langchain_core.prompts import PromptTemplate,load_prompt


load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
model = ChatOpenAI(temperature=0, openai_api_key=openai_api_key)
st.header("LangChain with Streamlit")
st.header("Reasearch Tool")

# user_input = st.text_input("Enter your query here:")

paper_input = st.selectbox( "Select Research Paper Name", ["Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"] )

style_input = st.selectbox( "Select Explanation Style", ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"] ) 

length_input = st.selectbox( "Select Explanation Length", ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"] )
user_input = f"Explain the research paper '{paper_input}' in a {style_input} style with a {length_input}."

template = load_prompt('template.json')



if st.button('Summarize'):
    chain = template | model
    result = chain.invoke({
        'paper_input':paper_input,
        'style_input':style_input,
        'length_input':length_input
    })
    st.write(result.content)