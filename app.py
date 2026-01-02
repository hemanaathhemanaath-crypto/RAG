import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# ---- UI ----
st.set_page_config(page_title="BLAST", layout="wide")
st.title("💥 BLAST")
st.write("Ask me anything!")

# ---- API Key UI ----
groq_api_key = st.sidebar.text_input("Enter Groq API Key", type="password")
st.sidebar.info("Get your FREE API key at https://console.groq.com/keys")

if not groq_api_key:
    st.warning("Please enter your Groq API key to continue.")
    st.stop()

# ---- LLM ----
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model="llama-3.3-70b-versatile",
    temperature=0.7
)

# ---- Chat Input ----
user_input = st.text_input("You:")

if user_input:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant."),
        ("user", "{question}")
    ])

    chain = prompt | llm
    response = chain.invoke({"question": user_input})

    st.subheader("Answer:")
    st.write(response.content)

