import streamlit as st
from langchain_groq import ChatGroq

st.title("🤖 RAG Chatbot Demo (No File Needed)")

# API key input
groq_api_key = st.sidebar.text_input("Enter Groq API Key", type="password")

if not groq_api_key:
    st.warning("Please enter your Groq API key to start chatting.")
    st.stop()

# Initialize ChatGroq model
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model="llama-3.3-70b-versatile"
)

# Chat UI
user_input = st.text_input("Ask something:")

if user_input:
    try:
        response = llm.invoke(user_input)
        st.markdown("### Answer:")
        st.write(response.content)

    except Exception as e:
        st.error(f"Error: {str(e)}")
