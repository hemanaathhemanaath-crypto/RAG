import streamlit as st
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain_groq import ChatGroq

# -------------------------------
# Streamlit UI Setup
# -------------------------------
st.set_page_config(page_title="Chatbot", layout="wide")
st.title("🤖 Chatbot")

# -------------------------------
# API Key Input UI
# -------------------------------
groq_api_key = st.sidebar.text_input("Enter Groq API Key", type="password")
st.sidebar.info("Get a FREE API key at https://console.groq.com/keys")

if not groq_api_key:
    st.warning("⚠ Please enter your Groq API key to continue.")
    st.stop()

# -------------------------------
# LLM Setup
# -------------------------------
llm = ChatGroq(
    api_key=groq_api_key,
    model="llama3-3-70b-versatile"
)

# -------------------------------
# Chat History
# -------------------------------
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(return_messages=True)

chat_chain = ConversationChain(
    llm=llm,
    memory=st.session_state.memory
)

# -------------------------------
# Chat Input Box
# -------------------------------
user_input = st.chat_input("Ask me anything...")

# -------------------------------
# Chat Handling
# -------------------------------
if user_input:
    with st.chat_message("user"):
        st.write(user_input)

    # MODEL RETURNS ONLY CLEAN TEXT
    answer = chat_chain.run(user_input)

    with st.chat_message("assistant"):
        st.write(answer)
