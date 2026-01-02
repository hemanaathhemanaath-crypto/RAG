import streamlit as st
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")

# -------------------------------
# SIDEBAR → API KEY
# -------------------------------
st.sidebar.header("🔐 Enter Groq API Key")
groq_api_key = st.sidebar.text_input("Groq API Key", type="password")

if not groq_api_key:
    st.warning("⚠️ Enter your Groq API key to continue.")
    st.stop()

# -------------------------------
# LLM
# -------------------------------
llm = ChatGroq(
    api_key=groq_api_key,
    model="llama-3.1-8b-instant"
)

st.title("🤖 RAG Chatbot Demo (Upload Optional)")
st.write("Ask anything. Uploading a file enables RAG mode.")

# -------------------------------
# FILE UPLOAD (optional)
# -------------------------------
uploaded_file = st.file_uploader("Upload a .txt file (optional)", type=["txt"])
vectorstore = None

if uploaded_file:
    raw_text = uploaded_file.read().decode("utf-8")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(raw_text)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)

# -------------------------------
# CHAT INPUT
# -------------------------------
user_query = st.text_input("Ask a question:")

if user_query:
    if vectorstore:
        # RAG MODE
        retriever = vectorstore.as_retriever()
        docs = retriever.get_relevant_documents(user_query)

        context_text = "\n\n".join([doc.page_content for doc in docs])
        prompt = f"""
You are an AI assistant. Use the CONTEXT below only if it is relevant.

CONTEXT:
{context_text}

QUESTION:
{user_query}

Answer clearly:
"""

        response = llm.invoke(prompt)
        st.write("📄 **Answer (RAG MODE):**")
        st.write(response.content)

    else:
        # NORMAL LLM CHAT
        response = llm.invoke(user_query)
        st.write("💬 **Answer:**")
        st.write(response.content)
