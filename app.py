import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter


st.set_page_config(page_title="RAG Chatbot", layout="wide")

st.title("🤖 RAG Chatbot Demo")

# Sidebar
groq_api_key = st.sidebar.text_input("Enter Groq API Key", type="password")

# Load LLM
def load_llm():
    return ChatGroq(api_key=groq_api_key, model="llama-3.3-70b-versatile")

# File Upload
uploaded_file = st.file_uploader("Upload a text file", type=["txt"])

vectorstore = None

# If file uploaded → build vector DB
if uploaded_file:
    file_text = uploaded_file.read().decode("utf-8")
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
    chunks = splitter.split_text(file_text)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(chunks, embeddings)

    st.success("File processed! Using RAG mode.")

# Chat input (always visible)
query = st.text_input("Ask something:")

# Chat handling
if query and groq_api_key:
    llm = load_llm()

    # If RAG available → use it
    if vectorstore:
        docs = vectorstore.similarity_search(query, k=3)
        context = "\n\n".join(d.page_content for d in docs)
        prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    else:
        # No RAG → pure LLM mode
        prompt = query

    response = llm.invoke(prompt)

    st.write("### Answer:")
    st.write(response)
else:
    st.info("Enter a query + Groq API Key to continue.")
