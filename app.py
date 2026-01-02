import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter


st.set_page_config(page_title="RAG Chatbot", layout="wide")

st.title("🤖 RAG Chatbot Demo")


# Sidebar for API key
groq_api_key = st.sidebar.text_input("Enter Groq API Key", type="password")


# Load model
def load_llm():
    return ChatGroq(api_key=groq_api_key, model="llama-3.3-70b-versatile")


# File upload
uploaded_file = st.file_uploader("Upload a text file", type=["txt"])

if uploaded_file and groq_api_key:

    # Read file
    file_text = uploaded_file.read().decode("utf-8")

    # Split text
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
    chunks = splitter.split_text(file_text)

    # Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # FAISS database
    vectorstore = FAISS.from_texts(chunks, embeddings)

    llm = load_llm()

    st.success("File processed! Ask your question below:")

    query = st.text_input("Ask something:")

    if query:
        # Retrieve relevant chunks
        docs = vectorstore.similarity_search(query, k=3)

        # Build final prompt
        context = "\n\n".join([d.page_content for d in docs])
        prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"

        # Generate response
        response = llm.invoke(prompt)

        st.write("### Answer:")
        st.write(response)
else:
    st.info("Upload a file + Enter Groq API key to continue.")
