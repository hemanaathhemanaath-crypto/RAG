import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough


st.set_page_config(page_title="Free RAG Chatbot", layout="wide")

st.title("🤖 100% FREE RAG Chatbot")
st.write("Ask anything about AI, Generative AI, or your documents — completely free!")

# Sidebar Groq API key
groq_api_key = st.sidebar.text_input("Enter Groq API Key", type="password")

if not groq_api_key:
    st.warning("Please enter your FREE Groq API key.")
    st.stop()

uploaded_file = st.file_uploader("Upload a document (txt)", type=["txt"])

if uploaded_file:
    # Save uploaded file
    with open("data.txt", "wb") as f:
        f.write(uploaded_file.read())

    loader = TextLoader("data.txt")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    # FREE local embeddings – no paid API needed
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever()

    # FREE Groq LLM
    llm = ChatGroq(
        api_key=groq_api_key,
        model="llama-3.1-8b-instant"
    )

    prompt = ChatPromptTemplate.from_template("""
You are a friendly AI RAG chatbot.

Guidelines:
- Greet when user says hi/hello.
- Explain AI & Generative AI clearly.
- Use retrieved document context.
- Say goodbye kindly when user says bye.

Context:
{context}

Question:
{question}
""")

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
    )

    query = st.text_input("Ask a question")

    if query:
        response = chain.invoke(query)
        st.write(response.content)

        if "bye" in query.lower():
            st.success("Goodbye! See you soon 😊")
