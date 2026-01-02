import streamlit as st
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")

# -------------------------------
# 1. SIDEBAR – API KEY INPUT
# -------------------------------
st.sidebar.header("🔐 Enter Groq API Key")
groq_api_key = st.sidebar.text_input("Groq API Key", type="password")

if not groq_api_key:
    st.warning("⚠️ Please enter your Groq API Key to continue.")
    st.stop()

# -------------------------------
# 2. LLM (works even without RAG)
# -------------------------------
llm = ChatGroq(
    api_key=groq_api_key,
    model="llama-3.1-8b-instant"
)

st.title("🤖 RAG Chatbot Demo")
st.write("Ask me anything — upload a document *optional*.")

# -------------------------------
# 3. Optional File Upload
# -------------------------------
uploaded_file = st.file_uploader("Upload a text file (optional)", type=["txt"])

vectorstore = None

if uploaded_file:
    text = uploaded_file.read().decode("utf-8")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(chunks, embeddings)

# -------------------------------
# 4. Chat Input
# -------------------------------
query = st.text_input("Ask a question:")

if query:
    if vectorstore:
        # --------- RAG MODE ---------
        retriever = vectorstore.as_retriever()
        chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever)

        response = chain({"question": query, "chat_history": []})
        st.write("📄 **Answer (RAG):**")
        st.write(response["answer"])
    else:
        # --------- NORMAL LLM CHAT ---------
        response = llm.invoke(query)
        st.write("💬 **Answer:**")
        st.write(response.content)


