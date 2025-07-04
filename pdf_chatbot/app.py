import os
import streamlit as st
from langchain_community.document_loaders import PDFMinerLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import google.generativeai as genai

from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# ---- Setup ----
st.set_page_config(page_title="🧠 Chat with Your PDF", layout="centered")
st.title("📄 Chat with Your PDF")

# ---- API Keys ----
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
genai.configure(api_key=GOOGLE_API_KEY)

# ---- File Upload ----
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

# ---- Session Chat History Storage ----
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if uploaded_file:
    with st.spinner("Processing your PDF..."):
        os.makedirs("data", exist_ok=True)
        file_path = os.path.join("data", uploaded_file.name)

        with open(file_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        try:
            loader = PDFMinerLoader(file_path)
            documents = loader.load()
        except Exception as e:
            st.error(f"❌ Error reading PDF: {e}")
            st.stop()

        # Split into chunks
        splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(documents)

        # HuggingFace Embeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(chunks, embeddings)
        retriever = vectorstore.as_retriever()

        st.success("✅ PDF processed successfully!")

        # Memory for chat history
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        # Prompt with memory
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant. Answer the question based on the provided context."),
            ("human", "{question}"),
            ("ai", "{chat_history}")
        ])

        # Initialize Gemini
        model = genai.GenerativeModel(model_name="models/gemini-1.5-flash-latest")

        # Ask question
        question = st.text_input("Ask a question about the PDF")

        if question:
            with st.spinner("Retrieving context..."):
                docs = retriever.invoke(question)
                context = "\n\n".join([doc.page_content for doc in docs[:3]])

            # Construct full prompt with memory
            history_str = "\n".join([f"User: {q}\nBot: {a}" for q, a in st.session_state.chat_history])
            full_prompt = f"""
You are a helpful assistant. Use the following context to answer the user's question.

Context:
{context}

Chat History:
{history_str}

Current Question:
{question}
"""

            with st.spinner("Generating answer..."):
                response = model.generate_content(full_prompt)
                answer = response.text.strip()

                # Save to session chat history
                st.session_state.chat_history.append((question, answer))

                # Show result
                st.markdown("### 💬 Answer")
                st.write(answer)

        # Show full chat history
        if st.session_state.chat_history:
            st.markdown("### 🗂️ Chat History")
            for q, a in st.session_state.chat_history:
                st.markdown(f"**🧑‍💻 You:** {q}")
                st.markdown(f"**🤖 Bot:** {a}")
