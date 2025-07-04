# pdf_chatbot
Chat with your PDF using free Hugging Face embeddings and Google Gemini! Upload any PDF and have multi-turn conversations with it. Powered by LangChain, FAISS, and Streamlit — no OpenAI needed.
# 📄 Chat with Your PDF using Gemini + Hugging Face + FAISS

This project lets you **interact with any PDF using natural language**.

🚀 It combines:
- **Hugging Face sentence embeddings** (free & local)
- **Google Gemini 1.5 Flash** for intelligent, fast answers
- **FAISS** for semantic document search
- **Streamlit** for a beautiful chat-based web UI
- **LangChain** to orchestrate everything (loader, splitter, memory, etc.)

---

## 🔍 Features

- 📤 Upload any PDF file
- 🧠 Uses **real semantic embeddings** (via HuggingFace Transformers)
- 🔍 Finds the most relevant chunks using **vector search with FAISS**
- 💬 Multi-turn conversational memory (remembers your chat)
- 🤖 Answers generated using **Gemini 1.5 Flash** (Google Generative AI)
- ⚡ Runs fast with no cost for embeddings

---


---

## 🧠 Tech Stack

| Component          | Tool / Library                              |
|--------------------|---------------------------------------------|
| LLM (answering)    | [Google Gemini 1.5 Flash](https://makersuite.google.com/) |
| Embeddings         | [Hugging Face Sentence Transformers](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) |
| Document Loader    | LangChain's `PDFMinerLoader`                |
| Text Splitter      | LangChain `CharacterTextSplitter`           |
| Vector DB          | [FAISS (Facebook AI Similarity Search)](https://github.com/facebookresearch/faiss) |
| Memory             | LangChain's `ConversationBufferMemory`      |
| UI Framework       | [Streamlit](https://streamlit.io)           |
| Orchestrator       | [LangChain](https://www.langchain.com)      |

---

## 📦 Installation

### 1. Clone the repo

```bash
git clone https://github.com/your-username/pdf-chatbot
cd pdf-chatbot

##2. Install dependencies


pip install -r requirements.txt

###3. Add your Gemini API Key

GOOGLE_API_KEY = "your_google_gemini_api_key_here"

###Run the App

streamlit run app.py

####file structure 
pdf-chatbot/
│
├── app.py                    # Main Streamlit app
├── requirements.txt          # Dependencies
├── .streamlit/secrets.toml  # Gemini API Key (user-provided)
├── /data                     # Uploaded PDFs
├── /venv (optional)          # Virtual environment
