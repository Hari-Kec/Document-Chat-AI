

# 🧠 MAWAD Online RAG Chatbot

An AI-powered chatbot designed for **Mawad Online**, a Dubai-based enterprise, that answers complex questions from uploaded **PDF documents** using **hybrid retrieval** (keyword + semantic search). Built with LangChain, OpenAI, and Streamlit, it ensures accurate, conversational, and context-aware responses with self-correction and memory handling.

---

## ✨ Features

* 📄 Upload multiple PDFs and ask natural-language questions
* 🔍 Combines **BM25 keyword search** and **semantic vector retrieval** (hybrid search)
* 🧠 Maintains **conversational memory** using LangChain memory buffer
* 🔁 Built-in **self-correction** when documents are insufficient or ambiguous
* 💬 Intelligent clarifications for vague queries (e.g., "it", "that")
* 📊 Token & cost logging for usage tracking
* 🌐 Streamlit-powered interactive UI

---

## 🚀 How It Works

1. **Upload PDFs** via Streamlit UI
2. Documents are:

   * Loaded and chunked
   * Embedded and stored in a **Chroma vectorstore**
   * Indexed with **BM25** for keyword relevance
3. User queries are run through:

   * Hybrid search for relevant chunks
   * LangChain chain with memory + context + OpenAI LLM
4. Self-correction ensures hallucination-free responses
5. Token usage and response cost are logged for analysis

---

## 📦 Tech Stack

| Layer         | Tools/Frameworks                   |
| ------------- | ---------------------------------- |
| Frontend (UI) | Streamlit                          |
| LLM           | OpenAI GPT-4o-mini (`gpt-4o-mini`) |
| Embeddings    | `text-embedding-3-small`           |
| Vectorstore   | ChromaDB + BM25 hybrid             |
| Framework     | LangChain                          |
| Logging       | Python `logging`, stored by date   |

---

## 🧬 File Structure

```
mawad-chatbot/
├── chatbot.py               # Main Streamlit chatbot UI
├── chain.py                 # LangChain chain creation with memory
├── config.py                # API keys, directories, model config
├── document_processing.py   # PDF loader & chunk splitter
├── memory.py                # Memory buffer and ambiguity handler
├── vectorstore.py           # Embedding & Chroma vector store logic
├── retrieval.py             # Hybrid search logic (BM25 + semantic)
├── data/
│   ├── tmp/                 # Temp document handling
│   └── vector_stores/       # Persistent vectorstore storage
├── logs/                    # Daily logs with cost/token tracking
├── .env                     # API key and environment variables
```

---

## ⚙️ Setup Instructions

### 1. Clone the Repo

```bash
git clone https://github.com/Hari-Kec/Document-Chat-AI.git
cd Document-Chat-AI
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure `.env`

Create a `.env` file in the root directory:

```env
OPENAI_API_KEY=your_openai_key_here
```

### 4. Run the App

```bash
streamlit run chatbot.py
```

---

## 🧪 Sample Usage

* Upload one or more PDFs
* Ask: *"What is the refund policy mentioned in the document?"*
* Ask follow-up: *"What about late delivery?"*
* Bot retains context and answers smoothly

---

## 📊 Logging & Monitoring

* Logs saved daily to `/logs/YYYYMMDD.log`
* Includes:

  * Prompt + completion tokens
  * Cost breakdown
  * Model used

---

## 🛡 Self-Correction

If no document context is found, or the model is unsure, it safely replies:

> *"I don't know."*

It never hallucinates answers — a key design goal.

---

## 📖 Powered By

* [LangChain](https://www.langchain.com/)
* [OpenAI GPT-4o-mini](https://platform.openai.com/)
* [Chroma Vector DB](https://www.trychroma.com/)
* [Streamlit](https://streamlit.io/)

---

## 🔐 License

This freelance project is proprietary and developed for **Mawad Online** (Dubai). Redistribution without consent is prohibited. For enterprise chatbot inquiries, please contact the author.


