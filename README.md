# RAG Pipeline Project

Welcome to the **Retrieval-Augmented Generation (RAG) Pipeline Project**!  
This project implements a customizable RAG system that integrates document retrieval with language model inference to provide context-based answers from PDF documents for fast and accurate results.

## 🎥 Demo Video

👉 [Click here to watch the demo on YouTube]([https://www.youtube.com/watch?v=NHYiGuLC9Bc](https://youtu.be/MtnL3I1NFtw))

## 🔍 Overview

This project uses the **LangChain** framework to build a flexible RAG pipeline with the following capabilities:

- Download and process PDF documents (e.g., from arXiv).
- Chunk documents into smaller segments.
- Embed and store text chunks in a vector database (**Chroma** or **FAISS**).
- Query the stored data using a **local LLM** (e.g., LLaMA-3.2-3B-Instruct) or **Groq API**.
- Support for **Vietnamese**, **English**, and other languages.
- Command-Line Interface (CLI) for easy interaction.

## ✨ Features

- **Document Processing**: Automatically download, chunk, and prepare PDFs.
- **Vector Storage**: Persistent vector DB support with Chroma and FAISS.
- **Model Flexibility**: Use Hugging Face models locally or call Groq APIs.
- **Multilingual Support**: Works with Vietnamese, English, etc.
- **Customizable CLI**: Interactive mode with configurable parameters.
- **Scalable & Modular**: Easy to expand, tweak, and deploy.

## 📁 Project Structure

```
project_root/
├── src/
│   ├── model/               # Language model logic (local & API)
│   ├── rag/                 # Core RAG components
|   ├── run_rag_new_data.py  # CLI: new data
|   |__ run_rag_old_db.py    # CLI: existing vector DB
│ 
├── data_source/             # Raw PDF storage or your PDF file
├── vector_db/               # Stored vector databases (Chroma/FAISS)
├── requirements.txt
├── venv
└── README.md
```

## ⚙️ Installation

### Prerequisites

- Python 3.9+
- Git
- (Optional) GPU for local model inference

### Setup

```bash
git clone https://github.com/NguyenTienAnh8365/Chatbot-AI-with-RAG.git
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Environment Variables

```
HUGGINGFACE_TOKEN=your_hf_token_here
GROQ_API_KEY=your_groq_api_key_here
DATA_DIR=./data_source/generative_ai
PERSIST_PATH=./vector_db
VECTOR_DB_TYPE=Chroma
```

## 🚀 Usage

### Run with New Data

```bash
python -m src.run_rag_new_data
```

- Choose model (1: API call model, 2: Local)
- Ask questions: "..."
- Type `exit` to quit

### Run with Existing Vector DB

```bash
python -m src.run_rag_old_db
```

- Choose model (1: API call model, 2: Local)
- Ask questions: "..."
- Type `exit` to quit

## 🧪 Example Queries

- 🇻🇳 *"Tóm tắt khái niệm Attention trong bài 'Attention Is All You Need'."*  
  → Summarizes the attention mechanism in Transformer.

- 🇺🇸 *"What is the main contribution of the Attention Is All You Need paper?"*  
  → Describes the Transformer architecture.

- 🤖 *"Bạn khỏe không?"*  
  → Friendly reply: "Tôi khỏe, cảm ơn bạn! 😊"

## 🔧 Configuration

- **Model temperature**: adjust in `get_hf_llm` or `get_groq_llm`
- **Retriever params**: modify `search_kwargs` (e.g., `{"k": 5}`)

## 🛠️ Future Improvements

- ✅ Improve RAG system
- ✅ API Deployment
- ✅ Unit testing for core modules
- ✅ Optimize for large datasets (GPU FAISS or distributed vector DBs)

## 🔎 Notes

- Local models require GPU for good performance
- Test with multilingual queries for robustness

## 👤 Author

- [Nguyen Tien Anh]
- [anhnguyentien8365@gmail.com]
