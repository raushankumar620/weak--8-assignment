# RAG Q&A Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that can answer questions based on your documents using document retrieval and generative AI.

## Features

- 📄 Process PDF and TXT documents
- 🔍 Vector-based document retrieval using FAISS
- 🤖 AI-powered response generation using Hugging Face models
- 💬 Interactive web interface with Streamlit
- 💾 Persistent vector store for quick loading
- 🎯 Context-aware responses

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create documents folder:
```bash
mkdir documents
```

3. Add your PDF or TXT files to the `documents` folder

## Usage

### Web Interface (Recommended)
```bash
streamlit run streamlit_app.py
```

### Console Interface
```bash
python main.py
```

## How it Works

1. **Document Processing**: Extracts text from PDFs/TXT files and splits into chunks
2. **Vector Embeddings**: Creates embeddings using SentenceTransformers
3. **Vector Store**: Uses FAISS for efficient similarity search
4. **Retrieval**: Finds relevant document chunks for user questions
5. **Generation**: Uses Hugging Face models to generate contextual responses

## Project Structure

