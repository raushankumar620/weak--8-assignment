import os
import logging
from typing import List, Tuple
import streamlit as st

try:
    from langchain.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_community.llms import HuggingFacePipeline

    from langchain.vectorstores import FAISS
    from langchain.llms import HuggingFacePipeline
    from langchain.chains import RetrievalQA
    from langchain.schema import Document
except ImportError as e:
    st.error(f"Required packages not installed: {e}")

class RAGChatbot:
    def __init__(self, use_simple_llm=True):
        self.vector_store = None
        self.qa_chain = None
        self.is_initialized = False
        self.use_simple_llm = use_simple_llm
        self.embeddings = None
        
        # Initialize embeddings
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
        except Exception as e:
            st.error(f"Failed to initialize embeddings: {e}")
    
    def load_documents(self, folder_path: str) -> bool:
        """Load documents from folder"""
        try:
            if not os.path.exists(folder_path):
                st.error(f"Folder path does not exist: {folder_path}")
                return False
            
            # Check if folder has any files
            files = os.listdir(folder_path)
            if not files:
                st.error(f"No files found in folder: {folder_path}")
                return False
            
            documents = []
            
            # Load different file types
            for file in files:
                file_path = os.path.join(folder_path, file)
                try:
                    if file.endswith('.txt'):
                        loader = TextLoader(file_path, encoding='utf-8')
                        docs = loader.load()
                        documents.extend(docs)
                    elif file.endswith('.pdf'):
                        loader = PyPDFLoader(file_path)
                        docs = loader.load()
                        documents.extend(docs)
                    else:
                        # Try to read as text file
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            doc = Document(page_content=content, metadata={"source": file_path})
                            documents.append(doc)
                except Exception as e:
                    st.warning(f"Could not load file {file}: {e}")
                    continue
            
            if not documents:
                st.error("No documents could be loaded. Check file formats.")
                return False
            
            # Split documents
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            splits = text_splitter.split_documents(documents)
            
            # Create vector store
            if self.embeddings:
                self.vector_store = FAISS.from_documents(splits, self.embeddings)
                self._initialize_qa_chain()
                self.is_initialized = True
                st.success(f"Loaded {len(documents)} documents with {len(splits)} chunks")
                return True
            else:
                st.error("Embeddings not initialized")
                return False
                
        except Exception as e:
            st.error(f"Error loading documents: {e}")
            return False
    
    def save_vector_store(self, path: str):
        """Save vector store to disk"""
        try:
            if self.vector_store:
                self.vector_store.save_local(path)
                st.success("Vector store saved successfully!")
        except Exception as e:
            st.error(f"Error saving vector store: {e}")
    
    def load_vector_store(self, path: str):
        """Load vector store from disk"""
        try:
            if os.path.exists(path) and self.embeddings:
                self.vector_store = FAISS.load_local(path, self.embeddings)
                self._initialize_qa_chain()
                self.is_initialized = True
                return True
            else:
                st.error("Vector store path not found or embeddings not initialized")
                return False
        except Exception as e:
            st.error(f"Error loading vector store: {e}")
            return False
    
    def _initialize_qa_chain(self):
        """Initialize QA chain with simple response"""
        try:
            if self.vector_store:
                # Simple retriever
                self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
                st.success("QA chain initialized successfully!")
        except Exception as e:
            st.error(f"Error initializing QA chain: {e}")
    
    def ask_question(self, question: str) -> str:
        """Ask a question and get answer"""
        try:
            if not self.is_initialized:
                return "Please load documents first."
            
            # Get relevant documents
            relevant_docs = self.retriever.get_relevant_documents(question)
            
            if not relevant_docs:
                return "No relevant information found in the documents."
            
            # Simple response generation (without LLM)
            context = "\n".join([doc.page_content[:500] for doc in relevant_docs[:2]])
            
            response = f"""Based on the documents, here's what I found:

{context}

This information is extracted from your documents and may help answer your question: "{question}"
"""
            return response
            
        except Exception as e:
            return f"Error processing question: {e}"
    
    def get_relevant_contexts(self, question: str) -> List[Tuple[str, float]]:
        """Get relevant contexts with scores"""
        try:
            if not self.is_initialized:
                return []
            
            # Get documents with scores
            docs_with_scores = self.vector_store.similarity_search_with_score(question, k=3)
            return [(doc.page_content, score) for doc, score in docs_with_scores]
            
        except Exception as e:
            st.error(f"Error getting contexts: {e}")
            return []
