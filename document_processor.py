import PyPDF2
import os
from typing import List
import re

class DocumentProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file"""
        text = ""
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text()
        return text
    
    def extract_text_from_txt(self, txt_path: str) -> str:
        """Extract text from TXT file"""
        with open(txt_path, 'r', encoding='utf-8') as file:
            return file.read()
    
    def clean_text(self, text: str) -> str:
        """Clean and preprocess text"""
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        return text
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks with overlap"""
        text = self.clean_text(text)
        chunks = []
        
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunk = text[i:i + self.chunk_size]
            if chunk.strip():
                chunks.append(chunk.strip())
        
        return chunks
    
    def process_document(self, file_path: str) -> List[str]:
        """Process document and return chunks"""
        if file_path.endswith('.pdf'):
            text = self.extract_text_from_pdf(file_path)
        elif file_path.endswith('.txt'):
            text = self.extract_text_from_txt(file_path)
        else:
            raise ValueError("Unsupported file format. Use PDF or TXT files.")
        
        return self.chunk_text(text)
