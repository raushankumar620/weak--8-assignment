import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from typing import List, Tuple
import os

class VectorStore:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.embedding_model = SentenceTransformer(model_name)
        self.dimension = self.embedding_model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
        self.documents = []
        self.embeddings = []
    
    def add_documents(self, documents: List[str]):
        """Add documents to the vector store"""
        # Generate embeddings
        embeddings = self.embedding_model.encode(documents, normalize_embeddings=True)
        
        # Add to FAISS index
        self.index.add(embeddings.astype(np.float32))
        
        # Store documents and embeddings
        self.documents.extend(documents)
        self.embeddings.extend(embeddings)
    
    def search(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        """Search for similar documents"""
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query], normalize_embeddings=True)
        
        # Search in FAISS index
        scores, indices = self.index.search(query_embedding.astype(np.float32), k)
        
        # Return documents with scores
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.documents):
                results.append((self.documents[idx], float(score)))
        
        return results
    
    def save(self, filepath: str):
        """Save vector store to disk"""
        data = {
            'documents': self.documents,
            'embeddings': self.embeddings
        }
        
        # Save FAISS index
        faiss.write_index(self.index, f"{filepath}.faiss")
        
        # Save metadata
        with open(f"{filepath}.pkl", 'wb') as f:
            pickle.dump(data, f)
    
    def load(self, filepath: str):
        """Load vector store from disk"""
        # Load FAISS index
        self.index = faiss.read_index(f"{filepath}.faiss")
        
        # Load metadata
        with open(f"{filepath}.pkl", 'rb') as f:
            data = pickle.load(f)
            self.documents = data['documents']
            self.embeddings = data['embeddings']
