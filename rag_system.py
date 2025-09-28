import os
import re
import json
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from typing import List, Dict, Any

class EfficientRAGSystem:
    def __init__(self, knowledge_base_path: str = "knowledge_base"):
        """Initialize efficient RAG system with document chunking"""
        self.knowledge_base_path = knowledge_base_path
        self.documents = {}
        self.chunk_size = 300  # characters per chunk
        self.overlap = 50     # overlap between chunks
        
        # Initialize vectorizer with controlled vocabulary
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1
        )
        
        self.document_vectors = None
        self.is_fitted = False
        
        # Load and index documents
        self._load_documents()
        if self.documents:
            self._build_index()
        
        logging.info(f"Loaded {len(self.documents)} document chunks")
    
    def _load_documents(self):
        """Load and chunk documents from knowledge base"""
        if not os.path.exists(self.knowledge_base_path):
            logging.warning(f"Knowledge base path not found: {self.knowledge_base_path}")
            return
        
        for filename in os.listdir(self.knowledge_base_path):
            if filename.endswith(".md") or filename.endswith(".txt"):
                file_path = os.path.join(self.knowledge_base_path, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Extract document metadata
                    doc_id = filename.replace(".md", "").replace(".txt", "")
                    title = doc_id.replace("_", " ").title()
                    
                    # Split into chunks
                    chunks = self._split_into_chunks(content)
                    
                    # Store chunks
                    for i, chunk in enumerate(chunks):
                        chunk_id = f"{doc_id}_{i}"
                        self.documents[chunk_id] = {
                            "id": chunk_id,
                            "title": f"{title} (Part {i+1})",
                            "content": chunk,
                            "source": doc_id,
                            "chunk_index": i,
                            "total_chunks": len(chunks)
                        }
                        
                except Exception as e:
                    logging.error(f"Error loading {filename}: {e}")
    
    def _split_into_chunks(self, content: str) -> List[str]:
        """Split content into overlapping chunks"""
        if len(content) <= self.chunk_size:
            return [content.strip()]
        
        chunks = []
        start = 0
        
        while start < len(content):
            # Find a good breaking point
            end = start + self.chunk_size
            
            if end >= len(content):
                # Last chunk
                chunk = content[start:].strip()
                if chunk:
                    chunks.append(chunk)
                break
            
            # Look for sentence boundary
            sentence_end = content.rfind('.', start, end)
            if sentence_end == -1:
                sentence_end = content.rfind('!', start, end)
            if sentence_end == -1:
                sentence_end = content.rfind('?', start, end)
            
            if sentence_end == -1 or sentence_end < start + self.chunk_size * 0.7:
                # No good sentence boundary, break at word
                word_end = content.rfind(' ', start, end)
                if word_end > start + self.chunk_size * 0.5:
                    end = word_end
                else:
                    end = start + self.chunk_size
            else:
                end = sentence_end + 1
            
            chunk = content[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move to next chunk with overlap
            start = end - self.overlap
            if start <= 0:
                start = end
        
        return chunks
    
    def _build_index(self):
        """Build search index using TF-IDF"""
        try:
            # Prepare documents
            doc_contents = [doc["content"] for doc in self.documents.values()]
            doc_ids = list(self.documents.keys())
            
            # Fit vectorizer
            self.document_vectors = self.vectorizer.fit_transform(doc_contents)
            self.document_ids = doc_ids
            self.is_fitted = True
            
            logging.info(f"Built index with {len(doc_ids)} documents")
            
        except Exception as e:
            logging.error(f"Error building index: {e}")
            self.is_fitted = False
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve relevant documents for a query"""
        if not self.is_fitted or not self.documents:
            return []
        
        try:
            # Transform query
            query_vector = self.vectorizer.transform([query])
            
            # Compute similarities
            similarities = (query_vector @ self.document_vectors.T).toarray()[0]
            
            # Get top k results
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            # Format results
            results = []
            for idx in top_indices:
                if similarities[idx] > 0.01:  # Minimum relevance threshold
                    doc_id = self.document_ids[idx]
                    doc = self.documents[doc_id]
                    results.append({
                        "id": doc_id,
                        "title": doc["title"],
                        "content": doc["content"],
                        "source": doc["source"],
                        "relevance": float(similarities[idx]),
                        "chunk_index": doc["chunk_index"],
                        "total_chunks": doc["total_chunks"]
                    })
            
            return results
            
        except Exception as e:
            logging.error(f"Retrieval error: {e}")
            return []
    
    def add_document(self, content: str, title: str, source: str = "user") -> bool:
        """Add a new document to the knowledge base"""
        try:
            # Split into chunks
            chunks = self._split_into_chunks(content)
            
            # Generate unique doc_id
            doc_id = f"{source}_{int(time.time())}"
            
            # Add chunks
            for i, chunk in enumerate(chunks):
                chunk_id = f"{doc_id}_{i}"
                self.documents[chunk_id] = {
                    "id": chunk_id,
                    "title": f"{title} (Part {i+1})",
                    "content": chunk,
                    "source": source,
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                }
            
            # Rebuild index
            self._build_index()
            return True
            
        except Exception as e:
            logging.error(f"Error adding document: {e}")
            return False
    
    def get_document_stats(self) -> Dict[str, Any]:
        """Get statistics about the document collection"""
        if not self.documents:
            return {
                "total_documents": 0,
                "total_chunks": 0,
                "average_chunk_size": 0,
                "sources": []
            }
        
        sources = set(doc["source"] for doc in self.documents.values())
        total_content = sum(len(doc["content"]) for doc in self.documents.values())
        
        return {
            "total_documents": len(sources),
            "total_chunks": len(self.documents),
            "average_chunk_size": total_content / len(self.documents),
            "sources": list(sources),
            "is_indexed": self.is_fitted
        }