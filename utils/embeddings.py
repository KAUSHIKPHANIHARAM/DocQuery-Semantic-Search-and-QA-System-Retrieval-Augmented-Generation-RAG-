"""
Embeddings Module

Handles generation of semantic embeddings using sentence-transformers.
Uses Streamlit caching to avoid reloading models.
"""

from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
import streamlit as st


# Model constants
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384  # Output dimension for all-MiniLM-L6-v2


@st.cache_resource
def load_embedding_model() -> SentenceTransformer:
    """
    Load the sentence-transformers embedding model.
    
    Uses Streamlit @cache_resource to avoid reloading on each run.
    First initialization takes ~30-60s (model download ~80MB).
    Subsequent calls are instant (< 100ms).
    
    Model: all-MiniLM-L6-v2
    - Dimension: 384
    - Size: ~80MB
    - Speed: ~6000 embeddings/second on CPU
    - Quality: Good balance between speed and quality for RAG tasks
    
    Returns:
        SentenceTransformer model instance
    """
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return model


def generate_embeddings(chunks: List[str], batch_size: int = 32) -> np.ndarray:
    """
    Generate embeddings for a list of text chunks.
    
    Processes chunks in batches for memory efficiency.
    
    Args:
        chunks: List of text strings to embed
        batch_size: Number of texts to process at once (default 32)
        
    Returns:
        NumPy array of shape (num_chunks, EMBEDDING_DIMENSION)
        dtype: float32
        
    Raises:
        ValueError: If chunks list is empty
        Exception: If embedding generation fails
    """
    if not chunks:
        raise ValueError("chunks list cannot be empty")
    
    try:
        model = load_embedding_model()
        embeddings = model.encode(chunks, batch_size=batch_size, show_progress_bar=False)
        return np.array(embeddings, dtype=np.float32)
    
    except Exception as e:
        raise Exception(f"Error generating embeddings: {str(e)}")


def encode_query(query: str) -> np.ndarray:
    """
    Generate embedding for a single query string.
    
    Helper function for query encoding in retrieval.
    
    Args:
        query: Query text to embed
        
    Returns:
        NumPy array of shape (1, EMBEDDING_DIMENSION)
    """
    if not query or not isinstance(query, str):
        raise ValueError("query must be a non-empty string")
    
    model = load_embedding_model()
    embedding = model.encode([query], show_progress_bar=False)
    return np.array(embedding, dtype=np.float32)


def get_embedding_model_info() -> dict:
    """
    Get information about the embedding model being used.
    
    Returns:
        Dict with keys: {
            'model_name': str,
            'dimension': int,
            'description': str
        }
    """
    return {
        'model_name': EMBEDDING_MODEL_NAME,
        'dimension': EMBEDDING_DIMENSION,
        'description': 'Lightweight, fast embedding model optimized for semantic search. ~80MB.'
    }
