"""
Vector Store Module

Handles FAISS index creation, management, persistence, and semantic search.
"""

from typing import List, Tuple
import numpy as np
import faiss
import os
import pickle
from pathlib import Path


# Constants
FAISS_SAVE_DIR = "data/faiss_index"
INDEX_FILENAME = "index.faiss"
CHUNKS_FILENAME = "chunks.pkl"


def create_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    """
    Create a FAISS index from embeddings.
    
    Uses IndexFlatL2 (Euclidean distance search):
    - Simple, exact search (no approximation)
    - Suitable for datasets up to ~100k vectors on CPU
    - Reasonable performance: ~O(1) search latency, ~O(n) linear scan
    
    Args:
        embeddings: NumPy array of shape (n_samples, embedding_dim)
                   dtype must be float32
        
    Returns:
        faiss.IndexFlatL2 instance
        
    Raises:
        ValueError: If embeddings is invalid or empty
    """
    if embeddings is None or len(embeddings) == 0:
        raise ValueError("embeddings cannot be None or empty")
    
    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(np.float32)
    
    # Determine dimension from embedding size
    dimension = embeddings.shape[1]
    
    # Create L2 distance index
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    return index


def save_index(
    index: faiss.Index,
    chunks: List[str],
    save_dir: str = FAISS_SAVE_DIR
) -> str:
    """
    Save FAISS index and associated chunks to disk.
    
    Saves:
    - FAISS index as .faiss file
    - Chunks metadata as pickled list
    
    Args:
        index: FAISS index instance
        chunks: List of text chunks corresponding to index
        save_dir: Directory to save files (created if not exists)
        
    Returns:
        Path to saved index directory
        
    Raises:
        Exception: If save operation fails
    """
    try:
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Save FAISS index
        index_path = os.path.join(save_dir, INDEX_FILENAME)
        faiss.write_index(index, index_path)
        
        # Save chunks
        chunks_path = os.path.join(save_dir, CHUNKS_FILENAME)
        with open(chunks_path, 'wb') as f:
            pickle.dump(chunks, f)
        
        return save_dir
        
    except Exception as e:
        raise Exception(f"Error saving FAISS index: {str(e)}")


def load_index(load_dir: str = FAISS_SAVE_DIR) -> Tuple[faiss.Index, List[str]]:
    """
    Load FAISS index and chunks from disk.
    
    Returns saved index and metadata.
    
    Args:
        load_dir: Directory containing saved index and chunks
        
    Returns:
        Tuple of (faiss.Index, list_of_chunks)
        
    Raises:
        FileNotFoundError: If index or chunks file not found
        Exception: If load operation fails
    """
    try:
        index_path = os.path.join(load_dir, INDEX_FILENAME)
        chunks_path = os.path.join(load_dir, CHUNKS_FILENAME)
        
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"FAISS index not found at {index_path}")
        
        if not os.path.exists(chunks_path):
            raise FileNotFoundError(f"Chunks file not found at {chunks_path}")
        
        # Load FAISS index
        index = faiss.read_index(index_path)
        
        # Load chunks
        with open(chunks_path, 'rb') as f:
            chunks = pickle.load(f)
        
        return index, chunks
        
    except Exception as e:
        raise Exception(f"Error loading FAISS index: {str(e)}")


def retrieve_chunks(
    index: faiss.Index,
    chunks: List[str],
    query_embedding: np.ndarray,
    k: int = 5
) -> List[Tuple[str, float]]:
    """
    Retrieve top-k most similar chunks to a query embedding.
    
    Args:
        index: FAISS index instance
        chunks: List of text chunks (must correspond to index)
        query_embedding: Query embedding array of shape (1, embedding_dim)
        k: Number of top results to return (default 5)
        
    Returns:
        List of tuples: [(chunk_text, L2_distance), ...]
        Sorted by distance (lower distance = more similar)
        
    Raises:
        ValueError: If inputs are invalid
    """
    if query_embedding.dtype != np.float32:
        query_embedding = query_embedding.astype(np.float32)
    
    if query_embedding.shape[0] != 1:
        raise ValueError("query_embedding must have shape (1, embedding_dim)")
    
    k = min(k, len(chunks))  # Can't retrieve more chunks than exist
    
    # Search FAISS index
    distances, indices = index.search(query_embedding, k)
    
    # Retrieve corresponding chunks
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < len(chunks):
            results.append((chunks[idx], float(dist)))
    
    return results


def index_exists(load_dir: str = FAISS_SAVE_DIR) -> bool:
    """
    Check if a saved FAISS index exists.
    
    Args:
        load_dir: Directory to check
        
    Returns:
        True if both index and chunks files exist, False otherwise
    """
    index_path = os.path.join(load_dir, INDEX_FILENAME)
    chunks_path = os.path.join(load_dir, CHUNKS_FILENAME)
    
    return os.path.exists(index_path) and os.path.exists(chunks_path)


def delete_index(load_dir: str = FAISS_SAVE_DIR) -> bool:
    """
    Delete saved FAISS index and chunks.
    
    Args:
        load_dir: Directory containing index to delete
        
    Returns:
        True if deletion successful, False if directory didn't exist
    """
    try:
        if os.path.exists(load_dir):
            # Remove files
            index_path = os.path.join(load_dir, INDEX_FILENAME)
            chunks_path = os.path.join(load_dir, CHUNKS_FILENAME)
            
            if os.path.exists(index_path):
                os.remove(index_path)
            if os.path.exists(chunks_path):
                os.remove(chunks_path)
            
            return True
        return False
    except Exception as e:
        print(f"Error deleting index: {str(e)}")
        return False


def get_index_stats(index: faiss.Index) -> dict:
    """
    Get statistics about a FAISS index.
    
    Args:
        index: FAISS index instance
        
    Returns:
        Dict with keys: {
            'num_vectors': int,
            'dimension': int,
            'index_type': str
        }
    """
    return {
        'num_vectors': index.ntotal,
        'dimension': index.d,
        'index_type': type(index).__name__
    }
