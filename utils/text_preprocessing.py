"""
Text Preprocessing Module

Handles text cleaning, normalization, and intelligent chunking for RAG systems.
"""

from typing import List
import re
import nltk
from nltk.tokenize import sent_tokenize
import string

# Download NLTK data on first import
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)


# Constants
DEFAULT_CHUNK_SIZE = 400  # words
DEFAULT_OVERLAP = 50  # words


def preprocess_text(text: str) -> str:
    """
    Clean and normalize text for embedding.
    
    Operations:
    - Remove extra whitespace and newlines
    - Convert to lowercase
    - Remove HTML tags and special formatting
    - Normalize punctuation
    - Remove excessive special characters (but keep important ones like . ? !)
    
    Args:
        text: Raw text to preprocess
        
    Returns:
        Cleaned text ready for chunking and embedding
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Replace multiple whitespace characters with single space
    text = re.sub(r'\s+', ' ', text)
    
    # Remove non-ASCII characters but keep accented chars
    text = text.encode('ascii', 'ignore').decode('ascii')
    
    # Remove extra special characters (keep ., ?, !, -, ', ")
    text = re.sub(r'[^\w\s\.\?!,;\-\'":]', ' ', text)
    
    # Clean up spacing again after removing special chars
    text = re.sub(r'\s+', ' ', text)
    
    # Lowercase for consistency
    text = text.lower()
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    return text


def smart_chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP
) -> List[str]:
    """
    Split text into intelligent chunks based on sentences to avoid breaking mid-sentence.
    
    Strategy:
    1. Split text into sentences using NLTK
    2. Group sentences into chunks of approximately chunk_size words
    3. Add overlap between chunks for context preservation
    
    Args:
        text: Input text to chunk (should be preprocessed)
        chunk_size: Target chunk size in words (default 400)
        overlap: Number of words to overlap between chunks (default 50)
        
    Returns:
        List of text chunks
    """
    if not text or not isinstance(text, str):
        return []
    
    # Ensure chunk_size and overlap are reasonable
    chunk_size = max(50, min(chunk_size, 1000))  # Clamp between 50-1000
    overlap = max(0, min(overlap, chunk_size // 2))  # Max overlap is half chunk size
    
    # Handle very short texts
    if len(text.split()) < chunk_size:
        return [text] if text.strip() else []
    
    # Sentence tokenization
    try:
        sentences = sent_tokenize(text)
    except:
        # Fallback: split by periods if NLTK fails
        sentences = [s.strip() + '.' for s in text.split('.') if s.strip()]
    
    if not sentences:
        return [text] if text.strip() else []
    
    chunks = []
    current_chunk = []
    current_word_count = 0
    
    for sentence in sentences:
        sentence_words = sentence.split()
        sentence_word_count = len(sentence_words)
        
        # If adding this sentence exceeds chunk size, save current chunk and start new one
        if current_word_count + sentence_word_count > chunk_size and current_chunk:
            # Save chunk
            chunk_text = ' '.join(current_chunk)
            chunks.append(chunk_text)
            
            # Create overlap: keep last overlap_word_count words for next chunk
            overlap_text = ' '.join(current_chunk[-overlap:]) if overlap > 0 else ""
            current_chunk = overlap_text.split() if overlap_text else []
            current_word_count = len(current_chunk)
        
        # Add sentence to current chunk
        current_chunk.extend(sentence_words)
        current_word_count += sentence_word_count
    
    # Add remaining text as final chunk
    if current_chunk:
        chunk_text = ' '.join(current_chunk)
        chunks.append(chunk_text)
    
    # Filter out very small chunks (less than 20 words)
    chunks = [c for c in chunks if len(c.split()) > 20]
    
    return chunks if chunks else [text] if text.strip() else []


def get_chunking_summary(chunks: List[str]) -> dict:
    """
    Generate statistics about chunked text.
    
    Args:
        chunks: List of text chunks
        
    Returns:
        Dict with statistics: {
            'num_chunks': int,
            'avg_chunk_size': float (words),
            'min_chunk_size': int (words),
            'max_chunk_size': int (words),
            'total_words': int
        }
    """
    if not chunks:
        return {
            'num_chunks': 0,
            'avg_chunk_size': 0,
            'min_chunk_size': 0,
            'max_chunk_size': 0,
            'total_words': 0
        }
    
    chunk_sizes = [len(chunk.split()) for chunk in chunks]
    total_words = sum(chunk_sizes)
    
    return {
        'num_chunks': len(chunks),
        'avg_chunk_size': round(total_words / len(chunks), 1),
        'min_chunk_size': min(chunk_sizes),
        'max_chunk_size': max(chunk_sizes),
        'total_words': total_words
    }


def format_chunks_for_display(chunks: List[str], max_display: int = 5) -> str:
    """
    Format chunks into a readable display string.
    
    Args:
        chunks: List of text chunks
        max_display: Maximum chunks to display (default 5)
        
    Returns:
        Formatted string for display
    """
    if not chunks:
        return "No chunks to display."
    
    display = f"📦 **Total Chunks: {len(chunks)}**\n\n"
    
    for i, chunk in enumerate(chunks[:max_display], 1):
        preview = chunk[:200] + "..." if len(chunk) > 200 else chunk
        word_count = len(chunk.split())
        display += f"**Chunk {i}** ({word_count} words):\n{preview}\n\n"
    
    if len(chunks) > max_display:
        display += f"... and {len(chunks) - max_display} more chunks"
    
    return display
