"""
QA Model Module

Modular question-answering system with multiple backend implementations:
- HuggingFace QA (default, open-source)
- OpenAI QA (requires API key)
- Google Gemini QA (requires API key)

All models follow the same interface for easy swapping.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import os
from dotenv import load_dotenv
import streamlit as st

# Load environment variables from .env file
load_dotenv()


class QAModel(ABC):
    """Abstract base class for QA models."""
    
    @abstractmethod
    def answer(self, question: str, context: str) -> str:
        """
        Generate an answer based on context and question.
        
        Args:
            question: User's question
            context: Retrieved context/chunks from document
            
        Returns:
            Generated answer string
        """
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """Return the name of the QA model."""
        pass


class HuggingFaceQA(QAModel):
    """
    HuggingFace QA Model - Open source, no API key required.
    
    Uses text-generation model for question answering.
    Fast and suitable for CPU inference.
    """
    
    def __init__(self):
        """Initialize HuggingFace text generation pipeline (lazy loaded on first use)."""
        self.pipeline = None
        self.model_name = "distilgpt2"
    
    @st.cache_resource
    def _load_pipeline(self):
        """Load text-generation pipeline with caching."""
        from transformers import pipeline
        return pipeline("text-generation", model=self.model_name)
    
    def answer(self, question: str, context: str) -> str:
        """
        Generate answer using simple context matching.
        
        Args:
            question: User's question
            context: Retrieved context from documents
            
        Returns:
            Extracted answer from context
        """
        try:
            # Simple extractive approach: find most relevant sentence from context
            from sentence_transformers import util
            from sentence_transformers import SentenceTransformer
            
            # Load embedding model to find best matching sentence
            model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            
            # Split context into sentences
            sentences = context.split('.')
            sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
            
            if not sentences:
                return "Unable to find relevant information in the provided context."
            
            # Encode question and sentences
            question_embedding = model.encode(question, convert_to_tensor=True)
            sentence_embeddings = model.encode(sentences, convert_to_tensor=True)
            
            # Find most similar sentence
            hits = util.semantic_search(question_embedding, sentence_embeddings, top_k=1)
            
            if hits and hits[0]:
                best_idx = hits[0][0]['corpus_id']
                answer = sentences[best_idx]
                return answer if answer else "No relevant answer found."
            
            return "Unable to generate an answer from the provided context."
        
        except Exception as e:
            return f"Error generating answer: {str(e)}"
    
    def get_model_name(self) -> str:
        """Return model identifier."""
        return f"HuggingFace (distilgpt2)"


class OpenAIQA(QAModel):
    """
    OpenAI QA Model - Uses GPT models for high-quality answers.
    
    Requires OPENAI_API_KEY environment variable or API key input.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize OpenAI QA model.
        
        Args:
            api_key: OpenAI API key. If None, reads from OPENAI_API_KEY env var.
        """
        self.api_key = api_key or os.environ.get('OPENAI_API_KEY')
        
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not provided. "
                "Set OPENAI_API_KEY environment variable or provide api_key parameter."
            )
        
        self.model_name = "gpt-3.5-turbo"
    
    def answer(self, question: str, context: str) -> str:
        """
        Generate answer using OpenAI API.
        
        Args:
            question: User's question
            context: Retrieved context from documents
            
        Returns:
            Generated answer from GPT model
        """
        try:
            from openai import OpenAI
            
            client = OpenAI(api_key=self.api_key)
            
            prompt = f"""Based on the following context, answer the question concisely and accurately.
            
Context:
{context}

Question: {question}

Answer:"""
            
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant answering questions based on provided context."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )
            
            return response.choices[0].message.content
        
        except Exception as e:
            raise Exception(f"OpenAI API error: {str(e)}")
    
    def get_model_name(self) -> str:
        """Return model identifier."""
        return f"OpenAI ({self.model_name})"


class GeminiQA(QAModel):
    """
    Google Gemini QA Model - Uses Google's Gemini Pro model.
    
    Requires GOOGLE_API_KEY environment variable or API key input.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Google Gemini QA model.
        
        Args:
            api_key: Google API key. If None, reads from GOOGLE_API_KEY env var.
        """
        self.api_key = api_key or os.environ.get('GOOGLE_API_KEY')
        
        if not self.api_key:
            raise ValueError(
                "Google API key not provided. "
                "Set GOOGLE_API_KEY environment variable or provide api_key parameter."
            )
        
        self.model_name = "gemini-pro"
    
    def answer(self, question: str, context: str) -> str:
        """
        Generate answer using Google Gemini API.
        
        Args:
            question: User's question
            context: Retrieved context from documents
            
        Returns:
            Generated answer from Gemini model
        """
        try:
            import google.generativeai as genai
            
            genai.configure(api_key=self.api_key)
            model = genai.GenerativeModel(self.model_name)
            
            prompt = f"""Based on the following context, answer the question concisely and accurately.

Context:
{context}

Question: {question}

Answer:"""
            
            response = model.generate_content(prompt)
            return response.text
        
        except Exception as e:
            raise Exception(f"Google Gemini API error: {str(e)}")
    
    def get_model_name(self) -> str:
        """Return model identifier."""
        return f"Google Gemini ({self.model_name})"


def get_qa_model(
    model_type: str,
    api_key: Optional[str] = None,
    fallback: bool = True
) -> QAModel:
    """
    Factory function to get QA model instance.
    
    Args:
        model_type: One of "huggingface", "openai", "gemini"
        api_key: API key if needed (for OpenAI or Gemini)
        fallback: If True, fallback to HuggingFace on error for API models
        
    Returns:
        QAModel instance
        
    Raises:
        ValueError: If model_type is unknown or API key missing (if fallback=False)
    """
    model_type = model_type.lower().strip()
    
    if model_type == "huggingface":
        return HuggingFaceQA()
    
    elif model_type == "openai":
        try:
            return OpenAIQA(api_key=api_key)
        except Exception as e:
            if fallback:
                print(f"Warning: Failed to initialize OpenAI model: {e}. Falling back to HuggingFace.")
                return HuggingFaceQA()
            else:
                raise
    
    elif model_type == "gemini":
        try:
            return GeminiQA(api_key=api_key)
        except Exception as e:
            if fallback:
                print(f"Warning: Failed to initialize Gemini model: {e}. Falling back to HuggingFace.")
                return HuggingFaceQA()
            else:
                raise
    
    else:
        raise ValueError(
            f"Unknown model_type: {model_type}. "
            "Must be one of: 'huggingface', 'openai', 'gemini'"
        )


def get_available_models() -> Dict[str, str]:
    """
    Get list of available QA models and their descriptions.
    
    Returns:
        Dict mapping model names to descriptions
    """
    return {
        "huggingface": "HuggingFace QA (Open-source, no API key required)",
        "openai": "OpenAI GPT-3.5 (Requires OPENAI_API_KEY)",
        "gemini": "Google Gemini Pro (Requires GOOGLE_API_KEY)"
    }
