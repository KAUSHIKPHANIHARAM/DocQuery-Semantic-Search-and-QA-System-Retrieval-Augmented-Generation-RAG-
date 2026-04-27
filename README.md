# DocuQuery: Semantic Search and QA System

A Retrieval-Augmented Generation (RAG) system that enables intelligent question-answering across multiple document formats. This application combines semantic embeddings with vector similarity search to provide accurate, context-aware responses from your documents.

## Overview

DocuQuery is built to handle real-world document processing challenges. Instead of keyword-based search, this system understands the semantic meaning of your queries and documents, retrieving the most relevant information regardless of exact word matches. It supports PDF, TXT, DOCX, and other common document formats.

## Key Features

- Multi-format document support: Process PDFs, Word documents, text files, and more
- Semantic search capabilities: Find information based on meaning, not just keywords
- Intelligent question answering: Get direct answers from your documents using contextual understanding
- FAISS vector indexing: Fast, scalable similarity search across large document collections
- Modular QA backends: Switch between HuggingFace, OpenAI, and Google Gemini models
- User-friendly interface: Built with Streamlit for easy interaction
- Environment-based configuration: Flexible setup for different deployment scenarios

## System Architecture

The system consists of three main components:

1. Document Processing: Extracts and preprocesses text from various document formats with intelligent chunking and cleaning

2. Vector Embeddings: Uses Sentence-Transformers to convert text into high-dimensional semantic embeddings that capture meaning beyond simple word matching

3. Semantic Search and QA: Implements FAISS indexing for fast similarity search combined with question-answering models to generate contextual responses

## Technical Stack

- Python 3.8+
- Streamlit: Web interface framework
- Sentence-Transformers: Semantic embedding generation
- FAISS: Vector similarity search engine
- Transformers: Deep learning models for NLP
- PyPDF2: PDF document processing
- PyTorch: Deep learning backend
- python-docx: Word document processing


### Working with Different Document Formats

The system automatically detects and processes various document formats:

- PDF files: Extracted with text layout preservation
- Word documents (.docx): Structured text and metadata preserved
- Plain text (.txt): Direct processing for immediate use

### Understanding the Semantic Search

The system converts documents and queries into semantic embeddings that capture meaning. This allows it to:

- Find relevant information even with different wording
- Understand context and relationships between concepts
- Provide more accurate results than traditional keyword search
- Handle complex, multi-part questions effectively

### Switching QA Models

The qa_model.py module supports multiple backend implementations:

- HuggingFace QA: Open-source, no API keys required, runs locally
- OpenAI: More advanced reasoning, requires API key and generates costs
- Google Gemini: Powerful multimodal capabilities, requires API key

Switch models by configuring environment variables or through the application interface.

## Performance Considerations

For optimal performance with large document collections:

- Document chunks are sized around 512 tokens for balanced context and search precision
- FAISS IndexFlatL2 provides exact similarity search suitable for up to approximately 100,000 vectors on standard CPU
- For larger collections, consider using approximate search indices like HNSW
- Embedding generation typically takes 50-100 milliseconds per chunk depending on hardware

### Memory Issues with Large Documents

If processing large PDF files causes memory issues:

- Process documents in smaller batches
- Increase chunk size to reduce total number of embeddings
- Monitor RAM usage during indexing

### FAISS Index Not Found

If the system cannot find a cached index, it will regenerate it. This is normal on first run or after clearing the cache. Delete the data/faiss_index directory to start fresh.

## Google Colab Usage

A Jupyter notebook is provided for running DocuQuery in Google Colab:

1. Open NLP_CBP_Colab.ipynb in Google Colab
2. Run the cells sequentially to set up the environment
3. Upload documents directly in the notebook interface
4. Execute queries interactively

## Contributing

Contributions are welcome. Please follow these guidelines:

- Maintain code style consistency with existing modules
- Add docstrings to new functions and classes
- Test changes with multiple document types
- Update requirements.txt if adding new dependencies
- Document any new features in this README

## Performance and Limitations

- Current implementation optimized for document sets up to several thousand pages
- Vector similarity search is approximate for very large collections
- Some special formatting in documents may be lost during text extraction
- LLM-based question answering response time depends on model selection and document context length

## Future Enhancements

Potential improvements for future versions include:

- Support for additional document formats including HTML, Markdown, and spreadsheets
- Approximate nearest neighbor search for faster retrieval on massive datasets
- Multi-language support and cross-language semantic search
- Document metadata preservation and advanced filtering
- Batch processing capabilities for automated document workflows
- API endpoints for integration with other applications
- Caching mechanisms to improve repeated query performance

