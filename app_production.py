"""
=============================================================================
DocuQuery: Semantic Search and QA System
Retrieval-Augmented Generation (RAG) with LLMs
=============================================================================

Multi-format document search and question-answering system.
Supports: PDF, TXT, DOCX, and more

Features:
- Multi-format text extraction
- Semantic embeddings (Sentence-Transformers)
- Vector similarity search (FAISS)
- Intelligent question answering

=============================================================================
"""

import streamlit as st
import time
import os
from typing import List, Tuple, Dict
import PyPDF2
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
from pathlib import Path
try:
    from docx import Document
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="DocuQuery - Semantic Search & QA",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional UI
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .answer-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .chunk-box {
        background-color: #e7f3ff;
        padding: 0.8rem;
        border-radius: 0.5rem;
        border-left: 4px solid #0066cc;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
    .error-box {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #dc3545;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# INITIALIZE SESSION STATE
# ============================================================================

def init_session_state():
    """Initialize Streamlit session state variables."""
    defaults = {
        'chat_history': [],
        'faiss_index': None,
        'chunks': [],
        'chunk_metadata': [],  # Track which file each chunk came from
        'embeddings': None,
        'loaded_files': [],
        'selected_files_filter': [],  # Files to search in
        'qa_model_type': 'huggingface',
        'top_k': 5,
        'debug_mode': False,
        'openai_api_key': os.environ.get('OPENAI_API_KEY', ''),
        'google_api_key': os.environ.get('GOOGLE_API_KEY', ''),
        'response_times': [],
        'index_stats': {}
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# ============================================================================
# CORE FUNCTIONS
# ============================================================================

@st.cache_resource
def load_embedding_model():
    """Load and cache the embedding model."""
    return SentenceTransformer('all-MiniLM-L6-v2')

def extract_pdf_text(pdf_file) -> str:
    """Extract text from PDF file."""
    text = ""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        total_pages = len(pdf_reader.pages)
        
        progress_bar = st.progress(0)
        for idx, page in enumerate(pdf_reader.pages):
            text += page.extract_text() + "\n"
            progress_bar.progress((idx + 1) / total_pages)
        
        return text
    except Exception as e:
        st.error(f"Error extracting PDF: {str(e)}")
        return ""

def extract_txt_text(txt_file) -> str:
    """Extract text from TXT file."""
    try:
        return txt_file.read().decode('utf-8', errors='ignore')
    except Exception as e:
        st.error(f"Error extracting TXT: {str(e)}")
        return ""

def extract_docx_text(docx_file) -> str:
    """Extract text from DOCX file."""
    if not HAS_DOCX:
        st.warning("DOCX support requires 'python-docx'. Install it to enable DOCX support.")
        return ""
    try:
        doc = Document(docx_file)
        text = "\n".join([para.text for para in doc.paragraphs])
        return text
    except Exception as e:
        st.error(f"Error extracting DOCX: {str(e)}")
        return ""

def extract_file_text(uploaded_file) -> str:
    """Extract text from various file formats."""
    file_extension = Path(uploaded_file.name).suffix.lower()
    
    if file_extension == '.pdf':
        return extract_pdf_text(uploaded_file)
    elif file_extension == '.txt':
        return extract_txt_text(uploaded_file)
    elif file_extension == '.docx':
        return extract_docx_text(uploaded_file)
    else:
        st.error(f"Unsupported file format: {file_extension}")
        return ""

def preprocess_text(text: str) -> str:
    """Preprocess and clean text."""
    # Remove extra whitespace
    text = ' '.join(text.split())
    # Remove special characters but keep punctuation
    text = text.replace('\x00', '').replace('\n\n\n', '\n\n')
    return text

def smart_chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> List[str]:
    """
    Intelligently split text into overlapping chunks.
    
    Args:
        text: Input text
        chunk_size: Target chunk size (words)
        overlap: Overlap between chunks (words)
    
    Returns:
        List of text chunks
    """
    sentences = text.split('. ')
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        sentence = sentence.strip() + ". "
        
        if len(current_chunk.split()) + len(sentence.split()) <= chunk_size:
            current_chunk += sentence
        else:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            current_chunk = sentence
    
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return [c for c in chunks if len(c.split()) > 10]  # Filter very short chunks

def create_faiss_index(embeddings: np.ndarray):
    """Create FAISS index from embeddings."""
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype('float32'))
    return index

def extract_keywords(question: str) -> List[str]:
    """Extract meaningful keywords from question."""
    import re
    
    # Convert to lowercase
    q = question.lower()
    
    # Extract specific phrases
    keywords = []
    
    # Pattern 1: Code identifiers (KR0463, etc.)
    codes = re.findall(r'\b[A-Z]{1,3}\d{3,4}\b', question)
    keywords.extend(codes)
    
    # Pattern 2: Week/Chapter/Section + number
    time_matches = re.findall(r'\b(week|chapter|section|module|part|day|lesson|unit)\s+(\d+)\b', q)
    for match in time_matches:
        keywords.append(f"{match[0]} {match[1]}")
    
    # Pattern 3: Extract remaining important words (nouns)
    # Remove common words
    stop_words = {'tell', 'me', 'about', 'what', 'is', 'the', 'a', 'an', 'and', 'or', 'but', 
                  'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'as', 'if', 'then',
                  'can', 'could', 'would', 'should', 'do', 'does', 'did', 'you', 'your'}
    
    words = re.findall(r'\b[a-z]{3,}\b', q)
    for word in words:
        if word not in stop_words and len(word) > 2:
            keywords.append(word)
    
    return list(set(keywords))  # Remove duplicates

def retrieve_chunks(query: str, top_k: int = 5) -> List[Tuple[str, float, str]]:
    """Retrieve chunks using keyword matching first, then semantic ranking."""
    if st.session_state.faiss_index is None:
        return []
    
    try:
        import re
        
        # Extract keywords from query
        keywords = extract_keywords(query)
        
        # If no keywords found, fall back to semantic search
        if not keywords:
            keywords = query.split()
        
        # Find chunks containing keywords
        matching_chunks = []
        for idx, chunk in enumerate(st.session_state.chunks):
            chunk_lower = chunk.lower()
            matched_keywords = []
            
            for keyword in keywords:
                # Case-insensitive search
                if keyword.lower() in chunk_lower:
                    matched_keywords.append(keyword)
            
            # If chunk contains at least one keyword, include it
            if matched_keywords:
                source_file = st.session_state.chunk_metadata[idx] if idx < len(st.session_state.chunk_metadata) else "Unknown"
                matching_chunks.append({
                    'idx': idx,
                    'chunk': chunk,
                    'source_file': source_file,
                    'keyword_matches': len(matched_keywords),
                    'matched_keywords': matched_keywords
                })
        
        # If keyword matching found results, use those
        if matching_chunks:
            # Sort by number of keyword matches (more matches = more relevant)
            matching_chunks.sort(key=lambda x: x['keyword_matches'], reverse=True)
            
            # Rank by cross-encoder for final ordering
            cross_encoder = load_cross_encoder()
            chunks_list = [c['chunk'] for c in matching_chunks[:min(len(matching_chunks), 20)]]
            cross_scores = cross_encoder.predict([(query, chunk) for chunk in chunks_list])
            
            for i, score in enumerate(cross_scores):
                matching_chunks[i]['cross_score'] = float(score)
            
            # Sort by cross-encoder score
            matching_chunks.sort(key=lambda x: x['cross_score'], reverse=True)
            
            # Return top-k
            results = []
            for item in matching_chunks[:top_k]:
                results.append((
                    item['chunk'],
                    item['cross_score'],
                    item['source_file']
                ))
            
            return results
        
        # Fallback: If no keyword matches, use semantic search
        model = load_embedding_model()
        query_embedding = model.encode([query]).astype('float32')
        distances, indices = st.session_state.faiss_index.search(query_embedding, top_k * 2)
        
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            idx = int(idx)
            if idx < len(st.session_state.chunks):
                source_file = st.session_state.chunk_metadata[idx] if idx < len(st.session_state.chunk_metadata) else "Unknown"
                similarity = 1 / (1 + float(distance))
                results.append((st.session_state.chunks[idx], similarity, source_file))
                if len(results) >= top_k:
                    break
        
        return results
    except Exception as e:
        st.error(f"Error retrieving chunks: {str(e)}")
        return []

@st.cache_resource
def load_cross_encoder():
    """Load and cache the cross-encoder model for precise QA ranking."""
    from sentence_transformers import CrossEncoder
    return CrossEncoder('cross-encoder/qnli-distilroberta-base')

def generate_answer_huggingface(question: str, context: str) -> str:
    """Generate answer by extracting relevant paragraphs containing keywords."""
    try:
        import re
        
        # Extract keywords from question
        keywords = extract_keywords(question)
        
        if not keywords:
            return "Unable to process your question. Please rephrase it."
        
        # Split context into paragraphs/sections
        paragraphs = [p.strip() for p in context.split('\n---\n') if len(p.strip()) > 20]
        
        if not paragraphs:
            # Fallback: split by double newlines
            paragraphs = [p.strip() for p in context.split('\n\n') if len(p.strip()) > 20]
        
        if not paragraphs:
            return "No relevant information found."
        
        # Find paragraphs containing keywords
        relevant_paragraphs = []
        for para in paragraphs:
            para_lower = para.lower()
            keyword_count = 0
            
            for keyword in keywords:
                if keyword.lower() in para_lower:
                    keyword_count += 1
            
            if keyword_count > 0:
                relevant_paragraphs.append({
                    'text': para,
                    'keyword_matches': keyword_count
                })
        
        # If we found relevant paragraphs, return the best one
        if relevant_paragraphs:
            # Sort by keyword matches
            relevant_paragraphs.sort(key=lambda x: x['keyword_matches'], reverse=True)
            
            # Take the first (most relevant) paragraph
            answer = relevant_paragraphs[0]['text']
            
            # Clean up the answer
            answer = answer.strip()
            if not answer.endswith('.'):
                answer += "."
            
            return answer
        
        # Fallback: Return the longest paragraph
        longest_para = max(paragraphs, key=len)
        if longest_para:
            answer = longest_para.strip()
            if not answer.endswith('.'):
                answer += "."
            return answer
        
        return "No relevant information found in the context."
    except Exception as e:
        return f"Error generating answer: {str(e)}"

def generate_answer_openai(question: str, context: str, api_key: str) -> str:
    """Generate answer using OpenAI API."""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        
        prompt = f"""Based on the following context, provide a concise and accurate answer to the question.

Context:
{context}

Question: {question}

Answer:"""
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=300
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

def generate_answer_gemini(question: str, context: str, api_key: str) -> str:
    """Generate answer using Google Gemini API."""
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-pro')
        
        prompt = f"""Based on the following context, provide a concise and accurate answer to the question.

Context:
{context}

Question: {question}

Answer:"""
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"

# ============================================================================
# MAIN UI LAYOUT
# ============================================================================

# Header
st.markdown("""
<div class="main-header">
    <h1>� DocuQuery: Semantic Search and QA System</h1>
    <p><em>Multi-format document search powered by LLMs and RAG</em></p>
</div>
""", unsafe_allow_html=True)

# Sidebar configuration
with st.sidebar:
    st.title("⚙️ Configuration")
    
    st.subheader("🤖 QA Model Selection")
    model_type = st.radio(
        "Choose QA Backend:",
        ["huggingface", "openai", "gemini"],
        captions=[
            "Open-source (No API key needed)",
            "GPT-3.5-Turbo (Requires API key)",
            "Google Gemini (Requires API key)"
        ]
    )
    st.session_state.qa_model_type = model_type
    
    if model_type == "openai":
        api_key = st.text_input("OpenAI API Key:", type="password", key="openai_key")
        st.session_state.openai_api_key = api_key
        if not api_key:
            st.warning("⚠️ OpenAI key not provided. Will use HuggingFace fallback.")
    
    elif model_type == "gemini":
        api_key = st.text_input("Google API Key:", type="password", key="google_key")
        st.session_state.google_api_key = api_key
        if not api_key:
            st.warning("⚠️ Google key not provided. Will use HuggingFace fallback.")
    
    st.divider()
    
    st.subheader("🎚️ Retrieval Settings")
    top_k = st.slider(
        "Top-K Chunks to Retrieve:",
        min_value=3,
        max_value=20,
        value=5,
        help="Number of most relevant chunks to use for answer generation"
    )
    st.session_state.top_k = top_k
    
    st.divider()
    
    st.subheader("🔍 Debug Options")
    debug_mode = st.checkbox("Enable Debug Mode", value=False)
    st.session_state.debug_mode = debug_mode
    
    if debug_mode:
        st.info("📊 In debug mode, you'll see retrieved chunks and similarity scores.")
    
    # Clear data button
    if st.button("🗑️ Clear All Data"):
        st.session_state.faiss_index = None
        st.session_state.chunks = []
        st.session_state.chat_history = []
        st.session_state.loaded_files = []
        st.success("✅ Data cleared!")
        st.rerun()

# ============================================================================
# MAIN CONTENT AREA
# ============================================================================

# Tab 1: Document Upload & Processing
tab1, tab2, tab3 = st.tabs(["📤 Upload & Process", "💬 Chat Interface", "📊 Statistics"])

with tab1:
    st.header("Step 1: Upload Documents")
    st.info("📎 Supported formats: PDF, TXT, DOCX")
    
    uploaded_files = st.file_uploader(
        "Select one or more documents:",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True,
        key="doc_uploader"
    )
    
    if uploaded_files:
        st.markdown(f"**{len(uploaded_files)} file(s) selected**")
        
        if st.button(" Process Files", key="process_btn", use_container_width=True):
            with st.spinner("Processing documents..."):
                all_chunks = []
                all_metadata = []  # Track source file for each chunk
                file_names = []
                
                # Extract and process each file separately
                for idx, file in enumerate(uploaded_files):
                    st.write(f"Processing: {file.name}")
                    text = extract_file_text(file)
                    file_names.append(file.name)
                    
                    # Preprocess this file's text
                    cleaned_text = preprocess_text(text)
                    
                    # Chunk this file's text
                    chunks = smart_chunk_text(cleaned_text, chunk_size=400, overlap=50)
                    
                    # Add metadata for each chunk (which file it came from)
                    for chunk in chunks:
                        all_chunks.append(chunk)
                        all_metadata.append(file.name)
                
                if all_chunks:
                    st.session_state.chunks = all_chunks
                    st.session_state.chunk_metadata = all_metadata
                    st.session_state.loaded_files = file_names
                    st.session_state.selected_files_filter = file_names  # Default: search all files
                    
                    # Generate embeddings
                    st.write("Generating embeddings (this may take a minute)...")
                    model = load_embedding_model()
                    embeddings = model.encode(all_chunks, show_progress_bar=True)
                    st.session_state.embeddings = embeddings
                    
                    # Create FAISS index
                    st.write("Creating FAISS index...")
                    index = create_faiss_index(embeddings)
                    st.session_state.faiss_index = index
                    
                    # Update stats
                    total_chars = sum(len(chunk) for chunk in all_chunks)
                    st.session_state.index_stats = {
                        'num_files': len(file_names),
                        'num_chunks': len(all_chunks),
                        'total_chars': total_chars,
                        'avg_chunk_size': total_chars / len(all_chunks) if all_chunks else 0,
                        'embedding_dim': embeddings.shape[1]
                    }
                    
                    st.success("✅ Files processed successfully!")
                    
                    # Display stats
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Files", len(file_names))
                    with col2:
                        st.metric("Total Chunks", len(all_chunks))
                    with col3:
                        st.metric("Total Characters", total_chars)
                    with col4:
                        st.metric("Embedding Dimensions", embeddings.shape[1])
                    
                    # Show file names
                    with st.expander("📄 Uploaded Files"):
                        for fname in file_names:
                            st.write(f"• {fname}")
                    
                    # Show sample chunks with metadata
                    with st.expander("📋 Sample Chunks (First 3)"):
                        for i, (chunk, source) in enumerate(zip(all_chunks[:3], all_metadata[:3])):
                            st.write(f"**Chunk {i+1}** | Source: `{source}`")
                            st.write(chunk[:200] + "..." if len(chunk) > 200 else chunk)
                            st.divider()
                
                else:
                    st.error("❌ No chunks generated. File may be empty or unreadable.")
    
    else:
        st.info("👆 Upload PDF file(s) to get started")

# ============================================================================
# Tab 2: Chat Interface
# ============================================================================

with tab2:
    st.header("💬 Ask Questions About Your Documents")
    
    if st.session_state.faiss_index is None:
        st.warning("⚠️ Please upload and process PDF files first in the 'Upload & Process' tab.")
    else:
        # Display currently loaded files
        with st.expander(f"📄 Loaded Files ({len(st.session_state.loaded_files)})"):
            for fname in st.session_state.loaded_files:
                st.write(f"✅ {fname}")
        
        st.divider()
        
        # File filter - search specific PDFs or all
    
        
        # Chat input
        question = st.text_input(
            "Enter your question:",
            placeholder="e.g., What is the main topic of this document?",
            key="question_input"
        )
        
        if st.button("🔍 Get Answer", use_container_width=True, key="answer_btn"):
            if not question.strip():
                st.error("❌ Please enter a question.")
            else:
                with st.spinner("Searching for relevant information..."):
                    start_time = time.time()
                    
                    # Retrieve top-k most relevant chunks
                    retrieved_chunks = retrieve_chunks(question, st.session_state.top_k)
                    
                    if not retrieved_chunks:
                        st.error("❌ No relevant information found in the documents.")
                    else:
                        # Build context from all retrieved chunks
                        context = "\n---\n".join([chunk for chunk, _, _ in retrieved_chunks])
                        
                        # Generate answer
                        model_type = st.session_state.qa_model_type
                        
                        if model_type == "openai":
                            answer = generate_answer_openai(
                                question, context, st.session_state.openai_api_key
                            )
                        elif model_type == "gemini":
                            answer = generate_answer_gemini(
                                question, context, st.session_state.google_api_key
                            )
                        else:
                            answer = generate_answer_huggingface(question, context)
                        
                        response_time = time.time() - start_time
                        st.session_state.response_times.append(response_time)
                        
                        # Display answer
                        st.markdown(f"""
                        <div class="answer-box">
                            <h4>Answer:</h4>
                            <p>{answer}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Display metadata
                        st.metric("Response Time", f"{response_time:.2f}s")
                        
                        # Debug mode: Show retrieved chunks with keyword matches
                        if st.session_state.debug_mode:
                            with st.expander("🔍 Debug: Retrieved Chunks & Keyword Matches"):
                                keywords = extract_keywords(question)
                                st.write(f"**Keywords extracted from question:** {', '.join(keywords)}")
                                st.divider()
                                
                                for i, (chunk, relevance_score, source_file) in enumerate(retrieved_chunks, 1):
                                    relevance_pct = max(0, int(relevance_score * 100))
                                    
                                    # Highlight which keywords appear in this chunk
                                    chunk_lower = chunk.lower()
                                    matched_keywords_in_chunk = [kw for kw in keywords if kw.lower() in chunk_lower]
                                    
                                    st.write(f"**Chunk {i}** | Source: `{source_file}` | Score: {relevance_score:.3f}")
                                    if matched_keywords_in_chunk:
                                        st.write(f"📌 **Matched keywords:** {', '.join(matched_keywords_in_chunk)}")
                                    
                                    st.markdown(f"""
                                    <div class="chunk-box">
                                        {chunk[:300]}...
                                    </div>
                                    """, unsafe_allow_html=True)
                        
                        # Add to chat history
                        st.session_state.chat_history.append({
                            'question': question,
                            'answer': answer,
                            'model': model_type,
                            'response_time': response_time
                        })
        
        # Display chat history
        st.divider()
        if st.session_state.chat_history:
            with st.expander(f"📜 Chat History ({len(st.session_state.chat_history)} messages)"):
                for i, msg in enumerate(st.session_state.chat_history, 1):
                    st.write(f"**Q{i}:** {msg['question']}")
                    st.write(f"**A{i}:** {msg['answer'][:150]}...")
                    st.caption(f"⏱️ {msg['response_time']:.2f}s | 🤖 {msg['model']}")
                    st.divider()

# ============================================================================
# Tab 3: Statistics & Information
# ============================================================================

with tab3:
    st.header("📊 System Statistics")
    
    if st.session_state.faiss_index is None:
        st.info("No data loaded yet. Process PDF files first.")
    else:
        # Index statistics
        stats = st.session_state.index_stats
        
        st.subheader("📈 Index Statistics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Files", stats['num_files'])
        with col2:
            st.metric("Total Chunks", stats['num_chunks'])
        with col3:
            st.metric("Embedding Dimensions", stats['embedding_dim'])
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Characters", f"{stats['total_chars']:,}")
        with col2:
            st.metric("Avg. Chunk Size", f"{stats['avg_chunk_size']:.0f} chars")
        
        # Response time statistics
        if st.session_state.response_times:
            st.subheader("⏱️ Performance Metrics")
            avg_time = np.mean(st.session_state.response_times)
            max_time = np.max(st.session_state.response_times)
            min_time = np.min(st.session_state.response_times)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Avg Response Time", f"{avg_time:.2f}s")
            with col2:
                st.metric("Max Response Time", f"{max_time:.2f}s")
            with col3:
                st.metric("Min Response Time", f"{min_time:.2f}s")
        
        # Loaded files
        st.subheader("📄 Loaded Documents")
        for fname in st.session_state.loaded_files:
            st.write(f"✅ {fname}")
        
        # Model information
        st.subheader("🧠 Embedding Model")
        model = load_embedding_model()
        st.write(f"**Model Name:** {model.get_sentence_embedding_dimension()}-dim embeddings")
        st.write(f"**Framework:** Sentence-Transformers (all-MiniLM-L6-v2)")
        st.write(f"**Retrieval Method:** FAISS (Flat L2 Distance)")


