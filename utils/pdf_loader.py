"""
PDF Loader Module

Handles extraction of text from PDF files using PyPDF2.
Supports single and multiple PDF processing with error handling.
"""

from typing import List, Tuple
import PyPDF2
import io


def load_pdf_text(file_obj) -> Tuple[str, dict]:
    """
    Extract text from a single PDF file.
    
    Args:
        file_obj: File-like object from Streamlit file uploader or file path string
        
    Returns:
        Tuple of (extracted_text, metadata)
        metadata: dict with keys {
            'num_pages': int,
            'num_characters': int,
            'filename': str (if available)
        }
        
    Raises:
        ValueError: If PDF is corrupted or cannot be read
        Exception: General PDF processing errors
    """
    try:
        # Handle both file-like objects (from Streamlit) and file paths (string)
        if isinstance(file_obj, str):
            with open(file_obj, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                filename = file_obj.split('/')[-1]
                text, num_pages = _extract_text_from_reader(pdf_reader)
        else:
            # Streamlit UploadedFile object
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_obj.read()))
            filename = file_obj.name
            text, num_pages = _extract_text_from_reader(pdf_reader)
        
        metadata = {
            'num_pages': num_pages,
            'num_characters': len(text),
            'filename': filename
        }
        
        return text, metadata
        
    except PyPDF2.PdfReadError as e:
        raise ValueError(f"Failed to read PDF: {str(e)}. The file may be corrupted or encrypted.")
    except Exception as e:
        raise Exception(f"Error extracting text from PDF: {str(e)}")


def _extract_text_from_reader(pdf_reader: PyPDF2.PdfReader) -> Tuple[str, int]:
    """
    Internal helper to extract text from a PyPDF2 reader object.
    
    Args:
        pdf_reader: PyPDF2.PdfReader instance
        
    Returns:
        Tuple of (extracted_text, num_pages)
    """
    text_parts = []
    num_pages = len(pdf_reader.pages)
    
    for page_num, page in enumerate(pdf_reader.pages):
        try:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
        except Exception as e:
            print(f"Warning: Could not extract text from page {page_num + 1}: {str(e)}")
            continue
    
    combined_text = "\n".join(text_parts)
    return combined_text, num_pages


def load_multiple_pdfs(file_objects: List) -> Tuple[str, List[dict]]:
    """
    Extract and combine text from multiple PDF files.
    
    Args:
        file_objects: List of file-like objects (from Streamlit file uploader)
        
    Returns:
        Tuple of (combined_text, list_of_metadata_dicts)
        Combined text has explicit file separators for context
    """
    combined_text = ""
    metadata_list = []
    
    for file_obj in file_objects:
        try:
            text, metadata = load_pdf_text(file_obj)
            
            # Add file separator for clarity
            file_header = f"\n\n[FILE: {metadata['filename']}]\n"
            combined_text += file_header + text
            
            metadata_list.append(metadata)
            
        except Exception as e:
            print(f"Warning: Could not process file {file_obj.name}: {str(e)}")
            continue
    
    if not combined_text.strip():
        raise ValueError("No text could be extracted from any of the provided PDFs.")
    
    return combined_text, metadata_list


def get_file_summary(metadata_list: List[dict]) -> str:
    """
    Generate a summary string of loaded files.
    
    Args:
        metadata_list: List of metadata dicts from load_pdf_text
        
    Returns:
        Formatted summary string
    """
    summary = "📄 **Loaded Documents:**\n"
    total_chars = 0
    
    for i, meta in enumerate(metadata_list, 1):
        summary += f"{i}. **{meta['filename']}** - {meta['num_pages']} pages ({meta['num_characters']:,} characters)\n"
        total_chars += meta['num_characters']
    
    summary += f"\n**Total:** {len(metadata_list)} files, {total_chars:,} characters"
    return summary
