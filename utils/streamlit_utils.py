import os
import json
import torch
import tempfile
import shutil

# Try to import PDF processing modules, but don't fail if they're missing
try:
    from utils.pdf_loader import pdf_loader
    from utils.embedding import get_contents_with_embedding
    PDF_PROCESSING_AVAILABLE = True
except ImportError:
    PDF_PROCESSING_AVAILABLE = False

def process_uploaded_pdf(uploaded_file, config, database_path):
    """
    Process an uploaded PDF file and add it to the database
    
    Args:
        uploaded_file: StreamlitUploadedFile object
        config: Configuration dictionary
        database_path: Path to the database directory
    
    Returns:
        tuple: (success: bool, message: str)
    """
    if not PDF_PROCESSING_AVAILABLE:
        return False, "PDF processing is not available. Please install pdf2image and pdfminer.six dependencies."
    
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        # Get file name without extension
        file_name = uploaded_file.name.replace('.pdf', '')
        file_target_path = os.path.join(database_path, file_name)
        
        # Check if file already exists
        if os.path.exists(file_target_path):
            os.unlink(tmp_path)  # Clean up temp file
            return False, f"File {file_name}.pdf has already been imported. Duplicate imports cannot be made!"
        
        # Create target directory
        os.makedirs(file_target_path, exist_ok=False)
        
        # Load and analyze PDF file
        raw_doc = pdf_loader(file_name, tmp_path, model_type=config['MODEL_TYPES']['PDF_ANALYZE_MODEL'])
        
        # Save raw data
        with open(os.path.join(file_target_path, 'raw_data.json'), "w", encoding="utf-8") as f:
            json.dump(raw_doc, f, ensure_ascii=False, indent=4)
        
        # Get contents with embeddings
        contents_with_embed = get_contents_with_embedding(
            raw_doc, 
            overlap=config['DATABASE']['OVERLAP_LENGTH'], 
            text_length=config['DATABASE']['TEXT_LENGTH'], 
            model_type=config['MODEL_TYPES']['TEXT_EMBED_MODEL']
        )
        
        # Save contents with embeddings
        torch.save(contents_with_embed, os.path.join(file_target_path, "contents_with_embed.pth"))
        
        # Save contents without embeddings for human review
        contents_with_embed_copy = contents_with_embed.copy()
        contents_with_embed_copy.pop('embedding')
        with open(os.path.join(file_target_path, 'contents_without_embed.json'), "w", encoding="utf-8") as f:
            json.dump(contents_with_embed_copy, f, ensure_ascii=False, indent=4)
        
        # Clean up temporary file
        os.unlink(tmp_path)
        
        return True, f"Successfully processed and added {file_name}.pdf to the database!"
        
    except Exception as e:
        # Clean up temporary file if it exists
        if 'tmp_path' in locals():
            try:
                os.unlink(tmp_path)
            except:
                pass
        return False, f"Error processing file: {str(e)}"

def validate_pdf_file(uploaded_file):
    """
    Validate uploaded PDF file
    
    Args:
        uploaded_file: StreamlitUploadedFile object
    
    Returns:
        tuple: (is_valid: bool, message: str)
    """
    if uploaded_file is None:
        return False, "No file uploaded"
    
    # Check file extension
    if not uploaded_file.name.lower().endswith('.pdf'):
        return False, "File must be a PDF"
    
    # Check file size (limit to 50MB)
    max_size = 50 * 1024 * 1024  # 50MB in bytes
    if uploaded_file.size > max_size:
        return False, f"File size ({uploaded_file.size / 1024 / 1024:.1f}MB) exceeds maximum allowed size (50MB)"
    
    return True, "File is valid"

def get_database_info(database_path):
    """
    Get information about the current database
    
    Args:
        database_path: Path to the database directory
    
    Returns:
        dict: Database information
    """
    if not os.path.exists(database_path):
        return {"total_files": 0, "files": []}
    
    files = []
    for item in os.listdir(database_path):
        item_path = os.path.join(database_path, item)
        if os.path.isdir(item_path):
            # Check if it has the required files
            has_embed = os.path.exists(os.path.join(item_path, 'contents_with_embed.pth'))
            has_raw = os.path.exists(os.path.join(item_path, 'raw_data.json'))
            
            files.append({
                "name": item,
                "has_embeddings": has_embed,
                "has_raw_data": has_raw,
                "status": "complete" if has_embed and has_raw else "incomplete"
            })
    
    return {
        "total_files": len(files),
        "files": files
    } 