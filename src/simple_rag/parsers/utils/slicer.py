"""
Simple PDF Slicer utility for extracting specific pages from PDF documents.
"""

from typing import List
from PyPDF2 import PdfReader, PdfWriter
from pathlib import Path


def slice_pdf_pages(pdf_path: str, pages: List[int]) -> str:
    """
    Extract specific pages from a PDF and create a new PDF.
    
    Args:
        pdf_path: Path to the source PDF file
        pages: List of page numbers to extract (1-indexed, e.g., [1, 2, 3])
        
    Returns:
        str: Path to the created sliced PDF
        
    Example:
        slice_pdf_pages("document.pdf", [1, 2, 3])
    """
    # Read the source PDF
    reader = PdfReader(pdf_path)
    writer = PdfWriter()
    
    # Add specified pages to writer (convert to 0-indexed)
    for page_num in pages:
        writer.add_page(reader.pages[page_num - 1])
    
    # Create output filename with page numbers
    if len(pages) == 1:
        pages_str = str(pages[0])
    else:
        pages_str = f"{pages[0]}_to_{pages[-1]}"
    
    # Convert to Path object and handle both string and Path inputs
    pdf_path = Path(pdf_path)
    output_path = pdf_path.with_stem(f"{pdf_path.stem}_pages_{pages_str}")
    
    # Write the new PDF
    with open(output_path, 'wb') as output_file:
        writer.write(output_file)
    
    return output_path


if __name__ == "__main__":
    import os
    
    # Get the relative path to the attention PDF
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(current_dir, '..', '..', '..', '..')
    pdf_path = os.path.join(project_root, 'data', 'raw', 'book.pdf')
    
    # Extract page 6
    output_path = slice_pdf_pages(pdf_path, list(range(22, 67)))
    print(f"Successfully created sliced PDF: {output_path}")