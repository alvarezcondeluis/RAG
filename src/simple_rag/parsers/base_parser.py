from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, List, Dict, Any, Union


class ParserProcessor(ABC):
    """
    Abstract base class for document parser processors.
    Defines the common interface that all parser implementations must follow.
    """
    
    def __init__(self, root_path: Optional[Path] = None):
        """
        Initialize the parser processor.
        
        Args:
            root_path: Root directory for file operations. If None, uses script location
        """
        self.root_path = root_path or Path(__file__).resolve().parents[2]
    
    @abstractmethod
    def parse_document(self, file_path: Path, verbose: bool = True) -> List[Any]:
        """
        Parse a document using the specific parser implementation.
        
        Args:
            file_path: Path to the document to parse
            verbose: Whether to print progress messages
            
        Returns:
            List of parsed document objects
            
        Raises:
            Exception: If parsing fails
        """
        pass
  
    
    def slice_pdf(self, pdf_path: Path, start_page: int, end_page: int, output_path: Path) -> Path:
        """
        Extract a range of pages from a PDF file.
        This is a common utility method available to all parser implementations.
        
        Args:
            pdf_path: Path to the source PDF
            start_page: Starting page number (1-indexed)
            end_page: Ending page number (1-indexed, inclusive)
            output_path: Path for the sliced PDF output
            
        Returns:
            Path to the created slice file
        """
        from pypdf import PdfReader, PdfWriter
        
        reader = PdfReader(str(pdf_path))
        writer = PdfWriter()
        
        for p in range(start_page - 1, end_page):
            if p < len(reader.pages):
                writer.add_page(reader.pages[p])
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            writer.write(f)
        
        return output_path
    
    def process_document(
        self,
        file_path: Path,
        output_path: Optional[Path] = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Complete workflow: parse document and generate output.
        This template method can be overridden by subclasses for custom workflows.
        
        Args:
            file_path: Path to the document to parse
            output_path: Path for the output file. If None, auto-generates
            verbose: Whether to print progress messages
            
        Returns:
            Dictionary with processing results
        """
        if output_path is None:
            output_dir = self.root_path / "data" / "processed"
            output_path = output_dir / f"{file_path.stem}_processed"
        
        # Parse document
        docs = self.parse_document(file_path, verbose=verbose)
        
        # Generate output
        if docs:
            result_path = self.generate_output(docs, output_path)
            if verbose:
                print(f"Saved output → {result_path}")
            return {"success": True, "output_path": result_path, "docs_count": len(docs)}
        else:
            if verbose:
                print("No documents returned from parsing - skipping output generation")
            return {"success": False, "output_path": None, "docs_count": 0}
