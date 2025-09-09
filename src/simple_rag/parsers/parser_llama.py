
import os
import re
import html
from pathlib import Path
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv
from pypdf import PdfReader, PdfWriter
from llama_parse import LlamaParse
from .base_parser import ParserProcessor


class LlamaParseProcessor(ParserProcessor):
    """
    A class for processing PDF documents using LlamaParse with configurable options
    and markdown normalization capabilities.
    """
    
    # Regex patterns for markdown normalization
    HEADING_RE = re.compile(r"(?m)^(#{1,6})(\S)")
    TABLE_ROW_RE = re.compile(r"^\s*\|")
    TABLE_SEP_RE = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)+\|?\s*$")
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        result_type: str = "markdown",
        language: str = "en",
        num_workers: int = 4,
        parsing_instruction: Optional[str] = None,
        root_path: Optional[Path] = None
    ):
        """
        Initialize the LlamaParseProcessor.
        
        Args:
            api_key: LlamaParse API key. If None, loads from LLAMAPARSE_API_KEY env var
            result_type: Output format ("markdown", "text", "structured")
            language: Document language
            num_workers: Number of parallel workers
            parsing_instruction: Custom parsing instructions
            root_path: Root directory for file operations. If None, uses script location
        """
        # Initialize parent class
        super().__init__(root_path)
        
        # Load environment variables
        load_dotenv()
        
        # Set API key
        self.api_key = api_key or os.getenv("LLAMAPARSE_API_KEY")
        if not self.api_key:
            raise RuntimeError("LLAMAPARSE_API_KEY not set in .env or provided as parameter")
        
        # Set configuration attributes
        self.result_type = result_type
        self.language = language
        self.num_workers = num_workers
        self.parsing_instruction = parsing_instruction or (
            "Preserve the document structure and content maintaining, mathematical equations \n"
            "Pay special attention also to the table structure and content so that it matches the original one (headers included)"
        )
        
        # Initialize LlamaParse client
        self._init_parser()
    
    def _init_parser(self):
        """Initialize the LlamaParse client with current configuration."""
        self.parser = LlamaParse(
            api_key=self.api_key,
            result_type=self.result_type,
            language=self.language,
            num_workers=self.num_workers,
            parsing_instruction=self.parsing_instruction
        )
    
    def slice_pdf(self, pdf_path: Path, start_page: int, end_page: int, output_path: Path) -> Path:
        """
        Extract a range of pages from a PDF file.
        
        Args:
            pdf_path: Path to the source PDF
            start_page: Starting page number (1-indexed)
            end_page: Ending page number (1-indexed, inclusive)
            output_path: Path for the sliced PDF output
            
        Returns:
            Path to the created slice file
        """
        reader = PdfReader(str(pdf_path))
        writer = PdfWriter()
        
        for p in range(start_page - 1, end_page):
            if p < len(reader.pages):
                writer.add_page(reader.pages[p])
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            writer.write(f)
        
        return output_path
    
    def parse_document(self, file_path: Path, verbose: bool = True) -> List[Any]:
        """
        Parse a document using LlamaParse.
        
        Args:
            file_path: Path to the document to parse
            verbose: Whether to print progress messages
            
        Returns:
            List of parsed document objects
            
        Raises:
            Exception: If parsing fails
        """
        try:
            if verbose:
                print(f"[llamaparse] sending document: {file_path}")
            
            docs = self.parser.load_data(str(file_path))
            
            if verbose:
                print(f"[llamaparse] returned docs: {len(docs)}")
            
            # Print diagnostic info if no docs returned
            if len(docs) == 0 and verbose:
                print("[warn] LlamaParse returned 0 docs. Quick checklist:")
                print("  - Is the file really text/scan? Try a different page range.")
                print("  - Try result_type='text' or 'structured'.")
                print("  - Try without slicing: pass the full PDF once.")
                print("  - Check network/proxy; Llama Cloud needs outbound https.")
                print("  - Double-check key validity (a bad key can yield empty results).")
            
            return docs
            
        except Exception as e:
            if verbose:
                print(f"[llamaparse] exception: {repr(e)}")
            raise
    
    def normalize_markdown(self, md: str) -> str:
        """
        Normalize markdown text with consistent formatting.
        
        Args:
            md: Raw markdown text
            
        Returns:
            Normalized markdown text
        """
        if not md:
            return ""
        
        # 1) decode HTML entities (&#x3C; -> <)
        md = html.unescape(md)

        # 2) fix headings missing a space: '#Title' -> '# Title'
        md = self.HEADING_RE.sub(lambda m: f"{m.group(1)} {m.group(2)}", md)

        # 3) collapse >2 blank lines to max 2
        md = re.sub(r"\n{3,}", "\n\n", md)

        # 4) ensure a blank line BEFORE headings
        md = re.sub(r"(?m)([^\n])\n(#{1,6}\s)", r"\1\n\n\2", md)

        # 5) ensure a blank line AFTER headings (if next line is not blank)
        md = re.sub(r"(?m)^(#{1,6}\s.+)\n(?!\n|\||-{3,})", r"\1\n\n", md)

        # 6) ensure blank lines around tables (pipe-style)
        lines = md.splitlines()
        out = []
        for i, line in enumerate(lines):
            # insert blank line before a table row if previous non-empty line isn't blank or header
            if self.TABLE_ROW_RE.match(line):
                if out and out[-1].strip() and not self.TABLE_ROW_RE.match(out[-1]) and not self.TABLE_SEP_RE.match(out[-1]):
                    out.append("")  # blank line before table
            out.append(line)
        md = "\n".join(out)

        # 7) demote standalone page banners like "# 468" to smaller heading or italic line
        md = re.sub(r"(?m)^#\s+(\d{1,5})\s*$", r"### Page \1", md)

        # 8) strip stray trailing spaces
        md = re.sub(r"[ \t]+(\n)", r"\1", md)

        return md.strip() + "\n"
    
    def safe_anchor(self, s: str) -> str:
        """
        Create a safe anchor string for markdown links.
        
        Args:
            s: Input string
            
        Returns:
            Safe anchor string
        """
        s = re.sub(r"[^\w\s-]", "", s).strip().lower()
        s = re.sub(r"\s+", "-", s)
        return s[:80] or "chunk"
    
    def generate_output(self, docs: List[Any], output_path: Path) -> Path:
        """
        Generate output file from parsed documents (implements abstract method).
        For LlamaParse, this generates markdown with table of contents.
        
        Args:
            docs: List of parsed document objects
            output_path: Path for the output file
            
        Returns:
            Path to the created output file
        """
        return self.generate_markdown_output(docs, output_path)
    
    def generate_markdown_output(self, docs: List[Any], output_path: Path) -> Path:
        """
        Generate a single markdown file from parsed documents with table of contents.
        
        Args:
            docs: List of parsed document objects
            output_path: Path for the output markdown file
            
        Returns:
            Path to the created markdown file
        """
        toc = ["# Parsed Output", "## Table of Contents"]
        parts = []

        for i, d in enumerate(docs):
            raw = (getattr(d, "text", "") or "").strip()
            if not raw:
                continue
            
            md = self.normalize_markdown(raw)

            # build a visible section header so chunk boundaries are obvious
            meta = dict(getattr(d, "metadata", {}) or {})
            page = meta.get("page_number") or meta.get("page") or "n/a"
            
            # use first heading in the chunk as title, else "Chunk i"
            m = re.search(r"(?m)^\s*#{1,6}\s+(.+)$", md)
            title = m.group(1).strip() if m else f"Chunk {i}"
            anchor = f"chunk-{i}-{self.safe_anchor(title)}"

            toc.append(f"- [Chunk {i} — p. {page}: {title}](#{anchor})")
            parts.append(
                f"\n\n---\n\n<a id='{anchor}'></a>\n\n## Chunk {i} — Page {page}\n\n{md}"
            )

        final_md = "\n".join(toc) + "\n" + "".join(parts)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(final_md, encoding="utf-8")
        
        return output_path
    
    def process_pdf_slice(
        self,
        pdf_path: Path,
        start_page: int,
        end_page: int,
        output_dir: Optional[Path] = None,
        slice_filename: Optional[str] = None,
        markdown_filename: Optional[str] = None,
        verbose: bool = True
    ) -> Dict[str, Path]:
        """
        Complete workflow: slice PDF, parse with LlamaParse, and generate markdown.
        
        Args:
            pdf_path: Path to the source PDF
            start_page: Starting page number (1-indexed)
            end_page: Ending page number (1-indexed, inclusive)
            output_dir: Directory for output files. If None, uses root_path/data/processed
            slice_filename: Name for the sliced PDF. If None, auto-generates
            markdown_filename: Name for the markdown output. If None, auto-generates
            verbose: Whether to print progress messages
            
        Returns:
            Dictionary with paths to created files: {'slice': Path, 'markdown': Path}
        """
        # Set default output directory
        if output_dir is None:
            output_dir = self.root_path / "data" / "processed"
        
        # Generate filenames if not provided
        if slice_filename is None:
            slice_filename = f"{pdf_path.stem}_p{start_page}_{end_page}.pdf"
        if markdown_filename is None:
            markdown_filename = f"{pdf_path.stem}_p{start_page}_{end_page}.md"
        
        slice_path = output_dir / slice_filename
        markdown_path = output_dir / markdown_filename
        
        # Step 1: Slice PDF
        if verbose:
            print(f"Slicing PDF pages {start_page}-{end_page} from {pdf_path}")
        self.slice_pdf(pdf_path, start_page, end_page, slice_path)
        
        # Step 2: Parse with LlamaParse
        docs = self.parse_document(slice_path, verbose=verbose)
        
        # Step 3: Generate markdown output
        if docs:
            self.generate_markdown_output(docs, markdown_path)
            if verbose:
                print(f"Saved markdown output → {markdown_path}")
        else:
            if verbose:
                print("No documents returned from parsing - skipping markdown generation")
        
        return {
            'slice': slice_path,
            'markdown': markdown_path if docs else None
        }

