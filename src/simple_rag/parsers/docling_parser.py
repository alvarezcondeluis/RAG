import os
import json
import re
import ollama
import base64
from pathlib import Path
from typing import Optional, List, Dict, Any
from io import BytesIO
from PIL import Image
from tqdm import tqdm
    
import logging
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
import warnings
from docling.datamodel.base_models import InputFormat
from docling_core.types.doc import ImageRefMode
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode, EasyOcrOptions, TesseractOcrOptions, OcrMacOptions
from docling.datamodel.settings import settings
from .base_parser import ParserProcessor


class DoclingParserProcessor(ParserProcessor):
    """
    A class for processing PDF documents using Docling with configurable options
    and content extraction capabilities.
    """
    
    def __init__(
        self,
        do_table_structure: bool = True,
        table_structure_mode: TableFormerMode = TableFormerMode.ACCURATE,
        do_cell_matching: bool = True,
        generate_page_images: bool = True,
        generate_picture_images: bool = True,
        images_scale: float = 2.0,
        root_path: Optional[Path] = None
    ):
        """
        Initialize the DoclingParserProcessor.
        
        Args:
            do_table_structure: Whether to extract table structure
            table_structure_mode: TableFormer mode (FAST or ACCURATE)
            do_cell_matching: Use text cells predicted from table structure model
            generate_page_images: Enable page image generation
            generate_picture_images: Enable picture image generation
            images_scale: Image resolution scale (scale=1 corresponds to 72 DPI)
            root_path: Root directory for file operations
        """
        # Initialize parent class
        super().__init__(root_path)
        
        # Set configuration attributes
        self.do_table_structure = do_table_structure
        self.table_structure_mode = table_structure_mode
        self.do_cell_matching = do_cell_matching
        self.generate_page_images = generate_page_images
        self.generate_picture_images = generate_picture_images
        self.images_scale = images_scale
        
        # Initialize pipeline options
        self.pipeline_options = PdfPipelineOptions(
            do_table_structure=self.do_table_structure,
            table_structure_options=dict(
                do_cell_matching=self.do_cell_matching,
                mode=self.table_structure_mode
            ),
            generate_page_images=self.generate_page_images,
            generate_picture_images=self.generate_picture_images,
            images_scale=self.images_scale,
        )
        self.corrupted_pdfs = 0
        
        # Initialize the DocumentConverter
        self.doc_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=self.pipeline_options)
            }
        )

    def obtain_image_summary_specific(self, image_data: Image.Image, model: str = "llama3.2-vision:latest", prompt: str = "Analyze and extract briefly the details from the image ") -> str:
        """
        Generate a summary of an image using Ollama's vision model.
        
        Args:
            image_data: PIL Image object
            model: Ollama model to use for image analysis
            
        Returns:
            Text summary of the image content
        """
        try:
            # Convert PIL Image to base64
            buffered = BytesIO()
            image_data.save(buffered, format="PNG")
            image_base64 = base64.b64encode(buffered.getvalue()).decode()
            
           
            # Use ollama library to generate image description
            response = ollama.generate(
                model=model,
                prompt=prompt,
                images=[image_base64]
            )
            
            return response.get("response", "[Image description unavailable]")
            
        except Exception as e:
            return f"[Image analysis error: {str(e)}]"
    
    def obtain_image_summary(self, image_data: Image.Image, model: str = "llama3.2-vision:latest") -> str:
        """
        Generate a summary of an image using Ollama's vision model.
        
        Args:
            image_data: PIL Image object
            model: Ollama model to use for image analysis
            
        Returns:
            Text summary of the image content
        """
        try:
            # Convert PIL Image to base64
            buffered = BytesIO()
            image_data.save(buffered, format="PNG")
            image_base64 = base64.b64encode(buffered.getvalue()).decode()
            
            prompt = """Extract the information of the image with special focus on risk indicators and other relevant information
                        about some ETF and Index fund images and charts. Answer briefly without leaving relevant information.
                        The answer has to be ready to be added to a markdown file. 
                        In the case of the risk one the answer has to be the Risk is ... and the rest of the information found in the next line.    """
            
            # Use ollama library to generate image description
            response = ollama.generate(
                model=model,
                prompt=prompt,
                images=[image_base64]
            )
            
            return response.get("response", "[Image description unavailable]")
            
        except Exception as e:
            return f"[Image analysis error: {str(e)}]"

    def save_markdown(self, markdown_content: str, file_path: Path, verbose: bool = True):
        """
        Save the markdown content to a file.
        
        Args:
            markdown_content: Markdown content to save
            file_path: Path to save the markdown file
            verbose: Whether to print progress messages
        """
        try:
            if verbose:
                print(f"[docling] saving markdown to: {file_path}")
            
            # Save the markdown content to a file
            with open(file_path, "w") as f:
                f.write(markdown_content)
            
        except Exception as e:
            if verbose:
                print(f"[docling] exception: {repr(e)}")
            raise
    
    def parse_document(self, file_path: Path, verbose: bool = True, output_path: Path = None) -> Any:
        """
        Parse a document using Docling.
        
        Args:
            file_path: Path to the document to parse
            verbose: Whether to print progress messages
            
        Returns:
            Docling conversion result
            
        Raises:
            Exception: If parsing fails
        """
        try:
            if verbose:
                print(f"[docling] parsing document: {file_path}")
            
            # Convert the document
            try:
                result = self.doc_converter.convert(str(file_path))
            except Exception as e:
                if verbose:
                    print(f"[docling] exception: {repr(e)}")
                raise
            
            # Initialize text with default markdown export
            text = result.document.export_to_markdown()
            
            # Process based on document type
            if "fund" in str(file_path).lower():
                text = self.process_fund(result, verbose)
            elif "etf" in str(file_path).lower():
                text = self.process_etf(result, verbose)
            
            
            return text 
            
        except Exception as e:
            if verbose:
                print(f"[docling] exception: {repr(e)}")
            raise
 
    def process_etf(self, result: Any, verbose: bool = True):
        """
        Process an ETF document.
        
        Args:
            result: Docling conversion result
            verbose: Whether to print progress messages
        """
        try:
            if verbose:
                print(f"[docling] processing ETF: {result}")
            
            # Process the ETF
            picture_count = 0
            table_count = 0

            images = []
            for element, _level in result.document.iterate_items():
                if hasattr(element, 'get_image'):
                    if 'picture' in str(type(element)).lower():
                        picture_count += 1
                        images.append(element)
                    elif 'table' in str(type(element)).lower():
                        table_count += 1
            

            summary = self.obtain_image_summary_specific(images[2], prompt= "Analyze and extract briefly the details from the image and add it to a markdown file")
            
            text = result.document.export_to_markdown()
            image_tag = '<!-- image -->'

            parts = text.split(image_tag)
            target_parts = [2]
            result_parts = []
            for i, part in enumerate(parts):
                if i in target_parts:
                    result_parts.append(part +image_tag + "\n\n" + summary)
                else:
                    result_parts.append(part) 
           
            text = "\n\n".join(result_parts)

            
            return text 
        except Exception as e:
            if verbose:
                print(f"[docling] exception: {repr(e)}")
            raise

    def process_fund(self, result: Any, verbose: bool = True):
        """
        Process an Fund document.
        
        Args:
            result: Docling conversion result
            verbose: Whether to print progress messages
        """
        try:
            if verbose:
                print(f"[docling] processing Fund: {result}")
            
            # Process the Fund
            picture_count = 0
            table_count = 0

            images = []
            for element, _level in result.document.iterate_items():
                if hasattr(element, 'get_image'):
                    if 'picture' in str(type(element)).lower():
                        picture_count += 1
                        images.append(element)
                    elif 'table' in str(type(element)).lower():
                        table_count += 1
                        if table_count == 1:
                            images.append(element)
            
            summaries = []
            for image in [images[0], images[1], images[4]]:
                summaries.append(self.obtain_image_summary(image.get_image(result.document)))

            text = result.document.export_to_markdown()
            
            image_tag = '<!-- image -->'

            parts = text.split(image_tag)
            target_parts = [1,3,5]
            result_parts = []
            for i, part in enumerate(parts):
                if i in target_parts:
                    result_parts.append(part +image_tag + "\n\n" + summaries.pop(0))
                else:
                    result_parts.append(part)


            
            text = "\n\n".join(result_parts)

            
            return text 
        except Exception as e:
            if verbose:
                print(f"[docling] exception: {repr(e)}")
            raise 
        


    def process_factsheet_folder(
        self, 
        input_folder: Path, 
        output_folder: Path, 
        verbose: bool = True,
        suppress_warnings: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Process all documents containing 'factSheet' in their name from a folder.
        
        Args:
            input_folder: Path to the folder containing documents
            output_folder: Path to the folder where outputs will be saved
            verbose: Whether to print progress messages
            suppress_warnings: Whether to suppress docling's internal warnings/errors
            
        Returns:
            List of dictionaries containing processing results for each document
        """
        # Suppress docling's verbose logging if requested
        if suppress_warnings:
            # Save original log levels
            docling_logger = logging.getLogger('docling')
            pypdfium_logger = logging.getLogger('pypdfium2')
            original_docling_level = docling_logger.level
            original_pypdfium_level = pypdfium_logger.level
            
            # Set to only show critical errors
            docling_logger.setLevel(logging.CRITICAL)
            pypdfium_logger.setLevel(logging.CRITICAL)
            
            # Suppress warnings
            warnings.filterwarnings('ignore')
        
        try:
            # Ensure input folder exists
            input_folder = Path(input_folder)
            if not input_folder.exists():
                raise ValueError(f"Input folder does not exist: {input_folder}")
            
            # Ensure output folder exists
            output_folder = Path(output_folder)
            output_folder.mkdir(parents=True, exist_ok=True)
            
            # Find all files with 'factSheet' in the name
            factsheet_files = []
            for file_path in input_folder.rglob("*"):
                if file_path.is_file() and "factSheet" in file_path.name:
                    factsheet_files.append(file_path)
            
            if verbose:
                print(f"[docling] Found {len(factsheet_files)} factSheet documents in {input_folder}")
            
            # Process each document
            results = []
            corrupted_count = 0
            
            for file_path in tqdm(factsheet_files, desc="Processing factSheets", disable=not verbose):
                try:
                    # Parse the document
                    parsed_result = self.parse_document(file_path, verbose=False)
                    
                    # Create output file path with same name in output folder
                    output_file = output_folder / file_path.name
                    output_file = output_file.with_suffix('.md')
                    
                    # Save the result
                    if isinstance(parsed_result, str):
                        # Ensure output directory exists
                        output_file.parent.mkdir(parents=True, exist_ok=True)
                        # Delete existing file if it exists
                        if output_file.exists():
                            output_file.unlink()
                        # Write the new file
                        with open(output_file, 'w', encoding='utf-8') as f:
                            f.write(parsed_result)
                    
                    results.append({
                        "source_file": str(file_path),
                        "output_file": str(output_file),
                        "status": "success",
                        "file_name": file_path.name
                    })
                    
                except Exception as e:
                    error_msg = str(e)
                    
                    # Categorize the error
                    if "not valid" in error_msg.lower() or "data format error" in error_msg.lower():
                        corrupted_count += 1
                        error_type = "corrupted_pdf"
                    elif "list index out of range" in error_msg.lower():
                        error_type = "parsing_error"
                    else:
                        error_type = "unknown_error"
                    
                    results.append({
                        "source_file": str(file_path),
                        "output_file": None,
                        "status": "failed",
                        "error": error_msg,
                        "error_type": error_type,
                        "file_name": file_path.name
                    })
                    
                    if verbose:
                        print(f"[docling] ✗ Error processing {file_path.name}: {error_type}")
                        print(error_msg)
                        tqdm.write(f"[docling] ✗ Error processing {file_path.name}: {error_type}")
            
            # Print summary
            if verbose:
                successful = sum(1 for r in results if r["status"] == "success")
                failed = sum(1 for r in results if r["status"] == "failed")
                print(f"\n[docling] Processing complete:")
                print(f"  - Total files: {len(results)}")
                print(f"  - Successful: {successful}")
                print(f"  - Failed: {failed}")
                print(f"  - Corrupted PDFs: {corrupted_count}")
                
                # Show breakdown of error types
                if failed > 0:
                    error_types = {}
                    for r in results:
                        if r["status"] == "failed":
                            error_type = r.get("error_type", "unknown")
                            error_types[error_type] = error_types.get(error_type, 0) + 1
                    
                    print(f"\n  Error breakdown:")
                    for error_type, count in error_types.items():
                        print(f"    - {error_type}: {count}")
            
            return results
        
        finally:
            # Restore original logging levels
            if suppress_warnings:
                docling_logger.setLevel(original_docling_level)
                pypdfium_logger.setLevel(original_pypdfium_level)
                warnings.filterwarnings('default')
    
    def normalize_markdown(self, md: str) -> str:
        """
        Normalize markdown text with consistent formatting.
        
        Args:
            md: Raw markdown text
            
        Returns:
            Normalized markdown text
        """
        pass
    
    def generate_output(self, docs: List[Any], output_path: Path) -> Path:
        """
        Generate output file from parsed documents.
        
        Args:
            docs: List of parsed document objects
            output_path: Path for the output file
            
        Returns:
            Path to the created output file
        """
        pass

