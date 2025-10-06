import os
import json
import re
import ollama
from pathlib import Path
from typing import Optional, List, Dict, Any
from unstructured.partition.pdf import partition_pdf
from .base_parser import ParserProcessor


class UnstructuredParserProcessor(ParserProcessor):
    """
    A class for processing PDF documents using Unstructured with configurable options
    and content extraction capabilities.
    """
    
    def __init__(
        self,
        strategy: str = "hi_res",
        infer_table_structure: bool = True,
        extract_image_block_types: Optional[List[str]] = None,
        extract_image_block_to_payload: bool = True,
        chunking_strategy: str = "by_title",
        max_characters: int = 10000,
        combine_text_under_n_chars: int = 200,
        new_after_n_chars: int = 6000,
        root_path: Optional[Path] = None
    ):
        """
        Initialize the UnstructuredParserProcessor.
        
        Args:
            strategy: Parsing strategy ("hi_res", "fast", "ocr_only", "auto")
            infer_table_structure: Whether to infer table structure
            extract_image_block_types: List of image block types to extract
            extract_image_block_to_payload: Whether to extract image blocks to payload
            chunking_strategy: Strategy for chunking ("by_title", "by_page", etc.)
            max_characters: Maximum characters per chunk
            combine_text_under_n_chars: Combine text elements under this character count
            new_after_n_chars: Create new chunk after this character count
            root_path: Root directory for file operations
        """
        # Initialize parent class
        super().__init__(root_path)
        
        # Set configuration attributes
        self.strategy = strategy
        self.infer_table_structure = infer_table_structure
        self.extract_image_block_types = extract_image_block_types or ["Image"]
        self.extract_image_block_to_payload = extract_image_block_to_payload
        self.chunking_strategy = chunking_strategy
        self.max_characters = max_characters
        self.combine_text_under_n_chars = combine_text_under_n_chars
        self.new_after_n_chars = new_after_n_chars
    
    def obtain_image_summary(self, image_base64: str, mime_type: str = "image/png", model: str = "gemma3:4b") -> str:
        """
        Generate a summary of an image using Ollama's vision model.
        
        Args:
            image_base64: Base64 encoded image data
            mime_type: MIME type of the image
            
        Returns:
            Text summary of the image content
        """
        try:
            prompt = """Extract the information related to charts, mention of risks (For example a risk vector with the risk value highlighted), tables from this image. If there is no relevant information, return 'No relevant information'."""
            # Use ollama library to generate image description
            response = ollama.generate(
                model=model,
                prompt=prompt,
                images=[image_base64]
            )
            
            return response.get("response", "[Image description unavailable]")
            
        except Exception as e:
            return f"[Image analysis error: {str(e)}]"
    
    def parse_document(self, file_path: Path, verbose: bool = True) -> List[Any]:
        """
        Parse a document using Unstructured.
        
        Args:
            file_path: Path to the document to parse
            verbose: Whether to print progress messages
            
        Returns:
            List of parsed document chunks
            
        Raises:
            Exception: If parsing fails
        """
        try:
            if verbose:
                print(f"[unstructured] parsing document: {file_path}")
            
            chunks = partition_pdf(
                filename=str(file_path),
                infer_table_structure=self.infer_table_structure,
                strategy=self.strategy,
                extract_image_block_types=self.extract_image_block_types,
                extract_image_block_to_payload=self.extract_image_block_to_payload,
                chunking_strategy=self.chunking_strategy,
                max_characters=self.max_characters,
                combine_text_under_n_chars=self.combine_text_under_n_chars,
                new_after_n_chars=self.new_after_n_chars,
            )
            
            if verbose:
                print(f"[unstructured] extracted {len(chunks)} elements")
            
            # Show breakdown of element types
            if verbose:
                element_types = {}
                for chunk in chunks:
                    chunk_type = type(chunk).__name__
                    element_types[chunk_type] = element_types.get(chunk_type, 0) + 1
                
                if element_types:
                    print(f"[unstructured] element type breakdown:")
                    for elem_type, count in sorted(element_types.items()):
                        print(f"  - {elem_type}: {count}")
            
            return chunks
            
        except Exception as e:
            if verbose:
                print(f"[unstructured] exception: {repr(e)}")
            raise
    
    def normalize_markdown(self, md: str) -> str:
        """
        Normalize markdown text with basic formatting.
        
        Args:
            md: Raw markdown text
            
        Returns:
            Normalized markdown text
        """
        if not md:
            return ""
        
        # Basic markdown normalization for unstructured content
        # 1) collapse multiple blank lines
        md = re.sub(r"\n{3,}", "\n\n", md)
        
        # 2) ensure proper heading spacing
        md = re.sub(r"(?m)([^\n])\n(#{1,6}\s)", r"\1\n\n\2", md)
        md = re.sub(r"(?m)^(#{1,6}\s.+)\n(?!\n)", r"\1\n\n", md)
        
        # 3) strip trailing spaces
        md = re.sub(r"[ \t]+(\n)", r"\1", md)
        
        return md.strip() + "\n"
    
    def generate_output(self, docs: List[Any], output_path: Path, source_document: str = None) -> Path:
        """
        Generate output file from parsed documents (implements abstract method).
        For Unstructured, this generates a JSON file with preprocessed chunks.
        
        Args:
            docs: List of parsed document objects
            output_path: Path for the output file
            source_document: Name of the source PDF document
            
        Returns:
            Path to the created output file
        """
        # Extract source document name from output_path if not provided
        if source_document is None:
            source_document = output_path.stem + ".pdf"
        
        # Preprocess chunks to group by chunk_id and separate images
        processed_content = self.preprocess_chunks(docs, verbose=True, document_path=output_path)
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(processed_content, f, indent=2, ensure_ascii=False)
        
        return output_path
    
    def preprocess_chunks(self, chunks: List[Any], verbose: bool = False, document_path = None, model = "gemma3:4b") -> Dict[str, Any]:
        """
        Preprocess chunks by grouping text elements from Title to Title and creating separate image chunks.
        
        Args:
            chunks: List of elements from unstructured partition_pdf
            verbose: Whether to print progress messages
            document_path: Path to the source document
            
        Returns:
            dict: Dictionary containing Title-based text chunks and separate image chunks
        """
        # Collect all elements first
        all_elements = []
        image_chunks = []
        if document_path:
            if hasattr(document_path, 'stem'):
                # It's a Path object
                source_document = document_path.stem + ".pdf"
            else:
                # It's a string path
                from pathlib import Path
                source_document = Path(document_path).stem + ".pdf"
        else:
            source_document = None
        
        for chunk_idx, chunk in enumerate(chunks):
            chunk_page = getattr(chunk.metadata, "page_number", None) if hasattr(chunk, 'metadata') else None
            
            # Check if chunk has orig_elements in metadata
            if hasattr(chunk, 'metadata') and hasattr(chunk.metadata, 'orig_elements'):
                orig_elements = chunk.metadata.orig_elements
                
                for elem_idx, element in enumerate(orig_elements):
                    element_type = type(element).__name__
                    element_text = getattr(element, "text", "").strip()
                    element_page = getattr(element.metadata, "page_number", chunk_page) if hasattr(element, 'metadata') else chunk_page
                    
                    if element_type == "Image" or element_type == "FigureCaption":
                        # Create separate image chunk
                        image_info = {
                            "chunk_id": f"image_{chunk_idx}_{elem_idx}",
                            "original_chunk_id": chunk_idx,
                            "page": element_page,
                            "type": "image",
                            "text": element_text,
                            "source_document": source_document,
                            "section_title": "Image Content"  # Will be updated if we can find related title
                        }
                        
                        # Add image data if available
                        if hasattr(element, 'metadata') and hasattr(element.metadata, 'image_base64'):
                            image_info["image_data"] = element.metadata.image_base64
                        if hasattr(element, 'metadata') and hasattr(element.metadata, 'image_mime_type'):
                            image_info["mime_type"] = element.metadata.image_mime_type
                            
                        # Process image with AI summarization
                        if "image_data" in image_info:
                            if verbose:
                                print(f"🖼️  Processing image in chunk {chunk_idx} on page {element_page}")
                            
                            image_summary = self.obtain_image_summary(
                                image_info["image_data"], 
                                image_info.get("mime_type", "image/png"), model
                            )
                            
                            image_info["ai_summary"] = image_summary
                            image_info["text"] = f"[IMAGE: {image_summary}]"
                            
                            if verbose:
                                print(f"   ✓ Generated summary: {image_summary[:100]}...")
                        
                        image_chunks.append(image_info)
                        
                    else:
                        # Add to all elements list for Title-based grouping
                        if element_text and self._should_include_element(element_type, element_text):
                            all_elements.append({
                                "element_type": element_type,
                                "text": element_text,
                                "page": element_page,
                                "original_chunk_id": chunk_idx,
                                "element_id": f"chunk-{chunk_idx}-elem-{elem_idx}"
                            })
            else:
                # Fallback: treat the chunk itself as a text element
                chunk_category = getattr(chunk, "category", "")
                chunk_text = getattr(chunk, "text", "").strip()
                
                if chunk_text:
                    all_elements.append({
                        "element_type": chunk_category,
                        "text": chunk_text,
                        "page": chunk_page,
                        "original_chunk_id": chunk_idx,
                        "element_id": f"chunk-{chunk_idx}"
                    })
        
        # Group elements by Title boundaries
        title_based_chunks = []
        current_chunk = None
        current_title = None
        chunk_counter = 0
        
        for element in all_elements:
            if element["element_type"] == "Title":
                # Start new chunk when we encounter a Title
                if current_chunk is not None:
                    # Finalize previous chunk with proper text combination
                    current_chunk["combined_text"] = self._combine_chunk_text(current_chunk["text_elements"])
                    title_based_chunks.append(current_chunk)
                
                # Store the current title for subsequent chunks
                current_title = element["text"]
                
                # Start new chunk with this Title
                current_chunk = {
                    "chunk_id": f"title_chunk_{chunk_counter}",
                    "page": element["page"],
                    "text_elements": [element],
                    "combined_text": "",
                    "section_title": current_title,
                    "source_document": source_document
                }
                chunk_counter += 1
            else:
                # Add to current chunk if exists, otherwise create first chunk
                if current_chunk is None:
                    current_chunk = {
                        "chunk_id": f"title_chunk_{chunk_counter}",
                        "page": element["page"],
                        "text_elements": [],
                        "combined_text": "",
                        "section_title": current_title or "No Title",
                        "source_document": source_document
                    }
                    chunk_counter += 1
                
                # Check if this is a FigureCaption that might need AI summary
                if element["element_type"] == "FigureCaption":
                    element = self._enhance_figure_caption(element, image_chunks, verbose)
                
                current_chunk["text_elements"].append(element)
        
        # Don't forget the last chunk
        if current_chunk is not None:
            current_chunk["combined_text"] = self._combine_chunk_text(current_chunk["text_elements"])
            title_based_chunks.append(current_chunk)
        
        if verbose:
            print(f"📊 Title-based preprocessing results:")
            print(f"   Title-based chunks: {len(title_based_chunks)}")
            print(f"   Image chunks: {len(image_chunks)}")
            for i, chunk in enumerate(title_based_chunks):
                title_elem = next((elem for elem in chunk["text_elements"] if elem["element_type"] == "Title"), None)
                title_text = title_elem["text"][:50] + "..." if title_elem and len(title_elem["text"]) > 50 else title_elem["text"] if title_elem else "No title"
                print(f"     Chunk {i}: {title_text} ({len(chunk['text_elements'])} elements)")
        
        return {
            "text_chunks": title_based_chunks,
            "image_chunks": image_chunks,
            "summary": {
                "text_chunks_count": len(title_based_chunks),
                "image_chunks_count": len(image_chunks),
                "total_chunks": len(title_based_chunks) + len(image_chunks)
            }
        }
    
    def _should_include_element(self, element_type: str, element_text: str) -> bool:
        """
        Determine if an element should be included in the final output.
        
        Args:
            element_type: Type of the element
            element_text: Text content of the element
            
        Returns:
            bool: True if element should be included
        """
        # Filter out standalone numbers or very short text elements
        if element_type == "Text":
            # Skip if it's just a number or very short non-meaningful text
            if element_text.strip().isdigit() or len(element_text.strip()) <= 2:
                return False
        
        # Skip empty or whitespace-only elements
        if not element_text.strip():
            return False
            
        return True
    
    def _combine_chunk_text(self, text_elements: List[Dict]) -> str:
        """
        Combine text elements of each chunk into a single string with proper paragraph separation.
        
        Args:
            text_elements: List of text element dictionaries
            
        Returns:
            str: Combined text with proper formatting
        """
        if not text_elements:
            return ""
        
        combined_parts = []
        
        for elem in text_elements:
            text = elem["text"].strip()
            if text:
                element_type = elem["element_type"]
                
                # Handle different element types with appropriate formatting
                if element_type == "Title":
                    # Titles should be on new lines and prominent
                    if combined_parts:
                        combined_parts.append("\n" + text)
                    else:
                        combined_parts.append(text)
                elif element_type in ["Header", "Footer"]:
                    # Headers and footers on new lines
                    if combined_parts:
                        combined_parts.append("\n" + text)
                    else:
                        combined_parts.append(text)
                elif element_type == "ListItem":
                    # List items should be properly formatted
                    if combined_parts:
                        combined_parts.append(" " + text)
                    else:
                        combined_parts.append(text)
                elif element_type == "NarrativeText":
                    # Check if this might be a list item based on content
                    if self._looks_like_list_item(text):
                        if combined_parts:
                            combined_parts.append(" " + text)
                        else:
                            combined_parts.append(text)
                    else:
                        # Regular narrative text with paragraph breaks
                        if combined_parts:
                            combined_parts.append("\n" + text)
                        else:
                            combined_parts.append(text)
                else:
                    # Default handling for other element types
                    if combined_parts:
                        combined_parts.append(" " + text)
                    else:
                        combined_parts.append(text)
        
        # Join and clean up the text
        result = "".join(combined_parts)
        # Clean up multiple newlines and extra spaces
        result = re.sub(r'\n\s*\n', '\n', result)
        result = re.sub(r' +', ' ', result)
        return result.strip()
    
    def _looks_like_list_item(self, text: str) -> bool:
        """
        Determine if text looks like a list item (single word/short phrase).
        
        Args:
            text: Text to check
            
        Returns:
            bool: True if text looks like a list item
        """
        # Simple heuristic: if it's a single word or very short phrase, might be a list item
        words = text.strip().split()
        return len(words) <= 3 and len(text.strip()) <= 50
    
    def _enhance_figure_caption(self, caption_element: Dict, image_chunks: List[Dict], verbose: bool = False) -> Dict:
        """
        Enhance FigureCaption elements by adding AI summaries from related images.
        
        Args:
            caption_element: The FigureCaption element dictionary
            image_chunks: List of image chunks with AI summaries
            verbose: Whether to print progress messages
            
        Returns:
            dict: Enhanced caption element with AI summary if available
        """
        caption_text = caption_element["text"]
        page = caption_element["page"]
        
        # Check if caption already contains AI descriptions to avoid duplication
        if "[AI Description" in caption_text or "[Image Description" in caption_text:
            if verbose:
                print(f"   ⚠️ FigureCaption on page {page} already contains AI descriptions, skipping enhancement")
            return caption_element
        
        # Look for related images on the same page or adjacent pages
        related_images = []
        for img_chunk in image_chunks:
            img_page = img_chunk.get("page")
            if img_page and abs(img_page - page) <= 1:  # Same page or adjacent
                if "ai_summary" in img_chunk:
                    related_images.append(img_chunk["ai_summary"])
        
        # If we found related images, enhance the caption
        if related_images:
            enhanced_text = caption_text
            
            # Add AI descriptions in a cleaner format
            if len(related_images) == 1:
                enhanced_text += f" [AI Description 1: {related_images[0]}]"
            else:
                for i, summary in enumerate(related_images, 1):
                    enhanced_text += f" [AI Description {i}: {summary}]"
            
            if verbose:
                print(f"   ✓ Enhanced FigureCaption on page {page} with {len(related_images)} AI description(s)")
            
            # Create enhanced element
            enhanced_element = caption_element.copy()
            enhanced_element["text"] = enhanced_text
            enhanced_element["original_text"] = caption_text
            enhanced_element["ai_enhanced"] = True
            return enhanced_element
        
        return caption_element

    def get_image_chunks(self, chunks: List[Any], verbose: bool = False) -> List[Dict[str, Any]]:
        """
        Extract only the image chunks from parsed document chunks without AI processing.
        
        Args:
            chunks: List of elements from unstructured partition_pdf
            verbose: Whether to print progress messages
            
        Returns:
            List of image chunk dictionaries with raw image data
        """
        image_chunks = []
        
        if verbose:
            print("🔍 Extracting image chunks...")
        
        for chunk_idx, chunk in enumerate(chunks):
            chunk_page = getattr(chunk.metadata, "page_number", None) if hasattr(chunk, 'metadata') else None
            
            # Check if chunk has orig_elements in metadata
            if hasattr(chunk, 'metadata') and hasattr(chunk.metadata, 'orig_elements'):
                orig_elements = chunk.metadata.orig_elements
                
                for elem_idx, element in enumerate(orig_elements):
                    element_type = type(element).__name__
                    
                    if element_type == "Image":
                        element_text = getattr(element, "text", "").strip()
                        element_page = getattr(element.metadata, "page_number", chunk_page) if hasattr(element, 'metadata') else chunk_page
                        
                        # Create image chunk
                        image_info = {
                            "chunk_id": f"image_{chunk_idx}_{elem_idx}",
                            "original_chunk_id": chunk_idx,
                            "page": element_page,
                            "type": "image",
                            "text": element_text,
                            "section_title": "Image Content"
                        }
                        
                        # Add image data if available
                        if hasattr(element, 'metadata') and hasattr(element.metadata, 'image_base64'):
                            image_info["image_data"] = element.metadata.image_base64
                        if hasattr(element, 'metadata') and hasattr(element.metadata, 'image_mime_type'):
                            image_info["mime_type"] = element.metadata.image_mime_type
                            
                        if verbose and "image_data" in image_info:
                            print(f"🖼️  Found image {elem_idx} in chunk {chunk_idx} on page {element_page}")
                        
                        image_chunks.append(image_info)
        
        if verbose:
            print(f"📊 Found {len(image_chunks)} image chunks")
        
        return image_chunks

    def extract_content_by_type(self, chunks: List[Any], process_images: bool = True, verbose: bool = False) -> Dict[str, Any]:
        """
        Iterate through chunks and their orig_elements to separate text, images, and tables.
        Optionally processes images with AI summarization.
        
        Args:
            chunks: List of elements from unstructured partition_pdf
            process_images: Whether to generate AI summaries for images
            verbose: Whether to print progress messages
            
        Returns:
            dict: Dictionary containing separated content types
        """
        text_elements = []
        images = []
        tables = []
        
        for chunk_idx, chunk in enumerate(chunks):
            chunk_page = getattr(chunk.metadata, "page_number", None) if hasattr(chunk, 'metadata') else None
            
            # Check if chunk has orig_elements in metadata
            if hasattr(chunk, 'metadata') and hasattr(chunk.metadata, 'orig_elements'):
                orig_elements = chunk.metadata.orig_elements
                
                for elem_idx, element in enumerate(orig_elements):
                    element_id = f"chunk-{chunk_idx}-elem-{elem_idx}"
                    element_type = type(element).__name__
                    element_text = getattr(element, "text", "").strip()
                    element_page = getattr(element.metadata, "page_number", chunk_page) if hasattr(element, 'metadata') else chunk_page
                    
                    base_info = {
                        "id": element_id,
                        "chunk_id": chunk_idx,
                        "element_type": element_type,
                        "page": element_page,
                        "text": element_text
                    }
                    
                    # Categorize based on element type
                    if element_type == "Table":
                        table_info = base_info.copy()
                        # Extract table-specific data
                        if hasattr(element, 'metadata') and hasattr(element.metadata, 'text_as_html'):
                            table_info["html"] = element.metadata.text_as_html
                        if hasattr(element, 'metadata') and hasattr(element.metadata, 'table_as_cells'):
                            table_info["cells"] = element.metadata.table_as_cells
                        tables.append(table_info)
                        
                    elif element_type == "Image":
                        image_info = base_info.copy()
                        # Extract image-specific data
                        if hasattr(element, 'metadata') and hasattr(element.metadata, 'image_base64'):
                            image_info["image_data"] = element.metadata.image_base64
                        if hasattr(element, 'metadata') and hasattr(element.metadata, 'image_mime_type'):
                            image_info["mime_type"] = element.metadata.image_mime_type
                        
                        # Process image with AI summarization if requested
                        if process_images and "image_data" in image_info:
                            if verbose:
                                print(f"🖼️  Processing image {image_info['id']} on page {image_info.get('page', 'unknown')}")
                            
                            image_summary = self.obtain_image_summary(
                                image_info["image_data"], 
                                image_info.get("mime_type", "image/png")
                            )
                            
                            image_info["ai_summary"] = image_summary
                            image_info["text"] = f"[IMAGE: {image_summary}]"  # Replace text with summary
                            
                            if verbose:
                                print(f"   ✓ Generated summary: {image_summary[:100]}...")
                        
                        images.append(image_info)
                        
                    else:
                        # Text elements (Title, NarrativeText, Formula, Footer, etc.)
                        if element_text:  # Only add non-empty text elements
                            text_elements.append(base_info)
            else:
                # Fallback: if no orig_elements, treat the chunk itself as a text element
                chunk_id = f"chunk-{chunk_idx}"
                chunk_category = getattr(chunk, "category", "")
                chunk_text = getattr(chunk, "text", "").strip()
                
                if chunk_text:
                    fallback_info = {
                        "id": chunk_id,
                        "chunk_id": chunk_idx,
                        "element_type": chunk_category,
                        "page": chunk_page,
                        "text": chunk_text
                    }
                    text_elements.append(fallback_info)
        
        return {
            "text_elements": text_elements,
            "images": images,
            "tables": tables,
            "summary": {
                "total_chunks": len(chunks),
                "text_elements_count": len(text_elements),
                "images_count": len(images),
                "tables_count": len(tables)
            }
        }
