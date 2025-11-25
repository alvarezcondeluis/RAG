import os
import ollama
import base64
from pathlib import Path
from typing import Optional, List, Dict, Any
from io import BytesIO
from PIL import Image
from tqdm import tqdm
import cv2
import numpy as np
from IPython.display import display
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from .base_parser import ParserProcessor
import socket
import subprocess
import time
import shutil

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
        # Initialize the DocumentConverter
        self.doc_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=self.pipeline_options)
            }
        )
    
    def ensure_ollama_is_running(self):
        """
        Checks if the Ollama server is running on port 11434.
        If not, it starts 'ollama serve' as a background subprocess.
        """
        host = '127.0.0.1'
        port = 11434
        
        # Use a with statement for the socket to ensure it's closed automatically
        try:
            with socket.create_connection((host, port), timeout=1):
                # If connection succeeds, the server is already running
                print("✅ Ollama is already running.")
                return
        except (ConnectionRefusedError, socket.timeout):
            # If connection is refused or times out, the server is not running
            print("🟡 Ollama not detected. Starting the server...")
            try:
                # Use Popen to start 'ollama serve' in the background
                # Redirect stdout and stderr to DEVNULL to keep the console clean
                subprocess.Popen(['ollama', 'serve'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                print("🚀 Ollama server started in the background.")
                # Give the server a few seconds to initialize
                time.sleep(3)
            except FileNotFoundError:
                print("❌ Error: 'ollama' command not found.")
                print("Please ensure the Ollama CLI is installed and in your system's PATH.")
                raise
            except Exception as e:
                print(f"❌ An unexpected error occurred while starting Ollama: {e}")
                raise


    def obtain_image_summary_specific(self, image_data: Image.Image, model: str = "gemma3:4b", prompt: str = """Obtain the information of the distribution of the assets into a bulleted list
            The instruction is to only answer with that list""") -> str:
            """
            Generate a summary of an image using Ollama's vision model.
            
            Args:
                image_data: PIL Image object
                model: Ollama model to use for image analysis
                prompt: Prompt to use for image analysis
            Returns:
                Text summary of the image content
            """
            try:

                self.ensure_ollama_is_running()
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


    def save_markdown(self, markdown_content: str, folder: Path, file_name: str, verbose: bool = True):
        """
        Save the markdown content to a file.
        
        Args:
            markdown_content: Markdown content to save
            folder: Path to save the markdown file
            file_name: Name of the file to save (extension will be changed to .md)
            verbose: Whether to print progress messages
        """
        try:
            # Change file extension to .md
            file_path = Path(file_name)
            md_file_name = file_path.stem + ".md"
            
            if verbose:
                print(f"[docling] saving markdown to: {folder}/{md_file_name}")
            
            # Create folder if it doesn't exist
            folder.mkdir(parents=True, exist_ok=True)
            
            # Save the markdown content to a file
            with open(folder / md_file_name, "w") as f:
                f.write(markdown_content)
            
        except Exception as e:
            if verbose:
                print(f"[docling] exception: {repr(e)}")
            raise
    
    def parse_document(self, doc_path: Path):

        IMAGE_RESOLUTION_SCALE = 2.0

        # Define pipeline options for PDF processing
        pipeline_options = PdfPipelineOptions(
            do_table_structure=True,  
            
            
            table_structure_options=dict(
                do_cell_matching=True,  # Use text cells predicted from table structure model
                mode=TableFormerMode.ACCURATE  # Use more accurate TableFormer model
            ),
            generate_page_images=True,  # Enable page image generation
            generate_picture_images=True,  # Enable picture image generation
            images_scale=IMAGE_RESOLUTION_SCALE, # Set image resolution scale (scale=1 corresponds to a standard 72 DPI image)
        
            
        )

        # Initialize the DocumentConverter with the specified pipeline options
        doc_converter_global = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )


        try:
            result = doc_converter_global.convert(doc_path)
        
        except Exception as e:
            print(f"[docling] exception: {repr(e)}")
            raise

        return result
 
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
            summary = ""
            expense_chart_image = None

            images = []
            for element, _level in result.document.iterate_items():
                if hasattr(element, 'get_image'):
                    if 'picture' in str(type(element)).lower():
                        picture_count += 1
                        image = element.get_image(result.document)
                        width,height = image.size
                        if 300 < width < 320 or 240 < height < 260:
                            tqdm.write(f"Processing ETF expense ratio chart: {width}x{height}")
                            display(image)
                            expense_chart_image = image
                            summary = self.obtain_image_summary_specific(image, model="gemma3:4b", prompt="Extract the numerical data from the chart briefly with the title of each of the bars")
                            tqdm.write(f"Expense ratio chart summary: {summary}")
                    elif 'table' in str(type(element)).lower():
                        table_count += 1
                         
            # Lets add the summary of the image into the correct section
            target = "## Expense ratio comparison"
            content = target + "\n\n" + summary 
            text = result.document.export_to_markdown()
            start_index = text.find(target)
            if start_index != -1:

                # Find the start of the *next* section to determine where the current one ends
                end_index = text.find("\n## ", start_index + 1)

                # If a next section exists, combine the text before the target and after the target section
                if end_index != -1:
                    text = text[:start_index] + content + text[end_index:]
                # If the target is the last section in the document
                else:
                    text = text[:start_index] + content
            else:
                # Fallback: If the target section wasn't found, just append the summary
                text += "\n\n" + content
            
            # Embed the expense ratio chart image at the end of the document
            if expense_chart_image:
                # Convert the detected chart image to base64 and embed in markdown
                tqdm.write("Embedding expense ratio chart image in markdown")
                buffered = BytesIO()
                expense_chart_image.save(buffered, format="PNG")
                img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
                
                text += "\n\n## Expense Ratio Chart:\n\n"
                text += f"![Expense Ratio Chart](data:image/png;base64,{img_b64})\n"
            
            return text 
        except Exception as e:
            if verbose:
                print(f"[docling] exception: {repr(e)}")
            raise

    def process_etf_factSheets(self, input_folder: str, output_folder: str, num_files: int = 20):
        """
        Process ETF fact sheets from a folder.
        
        Args:
            input_folder: Path to the folder containing the ETF fact sheets
            output_folder: Path to the folder where the processed fact sheets will be saved
        """
        try:
            # Get the list of files in the input folder
            files = os.listdir(input_folder)
            files = files[:num_files]
            # Process each file
            for file in tqdm(files):
                if file.endswith(".pdf"):
                    # Get the full path of the file
                    file_path = Path(input_folder) / file
                    
                    # Process the file
                    result = self.parse_document(file_path)
                    content = self.process_etf(result, True)
                    
                    # Prepare output paths
                    output_folder_path = Path(output_folder)
                    output_md_path = output_folder_path / f"{file_path.stem}.md"
                    output_pdf_path = output_folder_path / file_path.name
                    output_folder_path.mkdir(parents=True, exist_ok=True)
                    
                    # Save the markdown file
                    with open(output_md_path, 'w') as f:
                        f.write(content)
                    
                    # Copy the original PDF to the output folder
                    shutil.copy2(file_path, output_pdf_path)
                    
                    tqdm.write(f"SUCCESS: Saved {output_md_path.name} and {output_pdf_path.name}")
        except Exception as e:
            
            print(f"[docling] exception: {repr(e)}")
            raise

    def detect_risk_from_image(self, image: Image.Image) -> int | None:
        """
        Analyzes a risk meter image by detecting a specific colored highlight box
        and determining its position on a 1-5 scale.
        
        This method uses computer vision to:
        1. Convert the image to OpenCV BGR format
        2. Isolate pixels matching a specific color 
        3. Find the largest contour (the highlighted risk box)
        4. Calculate the horizontal position of the box center
        5. Map the position to a risk level (1-5)
        
        Args:
            image: PIL Image object containing the risk meter visualization
            
        Returns:
            int: Risk level from 1 (lowest) to 5 (highest)
            None: If no highlight box is detected in the image
            
        Note:
            The method assumes the risk scale is divided into 5 equal horizontal segments.
            The target color is BGR (104, 90, 0) with a tolerance of ±20 for each channel.
        """
        # Convert PIL Image to OpenCV format (which is BGR)
        open_cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        target_bgr_color = np.array([104, 90, 0])
        
        delta = 20
        
        # Calculate the lower and upper bounds for the color range
        lower_bound = np.clip(target_bgr_color - delta, 0, 255)
        upper_bound = np.clip(target_bgr_color + delta, 0, 255)
        
        # Create a mask that isolates only the pixels within our color range
        mask = cv2.inRange(open_cv_image, lower_bound, upper_bound)
        
        # Find contours (shapes) in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            print("DEBUG: No contours found for the specified color.")
            return None # No highlight found

        # Assume the largest contour is our highlighted box
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get the bounding box of the contour
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Calculate the center of the box
        center_x = x + w / 2
        
        # Determine the risk level based on the horizontal position
        image_width = open_cv_image.shape[1]
        # Assuming the scale is divided into 5 equal parts for levels 1-5
        segment_width = image_width / 5
        
        risk_level = int(center_x // segment_width) + 1
        
        # Clamp the value between 1 and 5 to be safe
        return max(1, min(5, risk_level))

    def _process_fund_images(self, result: Any, crop_width_percent: float = 0.15) -> Dict[str, Any]:
        """
        Extract and process images from a fund factsheet document.
        
        This method:
        1. Extracts all pictures and the first table image from the document
        2. Detects the risk level from the first image (cropped)
        3. Generates AI summaries for asset allocation images
        4. Identifies and extracts the growth chart image
        
        Args:
            result: Docling conversion result containing the parsed document
            crop_width_percent: Percentage of image width to crop for risk detection (default: 0.15)
            
        Returns:
            Dictionary containing:
                - 'risk': Detected risk level (1-5)
                - 'summaries': List of AI-generated summaries for asset allocation images
                - 'chart_image': PIL Image of the growth chart (or None if not found)
        """
        images = []
        picture_count = 0
        table_count = 0
        
        # Extract images from document elements
        for element, _level in result.document.iterate_items():
            if hasattr(element, 'get_image'):
                if 'picture' in str(type(element)).lower():
                    picture_count += 1
                    images.append(element.get_image(result.document))
                elif 'table' in str(type(element)).lower():
                    table_count += 1
                    if table_count == 1:
                        images.append(element.get_image(result.document))
        
        summaries = []
        chart_image = None
        risk = None
        counter = 0
        
        print(f"Total images extracted: {len(images)}")
        
        # Process each image
        for i, image in enumerate(images):
            width, height = image.size
            
            if i == 0:
                # First image: detect risk level from cropped left portion
                tqdm.write("Processing risk meter image (cropped)")
                crop_box = (0, 0, int(width * crop_width_percent), height)
                cropped_image = image.crop(crop_box)
                
                risk = self.detect_risk_from_image(cropped_image)
            else:
                tqdm.write(f"Image {i}: Height={height}, Width={width}")
                
                # Asset allocation image (small, specific dimensions)
                if 110 < height < 135 and 390 < width < 420:
                    tqdm.write("Processing asset allocation image")
                    display(image)
                    summary = self.obtain_image_summary_specific(image)
                    summaries.append(summary)
                    tqdm.write(f"Asset allocation summary: {summary}")
                
                # Growth chart image (larger, specific dimensions)
                elif 105 < height < 150 and 690 < width < 720:
                    counter += 1
                    if counter == 1:
                        tqdm.write("Growth chart image detected")
                        
                        chart_image = image
                    else:
                        tqdm.write("WARNING: Multiple chart images detected; using the first match")
                        break
                else:
                    tqdm.write("Other image type detected")
        
        return {
            'risk': risk,
            'summaries': summaries,
            'chart_image': chart_image
        }

    def process_fund_factSheets(self, input_folder: str, output_folder: str, crop_width_percent: float = 0.15, num_files: int = 20):
    
        # --- 1. Configuration ---
        input_path = Path(input_folder)

        # --- 2. Find all relevant documents ---
        all_pdfs = input_path.glob("*.pdf")

        files_to_process = []
        for pdf in all_pdfs:
            if "factsheet" in pdf.name.lower() and "fund" in pdf.name.lower():
                files_to_process.append(pdf)

        files_to_process = files_to_process[:num_files]
        # --- 3. Main Processing Loop ---
        for doc_path in tqdm(files_to_process, desc="Processing Documents"):
        
            try:
                
                tqdm.write("Processing document "+ str(doc_path))
                result = self.parse_document(doc_path)
                
                # Process images and extract information
                image_data = self._process_fund_images(result, crop_width_percent)
                risk = image_data['risk']
                summaries = image_data['summaries']
                chart_image = image_data['chart_image']
                
                text = result.document.export_to_markdown()


                # Define possible target texts to search for
                possible_targets = [
                    "## Largest state concentrations",
                    "## Geographic diversification",
                    "## Asset allocation",
                    "## Market allocation–stocks",
                    "## Market allocation-stocks",
                    "## Sector Diversification",
                    "## Allocation of underlying funds†",
                    "## Allocation of underlying funds"
                    "## Distribution by credit quality",
                    "## Distribution by credit quality†"
                ]
                
                # Find all target texts that exist in the document and match with summaries
                found_targets = []
                for possible_target in possible_targets:
                    if possible_target in text:
                        found_targets.append(possible_target)
                
                # Match found targets with available summaries
                if found_targets and summaries:
                    num_replacements = min(len(found_targets), len(summaries))
                    tqdm.write(f"Found {len(found_targets)} target section(s) and {len(summaries)} summary(ies)")
                    
                    # Perform replacements for each matched pair
                    for i in range(num_replacements):
                        target_text = found_targets[i]
                        summary = summaries[i]
                        replacement_block = f"{target_text}\n\n{summary}"
                        text = text.replace(target_text, replacement_block, 1)  # Replace only first occurrence
                        tqdm.write(f"Replaced '{target_text}' with summary {i+1}")
                    
                    # Warn if there are unmatched summaries or targets
                    if len(summaries) > num_replacements:
                        tqdm.write(f"Warning: {len(summaries) - num_replacements} summary(ies) not used (no matching target section)")
                    if len(found_targets) > num_replacements:
                        tqdm.write(f"Warning: {len(found_targets) - num_replacements} target section(s) not replaced (no matching summary)")
                else:
                    if not found_targets:
                        tqdm.write("Warning: None of the target sections found in document")
                    if not summaries:
                        tqdm.write("Warning: No summaries generated from images")

                # --- Now, append the other information as before ---

                tqdm.write("Number of summaries: " + str(len(summaries)))
                for summary in summaries:
                    tqdm.write(summary)
                text += ("\n\n## Risk Level (1-5): " + str(risk))
                
                if chart_image:
                    # Convert the detected chart image to base64 and embed in markdown
                    tqdm.write("Embedding chart image in markdown")
                    buffered = BytesIO()
                    chart_image.save(buffered, format="PNG")
                    img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
                    
                    text += "\n\n## Growth of a $10,000 investment :\n\n"
                    text += f"![Growth of $10,000](data:image/png;base64,{img_b64})\n"
                
                
                output_folder = Path(output_folder)
                output_md_path = output_folder / f"{doc_path.stem}.md"
                output_pdf_path = output_folder / doc_path.name
                output_folder.mkdir(parents=True, exist_ok=True)
                
                with open(output_md_path, 'w') as f:
                    f.write(text)
                
                # Copy the original PDF to the output folder
                shutil.copy2(doc_path, output_pdf_path)

                # Log the success message
                tqdm.write(f"SUCCESS: Saved {output_md_path.name} and {output_pdf_path.name} with risk level '{risk}'")

                
            except Exception as e:
                tqdm.write(f"ERROR processing {doc_path.name}: {e}")
        
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

