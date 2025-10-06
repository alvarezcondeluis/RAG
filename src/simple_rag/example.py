import fitz  # PyMuPDF
import os
from PIL import Image
import io
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import json

def detect_visual_elements(pdf_path):
    """
    Detects and extracts raster images and tables from a PDF,
    storing them as image data in memory.

    Args:
        pdf_path (str): The path to the input PDF file.
        
    Returns:
        list: List of dictionaries containing image data and metadata
    """
    doc = fitz.open(pdf_path)
    print(f"Opened '{pdf_path}' with {doc.page_count} pages.")

    element_count = 0
    extracted_elements = []  # Unified list for all visual elements

    for page_num in range(len(doc)):
        page = doc[page_num]
        page_element_count = 0

        # 1. --- Detect and Extract RASTER IMAGES ---
        image_list = page.get_images(full=True)
        for img_index, img in enumerate(image_list, start=1):
            element_count += 1
            page_element_count += 1
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            
            # Convert bytes to PIL Image for easier handling
            pil_image = Image.open(io.BytesIO(image_bytes))
            
            extracted_elements.append({
                'image_data': pil_image,
                'image_bytes': image_bytes,
                'format': image_ext,
                'type': 'Raster Image',
                'page': page_num + 1,
                'element_id': f"element_{page_num+1}_{element_count}_raster"
            })

        # 2. --- Detect and Extract TABLES ---
        tables = page.find_tables()
        for table_index, table in enumerate(tables, start=1):
            element_count += 1
            page_element_count += 1
            # Get the bounding box of the table and take a "screenshot"
            rect = table.bbox  # Use .bbox attribute for Table objects
            pix = page.get_pixmap(clip=rect, alpha=False)  # Call get_pixmap on the page
            
            # Convert pixmap to PIL Image
            img_data = pix.tobytes("png")
            pil_image = Image.open(io.BytesIO(img_data))

            # Try to extract structured table content
            table_rows = None
            table_json = None
            try:
                extracted = table.extract()
                # Newer PyMuPDF returns dict with 'cells'; older may return list-of-rows
                if isinstance(extracted, dict):
                    table_json = extracted
                    cells = extracted.get('cells', [])
                    nrows = extracted.get('nrows') or (max((c['row'] + (c.get('rowspan', 1) or 1) for c in cells), default=0))
                    ncols = extracted.get('ncols') or (max((c['col'] + (c.get('colspan', 1) or 1) for c in cells), default=0))
                    if nrows and ncols:
                        grid = [["" for _ in range(ncols)] for _ in range(nrows)]
                        for c in cells:
                            r = c['row']
                            c_idx = c['col']
                            text = c.get('text', '') or ''
                            # Note: colspan/rowspan not expanded; place top-left only
                            if 0 <= r < nrows and 0 <= c_idx < ncols:
                                grid[r][c_idx] = text
                        table_rows = grid
                elif isinstance(extracted, list):
                    table_rows = extracted
                    table_json = {"rows": extracted}
            except Exception as e:
                print(f"[Page {page_num+1} | Table {table_index}] table.extract() failed: {e}")
            
            extracted_elements.append({
                'image_data': pil_image,
                'image_bytes': img_data,
                'format': 'png',
                'type': 'Table',
                'page': page_num + 1,
                'element_id': f"element_{page_num+1}_{element_count}_table",
                'table_rows': table_rows,
                'table_json': table_json,
            })

        if page_element_count > 0:
            print(f"---> Found {page_element_count} visual elements on page {page_num + 1}")

    if element_count == 0:
        print("No visual elements (images or tables) found in the document.")
    else:
        print(f"\n✅ Done! Extracted {element_count} visual elements in memory.")
        # Print JSON for each detected table
        table_counter = 0
        for elem in extracted_elements:
            if elem.get('type') == 'Table' and elem.get('table_json') is not None:
                table_counter += 1
                print(f"\n=== Table {table_counter} (Page {elem['page']}) ===")
                try:
                    print(json.dumps(elem['table_json'], ensure_ascii=False, indent=2))
                except Exception as e:
                    print(f"(Failed to serialize table JSON): {e}")
        if extracted_elements:
            display_elements_grid_from_memory(extracted_elements, os.path.basename(pdf_path))
            
    doc.close()
    return extracted_elements


def display_elements_grid_from_memory(extracted_elements, pdf_filename, elements_per_page=12):
    """
    Displays all extracted visual elements in a paginated grid from memory.
    
    Args:
        extracted_elements: List of element dictionaries with image_data
        pdf_filename: Name of the PDF file
        elements_per_page: Number of elements to show per page
    """
    if not extracted_elements:
        print("No elements to display.")
        return

    n_elements = len(extracted_elements)
    n_pages = (n_elements + elements_per_page - 1) // elements_per_page
    
    print(f"\n🖼️ Displaying {n_elements} extracted elements in {n_pages} page(s)...")
    print("📋 Navigation: Close current window to see next page, or press Ctrl+C to stop.")
    
    for page_idx in range(n_pages):
        start_idx = page_idx * elements_per_page
        end_idx = min(start_idx + elements_per_page, n_elements)
        page_elements = extracted_elements[start_idx:end_idx]
        
        print(f"\n📄 Showing page {page_idx + 1}/{n_pages} (elements {start_idx + 1}-{end_idx})")
        
        # Calculate grid dimensions for this page
        n_page_elements = len(page_elements)
        cols = min(4, n_page_elements)
        rows = (n_page_elements + cols - 1) // cols
        
        # Create the plot
        fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
        fig.suptitle(f'Visual Elements from "{pdf_filename}" - Page {page_idx + 1}/{n_pages}\n(Elements {start_idx + 1}-{end_idx} of {n_elements})', 
                    fontsize=14, fontweight='bold')
        
        # Ensure axes is always a list/array for consistent indexing
        if n_page_elements == 1:
            axes = [axes]
        elif rows == 1:
            axes = [axes] if cols == 1 else axes
        else:
            axes = axes.flatten()

        # Display elements for this page
        for i, elem_info in enumerate(page_elements):
            try:
                # Use the PIL image directly
                pil_image = elem_info['image_data']
                axes[i].imshow(pil_image)
                
                # Create title with element number
                global_idx = start_idx + i + 1
                title = f"#{global_idx} - Page {elem_info['page']}\n{elem_info['type']}"
                axes[i].set_title(title, fontsize=9, pad=5)
                axes[i].axis('off')
                
            except Exception as e:
                axes[i].text(0.5, 0.5, f"Error loading:\n{elem_info['element_id']}", 
                           ha='center', va='center', color='red', fontsize=8)
                axes[i].set_title(f"#{start_idx + i + 1} - Error", fontsize=9, color='red')
                axes[i].axis('off')

        # Hide unused subplots
        for j in range(n_page_elements, len(axes)):
            axes[j].axis('off')

        plt.tight_layout(rect=[0, 0.03, 1, 0.92])
        
        try:
            plt.show()
        except KeyboardInterrupt:
            print("\n⏹️ Display interrupted by user.")
            plt.close('all')
            break
        
        # Close the figure to free memory
        plt.close(fig)
    
    print("✅ Image display completed!")


# --- How to use the function ---
if __name__ == "__main__":
    # 1. Replace this with the actual path to your PDF file
    pdf_to_process = "data/ETF/Ishares-SP500KIID.pdf"

    # 2. Check if the file exists before running
    if os.path.exists(pdf_to_process):
        extracted_elements = detect_visual_elements(pdf_to_process)
        print(f"\n📊 Summary: Extracted {len(extracted_elements)} visual elements")
        
        # You can now access the images in memory:
        # for element in extracted_elements:
        #     print(f"Element: {element['element_id']} - Type: {element['type']} - Page: {element['page']}")
        #     # Access PIL image: element['image_data']
        #     # Access raw bytes: element['image_bytes']
        #     # Access format: element['format']
    else:
        print(f"Error: The file '{pdf_to_process}' was not found. Please update the path.")