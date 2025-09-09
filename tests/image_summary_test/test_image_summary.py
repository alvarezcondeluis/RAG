#!/usr/bin/env python3
"""
Test script for the obtain_image_summary method from UnstructuredParserProcessor.
This script extracts images from test_p1_7_unstructured.json and tests the 
qwen2.5vl:7b model's image analysis capabilities.
"""

import json
import sys
import base64
from pathlib import Path
from typing import Dict, List

# Add the src directory to Python path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from simple_rag.parsers.unstructured_parser import UnstructuredParserProcessor


def load_test_images(json_file_path: Path) -> List[Dict]:
    """
    Load images from the test JSON file.
    
    Args:
        json_file_path: Path to the test JSON file
        
    Returns:
        List of image dictionaries with metadata and base64 data
    """
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data.get("images", [])


def save_image_to_file(image_base64: str, output_path: Path, mime_type: str = "image/jpeg"):
    """
    Save base64 image data to a file for inspection.
    
    Args:
        image_base64: Base64 encoded image data
        output_path: Path to save the image file
        mime_type: MIME type of the image
    """
    try:
        # Decode base64 data
        image_data = base64.b64decode(image_base64)
        
        # Write to file
        with open(output_path, 'wb') as f:
            f.write(image_data)
        
        print(f"   💾 Saved image to: {output_path}")
        
    except Exception as e:
        print(f"   ❌ Failed to save image: {e}")


def test_image_summary():
    """
    Main test function to test the obtain_image_summary method.
    """
    print("🧪 Testing obtain_image_summary with qwen2.5vl:7b")
    print("=" * 60)
    
    # Paths
    test_json_path = Path("../../data/processed/test_p1_7_unstructured.json")
    output_dir = Path("extracted_images")
    
    # Create output directory for extracted images
    output_dir.mkdir(exist_ok=True)
    
    # Load test images
    print(f"\n📂 Loading images from: {test_json_path}")
    try:
        images = load_test_images(test_json_path)
        print(f"   ✓ Found {len(images)} images")
    except Exception as e:
        print(f"   ❌ Failed to load images: {e}")
        return
    
    if not images:
        print("   ⚠️  No images found in the test file")
        return
    
    # Initialize UnstructuredParserProcessor
    print(f"\n🔧 Initializing UnstructuredParserProcessor...")
    try:
        parser = UnstructuredParserProcessor()
        print(f"   ✓ Parser initialized successfully")
    except Exception as e:
        print(f"   ❌ Failed to initialize parser: {e}")
        return
    
    # Test each image
    print(f"\n🖼️  Testing image analysis...")
    
    for i, image in enumerate(images, 1):
        print(f"\n--- Image {i}/{len(images)} ---")
        print(f"ID: {image.get('id', 'unknown')}")
        print(f"Page: {image.get('page', 'unknown')}")
        print(f"Element Type: {image.get('element_type', 'unknown')}")
        print(f"Original Text: {image.get('text', 'N/A')}")
        print(f"MIME Type: {image.get('mime_type', 'unknown')}")
        
        # Save image to file for inspection
        image_filename = f"image_{i}_{image.get('id', 'unknown')}.jpg"
        image_path = output_dir / image_filename
        
        if "image_data" in image:
            save_image_to_file(image["image_data"], image_path, image.get("mime_type", "image/jpeg"))
            
            # Test the obtain_image_summary method
            print(f"\n🤖 Analyzing image with qwen2.5vl:7b...")
            try:
                summary = parser.obtain_image_summary(
                    image["image_data"], 
                    image.get("mime_type", "image/jpeg")
                )
                
                print(f"✅ AI Summary Generated:")
                print(f"   {summary}")
                
                # Save summary to text file
                summary_path = output_dir / f"summary_{i}_{image.get('id', 'unknown')}.txt"
                with open(summary_path, 'w', encoding='utf-8') as f:
                    f.write(f"Image ID: {image.get('id', 'unknown')}\n")
                    f.write(f"Page: {image.get('page', 'unknown')}\n")
                    f.write(f"Original Text: {image.get('text', 'N/A')}\n")
                    f.write(f"MIME Type: {image.get('mime_type', 'unknown')}\n")
                    f.write(f"\nAI Summary:\n{summary}\n")
                
                print(f"   💾 Summary saved to: {summary_path}")
                
            except Exception as e:
                print(f"   ❌ Failed to analyze image: {e}")
        else:
            print(f"   ⚠️  No image data found for this image")
    
    print(f"\n🎉 Test completed!")
    print(f"📁 Check the '{output_dir}' folder for extracted images and summaries")


def main():
    """
    Main entry point for the test script.
    """
    try:
        test_image_summary()
    except KeyboardInterrupt:
        print(f"\n⏹️  Test interrupted by user")
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
