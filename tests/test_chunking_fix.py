#!/usr/bin/env python3
"""
Test script to verify the updated Title-based chunking functionality.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from simple_rag.parsers.unstructured_parser import UnstructuredParserProcessor

def test_combine_chunk_text():
    """Test the _combine_chunk_text method with various element types."""
    parser = UnstructuredParserProcessor()
    
    # Test data similar to what we see in the JSON
    test_elements = [
        {"element_type": "Title", "text": "The Data Engineering Lifecycle"},
        {"element_type": "NarrativeText", "text": "It is all too easy to fixate on technology and miss the bigger picture myopically."},
        {"element_type": "FigureCaption", "text": "Figure 1-1. The data engineering lifecycle"},
        {"element_type": "NarrativeText", "text": "The data engineering lifecycle shifts the conversation away from technology."},
        {"element_type": "ListItem", "text": "Generation"},
        {"element_type": "NarrativeText", "text": "Storage"},
        {"element_type": "NarrativeText", "text": "Ingestion"},
        {"element_type": "ListItem", "text": "Transformation"},
        {"element_type": "ListItem", "text": "Serving"},
        {"element_type": "Header", "text": "history."}
    ]
    
    combined = parser._combine_chunk_text(test_elements)
    
    print("=== TESTING _combine_chunk_text ===")
    print("Input elements:")
    for elem in test_elements:
        print(f"  {elem['element_type']}: {elem['text']}")
    
    print(f"\nCombined text:\n{repr(combined)}")
    print(f"\nFormatted output:\n{combined}")
    
    # Check that all text is included
    for elem in test_elements:
        if elem['text'] not in combined:
            print(f"❌ Missing text: {elem['text']}")
        else:
            print(f"✓ Found: {elem['element_type']} - {elem['text'][:30]}...")

def test_enhance_figure_caption():
    """Test the _enhance_figure_caption method."""
    parser = UnstructuredParserProcessor()
    
    # Mock image chunks with AI summaries
    image_chunks = [
        {
            "page": 4,
            "ai_summary": "A circular diagram showing the data engineering lifecycle with stages: Generation, Storage, Ingestion, Transformation, and Serving, connected by arrows in a cyclical flow."
        }
    ]
    
    # Test FigureCaption element
    caption_element = {
        "element_type": "FigureCaption",
        "text": "Figure 1-1. The data engineering lifecycle",
        "page": 4
    }
    
    print("\n=== TESTING _enhance_figure_caption ===")
    print(f"Original caption: {caption_element['text']}")
    
    enhanced = parser._enhance_figure_caption(caption_element, image_chunks, verbose=True)
    
    print(f"Enhanced caption: {enhanced['text']}")
    
    if 'ai_enhanced' in enhanced:
        print("✓ Caption was enhanced with AI summary")
    else:
        print("❌ Caption was not enhanced")

if __name__ == "__main__":
    test_combine_chunk_text()
    test_enhance_figure_caption()
