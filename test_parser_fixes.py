#!/usr/bin/env python3
"""
Test script to verify the fixes in the unstructured parser.
"""

import sys
from pathlib import Path

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent / "src"))

from simple_rag.parsers.unstructured_parser import UnstructuredParserProcessor

def test_element_filtering():
    """Test the _should_include_element method."""
    parser = UnstructuredParserProcessor()
    
    # Test cases
    test_cases = [
        ("Text", "4", False),  # Should filter out standalone numbers
        ("Text", "42", False),  # Should filter out standalone numbers
        ("Text", "A", False),   # Should filter out very short text
        ("Text", "The", True),  # Should include meaningful short text
        ("Text", "This is a longer text", True),  # Should include longer text
        ("NarrativeText", "Storage", True),  # Should include narrative text
        ("ListItem", "Generation", True),  # Should include list items
        ("Title", "Chapter 1", True),  # Should include titles
        ("Text", "", False),  # Should filter out empty text
        ("Text", "   ", False),  # Should filter out whitespace-only text
    ]
    
    print("Testing element filtering:")
    for element_type, text, expected in test_cases:
        result = parser._should_include_element(element_type, text)
        status = "✓" if result == expected else "✗"
        print(f"  {status} {element_type}: '{text}' -> {result} (expected {expected})")

def test_list_item_detection():
    """Test the _looks_like_list_item method."""
    parser = UnstructuredParserProcessor()
    
    test_cases = [
        ("Storage", True),
        ("Ingestion", True),
        ("Generation", True),
        ("This is a long narrative text that should not be considered a list item", False),
        ("Data engineering is the development, implementation, and maintenance of systems", False),
        ("Short phrase", True),
        ("A", True),
        ("Very long text that exceeds the character limit for list items", False),
    ]
    
    print("\nTesting list item detection:")
    for text, expected in test_cases:
        result = parser._looks_like_list_item(text)
        status = "✓" if result == expected else "✗"
        print(f"  {status} '{text[:30]}...' -> {result} (expected {expected})")

def test_text_combination():
    """Test the _combine_chunk_text method."""
    parser = UnstructuredParserProcessor()
    
    # Mock text elements
    text_elements = [
        {"element_type": "Title", "text": "Chapter 1. Data Engineering Described"},
        {"element_type": "NarrativeText", "text": "If you work in data or software, you may have noticed data engineering emerging from the shadows."},
        {"element_type": "Title", "text": "What Is Data Engineering?"},
        {"element_type": "NarrativeText", "text": "Despite the current popularity of data engineering, there's a lot of confusion."},
        {"element_type": "ListItem", "text": "Generation"},
        {"element_type": "NarrativeText", "text": "Storage"},  # This should be treated as a list item
        {"element_type": "NarrativeText", "text": "Ingestion"},  # This should be treated as a list item
        {"element_type": "ListItem", "text": "Transformation"},
        {"element_type": "ListItem", "text": "Serving"},
    ]
    
    result = parser._combine_chunk_text(text_elements)
    
    print("\nTesting text combination:")
    print("Result:")
    print(result)
    print("\nExpected structure:")
    print("- Titles should be on new lines")
    print("- List items should be properly spaced")
    print("- No excessive whitespace or duplicate newlines")

def test_figure_caption_enhancement():
    """Test figure caption enhancement logic."""
    parser = UnstructuredParserProcessor()
    
    # Test case 1: Caption without existing AI descriptions
    caption1 = {
        "element_type": "FigureCaption",
        "text": "Figure 1-1. The data engineering lifecycle",
        "page": 4
    }
    
    # Test case 2: Caption with existing AI descriptions (should be skipped)
    caption2 = {
        "element_type": "FigureCaption", 
        "text": "Figure 1-1. The data engineering lifecycle [AI Description 1: The image shows...]",
        "page": 4
    }
    
    # Mock image chunks
    image_chunks = [
        {
            "page": 4,
            "ai_summary": "The image illustrates a Data Engineering Lifecycle diagram..."
        }
    ]
    
    print("\nTesting figure caption enhancement:")
    
    # Test enhancement of clean caption
    result1 = parser._enhance_figure_caption(caption1, image_chunks, verbose=True)
    print(f"Clean caption enhanced: {result1.get('ai_enhanced', False)}")
    
    # Test skipping of already enhanced caption
    result2 = parser._enhance_figure_caption(caption2, image_chunks, verbose=True)
    print(f"Already enhanced caption skipped: {not result2.get('ai_enhanced', False)}")

if __name__ == "__main__":
    print("Testing UnstructuredParserProcessor fixes...\n")
    
    test_element_filtering()
    test_list_item_detection()
    test_text_combination()
    test_figure_caption_enhancement()
    
    print("\n✓ All tests completed!")
