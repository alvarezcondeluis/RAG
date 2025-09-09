import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Mock ollama before importing the parser
sys.modules['ollama'] = Mock()

from simple_rag.parsers.unstructured_parser import UnstructuredParserProcessor


class TestUnstructuredParserEnhanced(unittest.TestCase):
    """Test the enhanced UnstructuredParserProcessor with section_title and source_document functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.processor = UnstructuredParserProcessor(
            strategy="hi_res",
            chunking_strategy="by_title",
            max_characters=10000
        )
        
        # Create mock elements for testing
        self.mock_title_element = Mock()
        self.mock_title_element.text = "Introduction to Machine Learning"
        self.mock_title_element.metadata.page_number = 1
        
        self.mock_text_element = Mock()
        self.mock_text_element.text = "Machine learning is a subset of artificial intelligence."
        self.mock_text_element.metadata.page_number = 1
        
        self.mock_image_element = Mock()
        self.mock_image_element.text = ""
        self.mock_image_element.metadata.page_number = 2
        self.mock_image_element.metadata.image_base64 = "fake_base64_data"
        self.mock_image_element.metadata.image_mime_type = "image/png"
        
        # Set element types
        type(self.mock_title_element).__name__ = "Title"
        type(self.mock_text_element).__name__ = "NarrativeText"
        type(self.mock_image_element).__name__ = "Image"
    
    def create_mock_chunk_with_orig_elements(self, orig_elements, page=1):
        """Create a mock chunk with orig_elements."""
        mock_chunk = Mock()
        mock_chunk.metadata = Mock()
        mock_chunk.metadata.page_number = page
        mock_chunk.metadata.orig_elements = orig_elements
        return mock_chunk
    
    def create_mock_fallback_chunk(self, text, category="NarrativeText", page=1):
        """Create a mock chunk without orig_elements (fallback case)."""
        mock_chunk = Mock()
        mock_chunk.metadata = Mock()
        mock_chunk.metadata.page_number = page
        mock_chunk.category = category
        mock_chunk.text = text
        # No orig_elements attribute
        delattr(mock_chunk.metadata, 'orig_elements')
        return mock_chunk
    
    @patch('simple_rag.parsers.unstructured_parser.UnstructuredParserProcessor.obtain_image_summary')
    def test_preprocess_chunks_with_titles_and_source_document(self, mock_image_summary):
        """Test that preprocess_chunks correctly assigns section titles and source document."""
        mock_image_summary.return_value = "A diagram showing neural network architecture"
        
        # Create chunks with Title -> Text -> Image pattern
        chunks = [
            self.create_mock_chunk_with_orig_elements([self.mock_title_element], page=1),
            self.create_mock_chunk_with_orig_elements([self.mock_text_element], page=1),
            self.create_mock_chunk_with_orig_elements([self.mock_image_element], page=2)
        ]
        
        source_document = "ml_textbook.pdf"
        result = self.processor.preprocess_chunks(chunks, verbose=False, source_document=source_document)
        
        # Verify structure
        self.assertIn('text_chunks', result)
        self.assertIn('image_chunks', result)
        
        # Check text chunks have section_title and source_document
        text_chunks = result['text_chunks']
        self.assertEqual(len(text_chunks), 1)
        
        chunk = text_chunks[0]
        self.assertEqual(chunk['section_title'], "Introduction to Machine Learning")
        self.assertEqual(chunk['source_document'], source_document)
        self.assertIn('chunk_id', chunk)
        self.assertIn('combined_text', chunk)
        
        # Check image chunks have source_document
        image_chunks = result['image_chunks']
        self.assertEqual(len(image_chunks), 1)
        
        img_chunk = image_chunks[0]
        self.assertEqual(img_chunk['source_document'], source_document)
        self.assertEqual(img_chunk['section_title'], "Image Content")
        self.assertIn('ai_summary', img_chunk)
    
    def test_preprocess_chunks_multiple_titles(self):
        """Test handling of multiple titles creating separate chunks."""
        # Create second set of elements
        mock_title2 = Mock()
        mock_title2.text = "Deep Learning Fundamentals"
        mock_title2.metadata.page_number = 3
        type(mock_title2).__name__ = "Title"
        
        mock_text2 = Mock()
        mock_text2.text = "Deep learning uses neural networks with multiple layers."
        mock_text2.metadata.page_number = 3
        type(mock_text2).__name__ = "NarrativeText"
        
        chunks = [
            self.create_mock_chunk_with_orig_elements([self.mock_title_element], page=1),
            self.create_mock_chunk_with_orig_elements([self.mock_text_element], page=1),
            self.create_mock_chunk_with_orig_elements([mock_title2], page=3),
            self.create_mock_chunk_with_orig_elements([mock_text2], page=3)
        ]
        
        source_document = "ai_handbook.pdf"
        result = self.processor.preprocess_chunks(chunks, verbose=False, source_document=source_document)
        
        text_chunks = result['text_chunks']
        self.assertEqual(len(text_chunks), 2)
        
        # First chunk
        self.assertEqual(text_chunks[0]['section_title'], "Introduction to Machine Learning")
        self.assertEqual(text_chunks[0]['source_document'], source_document)
        
        # Second chunk
        self.assertEqual(text_chunks[1]['section_title'], "Deep Learning Fundamentals")
        self.assertEqual(text_chunks[1]['source_document'], source_document)
    
    def test_preprocess_chunks_no_title_fallback(self):
        """Test handling of chunks without titles (fallback to 'No Title')."""
        chunks = [
            self.create_mock_chunk_with_orig_elements([self.mock_text_element], page=1)
        ]
        
        source_document = "notes.pdf"
        result = self.processor.preprocess_chunks(chunks, verbose=False, source_document=source_document)
        
        text_chunks = result['text_chunks']
        self.assertEqual(len(text_chunks), 1)
        self.assertEqual(text_chunks[0]['section_title'], "No Title")
        self.assertEqual(text_chunks[0]['source_document'], source_document)
    
    def test_preprocess_chunks_fallback_chunks(self):
        """Test handling of chunks without orig_elements (fallback case)."""
        chunks = [
            self.create_mock_fallback_chunk("This is fallback text content.", "NarrativeText", page=1)
        ]
        
        source_document = "simple_doc.pdf"
        result = self.processor.preprocess_chunks(chunks, verbose=False, source_document=source_document)
        
        text_chunks = result['text_chunks']
        self.assertEqual(len(text_chunks), 1)
        self.assertEqual(text_chunks[0]['section_title'], "No Title")
        self.assertEqual(text_chunks[0]['source_document'], source_document)
        self.assertIn("This is fallback text content.", text_chunks[0]['combined_text'])
    
    def test_generate_output_with_source_document(self):
        """Test generate_output method with explicit source_document parameter."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_output.json"
            
            # Create simple mock chunks
            chunks = [
                self.create_mock_chunk_with_orig_elements([self.mock_title_element], page=1),
                self.create_mock_chunk_with_orig_elements([self.mock_text_element], page=1)
            ]
            
            source_document = "research_paper.pdf"
            result_path = self.processor.generate_output(chunks, output_path, source_document=source_document)
            
            # Verify file was created
            self.assertTrue(result_path.exists())
            
            # Load and verify content
            with open(result_path, 'r', encoding='utf-8') as f:
                content = json.load(f)
            
            self.assertIn('text_chunks', content)
            text_chunks = content['text_chunks']
            self.assertEqual(len(text_chunks), 1)
            self.assertEqual(text_chunks[0]['source_document'], source_document)
    
    def test_generate_output_auto_source_document(self):
        """Test generate_output method with automatic source_document extraction."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "my_document.json"
            
            chunks = [
                self.create_mock_chunk_with_orig_elements([self.mock_text_element], page=1)
            ]
            
            # Don't provide source_document - should auto-extract from output_path
            result_path = self.processor.generate_output(chunks, output_path)
            
            with open(result_path, 'r', encoding='utf-8') as f:
                content = json.load(f)
            
            text_chunks = content['text_chunks']
            self.assertEqual(text_chunks[0]['source_document'], "my_document.pdf")
    
    def test_combine_chunk_text(self):
        """Test the _combine_chunk_text method."""
        text_elements = [
            {"element_type": "Title", "text": "Chapter 1: Introduction"},
            {"element_type": "NarrativeText", "text": "This chapter covers basic concepts."},
            {"element_type": "NarrativeText", "text": "We will explore various topics."}
        ]
        
        combined = self.processor._combine_chunk_text(text_elements)
        # The actual method adds newlines for Title, Header, and NarrativeText when combined_parts exist
        expected = "Chapter 1: Introduction\nThis chapter covers basic concepts.\nWe will explore various topics."
        self.assertEqual(combined, expected)
    
    def test_combine_chunk_text_empty(self):
        """Test _combine_chunk_text with empty input."""
        result = self.processor._combine_chunk_text([])
        self.assertEqual(result, "")
    
    @patch('simple_rag.parsers.unstructured_parser.UnstructuredParserProcessor.obtain_image_summary')
    def test_image_chunk_structure(self, mock_image_summary):
        """Test that image chunks have the correct structure."""
        mock_image_summary.return_value = "Chart showing data trends"
        
        chunks = [
            self.create_mock_chunk_with_orig_elements([self.mock_image_element], page=2)
        ]
        
        source_document = "data_report.pdf"
        result = self.processor.preprocess_chunks(chunks, verbose=False, source_document=source_document)
        
        image_chunks = result['image_chunks']
        self.assertEqual(len(image_chunks), 1)
        
        img_chunk = image_chunks[0]
        required_fields = ['chunk_id', 'page', 'type', 'source_document', 'section_title', 'ai_summary']
        for field in required_fields:
            self.assertIn(field, img_chunk)
        
        self.assertEqual(img_chunk['type'], 'image')
        self.assertEqual(img_chunk['source_document'], source_document)
        self.assertEqual(img_chunk['section_title'], "Image Content")
        self.assertEqual(img_chunk['ai_summary'], "Chart showing data trends")
    
    def test_summary_statistics(self):
        """Test that summary statistics are correctly calculated."""
        chunks = [
            self.create_mock_chunk_with_orig_elements([self.mock_title_element, self.mock_text_element], page=1),
            self.create_mock_chunk_with_orig_elements([self.mock_image_element], page=2)
        ]
        
        with patch.object(self.processor, 'obtain_image_summary', return_value="Test image"):
            result = self.processor.preprocess_chunks(chunks, verbose=False, source_document="test.pdf")
        
        summary = result['summary']
        self.assertEqual(summary['text_chunks_count'], 1)
        self.assertEqual(summary['image_chunks_count'], 1)
        self.assertEqual(summary['total_chunks'], 2)


class TestUnstructuredParserIntegration(unittest.TestCase):
    """Integration tests for the enhanced parser functionality."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.processor = UnstructuredParserProcessor()
    
    def test_full_workflow_structure(self):
        """Test that the full workflow produces correctly structured output."""
        # This would be a more comprehensive test with actual PDF processing
        # For now, we'll test the data structure expectations
        
        expected_text_chunk_fields = [
            'chunk_id', 'page', 'text_elements', 'combined_text', 
            'section_title', 'source_document'
        ]
        
        expected_image_chunk_fields = [
            'chunk_id', 'page', 'type', 'source_document', 'section_title'
        ]
        
        # Mock a simple workflow result
        mock_chunks = [Mock()]
        mock_chunks[0].metadata = Mock()
        mock_chunks[0].metadata.page_number = 1
        mock_chunks[0].metadata.orig_elements = []
        
        result = self.processor.preprocess_chunks(mock_chunks, source_document="test.pdf")
        
        # Verify the overall structure exists
        self.assertIn('text_chunks', result)
        self.assertIn('image_chunks', result)
        self.assertIn('summary', result)
        
        # The specific field tests would depend on having actual data
        # But we've verified the structure is in place


if __name__ == '__main__':
    # Create a test suite
    suite = unittest.TestSuite()
    
    # Add all test methods
    suite.addTest(unittest.makeSuite(TestUnstructuredParserEnhanced))
    suite.addTest(unittest.makeSuite(TestUnstructuredParserIntegration))
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    print(f"{'='*50}")
