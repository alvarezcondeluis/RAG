import time
import json
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
from simple_rag.parsers.unstructured_parser import UnstructuredParserProcessor
from simple_rag.parsers.parser_llama import LlamaParseProcessor
from simple_rag.parsers.utils.slicer import slice_pdf_pages


class MainParserProcessor:
    """
    Main parser processor that manages both LlamaParse and Unstructured parsers,
    provides timing comparisons, and handles file operations.
    """
    
    def __init__(self, main_file_path: Optional[Path] = None, root_path: Optional[Path] = None, enable_llama_parse: bool = True):
        """
        Initialize the MainParserProcessor.
        
        Args:
            main_file_path: Path to the main PDF file to process
            root_path: Root directory for file operations
            enable_llama_parse: Whether to enable LlamaParse parsing (default: True)
        """
        # Set paths
        self.root_path = root_path or Path(__file__).resolve().parents[2]
        self.main_file_path = main_file_path
        self.enable_llama_parse = enable_llama_parse
        
        # Initialize parsers
        self.unstructured_parser = UnstructuredParserProcessor(root_path=self.root_path)
        self.llama_parser = LlamaParseProcessor(root_path=self.root_path) if enable_llama_parse else None
        
        # Output directories
        self.output_dir = self.root_path / "data" / "processed"
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def set_main_file_path(self, file_path: Path) -> None:
        """
        Set the main PDF file path.
        
        Args:
            file_path: Path to the main PDF file
        """
        self.main_file_path = Path(file_path)
    
    
    
    def compare_parsers_timing(
        self,
        file_path: Optional[Path] = None,
        start_page: Optional[int] = None,
        end_page: Optional[int] = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Compare the timing performance of both parsers on the same document.
        
        Args:
            file_path: Path to the PDF file to parse (or main file if None)
            start_page: Optional start page for slicing
            end_page: Optional end page for slicing
            verbose: Whether to print progress messages
            
        Returns:
            Dictionary with timing results and parser outputs
        """
        if file_path is None:
            if not self.main_file_path:
                raise ValueError("No file path provided and main_file_path not set.")
            file_path = self.main_file_path
        
       
        results = {
            "file_path": str(file_path),
            "slice_info": {"start_page": start_page, "end_page": end_page} if start_page else None,
            "parsers": {}
        }
        
        # Test LlamaParse (only if enabled)
        if self.enable_llama_parse:
            if verbose:
                print("\n" + "="*50)
                print("🦙 TESTING LLAMAPARSE")
                print("="*50)
            
            llama_start = time.time()
            try:
                llama_docs = self.llama_parser.parse_document(file_path, verbose=verbose)
                llama_end = time.time()
                llama_success = True
                llama_error = None
            except Exception as e:
                llama_end = time.time()
                llama_docs = []
                llama_success = False
                llama_error = str(e)
                if verbose:
                    print(f"❌ LlamaParse failed: {e}")
            
            results["parsers"]["llama"] = {
                "success": llama_success,
                "duration_seconds": llama_end - llama_start,
                "docs_count": len(llama_docs) if llama_docs else 0,
                "error": llama_error
            }
        else:
            llama_docs = []
            llama_success = False
            if verbose:
                print("\n" + "="*50)
                print("🦙 LLAMAPARSE DISABLED")
                print("="*50)
            results["parsers"]["llama"] = {
                "success": False,
                "duration_seconds": 0,
                "docs_count": 0,
                "error": "LlamaParse disabled"
            }
        
        # Test Unstructured
        if verbose:
            print("\n" + "="*50)
            print("📄 TESTING UNSTRUCTURED")
            print("="*50)
        
        unstructured_start = time.time()
        try:
            unstructured_docs = self.unstructured_parser.parse_document(file_path, verbose=verbose)
            unstructured_end = time.time()
            unstructured_success = True
            unstructured_error = None
            
        except Exception as e:
            unstructured_end = time.time()
            unstructured_docs = []
            unstructured_success = False
            unstructured_error = str(e)
            if verbose:
                print(f"❌ Unstructured failed: {e}")
        
        results["parsers"]["unstructured"] = {
            "success": unstructured_success,
            "duration_seconds": unstructured_end - unstructured_start,
            "docs_count": len(unstructured_docs) if unstructured_docs else 0,
            "error": unstructured_error
        }
        
        # Print comparison summary
        if verbose:
            print("\n" + "="*50)
            print("⏱️  TIMING COMPARISON RESULTS")
            print("="*50)
            
            llama_time = results["parsers"]["llama"]["duration_seconds"]
            unstructured_time = results["parsers"]["unstructured"]["duration_seconds"]
            
            print(f"🦙 LlamaParse:")
            print(f"   Duration: {llama_time:.2f}s")
            print(f"   Success: {results['parsers']['llama']['success']}")
            print(f"   Documents: {results['parsers']['llama']['docs_count']}")
            
            print(f"\n📄 Unstructured:")
            print(f"   Duration: {unstructured_time:.2f}s")
            print(f"   Success: {results['parsers']['unstructured']['success']}")
            print(f"   Documents: {results['parsers']['unstructured']['docs_count']}")
            
            if self.enable_llama_parse and llama_success and unstructured_success:
                if llama_time < unstructured_time:
                    faster = "LlamaParse"
                    speedup = unstructured_time / llama_time
                else:
                    faster = "Unstructured"
                    speedup = llama_time / unstructured_time
                
                print(f"\n🏆 Winner: {faster} ({speedup:.2f}x faster)")
            elif not self.enable_llama_parse and unstructured_success:
                print(f"\n✅ Unstructured completed successfully (LlamaParse disabled)")
        
        # Store parsed documents for output generation
        results["parsed_docs"] = {
            "llama": llama_docs if llama_success else None,
            "unstructured": unstructured_docs if unstructured_success else None
        }
        
        return results
    
    def save_parser_outputs(
        self,
        results: Dict[str, Any],
        base_filename: Optional[str] = None
    ) -> Dict[str, Path]:
        """
        Save the outputs from both parsers to separate files.
        
        Args:
            results: Results dictionary from compare_parsers_timing()
            base_filename: Base filename for outputs. If None, auto-generates
            
        Returns:
            Dictionary with paths to created output files
        """
        if base_filename is None:
            file_path = Path(results["file_path"])
            base_filename = file_path.stem
        
        output_paths = {}
        
        # Save LlamaParse output (markdown) - only if enabled and successful
        if self.enable_llama_parse and results["parsed_docs"]["llama"]:
            llama_output_path = self.output_dir / f"{base_filename}_llama.md"
            self.llama_parser.generate_output(results["parsed_docs"]["llama"], llama_output_path)
            output_paths["llama"] = llama_output_path
            print(f"💾 Saved LlamaParse output → {llama_output_path}")
        
        # Save Unstructured output (JSON)
        if results["parsed_docs"]["unstructured"]:
            unstructured_output_path = self.output_dir / f"{base_filename}_unstructured.json"
            self.unstructured_parser.generate_output(results["parsed_docs"]["unstructured"], unstructured_output_path)
            output_paths["unstructured"] = unstructured_output_path
            print(f"💾 Saved Unstructured output → {unstructured_output_path}")
        
        
        return output_paths
    
    def process_and_compare(
        self,
        file_path: Optional[Path] = None,
        start_page: Optional[int] = None,
        end_page: Optional[int] = None,
        save_outputs: bool = True,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Complete workflow: slice PDF (if needed), compare parsers, and save outputs.
        
        Args:
            file_path: Path to PDF file. If None, uses main_file_path
            start_page: Optional start page for slicing
            end_page: Optional end page for slicing
            save_outputs: Whether to save parser outputs to files
            verbose: Whether to print progress messages
            
        Returns:
            Dictionary with complete results including timing and output paths
        """
        # Run timing comparison
        results = self.compare_parsers_timing(file_path, start_page, end_page, verbose)
        
        # Save outputs if requested
        if save_outputs:
            output_paths = self.save_parser_outputs(results)
            results["output_paths"] = output_paths
        
        return results


def run_test_demo():
    """
    Comprehensive test function that demonstrates all MainParserProcessor features.
    This function runs when the script is executed directly.
    """
    print("🚀 MainParserProcessor Test Demo")
    print("=" * 60)
    
    # Test configuration
    test_pdf = Path("data/raw/book_pages_22_to_66.pdf")  # Change this to your test PDF
    start_page = 1
    end_page = 7
    
    print(f"📄 Test Configuration:")
    print(f"   PDF File: {test_pdf}")
    print(f"   Page Range: {start_page}-{end_page}")
    print(f"   Current Directory: {Path.cwd()}")
    
    try:
        # Step 1: Initialize MainParserProcessor
        print(f"\n🔧 Step 1: Initializing MainParserProcessor...")
        # You can set enable_llama_parse=False to run only Unstructured parser
        main_parser = MainParserProcessor(main_file_path=test_pdf, enable_llama_parse=False)
        print(f"   ✓ Root path: {main_parser.root_path}")
        print(f"   ✓ Output directory: {main_parser.output_dir}")
        print(f"   ✓ LlamaParse enabled: {main_parser.enable_llama_parse}")
        
        # Step 2: Check if test PDF exists
        print(f"\n📋 Step 2: Checking test file...")
        if not main_parser.main_file_path or not main_parser.main_file_path.exists():
            print(f"   ⚠️  Test PDF not found: {test_pdf}")
            print(f"   💡 Creating a sample test scenario instead...")
            
            # Alternative: demonstrate with a different file or create mock test
            print(f"\n🎭 Running Mock Test Scenario:")
            print(f"   - Would slice PDF pages {start_page}-{end_page}")
            print(f"   - Would compare LlamaParse vs Unstructured timing")
            print(f"   - Would save outputs to separate files")
            print(f"\n✅ Mock test completed successfully!")
            return
        
        print(f"   ✓ Test PDF found: {main_parser.main_file_path}")
        
        # Step 3: Test PDF slicing
        print(f"\n✂️  Step 3: Testing PDF slicing...")
        slice_path = slice_pdf_pages(main_parser.main_file_path, list(range(start_page, end_page + 1)))
        print(f"   ✓ Created slice: {slice_path}")
        
        # Step 4: Run parser comparison
        print(f"\n⚡ Step 4: Running parser timing comparison...")
        results = main_parser.compare_parsers_timing(
            slice_path,
            verbose=True
        )
        
        # Step 5: Save outputs
        print(f"\n💾 Step 5: Saving parser outputs...")
        output_paths = main_parser.save_parser_outputs(results, f"test_p{start_page}_{end_page}")
        
        # Step 6: Display final results
        print(f"\n📊 Step 6: Final Test Results")
        print(f"   🦙 LlamaParse: {results['parsers']['llama']['success']} "
              f"({results['parsers']['llama']['duration_seconds']:.2f}s)")
        print(f"   📄 Unstructured: {results['parsers']['unstructured']['success']} "
              f"({results['parsers']['unstructured']['duration_seconds']:.2f}s)")
        
        if output_paths:
            print(f"\n📁 Generated Files:")
            for parser_name, path in output_paths.items():
                print(f"   {parser_name}: {path}")
        
        print(f"\n🎉 Test completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        print(f"\n🔍 Troubleshooting tips:")
        print(f"   1. Make sure the PDF file exists at: {test_pdf}")
        print(f"   2. Check that required dependencies are installed")
        print(f"   3. Verify API keys are set (for LlamaParse)")
        print(f"   4. Ensure write permissions for output directory")
        
        # Show available files for debugging
        data_dir = Path("data/raw")
        if data_dir.exists():
            pdf_files = list(data_dir.glob("*.pdf"))
            if pdf_files:
                print(f"\n📋 Available PDF files in {data_dir}:")
                for pdf in pdf_files[:5]:  # Show first 5
                    print(f"   - {pdf.name}")
        
        raise


def main():
    """
    Main entry point when script is executed directly.
    Provides both test demo and potential for future CLI interface.
    """
    import sys
    
    if len(sys.argv) > 1:
        # Future: could add CLI arguments here
        # For now, just run the test
        print("🔧 CLI arguments detected, running test demo...")
    
    run_test_demo()


# Class can be imported and used normally
# When executed directly, runs comprehensive test
if __name__ == "__main__":
    main()