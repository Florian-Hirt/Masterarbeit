#!/usr/bin/env python3
"""
Simple test script to verify SAFIM setup is working correctly
"""

import sys
import os

# Add the SAFIM directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from safim.data_utils import load_dataset
from safim.ast_utils import get_parser, ErrorCheckVisitor

def test_data_loading():
    """Test that we can load the dataset"""
    print("Testing dataset loading...")
    try:
        samples = load_dataset("block")
        python_samples = [s for s in samples if s["lang"] == "python"]
        print(f"✓ Loaded {len(python_samples)} Python samples from 'statement' dataset")
        
        # Show first sample
        if python_samples:
            first = python_samples[0]
            print(f"\nFirst sample ID: {first['task_id']}")
            print(f"Has unit tests: {'unit_tests' in first and bool(first['unit_tests'])}")
            print(f"Prompt preview: {first['prompt'][:100]}...")
        return True
    except Exception as e:
        print(f"✗ Failed to load dataset: {e}")
        return False

def test_syntax_checking():
    """Test syntax checking functionality"""
    print("\nTesting syntax checking...")
    try:
        parser = get_parser("python")
        
        # Test valid code
        valid_code = "x = 1 + 2\nprint(x)"
        tree = parser.parse(valid_code.encode("utf-8"))
        visitor = ErrorCheckVisitor()
        visitor(tree)
        print(f"✓ Valid code check: {visitor.error_cnt == 0}")
        
        # Test invalid code
        invalid_code = "x = 1 ++"
        tree = parser.parse(invalid_code.encode("utf-8"))
        visitor = ErrorCheckVisitor()
        visitor(tree)
        print(f"✓ Invalid code check: {visitor.error_cnt > 0}")
        return True
    except Exception as e:
        print(f"✗ Failed syntax checking: {e}")
        return False

def test_model_loading():
    """Test that we can load the model (with small version)"""
    print("\nTesting model loading...")
    try:
        from transformers import AutoTokenizer
        # Just test tokenizer loading
        tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-Python-hf")
        print("✓ Successfully loaded tokenizer")
        print(f"  Vocab size: {len(tokenizer)}")
        return True
    except Exception as e:
        print(f"✗ Failed to load tokenizer: {e}")
        print("  Make sure you have transformers installed and are logged in to HuggingFace if needed")
        return False

def main():
    print("SAFIM Setup Test")
    print("=" * 50)
    
    tests = [
        test_data_loading,
        test_syntax_checking,
        test_model_loading
    ]
    
    passed = sum(test() for test in tests)
    total = len(tests)
    
    print(f"\n{'=' * 50}")
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("\n✓ All tests passed! You're ready to run the main script.")
    else:
        print("\n✗ Some tests failed. Please check the errors above.")

if __name__ == "__main__":
    main()