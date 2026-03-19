#!/usr/bin/env python3
"""
Retrain Query Classifier with Updated Categories

This script retrains the SetFit query classifier to match the actual
schema slices in query_handler.py. The new model will classify queries
into 6 categories:
- not_related
- fund_basic (includes performance, returns, financial highlights)
- fund_portfolio
- fund_profile
- company_filing
- company_people

Usage:
    uv run src/simple_rag/rag/query/retrain_classifier.py
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.simple_rag.rag.query.query_classification import (
    QueryClassifier, 
    load_data, 
    train_model, 
    evaluate_model,
    LABELS
)

def main():
    """Retrain the query classifier with updated training data."""
    
    print("="*60)
    print("Query Classifier Retraining")
    print("="*60)
    print()
    print()
    print("Changes:")
    print("  - Removed: cross_entity, hybrid_graph_vector")
    print("  - Merged: fund_performance → fund_basic")
    print()
    print("="*60)
    print()
    
    # Paths
    query_dir = Path(__file__).parent
    train_data_path = query_dir / "training_data.json"
    test_data_path = query_dir / "classification_test_set.json"
    model_save_path = query_dir / "models" / "query_classifier"
    
    # Verify files exist
    if not train_data_path.exists():
        print(f"❌ Training data not found: {train_data_path}")
        return 1
    
    if not test_data_path.exists():
        print(f"❌ Test data not found: {test_data_path}")
        return 1
    
    print(f"📂 Training data: {train_data_path}")
    print(f"📂 Test data: {test_data_path}")
    print(f"📂 Model will be saved to: {model_save_path}")
    print()
    
    # Load data
    print("Loading training and test data...")
    train_dataset, test_dataset, mlb = load_data(str(train_data_path))
    
    # Train the model
    print()
    print("="*60)
    print("Starting training...")
    print("="*60)
    print()
    
    try:
        # Train the model
        trained_model = train_model(
            train_dataset=train_dataset,
            output_dir=str(model_save_path),
            num_epochs=3,
            batch_size=16,
            learning_rate=2e-5
        )
        
        # Evaluate the model
        print()
        print("="*60)
        print("Evaluating model...")
        print("="*60)
        print()
        
        evaluate_model(trained_model, test_dataset, mlb)
        
        print()
        print("="*60)
        print("✅ Training completed successfully!")
        print("="*60)
        print()
        print(f"Model saved to: {model_save_path}")
        print()
        print("Next steps:")
        print("1. Replace old training data:")
        print(f"   mv {train_data_path} {query_dir}/training_data.json")
        print()
        print("2. Replace old test data:")
        print(f"   mv {test_data_path} {query_dir}/classification_test_set.json")
        print()
        print("3. The new model is already saved and ready to use!")
        print()
        
        return 0
        
    except Exception as e:
        print()
        print("="*60)
        print("❌ Training failed!")
        print("="*60)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
