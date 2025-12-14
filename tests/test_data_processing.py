import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_processing import create_aggregate_features, build_pipeline

def test_create_aggregate_features():
    """Test that aggregate features are created correctly."""
    # Create dummy data
    data = {
        'customer_id': [1, 1, 2, 2, 3],
        'transaction_amount': [100, 200, 50, 150, 300]
    }
    df = pd.DataFrame(data)
    
    # Run function
    result = create_aggregate_features(df, 'customer_id', 'transaction_amount')
    
    # Assertions
    assert 'transaction_amount_sum' in result.columns
    assert 'transaction_amount_mean' in result.columns
    assert result[result['customer_id'] == 1]['transaction_amount_sum'].iloc[0] == 300
    assert result[result['customer_id'] == 2]['transaction_amount_mean'].iloc[0] == 100

def test_build_pipeline():
    """Test that the pipeline is built and can transform data."""
    # Create dummy data
    data = {
        'age': [25, 30, 35],
        'bmi': [20.0, 25.0, 30.0],
        'children': [0, 1, 2],
        'sex': ['male', 'female', 'male'],
        'smoker': ['yes', 'no', 'no'],
        'region': ['southwest', 'southeast', 'northwest']
    }
    df = pd.DataFrame(data)
    
    numerical_features = ['age', 'bmi', 'children']
    categorical_features = ['sex', 'smoker', 'region']
    
    # Build pipeline
    pipeline = build_pipeline(numerical_features, categorical_features)
    
    # Transform
    X_processed = pipeline.fit_transform(df)
    
    # Assertions
    assert X_processed is not None
    # Check shape: 3 numerical + (2 sex + 2 smoker + 3 region) = 10 columns (approx, depending on OneHot drop)
    # OneHotEncoder default doesn't drop first, so:
    # sex (2), smoker (2), region (3) -> 7 categorical columns
    # + 3 numerical + 1 interaction (bmi*age) = 11 columns
    assert X_processed.shape[0] == 3
    assert X_processed.shape[1] >= 10 # Ensure we have expanded features
