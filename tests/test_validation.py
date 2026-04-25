import pandas as pd

def test_data_validation_rules():
    # 1. Create a tiny mock dataset simulating your engineered data
    mock_data = {
        'isFraud': [0, 1, 0], 
        'TransactionAmt': [50.0, 150.0, 25.0],
        'card1': [1000, 2000, 3000]
    }
    df = pd.DataFrame(mock_data)
    
    # 2. Perform Data Validation Checks (Stage 1 Requirement)
    
    # Check if target column exists
    assert 'isFraud' in df.columns, "Target column 'isFraud' is missing!"
    
    # Check if critical features exist
    assert 'TransactionAmt' in df.columns, "Critical feature missing"
    
    # Check for invalid negative values in money
    assert df['TransactionAmt'].min() >= 0, "Transaction amount cannot be negative"
    
    # Check for missing values
    assert not df.isnull().values.any(), "Dataset contains unexpected missing values"
