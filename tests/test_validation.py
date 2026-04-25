import pandas as pd
import os

def test_real_data_validation():
    # In GitHub Actions, the code runs from the root of your repo
    file_path = 'Data/interim/04_engineered.csv'
    
    # 1. Check if the real file actually uploaded to GitHub
    assert os.path.exists(file_path), "Real data file not found! Git did not upload it."
    
    # 2. Read your actual dataset
    df = pd.read_csv(file_path)
    
    # 3. Perform real validation checks on your data
    assert 'isFraud' in df.columns, "Target column 'isFraud' is missing!"
    assert 'TransactionAmt' in df.columns, "Critical feature 'TransactionAmt' missing!"
    assert df['TransactionAmt'].min() >= 0, "Found invalid negative transaction amounts!"
    
    # 4. Critical Null Check (Check only the crucial columns instead of all 394)
    assert not df['isFraud'].isnull().any(), "Target column 'isFraud' contains missing values!"
    assert not df['TransactionAmt'].isnull().any(), "Critical column 'TransactionAmt' contains missing values!"
