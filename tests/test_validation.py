import pandas as pd
import os

def test_data_exists():
    # Check if the interim data folder exists
    assert os.path.exists('/opt/airflow/Data/interim') or os.path.exists('./Data/interim')

def test_data_columns():
    # If the file exists locally during GitHub Actions, validate it
    file_path = './Data/interim/04_engineered.csv'
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        # Ensure target column exists
        assert 'isFraud' in df.columns
