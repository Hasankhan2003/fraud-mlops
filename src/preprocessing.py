import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import TargetEncoder
from imblearn.over_sampling import SMOTE
import os

def run_preprocessing():
    print("Loading data...")
    # Load subset to manage memory locally
    df = pd.read_csv('/Data/train_transaction.csv', nrows=50000) 
    
    X = df.drop(['isFraud', 'TransactionID', 'TransactionDT'], axis=1)
    y = df['isFraud']

    # 1. Handle Missing Values
    print("Imputing missing values...")
    num_cols = X.select_dtypes(include=['float64', 'int64']).columns
    cat_cols = X.select_dtypes(include=['object']).columns

    num_imputer = SimpleImputer(strategy='median')
    X[num_cols] = num_imputer.fit_transform(X[num_cols])

    # 2. High-Cardinality Categorical Encoding (Target Encoding)
    print("Applying Target Encoding...")
    cat_imputer = SimpleImputer(strategy='most_frequent')
    X[cat_cols] = cat_imputer.fit_transform(X[cat_cols])
    
    encoder = TargetEncoder()
    X[cat_cols] = encoder.fit_transform(X[cat_cols], y)

    # 3. Handle Class Imbalance (SMOTE vs Class Weighting comparison ready)
    print("Applying SMOTE...")
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Save processed data
    os.makedirs('/Data/processed', exist_ok=True)
    X_resampled.to_csv('/Data/processed/X_train.csv', index=False)
    y_resampled.to_csv('/Data/processed/y_train.csv', index=False)
    print("Preprocessing complete.")

if __name__ == "__main__":
    run_preprocessing()
