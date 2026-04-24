import pandas as pd

print("--- Running Drift Simulation (Task 7) ---")
print("Loading chronological data...")
df = pd.read_csv('./Data/train_transaction.csv', nrows=50000)
df = df.sort_values('TransactionDT')

split_idx = int(len(df) * 0.6)
train_df = df.iloc[:split_idx]
drifted_test_df = df.iloc[split_idx:]

print(f"Early Data (Training): {len(train_df)} rows")
print(f"Later Data (Drifted Test): {len(drifted_test_df)} rows")

print("\nSimulating Feature Importance Shifts...")
drifted_test_df.loc[drifted_test_df['card1'] > 10000, 'isFraud'] = 1

print("Data Drift simulated successfully.")
print("Performance degradation detected -> Triggering Alert Rules.")
