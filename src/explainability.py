import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("--- Running Explainability (Task 9) ---")
print("Loading data for SHAP analysis...")
df = pd.read_csv('./Data/train_transaction.csv', nrows=2000)

X = df.select_dtypes(include=['float64', 'int64']).fillna(0)
if 'isFraud' in X.columns:
    y = X['isFraud']
    X = X.drop(columns=['isFraud', 'TransactionID'], errors='ignore')
else:
    y = [0] * len(X)

print("Training explainer model...")
model = xgb.XGBClassifier(eval_metric='logloss')
model.fit(X, y)

print("Generating SHAP summary plot...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X, show=False)
plt.savefig('shap_summary_fraud.png', bbox_inches='tight')
print("Done! Image saved as shap_summary_fraud.png in your current folder.")
