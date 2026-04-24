import pandas as pd
import joblib
import mlflow
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

def run_evaluation(**kwargs):
    mlflow.set_tracking_uri("http://localhost:5000")
    
    clf = joblib.load('/Data/processed/model.pkl')
    X_val, y_val = joblib.load('/Data/processed/val_data.pkl')
    
    y_pred = clf.predict(X_val)
    y_proba = clf.predict_proba(X_val)[:, 1]

    # Metrics
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    auc = roc_auc_score(y_val, y_proba)
    cm = confusion_matrix(y_val, y_pred)

    with mlflow.start_run(run_name="Evaluation_Run"):
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)  # Critical metric
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("auc_roc", auc)
        
        print(f"Metrics - Recall: {recall:.4f}, AUC: {auc:.4f}")
        print(f"Confusion Matrix:\n{cm}")

        # Register model if it meets baseline
        if auc > 0.80:
            mlflow.sklearn.log_model(clf, "fraud_model", registered_model_name="FraudDetectionModel")

    # Pass AUC to Airflow for branching
    if 'ti' in kwargs:
        kwargs['ti'].xcom_push(key='auc_roc', value=auc)

if __name__ == "__main__":
    run_evaluation()
