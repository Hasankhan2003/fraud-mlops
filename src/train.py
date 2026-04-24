import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
import mlflow
import mlflow.xgboost
import mlflow.lightgbm
import mlflow.sklearn
from sklearn.model_selection import train_test_split
import joblib

def run_training():
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("Fraud_Detection_Training")

    print("Loading preprocessed data...")
    X = pd.read_csv('/Data/processed/X_train.csv')
    # Use .squeeze() to ensure 'y' is a 1D Series, fixing the float bug!
    y = pd.read_csv('/Data/processed/y_train.csv').squeeze()

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Task 4: Assign higher penalty to false negatives
    scale_pos_weight = float((len(y_train) - y_train.sum()) / y_train.sum())

    # 1. XGBoost (Cost-Sensitive)
    print("Training Cost-Sensitive XGBoost...")
    with mlflow.start_run(run_name="XGBoost_Cost_Sensitive"):
        xgb_model = xgb.XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            max_depth=5,
            learning_rate=0.1,
            n_estimators=100
        )
        xgb_model.fit(X_train, y_train)
        mlflow.log_param("model", "XGBoost")
        mlflow.log_param("cost_sensitive", True)
        mlflow.xgboost.log_model(xgb_model, "model")
        
        # Save model locally for evaluation and explainability
        joblib.dump(xgb_model, '/Data/processed/model.pkl')
        joblib.dump((X_val, y_val), '/Data/processed/val_data.pkl')

    # 2. LightGBM
    print("Training LightGBM...")
    with mlflow.start_run(run_name="LightGBM_Standard"):
        lgb_model = lgb.LGBMClassifier(class_weight='balanced')
        lgb_model.fit(X_train, y_train)
        mlflow.log_param("model", "LightGBM")
        mlflow.lightgbm.log_model(lgb_model, "model")

    # 3. Hybrid Model (Random Forest + Feature Selection)
    print("Training Hybrid Model...")
    with mlflow.start_run(run_name="Hybrid_RF_FeatureSelection"):
        hybrid_model = Pipeline([
            ('feature_selection', SelectFromModel(RandomForestClassifier(n_estimators=10, random_state=42))),
            ('classification', RandomForestClassifier(n_estimators=50, class_weight='balanced', random_state=42))
        ])
        hybrid_model.fit(X_train, y_train)
        mlflow.log_param("model", "Hybrid_RF")
        mlflow.sklearn.log_model(hybrid_model, "model")

    print("All models successfully trained and logged to MLflow!")

if __name__ == "__main__":
    run_training()
