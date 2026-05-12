# 🚀 FraudOps: End-to-End Real-Time Fraud Detection MLOps

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker)
![Apache Airflow](https://img.shields.io/badge/Airflow-Orchestration-017CEE?logo=apacheairflow)
![MLflow](https://img.shields.io/badge/MLflow-Registry-0194E2?logo=mlflow)
![FastAPI](https://img.shields.io/badge/FastAPI-Serving-009688?logo=fastapi)
![Grafana](https://img.shields.io/badge/Grafana-Monitoring-F46800?logo=grafana)

An end-to-end, containerized Machine Learning Operations (MLOps) pipeline designed to detect fraudulent financial transactions in real-time. This project handles the entire ML lifecycle: automated data ingestion, model training, live API serving, and real-time monitoring with automated CI/CD retraining loops triggered by data drift.

## 🌟 Key Features
* **Automated Orchestration:** 9-step Apache Airflow DAG handling ingestion, KNN imputation, target encoding, and conditional deployment.
* **Experiment Tracking:** MLflow integration comparing XGBoost, LightGBM, and Hybrid Random Forest models (handling class imbalance via SMOTE and Cost-Sensitive learning).
* **High-Performance API:** FastAPI inference server serving champion models with **<250ms latency** and achieving **97% AUC**.
* **Live System Monitoring:** Prometheus and Grafana dashboards tracking API health, dynamic precision/recall, and prediction confidence.
* **Closed-Loop Retraining:** Simulated data drift alerts (confidence dropping <85%) trigger GitHub Actions Webhooks to automatically initiate model retraining.
* **AI Explainability:** Global and local SHAP value generation to interpret tree-based model decisions.

---

## 🏗️ System Architecture

```text
[Raw Data] --> (Apache Airflow) --> Data Validation & Feature Engineering
                                      |
                                      v
                             Model Training (XGBoost/LightGBM)
                                      |
                                      v
                                 (MLflow) <-- Logs Metrics, Artifacts, SHAP
                                      |
                               [Model Registry]
                                      |
[Client Request] --> (FastAPI) <------+ Pulls Champion Model
                         |
                         v
                    (Prometheus) <-- Scrapes API Metrics
                         |
                         v
                     (Grafana) <-- Visualizes Drift & Performance
                         |
                  [Alert Trigger] --> (GitHub Actions) --> Triggers Retraining
