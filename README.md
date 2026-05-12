```markdown
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

```

---

## 📂 Repository Structure

```text
fraud_detection_mlops/
│
├── dags/
│   └── fraud_pipeline.py           # Airflow DAG definition
├── src/
│   ├── 1_ingest_data.py            # Data subsetting
│   ├── 2_validate_data.py          # Schema and null checks
│   ├── 3_preprocess_data.py        # KNN Imputation
│   ├── 4_feature_engineering.py    # Target Encoding for high-cardinality features
│   ├── 5_train_models.py           # Model training (Standard, Cost-Sensitive, SMOTE)
│   ├── 6_evaluate_model.py         # AUC/ROC evaluation & SHAP logging
│   ├── 7_register_model.py         # Champion model selection
│   ├── app.py                      # FastAPI inference server & Prometheus metrics
│   ├── send_test.py                # Script to simulate normal live API traffic
│   ├── 7_simulate_drift.py         # Script to simulate hacker tactics & trigger alerts
│   └── 9_explain.py                # Local and Global SHAP explainer script
├── Data/
│   └── train_transaction.csv       # Raw dataset (Add this manually)
├── docker-compose.yml              # Multi-container orchestration
├── Dockerfile                      # Custom Airflow/ML image build
└── README.md

```

---

## 🚀 Quick Start Guide

### 1. Prerequisites

* **Docker Desktop** (or Docker Engine on Linux)
* **Git**
* Port `5000` must be available (Mac users: turn off AirPlay Receiver in system settings).

### 2. Installation

Clone the repository and navigate into the project directory:

```bash
git clone [https://github.com/Hasankhan2003/fraud-mlops.git](https://github.com/Hasankhan2003/fraud-mlops.git)
cd fraud-mlops

```

*(Ensure your `train_transaction.csv` file is placed inside the `Data/` directory).*

### 3. Spin Up the Infrastructure

Build and launch all microservices using Docker Compose. The `-d` flag runs them in the background.

```bash
docker-compose up --build -d

```

*Note: The initial build may take 5-10 minutes to install all ML dependencies and initialize the Postgres database.*

---

## 🌐 Service Map

Once the containers are running, access the ecosystem via the following URLs:

| Service | Description | URL | Credentials |
| --- | --- | --- | --- |
| **Apache Airflow** | Pipeline Orchestration | [localhost:8081](https://www.google.com/search?q=http://localhost:8081) | `airflow` / `airflow` |
| **MLflow** | Experiment Tracking | [localhost:5000](https://www.google.com/search?q=http://localhost:5000) | *None* |
| **FastAPI** | Live Inference API | [localhost:8001/docs](https://www.google.com/search?q=http://localhost:8001/docs) | *None* |
| **Prometheus** | Metrics Scraper | [localhost:9090](https://www.google.com/search?q=http://localhost:9090) | *None* |
| **Grafana** | Dashboards & Alerts | [localhost:3000](https://www.google.com/search?q=http://localhost:3000) | `admin` / `admin` |

---

## 🛠️ Usage

### 1. Run the Training Pipeline

1. Open **Airflow** (`localhost:8081`).
2. Locate `end_to_end_fraud_pipeline`.
3. Unpause the DAG (toggle left) and click **Trigger DAG** (play button).
4. Monitor the execution graph. Upon success, the champion model is pushed to MLflow.

### 2. View Experiments

Open **MLflow** (`localhost:5000`) to view training runs, compare AUC scores, and inspect the logged SHAP waterfall/summary plots in the artifacts section.

### 3. Simulate Traffic & Data Drift

To see the monitoring ecosystem in action, open your local terminal and run the test scripts:

**Normal Traffic:**

```bash
python3 src/send_test.py

```

*Action:* Watch the Grafana **System Health** and **Model Performance** dashboards map latency and precision/recall dynamically.

**Simulate an Attack (Data Drift):**

```bash
python3 src/7_simulate_drift.py

```

*Action:* This script multiplies transaction amounts and injects unseen categorical variables. Watch the Grafana **Data Drift** dashboard crash, which pushes the "Data Drift Alert" into a **Firing** state and sends a payload to GitHub Actions for retraining.

---

## 🧹 Teardown

To safely shut down the ecosystem and free up system resources:

```bash
docker-compose down

```

*(To completely wipe the database volumes and start entirely fresh, use `docker-compose down -v`).*

---

## 👨‍💻 Author

**Ahmad Hasan Khan** * Data Science Undergraduate @ FAST-NUCES, Islamabad

* [LinkedIn](https://www.linkedin.com/in/hasan-khan02/) | [GitHub](https://github.com/Hasankhan2003)

```

```
