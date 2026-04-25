FROM apache/airflow:2.7.1

# 1. Install the C++ system library for LightGBM
USER root
RUN apt-get update && apt-get install -y libgomp1

# 2. Create the artifact directory and give the airflow user ownership
RUN mkdir -p /mlflow && chown -R airflow:root /mlflow

# 3. Switch back to Airflow user
USER airflow

# 4. Patient installations
ENV PIP_DEFAULT_TIMEOUT=1000
RUN pip install pandas scikit-learn
RUN pip install category_encoders imbalanced-learn shap
RUN pip install protobuf==3.20.3 mlflow
RUN pip install lightgbm
RUN pip install xgboost
