from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

PROJECT_DIR = "/opt/airflow/project"

with DAG(
    dag_id="enterprise_forecasting_anomaly_ml_pipeline",
    description="End-to-end enterprise forecasting and anomaly detection MLOps workflow.",
    start_date=datetime(2024, 1, 1),
    schedule="@weekly",
    catchup=False,
    default_args={"owner": "mlops", "retries": 1, "retry_delay": timedelta(minutes=5)},
    tags=["mlops", "forecasting", "anomaly_detection"],
) as dag:
    generate_data = BashOperator(
        task_id="generate_or_ingest_data",
        bash_command=f"cd {PROJECT_DIR} && python src/data/generate_synthetic_data.py --rows 50000",
    )

    validate_data = BashOperator(
        task_id="validate_data",
        bash_command=f"cd {PROJECT_DIR} && python src/data/validation.py",
    )

    build_features = BashOperator(
        task_id="build_features",
        bash_command=f"cd {PROJECT_DIR} && python src/features/build_features.py",
    )

    train_forecasting = BashOperator(
        task_id="train_forecasting_model",
        bash_command=f"cd {PROJECT_DIR} && python src/models/train_forecasting_model.py",
    )

    train_anomaly = BashOperator(
        task_id="train_anomaly_detection_model",
        bash_command=f"cd {PROJECT_DIR} && python src/models/train_anomaly_model.py",
    )

    run_batch_inference = BashOperator(
        task_id="run_batch_inference",
        bash_command=f"cd {PROJECT_DIR} && python src/inference/batch_inference.py",
    )

    generate_monitoring_report = BashOperator(
        task_id="generate_monitoring_report",
        bash_command=f"cd {PROJECT_DIR} && python src/monitoring/generate_monitoring_report.py",
    )

    generate_data >> validate_data >> build_features >> train_forecasting >> train_anomaly >> run_batch_inference >> generate_monitoring_report
