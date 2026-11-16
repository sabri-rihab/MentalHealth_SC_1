from datetime import datetime, timedelta
from airflow import DAG
from airflow.decorators  import task

from src.clean import load_raw_data, clean_raw_data, save_data
from src.train import split_dataset, train_and_optimize, save_model
from src.evaluate import evaluate_model
from src.config import CLEANED_DATA_PATH

import pandas as pd


def task_clean_data():
    df= load_raw_data()
    df_clean = clean_raw_data(df)
    save_data(df_clean)

def task_train_model():
    df = pd.read_csv(CLEANED_DATA_PATH)
    X_train, X_test, y_train, y_test = split_dataset(df)
 
    X_test.to_parquet('data/X_test.parquet')
    y_test.to_frame().to_parquet('data/y_test.parquet')

    best_pipline = train_and_optimize(X_train, y_train)
    save_model(best_pipline)

def task_evaluate_model():
    X_test = pd.read_parquet('data/X_test.parquet')
    y_test = pd.read_parquet('data/y_test.parquet').squeeze()

    metrics = evaluate_model(X_test, y_test)
    print(f"Model Accuracy : {metrics['accuracy']:.4f}")


default_args = {
    'retries' : 2,
    'retry_delay': timedelta(minutes=2)
}
with DAG(
    'student_mental_health_training_pipline',
    start_date=datetime(2026,12,12),
    schedule = '@weekly',
    catchup= False,
    default_args = default_args
    
) as dag:
    @task
    def clean_step():
        return task_clean_data()

    @task
    def train_step():
        return task_train_model()

    @task
    def evaluate_step():
        return task_evaluate_model()


    clean = clean_step()
    train = train_step()
    evaluate = evaluate_step()

    clean >> train >> evaluate
