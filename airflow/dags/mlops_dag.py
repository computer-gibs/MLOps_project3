from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta

# Импорт скриптов проекта как модулей
from scripts.get_data import get_data
from scripts.prepare_data import prepare_data
from scripts.train_test_split import split_data
from scripts.training_model import train_model
from scripts.evaluating_model import evaluate_model

default_args = {
    'owner': 'mlops',
    'depends_on_past': False,
    'start_date': datetime(2023, 11, 25),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG('mlops_pipeline', default_args=default_args, schedule_interval=timedelta(days=1))

t1 = PythonOperator(
    task_id='get_data',
    python_callable=get_data,
    dag=dag,
)

t2 = PythonOperator(
    task_id='prepare_data',
    python_callable=prepare_data,
    dag=dag,
)

t3 = PythonOperator(
    task_id='split_data',
    python_callable=split_data,
    dag=dag,
)

t4 = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    dag=dag,
)

t5 = PythonOperator(
    task_id='evaluate_model',
    python_callable=evaluate_model,
    dag=dag,
)

t1 >> t2 >> t3 >> t4 >> t5