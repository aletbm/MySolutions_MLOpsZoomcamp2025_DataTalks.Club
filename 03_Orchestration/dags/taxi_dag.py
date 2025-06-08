from datetime import datetime, timedelta
from airflow import DAG
from airflow.exceptions import AirflowFailException
from airflow.operators.python import PythonOperator
import sqlite3
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
import numpy as np
from scipy import sparse
import mlflow
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error as RMSE

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("nyc_taxi_experiment")

SQLITE_PATH = './tmp/airflow_data.sqlite'

dag = DAG( "taxi_pipeline",
          default_args={
                "depends_on_past": False,
                "retries": 1,
                "retry_delay": timedelta(minutes=5),
                # 'queue': 'bash_queue',
                # 'pool': 'backfill',
                # 'priority_weight': 10,
                # 'end_date': datetime(2016, 1, 1),
                # 'wait_for_downstream': False,
                # 'execution_timeout': timedelta(seconds=300),
                # 'on_failure_callback': some_function, # or list of functions
                # 'on_success_callback': some_other_function, # or list of functions
                # 'on_retry_callback': another_function, # or list of functions
                # 'sla_miss_callback': yet_another_function, # or list of functions
                # 'on_skipped_callback': another_function, #or list of functions
                # 'trigger_rule': 'all_success'
            },
            description="My first DAG",
            schedule=timedelta(days=1),
            start_date=datetime(2021, 1, 1),
            catchup=False,
            tags=["example"])

def get_conn():
    return sqlite3.connect(SQLITE_PATH)

def load_data(**kwargs):
    df = pd.read_parquet("./data/yellow_tripdata_2023-03.parquet")
    
    print(f"The dataset contain {df.shape[0]} rows and {df.shape[1]} columns.")
    
    conn = get_conn()
    df.to_sql('yellow_march23', con=conn, if_exists='replace', index=False)
    conn.close()
    
    return

def transform_data(**kwargs):
    conn = get_conn()
    
    df = pd.read_sql('SELECT * FROM yellow_march23', 
                     con=conn,
                     parse_dates=['tpep_dropoff_datetime', 'tpep_pickup_datetime'])

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    
    print(f"The dataset contain {df.shape[0]} rows and {df.shape[1]} columns.")
    
    df.to_sql('yellow_march23_tf', con=conn, if_exists='replace', index=False)
    
    conn.close()
    
    return

def prepare_data(**kwargs):
    conn = get_conn()
    
    df = pd.read_sql('SELECT * FROM yellow_march23_tf', 
                     con=conn,
                     parse_dates=['tpep_dropoff_datetime', 'tpep_pickup_datetime'])

    conn.close()
    
    dicts = df[['PULocationID', 'DOLocationID']].to_dict(orient='records')
    dv = DictVectorizer()
    
    X = dv.fit_transform(dicts)
    sparse.save_npz('./tmp/X.npz', X)
    
    y = df["duration"].to_numpy()
    np.save('./tmp/y.npy', y)
    return

def train_model(**kwargs):
    
    X_train = sparse.load_npz('./tmp/X.npz')
    y_train = np.load('./tmp/y.npy')
    
    with mlflow.start_run():
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        
        mlflow.log_param("intercept_", lr.intercept_)
        
        mlflow.sklearn.log_model(sk_model=lr,
                                artifact_path="artifacts_local",
                                registered_model_name="MyLinearRegressor",
                                )
    return
        
tk_load = PythonOperator(
    task_id="load_data",
    python_callable=load_data,
    dag=dag
)

tk_transform = PythonOperator(
    task_id="transform_data",
    python_callable=transform_data,
    dag=dag
)

tk_prepare = PythonOperator(
    task_id="prepare_data",
    python_callable=prepare_data,
    dag=dag
)

tk_train = PythonOperator(
    task_id="train_model",
    python_callable=train_model,
    dag=dag
)

tk_load >> tk_transform >> tk_prepare >> tk_train