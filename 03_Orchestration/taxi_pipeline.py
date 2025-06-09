from datetime import timedelta
import sqlite3
import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from scipy import sparse
import mlflow
from prefect import flow, task
from prefect.tasks import task_input_hash
from pathlib import Path

SQLITE_PATH = './database.sqlite'
DATA_PATH = './data/yellow_tripdata_2023-03.parquet'
TMP_DIR = Path('./tmp')
TMP_DIR.mkdir(parents=True, exist_ok=True)

def get_conn():
    return sqlite3.connect(SQLITE_PATH)

@task
def load_data():
    df = pd.read_parquet(DATA_PATH)
    print(f"The dataset contains {df.shape[0]} rows and {df.shape[1]} columns.")
    
    conn = get_conn()
    df.to_sql('yellow_march23', con=conn, if_exists='replace', index=False)
    conn.close()

@task
def transform_data():
    conn = get_conn()
    df = pd.read_sql(
        'SELECT * FROM yellow_march23', 
        con=conn,
        parse_dates=['tpep_dropoff_datetime', 'tpep_pickup_datetime']
    )

    df['duration'] = (df.tpep_dropoff_datetime - df.tpep_pickup_datetime).dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)]
    df[['PULocationID', 'DOLocationID']] = df[['PULocationID', 'DOLocationID']].astype(str)

    print(f"The transformed dataset contains {df.shape[0]} rows and {df.shape[1]} columns.")
    
    df.to_sql('yellow_march23_tf', con=conn, if_exists='replace', index=False)
    conn.close()

@task
def prepare_data():
    conn = get_conn()
    df = pd.read_sql(
        'SELECT * FROM yellow_march23_tf', 
        con=conn,
        parse_dates=['tpep_dropoff_datetime', 'tpep_pickup_datetime']
    )
    conn.close()

    dicts = df[['PULocationID', 'DOLocationID']].to_dict(orient='records')
    dv = DictVectorizer()
    X = dv.fit_transform(dicts)
    sparse.save_npz(TMP_DIR / 'X.npz', X)

    y = df["duration"].to_numpy()
    np.save(TMP_DIR / 'y.npy', y)

@task
def train_model():
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("nyc_taxi_experiment")

    X_train = sparse.load_npz(TMP_DIR / 'X.npz')
    y_train = np.load(TMP_DIR / 'y.npy')

    with mlflow.start_run():
        lr = LinearRegression()
        lr.fit(X_train, y_train)

        mlflow.log_param("intercept_", lr.intercept_)
        mlflow.sklearn.log_model(
            sk_model=lr,
            artifact_path="artifacts_local",
            registered_model_name="MyLinearRegressor"
        )

@flow(name="Taxi ML Pipeline", retries=1, retry_delay_seconds=300)
def taxi_pipeline():
    load_data()
    transform_data()
    prepare_data()
    train_model()

if __name__ == "__main__":
    taxi_pipeline()