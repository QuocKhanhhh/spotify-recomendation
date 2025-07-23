from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
import logging
import traceback
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] {%(filename)s:%(lineno)d} - %(message)s'
)
logger = logging.getLogger(__name__)

from crawl_playlist import crawl_single_playlist
from checkpoint_manager import load_checkpoint, save_checkpoint
from preprocess import preprocess_data
from train_model import train_and_register_model

def crawl_pending_playlists():
    logger.info("Starting crawl_pending_playlists task")
    df_checkpoint = load_checkpoint()
    pending_playlists = df_checkpoint[df_checkpoint['status'] != 'done']['playlist_id'].tolist()
    logger.info(f"Found {len(pending_playlists)} pending playlists: {pending_playlists}")

    for playlist_id in pending_playlists:
        try:
            logger.info(f"Processing playlist {playlist_id}")
            success = crawl_single_playlist(playlist_id, output_file='/opt/airflow/data/crawled_tracks.csv')
            status = 'done' if success else 'failed'
            df_checkpoint.loc[df_checkpoint['playlist_id'] == playlist_id, 'status'] = status
            save_checkpoint(df_checkpoint)
            logger.info(f"Updated checkpoint for playlist {playlist_id} with status {status}")
        except Exception as e:
            logger.error(f"Unexpected error for playlist {playlist_id}: {str(e)}\n{traceback.format_exc()}")
            df_checkpoint.loc[df_checkpoint['playlist_id'] == playlist_id, 'status'] = 'failed'
            save_checkpoint(df_checkpoint)
            continue
    logger.info("Completed crawl_pending_playlists task")

def preprocess_data_wrapper(ti):
    logger.info("Starting preprocess_data_wrapper task")
    file_path = '/opt/airflow/data/crawled_tracks.csv'
    try:
        if not os.path.exists(file_path):
            logger.error(f"File {file_path} not found")
            raise FileNotFoundError(f"File {file_path} not found.")
        
        logger.info(f"Reading {file_path}")
        df = pd.read_csv(file_path, encoding='utf-8')
        logger.info(f"Loaded DataFrame with {len(df)} rows and columns: {list(df.columns)}")
        
        if df.empty:
            logger.error(f"File {file_path} is empty")
            raise ValueError(f"File {file_path} is empty.")
        
        logger.info("Starting data preprocessing")
        features_file = preprocess_data(df)
        logger.info(f"Preprocessed data, features saved to {features_file}")
        ti.xcom_push(key='features_file', value=features_file)
        logger.info("Completed preprocess_data_wrapper task")
        return features_file
    except Exception as e:
        logger.error(f"Error in preprocess_data_wrapper: {str(e)}\n{traceback.format_exc()}")
        raise

def train_model_wrapper(ti):
    logger.info("Starting train_model_wrapper task")
    try:
        features_file = ti.xcom_pull(key='features_file', task_ids='preprocess_data')
        if features_file is None:
            logger.error("No features file received from preprocess_data task")
            raise ValueError("No features file received from preprocess_data task")
        
        logger.info(f"Training model with features from {features_file}")
        train_and_register_model(features_file)
        logger.info("Completed train_model_wrapper task")
    except Exception as e:
        logger.error(f"Error in train_model_wrapper: {str(e)}\n{traceback.format_exc()}")
        raise

default_args = {
    'owner': 'airflow',
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

with DAG(
    dag_id='mlops_spotify_pipeline',
    default_args=default_args,
    start_date=datetime(2025, 7, 16),
    schedule_interval='0 1 * * *',
    catchup=False,
    max_active_runs=1,
    tags=['mlops', 'spotify']
) as dag:
    crawl_task = PythonOperator(
        task_id='crawl_data',
        python_callable=crawl_pending_playlists
    )
    preprocess_task = PythonOperator(
        task_id='preprocess_data',
        python_callable=preprocess_data_wrapper
    )
    train_task = PythonOperator(
        task_id='train_model',
        python_callable=train_model_wrapper
    )
    crawl_task >> preprocess_task >> train_task