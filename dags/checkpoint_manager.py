import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CHECKPOINT_FILE = '/opt/airflow/data/playlist_ids_checkpoint.csv'

def load_checkpoint():
    try:
        df = pd.read_csv(CHECKPOINT_FILE)
        logger.info(f"Loaded checkpoint with {len(df)} playlist IDs")
    except FileNotFoundError:
        logger.warning(f"Checkpoint file {CHECKPOINT_FILE} not found, creating new")
        df = pd.DataFrame(columns=['playlist_id', 'status'])
    return df

def save_checkpoint(df):
    logger.info(f"Saving checkpoint with {len(df)} playlist IDs")
    df.to_csv(CHECKPOINT_FILE, index=False, encoding='utf-8')

def add_new_playlists(new_ids):
    df = load_checkpoint()
    new_rows = [{'playlist_id': pid, 'status': 'pending'} for pid in new_ids if pid not in df['playlist_id'].values]
    if new_rows:
        logger.info(f"Adding {len(new_rows)} new playlist IDs to checkpoint")
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
        save_checkpoint(df)
    else:
        logger.info("No new playlist IDs to add")