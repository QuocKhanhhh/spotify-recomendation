import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import csr_matrix, hstack
from scipy.sparse import save_npz
import logging
import traceback

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] {%(filename)s:%(lineno)d} - %(message)s'
)
logger = logging.getLogger(__name__)

def fix_partial_date(x):
    if pd.isna(x) or not x:
        logger.warning(f"Invalid date value: {x}, returning default '1970-01-01'")
        return '1970-01-01'
    x = str(x).strip()
    try:
        if re.fullmatch(r'\d{4}', x):
            return x + '-01-01'
        elif re.fullmatch(r'\d{4}-\d{2}', x):
            return x + '-01'
        elif re.fullmatch(r'\d{4}-\d{2}-\d{2}', x):
            return x
        else:
            logger.warning(f"Invalid date format: {x}, returning default '1970-01-01'")
            return '1970-01-01'
    except Exception as e:
        logger.error(f"Error processing date {x}: {str(e)}\n{traceback.format_exc()}")
        return '1970-01-01'

def clean_text(df):
    logger.info("Starting text cleaning")
    for col in ['track_name', 'artist_name']:
        if col not in df.columns:
            logger.error(f"Column {col} not found in DataFrame")
            raise KeyError(f"Column {col} not found in DataFrame")
        try:
            df[col] = df[col].astype(str).str.replace(r'[^\w\s]', '', regex=True)
            df[col] = df[col].str.lower()
            df[col] = df[col].str.strip()
            df[col] = df[col].str.replace(r'\s+', ' ', regex=True)
            logger.info(f"Cleaned column {col}")
        except Exception as e:
            logger.error(f"Error cleaning column {col}: {str(e)}\n{traceback.format_exc()}")
            raise
    logger.info("Completed text cleaning")
    return df

def preprocess_data(df):
    logger.info(f"Starting preprocess_data with {len(df)} rows and columns: {list(df.columns)}")
    try:
        # Kiểm tra các cột bắt buộc
        required_columns = ['track_id', 'track_name', 'artist_name', 'release_date', 'popularity', 'explicit', 'artist_genres']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            raise KeyError(f"Missing required columns: {missing_columns}")

        # Xóa dữ liệu NaN và trùng lặp
        logger.info("Cleaning NaN and duplicates")
        df_clean = df.dropna(subset=['track_id', 'track_name']).drop_duplicates(subset=['track_id', 'track_name']).reset_index(drop=True)
        logger.info(f"After cleaning NaN and duplicates: {len(df_clean)} rows")

        if df_clean.empty:
            logger.error("No valid data after cleaning. DataFrame is empty.")
            raise ValueError("No valid data after cleaning. DataFrame is empty.")

        # Xử lý cột release_date
        logger.info("Processing release_date")
        try:
            df_clean['release_date'] = df_clean['release_date'].apply(fix_partial_date)
            df_clean['release_date'] = pd.to_datetime(df_clean['release_date'], errors='coerce')
            df_clean['month'] = df_clean['release_date'].dt.month.fillna(1).astype(int)
            df_clean['year'] = df_clean['release_date'].dt.year.fillna(1970).astype(int)
            logger.info("Completed release_date processing")
        except Exception as e:
            logger.error(f"Error processing release_date: {str(e)}\n{traceback.format_exc()}")
            raise

        # Xử lý text
        df_clean = clean_text(df_clean)

        # TF-IDF encoding cho artist_genres
        logger.info("Processing artist_genres")
        try:
            # Check if artist_genres has non-empty values
            genres = df_clean['artist_genres'].fillna('')
            non_empty_genres = genres[genres.str.strip() != '']
            logger.info(f"Found {len(non_empty_genres)} non-empty artist_genres entries out of {len(genres)}")

            if len(non_empty_genres) == 0:
                logger.warning("No valid genres found. Using dummy TF-IDF features.")
                # Create dummy sparse matrix with zeros
                genre_tfidf = csr_matrix((len(df_clean), 1))
            else:
                tfidf = TfidfVectorizer(tokenizer=lambda x: x.split(',') if x else [], stop_words=None)
                genre_tfidf = tfidf.fit_transform(genres)
                logger.info(f"TF-IDF encoding completed, shape: {genre_tfidf.shape}, vocabulary size: {len(tfidf.vocabulary_)}")
        except Exception as e:
            logger.error(f"Error in TF-IDF encoding for artist_genres: {str(e)}\n{traceback.format_exc()}")
            raise

        # Xử lý cột popularity
        logger.info("Processing popularity")
        try:
            df_clean['popularity'] = pd.to_numeric(df_clean['popularity'], errors='coerce').fillna(0)
            scaler = MinMaxScaler()
            df_clean['popularity_scaled'] = scaler.fit_transform(df_clean[['popularity']])
            logger.info("Completed popularity processing")
        except Exception as e:
            logger.error(f"Error processing popularity: {str(e)}\n{traceback.format_exc()}")
            raise

        # Xử lý cột explicit
        logger.info("Processing explicit")
        try:
            df_clean['explicit_flag'] = df_clean['explicit'].astype(bool).astype(int)
            logger.info("Completed explicit processing")
        except Exception as e:
            logger.error(f"Error processing explicit: {str(e)}\n{traceback.format_exc()}")
            raise

        # Tạo features
        logger.info("Creating features")
        try:
            popularity_sparse = csr_matrix(df_clean['popularity_scaled'].values.reshape(-1, 1))
            explicit_sparse = csr_matrix(df_clean['explicit_flag'].values.reshape(-1, 1))
            features = hstack([genre_tfidf, popularity_sparse, explicit_sparse])
            features_file = '/opt/airflow/data/features.npz'
            save_npz(features_file, features)
            df_clean.to_csv('/opt/airflow/data/clean_data.csv', index=False, encoding='utf-8')
            logger.info(f"Saved features to {features_file} and clean data to /opt/airflow/data/clean_data.csv")
            logger.info("Completed preprocess_data")
            return features_file
        except Exception as e:
            logger.error(f"Error creating or saving features: {str(e)}\n{traceback.format_exc()}")
            raise
    except Exception as e:
        logger.error(f"Error in preprocess_data: {str(e)}\n{traceback.format_exc()}")
        raise