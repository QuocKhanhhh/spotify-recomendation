import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from scipy.sparse import load_npz
import numpy as np
import logging
import traceback

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] {%(filename)s:%(lineno)d} - %(message)s'
)
logger = logging.getLogger(__name__)

def train_and_register_model(features_file):
    logger.info(f"Starting train_and_register_model with features file: {features_file}")
    try:
        # Đọc ma trận thưa và dữ liệu
        logger.info(f"Loading features from {features_file}")
        features = load_npz(features_file)
        logger.info(f"Loaded features with shape: {features.shape}")

        logger.info("Loading clean_data.csv")
        df = pd.read_csv('/opt/airflow/data/clean_data.csv', encoding='utf-8')
        logger.info(f"Loaded DataFrame with {len(df)} rows and columns: {list(df.columns)}")

        if df.empty:
            logger.error("No data in clean_data.csv to train the model.")
            raise ValueError("No data in clean_data.csv to train the model.")
        
        if not isinstance(features, csr_matrix):
            logger.error(f"Input features must be a scipy.sparse.csr_matrix, got {type(features)}")
            raise TypeError("Input features must be a scipy.sparse.csr_matrix")
        
        if features.shape[0] != len(df):
            logger.error(f"Features shape {features.shape[0]} does not match DataFrame length {len(df)}")
            raise ValueError(f"Features shape {features.shape[0]} does not match DataFrame length {len(df)}")

        # Check for dummy features (e.g., from empty genres)
        if features.shape[1] == 3:  # Only popularity_scaled and explicit_flag + dummy genre column
            logger.warning("Features contain only dummy genre data. Similarity matrix may be less accurate.")

        # Tính toàn bộ ma trận cosine similarity
        logger.info("Computing cosine similarity matrix")
        sim_matrix = cosine_similarity(features)
        logger.info(f"Computed similarity matrix with shape: {sim_matrix.shape}")
        
        # Đặt giá trị trên đường chéo = -1 để loại bỏ chính nó
        np.fill_diagonal(sim_matrix, -1)
        logger.info("Set diagonal of similarity matrix to -1")

        recommendations = []
        top_k = 10
        logger.info(f"Generating top-{top_k} recommendations for each track")
        
        # Lấy top-K cho mỗi bài hát
        for idx in range(len(df)):
            track_id = df.iloc[idx]['track_id']
            track_name = df.iloc[idx]['track_name']
            artist_name = df.iloc[idx]['artist_name']
            
            try:
                top_indices = np.argsort(sim_matrix[idx])[-top_k:][::-1]
                top_scores = sim_matrix[idx][top_indices]
                
                for i, score in zip(top_indices, top_scores):
                    rec_row = df.iloc[i]
                    recommendations.append({
                        'track_id': track_id,
                        'track_name': track_name,
                        'artist_name': artist_name,
                        'recommended_id': rec_row['track_id'],
                        'recommended_name': rec_row['track_name'],
                        'recommended_artist': rec_row['artist_name'],
                        'similarity_score': score
                    })
                if idx % 100 == 0:
                    logger.info(f"Processed recommendations for {idx+1}/{len(df)} tracks")
            except Exception as e:
                logger.error(f"Error generating recommendations for track {track_id}: {str(e)}\n{traceback.format_exc()}")
                continue
        
        # Chuyển thành DataFrame và lưu
        logger.info(f"Generated {len(recommendations)} recommendations")
        recommend_df = pd.DataFrame(recommendations)
        output_file = '/opt/airflow/data/recommend.csv'
        logger.info(f"Saving recommendations to {output_file}")
        recommend_df.to_csv(output_file, index=False, encoding='utf-8')
        logger.info(f"Recommendations saved to {output_file}")
        logger.info("Completed train_and_register_model")
    except Exception as e:
        logger.error(f"Error in train_and_register_model: {str(e)}\n{traceback.format_exc()}")
        raise