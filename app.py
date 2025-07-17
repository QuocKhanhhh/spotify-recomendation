from fastapi import FastAPI, HTTPException, Query
import pandas as pd

app = FastAPI(title="Spotify Recommendation API")

# Load recommendations khi app khởi chạy
recommend_df = pd.read_csv('recommendations.csv')

@app.get("/")
def read_root():
    return {"message": "Welcome to Spotify Recommendation API"}

@app.get("/recommend/{track_id}")
def recommend(track_id: str, top_k: int = 5):
    results = recommend_df[recommend_df['track_id'] == track_id].head(top_k)

    if results.empty:
        raise HTTPException(status_code=404, detail="Track ID not found or no recommendations available")

    return results.to_dict(orient='records')

@app.get("/recommend_by_name/")
def recommend_by_name(track_name: str = Query(..., description="Full or partial track name"), top_k: int = 5):
    # Tìm track_id theo tên track (case-insensitive contains)
    matched_tracks = recommend_df[recommend_df['recommended_name'].str.contains(track_name, case=False, na=False)]

    if matched_tracks.empty:
        raise HTTPException(status_code=404, detail="No tracks found matching the name")

    # Trả top_k kết quả đầu tiên
    return matched_tracks.head(top_k).to_dict(orient='records')