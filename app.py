from fastapi import FastAPI, HTTPException
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
