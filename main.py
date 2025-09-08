"""
Professional Feedback Processor API
Backend for analyzing uploaded CSV feedback with sentiment + emotion
and storing results in MongoDB.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline
from pymongo import MongoClient
import pandas as pd
import os
import uvicorn

# ------------------------------
# FastAPI app initialization
# ------------------------------
app = FastAPI(
    title="Professional Feedback Processor",
    description="Analyze customer feedback with sentiment + emotion",
    version="1.0.0",
)

# Allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all for testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------
# Database setup (MongoDB Atlas)
# ------------------------------
MONGO_URI = os.getenv(
    "MONGO_URI",
    "mongodb+srv://kavinayashravanthy_db_user:kavinaya2007@feedo.xetbc1f.mongodb.net/sample_mflix?retryWrites=true&w=majority"
)
client = MongoClient(MONGO_URI)
db = client["sample_mflix"]
feedback_collection = db["feedback"]

# ------------------------------
# Hugging Face Pipelines
# ------------------------------
try:
    sentiment_analyzer = pipeline("sentiment-analysis")
    emotion_analyzer = pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base"
    )
    print("✅ Sentiment and Emotion models loaded successfully")
except Exception as e:
    raise RuntimeError(f"Model loading failed: {str(e)}")

# ------------------------------
# API Routes
# ------------------------------
@app.get("/")
def home():
    return {"message": "Feedback Analyzer Backend is running!"}

@app.post("/analyze-csv/")
async def analyze_csv(file: UploadFile = File(...)):
    """Upload a CSV file with 'feedback' column and analyze."""
    try:
        file.file.seek(0)
        df = pd.read_csv(file.file)

        if "feedback" not in df.columns:
            raise HTTPException(status_code=400, detail="CSV must contain a 'feedback' column")

        results = []
        for _, row in df.iterrows():
            text = str(row["feedback"]).strip()
            if not text:
                continue

            # Sentiment
            sent_result = sentiment_analyzer(text)[0]
            sentiment = sent_result["label"]
            score = round(sent_result["score"], 2)

            # Emotion
            emo_result = emotion_analyzer(text)
            emotion = emo_result[0]["label"] if emo_result else "Unknown"

            result = {
                "feedback": text,
                "score": score,
                "sentiment": sentiment,
                "emotion": emotion,
            }

            try:
                feedback_collection.insert_one(result)
            except Exception as db_err:
                print(f"⚠️ Mongo insert failed: {db_err}")

            results.append(result)

        return {"results": results}

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing file: {str(e)}"
        )

# ------------------------------
# Run the app
# ------------------------------
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)