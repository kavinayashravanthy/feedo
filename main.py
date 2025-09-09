from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline
import pandas as pd
import os
import uvicorn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    sentiment_analyzer = pipeline("sentiment-analysis")
    emotion_analyzer = pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base"
    )
    print("✅ Sentiment and Emotion models loaded successfully")
except Exception as e:
    raise RuntimeError(f"Model loading failed: {str(e)}")

@app.get("/")
def home():
    return {"message": "Feedback Analyzer Backend is running!"}

@app.post("/analyze-csv/")
async def analyze_csv(file: UploadFile = File(...)):
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

            sent_result = sentiment_analyzer(text)[0]
            sentiment = sent_result["label"]
            score = round(sent_result["score"], 2)

            emo_result = emotion_analyzer(text)
            emotion = emo_result[0]["label"] if emo_result else "Unknown"

            result = {
                "feedback": text,
                "score": score,
                "sentiment": sentiment,
                "emotion": emotion,
            }

            results.append(result)

        return {"results": results}

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing file: {str(e)}"
        )

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
