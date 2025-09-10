
from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline
import pandas as pd
import os
import uvicorn

SECTOR_KEYWORDS = {
    "billing": {
        "primary": ["invoice", "payment", "charge", "bill", "cost", "price", "refund", "money"],
        "secondary": ["expensive", "cheap", "affordable", "subscription", "fee", "transaction"]
    },
    "support": {
        "primary": ["help", "customer service", "agent", "support", "assistance", "representative"],
        "secondary": ["staff", "employee", "team", "helpdesk", "response", "contact"]
    },
    "delivery": {
        "primary": ["shipping", "delivery", "arrival", "late", "fast", "slow", "tracking"],
        "secondary": ["package", "order", "dispatch", "logistics", "courier", "freight"]
    },
    "product": {
        "primary": ["quality", "feature", "functionality", "design", "performance", "defect"],
        "secondary": ["broken", "works", "excellent", "poor", "amazing", "terrible"]
    },
    "website": {
        "primary": ["website", "app", "interface", "login", "navigation", "loading"],
        "secondary": ["browser", "mobile", "desktop", "user experience", "ui", "ux"]
    }
}

def detect_sector(text):
    text_lower = text.lower()
    for sector, keywords in SECTOR_KEYWORDS.items():
        for word in keywords["primary"]:
            if word in text_lower:
                return sector
        for word in keywords["secondary"]:
            if word in text_lower:
                return sector
    return "general"

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
        nps_scores = []
        sector_scores = {}

        for _, row in df.iterrows():
            if isinstance(row["feedback"], str):
                text = row["feedback"].strip()
            else:
                text = ','.join([str(x) for x in row if pd.notnull(x)]).strip()
            if not text:
                continue

            sent_result = sentiment_analyzer(text)[0]
            sentiment = sent_result["label"]
            confidence = sent_result["score"]

            # Score mapping: 1-10 scale
            if sentiment == "POSITIVE":
                score = round(5 + confidence * 5, 1)  # 5.0 to 10.0
            else:
                score = round(6 - confidence * 5, 1)  # 1.0 to 5.0

            nps_scores.append(score)

            emo_result = emotion_analyzer(text)
            emotion = emo_result[0]["label"] if emo_result else "Unknown"

            sector = detect_sector(text)
            sector_scores.setdefault(sector, []).append(score)

            result = {
                "feedback": text,
                "score": score,
                "sentiment": sentiment,
                "emotion": emotion,
                "sector": sector
            }

            results.append(result)

        # NPS Calculation
        promoters = sum(1 for s in nps_scores if s >= 9)
        detractors = sum(1 for s in nps_scores if s <= 6)
        total = len(nps_scores)
        nps = round(((promoters / total) - (detractors / total)) * 100, 1) if total else 0

        # Find sector with lowest average score
        sector_least = "N/A"
        if sector_scores:
            sector_least = min(sector_scores.items(), key=lambda x: sum(x[1])/len(x[1]))[0]

        return {
            "results": results,
            "nps": nps,
            "sector_least": sector_least,
            "sector_scores": {k: round(sum(v)/len(v), 2) for k, v in sector_scores.items()}
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing file: {str(e)}"
        )

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
