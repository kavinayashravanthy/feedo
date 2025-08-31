

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator, Field
from transformers import pipeline
from typing import Optional, Dict, List, Any
from datetime import datetime
from fuzzywuzzy import fuzz
import re
import logging
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Professional Feedback Processor",
    description="""
    A comprehensive API for processing customer feedback with advanced analytics.
    
    ## Features
    - **Sentiment Analysis**: AI-powered sentiment detection with confidence scores
    - **Sector Classification**: Intelligent categorization of feedback by business sector
    - **NPS Processing**: Net Promoter Score extraction and classification
    - **Detailed Analytics**: Comprehensive feedback insights and trends
    
    ## NPS Classifications
    - **Promoters (9-10)**: Loyal enthusiasts who will keep buying and refer others
    - **Passives (7-8)**: Satisfied but unenthusiastic customers who are vulnerable to competitive offerings
    - **Detractors (0-6)**: Unhappy customers who can damage your brand through negative word-of-mouth
    
    ## Sentiment Analysis
    Uses DistilBERT model fine-tuned on Stanford Sentiment Treebank for accurate emotion detection.
    """,
    version="1.0.0",
    contact={
        "name": "API Support",
        "email": "support@yourcompany.com"
    }
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load sentiment analysis model with error handling
try:
    sentiment_analyzer = pipeline(
        "sentiment-analysis",
        model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
        revision="714eb0f"
    )
    logger.info("Sentiment analysis model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load sentiment model: {e}")
    raise RuntimeError("Failed to initialize sentiment analysis model")

# Enhanced sector keywords with fuzzy matching support
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

# Input Models
class FeedbackInput(BaseModel):
    feedback: str = Field(..., description="Customer feedback text", min_length=1, max_length=5000)
    customer_id: Optional[str] = Field(None, description="Optional customer identifier")
    source: Optional[str] = Field(None, description="Feedback source (email, survey, chat, etc.)")
    
    @validator('feedback')
    def feedback_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError('Feedback cannot be empty or whitespace only')
        return v.strip()

class BulkFeedbackInput(BaseModel):
    feedbacks: List[FeedbackInput] = Field(..., description="List of feedback items to process")
    
    @validator('feedbacks')
    def validate_feedback_count(cls, v):
        if len(v) > 100:
            raise ValueError('Maximum 100 feedback items allowed per bulk request')
        if len(v) == 0:
            raise ValueError('At least one feedback item is required')
        return v

# Response Models
class SentimentDetail(BaseModel):
    label: str = Field(..., description="Sentiment label (POSITIVE/NEGATIVE)")
    score: float = Field(..., description="Confidence score (0.0 to 1.0)")
    interpretation: str = Field(..., description="Human-readable sentiment interpretation")

class SectorDetail(BaseModel):
    sector: str = Field(..., description="Identified sector")
    confidence: float = Field(..., description="Classification confidence (0.0 to 1.0)")
    matched_keywords: List[str] = Field(..., description="Keywords that influenced the classification")

class NPSDetail(BaseModel):
    classification: str = Field(..., description="NPS classification (Promoter/Passive/Detractor/Unknown)")
    score: Optional[int] = Field(None, description="Extracted NPS score (0-10)")
    explanation: str = Field(..., description="Detailed explanation of the NPS classification")

class FeedbackResponse(BaseModel):
    message: str = Field(..., description="Processing status message")
    feedback_id: str = Field(..., description="Unique identifier for this feedback")
    original_feedback: str = Field(..., description="Original feedback text")
    processed_at: datetime = Field(..., description="Processing timestamp")
    sentiment: SentimentDetail = Field(..., description="Detailed sentiment analysis")
    sector: SectorDetail = Field(..., description="Sector classification details")
    nps: NPSDetail = Field(..., description="NPS analysis details")
    customer_id: Optional[str] = Field(None, description="Customer identifier if provided")
    source: Optional[str] = Field(None, description="Feedback source if provided")

class AnalyticsResponse(BaseModel):
    total_processed: int
    sentiment_distribution: Dict[str, int]
    sector_distribution: Dict[str, int]
    nps_distribution: Dict[str, int]
    average_sentiment_confidence: float
    processing_summary: str

# Enhanced sector extraction with fuzzy matching
def extract_sector_enhanced(feedback: str) -> tuple[str, float, List[str]]:
    """
    Enhanced sector extraction with confidence scoring and matched keywords tracking.
    
    Returns:
        tuple: (sector, confidence_score, matched_keywords)
    """
    feedback_lower = feedback.lower()
    sector_scores = {}
    matched_keywords_per_sector = {}
    
    for sector, keyword_groups in SECTOR_KEYWORDS.items():
        score = 0
        matched_keywords = []
        
        # Check primary keywords (higher weight)
        for keyword in keyword_groups["primary"]:
            if keyword in feedback_lower:
                score += 10
                matched_keywords.append(keyword)
            else:
                # Fuzzy matching for typos
                words = feedback_lower.split()
                for word in words:
                    if fuzz.ratio(keyword, word) > 80:
                        score += 8
                        matched_keywords.append(f"{keyword}~{word}")
        
        # Check secondary keywords (lower weight)
        for keyword in keyword_groups["secondary"]:
            if keyword in feedback_lower:
                score += 5
                matched_keywords.append(keyword)
        
        if score > 0:
            sector_scores[sector] = score
            matched_keywords_per_sector[sector] = matched_keywords
    
    if not sector_scores:
        return "general", 0.0, []
    
    # Get the sector with highest score
    best_sector = max(sector_scores.items(), key=lambda x: x[1])
    sector_name = best_sector[0]
    raw_score = best_sector[1]
    
    # Calculate confidence (normalize to 0-1)
    max_possible_score = len(SECTOR_KEYWORDS[sector_name]["primary"]) * 10 + \
                        len(SECTOR_KEYWORDS[sector_name]["secondary"]) * 5
    confidence = min(raw_score / max_possible_score, 1.0)
    
    return sector_name, confidence, matched_keywords_per_sector[sector_name]

# Enhanced NPS extraction with multiple patterns
def extract_nps_enhanced(feedback: str) -> tuple[Optional[int], str]:
    """
    Enhanced NPS extraction supporting multiple rating patterns.
    
    Returns:
        tuple: (score, classification)
    """
    patterns = [
        (r'(\d{1,2})\s*/\s*10', "X/10 format"),
        (r'(\d{1,2})\s*out\s*of\s*10', "X out of 10 format"),
        (r'rate\s*(?:it\s*)?(\d{1,2})', "rate X format"),
        (r'score\s*(?:of\s*)?(\d{1,2})', "score X format"),
        (r'(\d{1,2})\s*(?:\/|out\s*of)\s*10', "flexible format"),
        (r'give\s*(?:it\s*)?(?:a\s*)?(\d{1,2})', "give X format")
    ]
    
    for pattern, description in patterns:
        match = re.search(pattern, feedback.lower())
        if match:
            try:
                score = int(match.group(1))
                if 0 <= score <= 10:
                    if score >= 9:
                        return score, "Promoter"
                    elif score >= 7:
                        return score, "Passive"
                    else:
                        return score, "Detractor"
            except ValueError:
                continue
    
    return None, "Unknown"

# Generate sentiment interpretation
def get_sentiment_interpretation(label: str, score: float) -> str:
    """Generate human-readable sentiment interpretation."""
    confidence_level = "high" if score > 0.8 else "medium" if score > 0.6 else "low"
    
    if label == "POSITIVE":
        return f"Customer expresses positive sentiment with {confidence_level} confidence"
    else:
        return f"Customer expresses negative sentiment with {confidence_level} confidence"

# Generate NPS explanation
def get_nps_explanation(classification: str, score: Optional[int]) -> str:
    """Generate detailed NPS explanation."""
    explanations = {
        "Promoter": "Loyal enthusiasts who will keep buying and refer others, fueling growth",
        "Passive": "Satisfied but unenthusiastic customers vulnerable to competitive offerings",
        "Detractor": "Unhappy customers who can damage brand through negative word-of-mouth",
        "Unknown": "No clear NPS score detected in the feedback text"
    }
    
    base_explanation = explanations[classification]
    
    if score is not None:
        return f"Score {score}/10 - {base_explanation}"
    return base_explanation

# In-memory storage for analytics (replace with database in production)
feedback_storage: List[Dict[str, Any]] = []

# API Endpoints
@app.get("/", tags=["Health"])
def root():
    """Root endpoint with API information."""
    return {
        "message": "Professional Feedback Processor API",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "process_feedback": "/feedback",
            "bulk_process": "/feedback/bulk",
            "analytics": "/analytics",
            "health": "/health"
        }
    }

@app.get("/health", tags=["Health"])
def health_check():
    """Health check endpoint for monitoring."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_status": "loaded",
        "total_processed": len(feedback_storage)
    }

@app.post("/feedback", response_model=FeedbackResponse, tags=["Feedback Processing"])
def process_feedback(data: FeedbackInput):
    """
    Process individual customer feedback with comprehensive analysis.
    
    This endpoint analyzes customer feedback and provides:
    - Sentiment analysis with confidence scores
    - Sector classification based on content
    - NPS score extraction and classification
    - Detailed explanations for all metrics
    """
    try:
        feedback = data.feedback
        feedback_id = f"fb_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(feedback_storage) + 1}"
        
        # Sentiment Analysis
        sentiment_result = sentiment_analyzer(feedback)[0]
        sentiment_interpretation = get_sentiment_interpretation(
            sentiment_result["label"], 
            sentiment_result["score"]
        )
        
        # Sector Classification
        sector, sector_confidence, matched_keywords = extract_sector_enhanced(feedback)
        
        # NPS Analysis
        nps_score, nps_classification = extract_nps_enhanced(feedback)
        nps_explanation = get_nps_explanation(nps_classification, nps_score)
        
        # Create response
        response = FeedbackResponse(
            message="Feedback processed successfully",
            feedback_id=feedback_id,
            original_feedback=feedback,
            processed_at=datetime.now(),
            sentiment=SentimentDetail(
                label=sentiment_result["label"],
                score=round(sentiment_result["score"], 4),
                interpretation=sentiment_interpretation
            ),
            sector=SectorDetail(
                sector=sector,
                confidence=round(sector_confidence, 4),
                matched_keywords=matched_keywords
            ),
            nps=NPSDetail(
                classification=nps_classification,
                score=nps_score,
                explanation=nps_explanation
            ),
            customer_id=data.customer_id,
            source=data.source
        )
        
        # Store for analytics (replace with database in production)
        feedback_storage.append(response.dict())
        
        logger.info(f"Processed feedback {feedback_id} - Sentiment: {sentiment_result['label']}, Sector: {sector}, NPS: {nps_classification}")
        
        return response
        
    except Exception as e:
        logger.error(f"Error processing feedback: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing feedback: {str(e)}"
        )

@app.post("/feedback/bulk", tags=["Feedback Processing"])
def process_bulk_feedback(data: BulkFeedbackInput):
    """
    Process multiple feedback items in a single request.
    
    Efficiently processes up to 100 feedback items with detailed analytics.
    """
    try:
        results = []
        errors = []
        
        for idx, feedback_item in enumerate(data.feedbacks):
            try:
                result = process_feedback(feedback_item)
                results.append(result)
            except Exception as e:
                errors.append({
                    "index": idx,
                    "feedback": feedback_item.feedback[:100] + "..." if len(feedback_item.feedback) > 100 else feedback_item.feedback,
                    "error": str(e)
                })
        
        return {
            "message": f"Bulk processing completed",
            "total_submitted": len(data.feedbacks),
            "successfully_processed": len(results),
            "errors": len(errors),
            "results": results,
            "processing_errors": errors if errors else None
        }
        
    except Exception as e:
        logger.error(f"Error in bulk processing: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Bulk processing error: {str(e)}"
        )

@app.get("/analytics", response_model=AnalyticsResponse, tags=["Analytics"])
def get_analytics():
    """
    Get comprehensive analytics for all processed feedback.
    
    Returns detailed statistics including:
    - Sentiment distribution
    - Sector breakdown
    - NPS classification summary
    - Processing insights
    """
    if not feedback_storage:
        return AnalyticsResponse(
            total_processed=0,
            sentiment_distribution={},
            sector_distribution={},
            nps_distribution={},
            average_sentiment_confidence=0.0,
            processing_summary="No feedback processed yet"
        )
    
    # Calculate distributions
    sentiment_dist = {}
    sector_dist = {}
    nps_dist = {}
    confidence_sum = 0
    
    for feedback in feedback_storage:
        # Sentiment distribution
        sentiment_label = feedback["sentiment"]["label"]
        sentiment_dist[sentiment_label] = sentiment_dist.get(sentiment_label, 0) + 1
        confidence_sum += feedback["sentiment"]["score"]
        
        # Sector distribution
        sector = feedback["sector"]["sector"]
        sector_dist[sector] = sector_dist.get(sector, 0) + 1
        
        # NPS distribution
        nps_class = feedback["nps"]["classification"]
        nps_dist[nps_class] = nps_dist.get(nps_class, 0) + 1
    
    avg_confidence = confidence_sum / len(feedback_storage)
    
    # Generate summary
    total = len(feedback_storage)
    positive_pct = round((sentiment_dist.get("POSITIVE", 0) / total) * 100, 1)
    promoters_pct = round((nps_dist.get("Promoter", 0) / total) * 100, 1)
    top_sector = max(sector_dist.items(), key=lambda x: x[1])[0] if sector_dist else "None"
    
    summary = f"Processed {total} feedback items. {positive_pct}% positive sentiment, {promoters_pct}% promoters. Top concern area: {top_sector}"
    
    return AnalyticsResponse(
        total_processed=total,
        sentiment_distribution=sentiment_dist,
        sector_distribution=sector_dist,
        nps_distribution=nps_dist,
        average_sentiment_confidence=round(avg_confidence, 4),
        processing_summary=summary
    )

@app.get("/analytics/nps-score", tags=["Analytics"])
def get_nps_score():
    """
    Calculate the overall Net Promoter Score.
    
    NPS = % Promoters - % Detractors
    Range: -100 to +100
    """
    if not feedback_storage:
        return {"nps_score": None, "message": "No feedback data available"}
    
    total_with_nps = 0
    promoters = 0
    detractors = 0
    
    for feedback in feedback_storage:
        nps_class = feedback["nps"]["classification"]
        if nps_class in ["Promoter", "Passive", "Detractor"]:
            total_with_nps += 1
            if nps_class == "Promoter":
                promoters += 1
            elif nps_class == "Detractor":
                detractors += 1
    
    if total_with_nps == 0:
        return {"nps_score": None, "message": "No valid NPS scores found"}
    
    promoter_pct = (promoters / total_with_nps) * 100
    detractor_pct = (detractors / total_with_nps) * 100
    nps_score = promoter_pct - detractor_pct
    
    # Interpretation
    if nps_score > 70:
        interpretation = "Excellent - World-class customer loyalty"
    elif nps_score > 50:
        interpretation = "Great - Strong customer loyalty"
    elif nps_score > 30:
        interpretation = "Good - Above average loyalty"
    elif nps_score > 0:
        interpretation = "Fair - Room for improvement"
    else:
        interpretation = "Poor - Significant loyalty issues"
    
    return {
        "nps_score": round(nps_score, 2),
        "promoters": promoters,
        "passives": total_with_nps - promoters - detractors,
        "detractors": detractors,
        "total_scored_feedback": total_with_nps,
        "interpretation": interpretation,
        "promoter_percentage": round(promoter_pct, 2),
        "detractor_percentage": round(detractor_pct, 2)
    }

@app.get("/feedback/{feedback_id}", tags=["Feedback Processing"])
def get_feedback_by_id(feedback_id: str):
    """Retrieve specific feedback analysis by ID."""
    for feedback in feedback_storage:
        if feedback["feedback_id"] == feedback_id:
            return feedback
    
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Feedback with ID {feedback_id} not found"
    )

@app.delete("/analytics/reset", tags=["Analytics"])
def reset_analytics():
    """Reset all stored feedback data (development use only)."""
    global feedback_storage
    old_count = len(feedback_storage)
    feedback_storage.clear()
    return {
        "message": f"Analytics reset successfully. Cleared {old_count} feedback records.",
        "timestamp": datetime.now().isoformat()
    }

# Exception handlers
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    return HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        detail=str(exc)
    )

# Startup event
@app.on_event("startup")
async def startup_event():
    logger.info("Professional Feedback Processor API started successfully")
    logger.info("Sentiment model loaded and ready")
    logger.info(f"Supporting {len(SECTOR_KEYWORDS)} sector classifications")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

"""
DEPLOYMENT INSTRUCTIONS:

1. Install dependencies:
   pip install fastapi uvicorn transformers torch pydantic fuzzywuzzy python-levenshtein

2. Run the application:
   python main.py
   
   Or with uvicorn directly:
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload

3. Access the API:
   - API Documentation: http://localhost:8000/docs
   - Alternative docs: http://localhost:8000/redoc
   - Health check: http://localhost:8000/health

4. Example usage:
   curl -X POST "http://localhost:8000/feedback" \
   -H "Content-Type: application/json" \
   -d '{"feedback": "The delivery was late but customer service was helpful. I rate it 7/10"}'

PRODUCTION CONSIDERATIONS:
- Replace in-memory storage with a proper database (PostgreSQL, MongoDB)
- Add authentication and rate limiting
- Implement caching for model predictions
- Add comprehensive monitoring and alerting
- Use environment variables for configuration
- Add data retention policies
- Implement backup and recovery procedures

API FEATURES:
- Comprehensive sentiment analysis with confidence scoring
- Multi-sector classification with fuzzy keyword matching
- Advanced NPS extraction supporting multiple rating formats
- Detailed analytics and reporting
- Bulk processing capabilities
- Individual feedback retrieval
- Overall NPS score calculation
- Professional error handling and validation

"""
