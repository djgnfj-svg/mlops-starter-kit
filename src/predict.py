"""FastAPI Model Serving Application"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import logging
from typing import List, Dict
from model import MLModel
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="MLOps Model Serving API",
    description="Simple ML model serving with FastAPI",
    version="1.0.0"
)

# Load model
model_path = os.environ.get("MODEL_PATH", "models/latest_model.pkl")
try:
    model = MLModel(model_path=model_path)
    logger.info(f"Model loaded from {model_path}")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    model = None


# Request/Response models
class PredictionRequest(BaseModel):
    features: List[float]
    
    class Config:
        schema_extra = {
            "example": {
                "features": [5.1, 3.5, 1.4, 0.2]
            }
        }


class PredictionResponse(BaseModel):
    prediction: str
    prediction_id: int
    probabilities: Dict[str, float]
    features: List[float]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_path: str


@app.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        model_path=model_path
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make prediction endpoint"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Validate input
        if len(request.features) != 4:
            raise HTTPException(
                status_code=400, 
                detail="Expected 4 features (sepal_length, sepal_width, petal_length, petal_width)"
            )
        
        # Make prediction
        result = model.predict(request.features)
        
        logger.info(f"Prediction made: {result['prediction']} for features {request.features}")
        
        return PredictionResponse(**result)
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/model/info")
async def model_info():
    """Get model information"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": "RandomForestClassifier",
        "features": model.feature_names,
        "targets": model.target_names,
        "model_path": model_path
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)