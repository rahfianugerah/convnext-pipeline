#!/usr/bin/env python3
"""
FastAPI REST API Server for Model Serving
"""

import io
import os
import time
import json
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

import torch
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Add parent directory to path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from production.inference.inference import InferenceEngine
from production.validation.input_validator import InputValidator
from production.monitoring.logger import PredictionLogger

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="ConvNeXt Multi-Task API",
    description="Production API for Classification, Object Detection, and OCR",
    version="1.0.0"
)

# Global state
inference_engine: Optional[InferenceEngine] = None
validator: Optional[InputValidator] = None
prediction_logger: Optional[PredictionLogger] = None

# Configuration from environment
MODEL_PATH = os.getenv("MODEL_PATH", "./artifacts/classification/best_model.pth")
THEME = os.getenv("THEME", "classification")
DEVICE = os.getenv("DEVICE", "auto")
LOG_DIR = os.getenv("LOG_DIR", "./logs")


# Pydantic models for request/response
class PredictionResponse(BaseModel):
    """Standard prediction response"""
    success: bool
    theme: str
    prediction: Dict[str, Any]
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: str


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    theme: str
    device: str
    timestamp: str


class ErrorResponse(BaseModel):
    """Error response"""
    error: str
    detail: Optional[str] = None
    timestamp: str


@app.on_event("startup")
async def startup_event():
    """Initialize models and services on startup"""
    global inference_engine, validator, prediction_logger
    
    logger.info("Starting API server...")
    logger.info(f"Model path: {MODEL_PATH}")
    logger.info(f"Theme: {THEME}")
    logger.info(f"Device: {DEVICE}")
    
    try:
        # Initialize inference engine
        inference_engine = InferenceEngine(MODEL_PATH, THEME, DEVICE)
        logger.info("✓ Inference engine loaded")
        
        # Initialize validator
        validator = InputValidator(THEME, strict=False)
        logger.info("✓ Input validator initialized")
        
        # Initialize logger
        os.makedirs(LOG_DIR, exist_ok=True)
        prediction_logger = PredictionLogger(LOG_DIR)
        logger.info("✓ Prediction logger initialized")
        
        logger.info("API server ready!")
        
    except Exception as e:
        logger.error(f"Failed to initialize: {str(e)}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down API server...")
    if prediction_logger:
        prediction_logger.close()


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "ConvNeXt Multi-Task API",
        "version": "1.0.0",
        "theme": THEME,
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "predict_batch": "/predict/batch",
            "metrics": "/metrics"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if inference_engine is not None else "unhealthy",
        model_loaded=inference_engine is not None,
        theme=THEME,
        device=str(inference_engine.device) if inference_engine else "unknown",
        timestamp=datetime.now().isoformat()
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None,
    img_size: int = 224,
    det_size: int = 640,
    ocr_h: int = 32,
    ocr_w: int = 256,
    conf_threshold: float = 0.5
):
    """
    Predict on a single image
    
    Args:
        file: Image file
        img_size: Image size for classification
        det_size: Image size for detection
        ocr_h: OCR image height
        ocr_w: OCR image width
        conf_threshold: Confidence threshold for detection
    """
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = time.time()
    
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Convert to numpy array (OpenCV format)
        import cv2
        img_array = np.array(image)
        if len(img_array.shape) == 2:  # Grayscale
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
        elif img_array.shape[2] == 4:  # RGBA
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
        elif img_array.shape[2] == 3:  # RGB
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Save to temporary file for processing
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
            tmp_path = tmp_file.name
            cv2.imwrite(tmp_path, img_array)
        
        try:
            # Validate image
            if validator:
                is_valid, validation_report = validator.validate_and_preprocess(tmp_path)
                if not is_valid:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid image: {validation_report['errors']}"
                    )
            
            # Run inference
            kwargs = {
                "img_size": img_size,
                "det_size": det_size,
                "ocr_h": ocr_h,
                "ocr_w": ocr_w,
                "conf_threshold": conf_threshold
            }
            
            prediction = inference_engine.predict(tmp_path, **kwargs)
            
            # Calculate inference time
            inference_time = time.time() - start_time
            
            # Build response
            response = PredictionResponse(
                success=True,
                theme=THEME,
                prediction=prediction,
                metadata={
                    "filename": file.filename,
                    "inference_time_ms": round(inference_time * 1000, 2),
                    "image_size": image.size,
                    "device": str(inference_engine.device)
                },
                timestamp=datetime.now().isoformat()
            )
            
            # Log prediction in background
            if prediction_logger and background_tasks:
                background_tasks.add_task(
                    prediction_logger.log_prediction,
                    theme=THEME,
                    prediction=prediction,
                    metadata=response.metadata
                )
            
            return response
            
        finally:
            # Cleanup temp file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch")
async def predict_batch(
    files: List[UploadFile] = File(...),
    background_tasks: BackgroundTasks = None,
    img_size: int = 224,
    det_size: int = 640,
    ocr_h: int = 32,
    ocr_w: int = 256,
    conf_threshold: float = 0.5
):
    """
    Predict on multiple images
    
    Args:
        files: List of image files
        Other args same as predict endpoint
    """
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    results = []
    
    for file in files:
        try:
            # Call predict for each file
            result = await predict(
                file=file,
                background_tasks=background_tasks,
                img_size=img_size,
                det_size=det_size,
                ocr_h=ocr_h,
                ocr_w=ocr_w,
                conf_threshold=conf_threshold
            )
            results.append(result.dict())
        except Exception as e:
            results.append({
                "success": False,
                "filename": file.filename,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
    
    return {
        "total": len(files),
        "successful": sum(1 for r in results if r.get("success", False)),
        "failed": sum(1 for r in results if not r.get("success", True)),
        "results": results
    }


@app.get("/metrics")
async def get_metrics():
    """Get API metrics"""
    if prediction_logger:
        return prediction_logger.get_metrics()
    return {"error": "Metrics not available"}


@app.get("/model/info")
async def get_model_info():
    """Get model information"""
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "theme": THEME,
        "model_path": MODEL_PATH,
        "device": str(inference_engine.device),
        "metadata": inference_engine.metadata
    }


def main():
    """Run the API server"""
    import argparse
    
    parser = argparse.ArgumentParser(description="ConvNeXt API Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    
    uvicorn.run(
        "production.api.server:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=args.reload
    )


if __name__ == "__main__":
    main()
