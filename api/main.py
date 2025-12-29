# api/main.py
"""
FastAPI service for speech enhancement.
Phase 3: Model Deployment
"""
import os
import io
import time
import logging
from datetime import datetime
from typing import Optional

import torch
import librosa
import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse, Response
from pydantic import BaseModel
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST, REGISTRY
from fastapi.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware

# Import model
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import UNet
from src.metrics import (
    request_count, 
    request_duration, 
    predictions_total,
    prediction_duration,
    audio_file_size,
    api_health,
    model_loaded,
    processing_success_rate,
    processing_rtf
)

# Ensure logs directory exists before configuring logging
os.makedirs('logs', exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/api.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

# --- Configuration ---
SAMPLE_RATE = 16000
N_FFT = 512
HOP_LENGTH = 256
MODEL_PATH = os.getenv("MODEL_PATH", "models/trained/best_model.pth")

# Initialize FastAPI app
app = FastAPI(
    title="SpeechOps API",
    description="Speech Enhancement API using U-Net model",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Global model variable
model = None
device = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    timestamp: str


class PredictResponse(BaseModel):
    success: bool
    message: str
    processing_time_ms: float
    input_duration_sec: float
    output_duration_sec: float


def load_model():
    """Load the model into memory."""
    global model, device
    
    # Set device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    logger.info(f"Using device: {device}")
    
    # Load model
    model = UNet(in_channels=1, out_channels=1).to(device)
    
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        logger.info(f"Model loaded from {MODEL_PATH}")
    else:
        logger.warning(f"Model file not found at {MODEL_PATH}. Using untrained model.")
    
    model.eval()
    return model


def denoise_audio_bytes(audio_bytes: bytes) -> tuple[bytes, float, float]:
    """
    Denoise audio from bytes and return denoised audio bytes.
    Returns: (denoised_bytes, input_duration, output_duration)
    """
    global model, device
    
    # Load audio from bytes
    audio_buffer = io.BytesIO(audio_bytes)
    noisy_wav, sr = librosa.load(audio_buffer, sr=SAMPLE_RATE)
    input_duration = len(noisy_wav) / SAMPLE_RATE
    
    # Create spectrogram and phase
    noisy_spec = librosa.stft(noisy_wav, n_fft=N_FFT, hop_length=HOP_LENGTH)
    noisy_mag, noisy_phase = librosa.magphase(noisy_spec)
    
    # Convert to tensor: [Batch, Channel, Frequency, Time]
    noisy_tensor = torch.from_numpy(noisy_mag).float().unsqueeze(0).unsqueeze(0).to(device)
    
    # Inference
    with torch.no_grad():
        denoised_mag_tensor = model(noisy_tensor)
    
    # Convert back to numpy
    denoised_mag = denoised_mag_tensor.cpu().squeeze(0).squeeze(0).numpy()
    
    # Reconstruct audio waveform
    denoised_wav = librosa.istft(denoised_mag * noisy_phase, hop_length=HOP_LENGTH, length=len(noisy_wav))
    output_duration = len(denoised_wav) / SAMPLE_RATE
    
    # Convert to bytes
    output_buffer = io.BytesIO()
    sf.write(output_buffer, denoised_wav, SAMPLE_RATE, format='WAV')
    output_buffer.seek(0)
    
    return output_buffer.getvalue(), input_duration, output_duration


# Middleware: Record request metrics
class MetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        start_time = time.time()
        
        response = await call_next(request)
        
        duration = time.time() - start_time
        request_count.labels(
            method=request.method,
            endpoint=request.url.path,
            status=response.status_code
        ).inc()
        
        request_duration.labels(
            method=request.method,
            endpoint=request.url.path
        ).observe(duration)
        
        return response

app.add_middleware(MetricsMiddleware)

@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    os.makedirs("logs", exist_ok=True)
    load_model()
    logger.info("API started successfully")


@app.get("/", response_class=JSONResponse)
async def root():
    """Root endpoint."""
    return {
        "message": "SpeechOps API",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict"
    }


# Add Metrics endpoint
@app.get("/metrics", include_in_schema=False)
async def metrics():
    """Prometheus metrics"""
    return Response(
        content=generate_latest(REGISTRY),
        media_type=CONTENT_TYPE_LATEST
    )


# Health check
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    health_status = 1 if model is not None else 0
    api_health.set(health_status)
    model_loaded.set(health_status)
    
    return HealthResponse(
        status="healthy" if model else "unhealthy",
        model_loaded=model is not None,
        device=str(device) if device else "not initialized",
        timestamp=datetime.now().isoformat()
    )


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Denoise audio file"""
    start_time = time.time()
    
    try:
        # Record file size
        audio_bytes = await file.read()
        audio_file_size.observe(len(audio_bytes))
        
        # Make prediction
        pred_start = time.time()
        denoised_bytes, input_dur, output_dur = denoise_audio_bytes(audio_bytes)
        pred_duration = time.time() - pred_start
        
        # Update metrics
        prediction_duration.observe(pred_duration)
        predictions_total.labels(status="success").inc()
        processing_success_rate.set(1.0)
        
        # Calculate and observe RTF
        if input_dur > 0:
            rtf = pred_duration / input_dur
            processing_rtf.observe(rtf)
            
            logger.info(f"RTF: {rtf:.4f} (Pred: {pred_duration:.2f}s / Input: {input_dur:.2f}s)")
        
        processing_time = (time.time() - start_time) * 1000
        
        logger.info(
            f"Processed: {file.filename} | "
            f"Duration: {input_dur:.2f}s | "
            f"Processing: {processing_time:.0f}ms"
        )
        
        output_filename = f"denoised_{file.filename.rsplit('.', 1)[0]}.wav"
        return StreamingResponse(
            io.BytesIO(denoised_bytes),
            media_type="audio/wav",
            headers={
                "Content-Disposition": f"attachment; filename={output_filename}",
                "X-Processing-Time-Ms": str(round(processing_time, 2)),
                "X-Input-Duration-Sec": str(round(input_dur, 2)),
                "X-Output-Duration-Sec": str(round(output_dur, 2))
            }
        )
    
    except Exception as e:
        predictions_total.labels(status="error").inc()
        processing_success_rate.set(0.0)
        logger.error(f"Error processing audio: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")


@app.post("/predict/json", response_model=PredictResponse)
async def predict_json(file: UploadFile = File(...)):
    """
    Denoise an audio file and return metadata (without audio).
    Useful for testing and monitoring.
    """
    start_time = time.time()
    
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    try:
        audio_bytes = await file.read()
        _, input_dur, output_dur = denoise_audio_bytes(audio_bytes)
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    processing_time = (time.time() - start_time) * 1000
    
    return PredictResponse(
        success=True,
        message=f"Successfully processed {file.filename}",
        processing_time_ms=round(processing_time, 2),
        input_duration_sec=round(input_dur, 2),
        output_duration_sec=round(output_dur, 2)
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)

