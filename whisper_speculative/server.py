"""REST API for Whisper Speculative Decoding using FastAPI."""

import os
import tempfile
from typing import List, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .api import SpeculativeWhisper

app = FastAPI(
    title="Whisper Speculative Decoding API",
    description="Fast speech-to-text using speculative decoding with Whisper",
    version="1.0.0",
)

# Global model instance (lazy loaded)
_model: Optional[SpeculativeWhisper] = None


class TranscriptionResponse(BaseModel):
    """Response model for transcription."""
    text: str
    duration: float


class BatchTranscriptionResponse(BaseModel):
    """Response model for batch transcription."""
    results: List[TranscriptionResponse]
    total_time: float


class ModelConfig(BaseModel):
    """Model configuration."""
    draft_model: str = "tiny"
    final_model: str = "large-v3"
    draft_k: int = 6
    top_p: float = 0.0
    device: Optional[str] = None


def get_model() -> SpeculativeWhisper:
    """Get or initialize the global model instance."""
    global _model
    if _model is None:
        _model = SpeculativeWhisper(
            draft_model=os.getenv("DRAFT_MODEL", "tiny"),
            final_model=os.getenv("FINAL_MODEL", "large-v3"),
            draft_k=int(os.getenv("DRAFT_K", "6")),
            top_p=float(os.getenv("TOP_P", "0.0")),
            device=os.getenv("DEVICE", None),
        )
    return _model


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "message": "Whisper Speculative Decoding API"}


@app.get("/health")
async def health():
    """Health check with model status."""
    return {
        "status": "ok",
        "model_loaded": _model is not None,
    }


@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe(
    file: UploadFile = File(...),
    max_tokens: int = Query(128, ge=1, le=448),
    language: str = Query("en"),
):
    """
    Transcribe a single audio file.
    
    Args:
        file: Audio file (WAV, MP3, FLAC, etc.)
        max_tokens: Maximum tokens to generate
        language: Target language code
    """
    try:
        model = get_model()
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        try:
            result = model.transcribe(tmp_path, max_tokens=max_tokens, return_timing=True)
            return TranscriptionResponse(
                text=result["texts"],
                duration=result["total_time"],
            )
        finally:
            os.unlink(tmp_path)
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/transcribe/batch", response_model=BatchTranscriptionResponse)
async def transcribe_batch(
    files: List[UploadFile] = File(...),
    max_tokens: int = Query(128, ge=1, le=448),
):
    """
    Transcribe multiple audio files.
    
    Args:
        files: List of audio files
        max_tokens: Maximum tokens per transcription
    """
    try:
        model = get_model()
        
        # Save all files temporarily
        tmp_paths = []
        for file in files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                content = await file.read()
                tmp.write(content)
                tmp_paths.append(tmp.name)
        
        try:
            result = model.transcribe(tmp_paths, max_tokens=max_tokens, return_timing=True)
            
            texts = result["texts"] if isinstance(result["texts"], list) else [result["texts"]]
            avg_time = result["total_time"] / len(texts)
            
            return BatchTranscriptionResponse(
                results=[
                    TranscriptionResponse(text=text, duration=avg_time)
                    for text in texts
                ],
                total_time=result["total_time"],
            )
        finally:
            for path in tmp_paths:
                os.unlink(path)
                
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reload")
async def reload_model(config: ModelConfig):
    """Reload model with new configuration."""
    global _model
    try:
        _model = SpeculativeWhisper(
            draft_model=config.draft_model,
            final_model=config.final_model,
            draft_k=config.draft_k,
            top_p=config.top_p,
            device=config.device,
        )
        return {"status": "ok", "message": "Model reloaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the FastAPI server."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
