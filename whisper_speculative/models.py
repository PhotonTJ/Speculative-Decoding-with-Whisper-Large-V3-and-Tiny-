"""Model loading utilities."""

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor


def load_models(config):
    """Load target and draft models with processors."""
    device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    target_processor = AutoProcessor.from_pretrained(config.target_model)
    draft_processor = AutoProcessor.from_pretrained(config.draft_model)
    
    target_model = AutoModelForSpeechSeq2Seq.from_pretrained(
        config.target_model,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    ).to(device).eval()
    
    draft_model = AutoModelForSpeechSeq2Seq.from_pretrained(
        config.draft_model,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    ).to(device).eval()
    
    target_model.config.use_cache = True
    draft_model.config.use_cache = True
    
    return {
        "target_model": target_model,
        "draft_model": draft_model,
        "target_processor": target_processor,
        "draft_processor": draft_processor,
        "device": device,
        "dtype": dtype
    }
