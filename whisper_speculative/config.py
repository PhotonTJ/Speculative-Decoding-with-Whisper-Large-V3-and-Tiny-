"""Configuration for speculative decoding."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    """Speculative decoding configuration."""
    target_model: str = "openai/whisper-large-v3"
    draft_model: str = "openai/whisper-tiny"
    draft_k: int = 6
    max_new_tokens: int = 128
    top_p: float = 0.0
    temperature: float = 1.0
    device: Optional[str] = None
    language: str = "en"
    task: str = "transcribe"
    max_samples: int = 50
    dataset: str = "hf-internal-testing/librispeech_asr_dummy"
    split: str = "validation"
