"""High-level API for Whisper Speculative Decoding."""

import torch
import numpy as np
from typing import List, Union, Optional
from pathlib import Path

from .config import Config
from .models import load_models
from .speculative import speculative_decode


class SpeculativeWhisper:
    """
    High-level API for speculative decoding with Whisper models.
    
    Example:
        sw = SpeculativeWhisper(draft_model="tiny", final_model="large-v3")
        outputs = sw.transcribe(["audio1.wav", "audio2.wav"])
        for audio, text in zip(audio_files, outputs):
            print(f"{audio}: {text}")
    """
    
    MODEL_MAP = {
        "tiny": "openai/whisper-tiny",
        "base": "openai/whisper-base",
        "small": "openai/whisper-small",
        "medium": "openai/whisper-medium",
        "large": "openai/whisper-large",
        "large-v2": "openai/whisper-large-v2",
        "large-v3": "openai/whisper-large-v3",
    }
    
    def __init__(
        self,
        draft_model: str = "tiny",
        final_model: str = "large-v3",
        device: Optional[str] = None,
        draft_k: int = 6,
        top_p: float = 0.0,
        temperature: float = 1.0,
        language: str = "en",
        task: str = "transcribe",
    ):
        """
        Initialize SpeculativeWhisper.
        
        Args:
            draft_model: Draft model name ("tiny", "base", "small", etc.)
            final_model: Target model name ("large-v3", "large-v2", etc.)
            device: Device to use ("cuda", "cpu", or None for auto)
            draft_k: Number of draft tokens per speculation step
            top_p: Top-p sampling (0.0 for greedy)
            temperature: Sampling temperature
            language: Target language code
            task: Task type ("transcribe" or "translate")
        """
        self.draft_k = draft_k
        self.top_p = top_p
        self.temperature = temperature
        self.language = language
        self.task = task
        
        # Resolve model names
        draft_name = self.MODEL_MAP.get(draft_model, draft_model)
        final_name = self.MODEL_MAP.get(final_model, final_model)
        
        # Load models
        config = Config(
            target_model=final_name,
            draft_model=draft_name,
            device=device,
            language=language,
            task=task,
        )
        
        print(f"Loading models: {draft_model} (draft) + {final_model} (target)...")
        self.models = load_models(config)
        self.device = self.models["device"]
        self.dtype = self.models["dtype"]
        print(f"Ready on {self.device}")
    
    def transcribe(
        self,
        audio: Union[str, Path, np.ndarray, List[Union[str, Path, np.ndarray]]],
        max_tokens: int = 128,
        batch_size: int = 1,
        return_timing: bool = False,
    ) -> Union[str, List[str], dict]:
        """
        Transcribe audio file(s) using speculative decoding.
        
        Args:
            audio: Single audio file path/array or list of audio files/arrays
            max_tokens: Maximum tokens to generate
            batch_size: Batch size for processing (currently processes sequentially)
            return_timing: If True, return dict with text and timing info
            
        Returns:
            Transcribed text (str) or list of texts, or dict with timing if requested
        """
        import time
        import soundfile as sf
        
        # Handle single input
        single_input = not isinstance(audio, list)
        if single_input:
            audio = [audio]
        
        target_model = self.models["target_model"]
        draft_model = self.models["draft_model"]
        target_processor = self.models["target_processor"]
        draft_processor = self.models["draft_processor"]
        
        forced_decoder_ids = target_processor.get_decoder_prompt_ids(
            language=self.language, task=self.task
        )
        eos_token_id = target_model.config.eos_token_id
        
        results = []
        total_time = 0.0
        
        for audio_input in audio:
            # Load audio if path
            if isinstance(audio_input, (str, Path)):
                audio_array, sample_rate = sf.read(str(audio_input))
                if sample_rate != 16000:
                    import librosa
                    audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=16000)
            else:
                audio_array = audio_input
            
            # Prepare inputs
            target_inputs = target_processor(
                audio_array, sampling_rate=16000, return_tensors="pt"
            )
            draft_inputs = draft_processor(
                audio_array, sampling_rate=16000, return_tensors="pt"
            )
            
            target_inputs = {
                k: v.to(self.device, dtype=self.dtype if v.dtype.is_floating_point else v.dtype)
                for k, v in target_inputs.items()
            }
            draft_inputs = {
                k: v.to(self.device, dtype=self.dtype if v.dtype.is_floating_point else v.dtype)
                for k, v in draft_inputs.items()
            }
            
            with torch.no_grad():
                start = time.perf_counter()
                
                # Encode audio
                target_enc = target_model.get_encoder()(**target_inputs)
                draft_enc = draft_model.get_encoder()(**draft_inputs)
                
                # Prepare prefix
                prefix_ids = [target_model.config.decoder_start_token_id] + \
                             [fid[1] for fid in forced_decoder_ids]
                prefix = torch.tensor([prefix_ids], device=self.device, dtype=torch.long)
                
                # Speculative decode
                output_tokens = speculative_decode(
                    target_model, draft_model, target_enc, draft_enc,
                    prefix, max_tokens, self.draft_k, eos_token_id,
                    self.top_p, self.temperature
                )
                
                elapsed = time.perf_counter() - start
                total_time += elapsed
            
            text = target_processor.batch_decode(output_tokens, skip_special_tokens=True)[0]
            results.append(text.strip())
        
        # Return format
        if return_timing:
            return {
                "texts": results if not single_input else results[0],
                "total_time": total_time,
                "avg_time": total_time / len(audio),
            }
        
        return results if not single_input else results[0]
    
    def __repr__(self):
        return (
            f"SpeculativeWhisper("
            f"draft_k={self.draft_k}, "
            f"top_p={self.top_p}, "
            f"device={self.device})"
        )
