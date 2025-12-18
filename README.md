# Whisper Speculative Decoding

This project implements speculative decoding to speed up speech-to-text transcription using OpenAI's Whisper models. It uses a small, fast model (Whisper Tiny) to generate draft transcriptions, which are then verified and corrected by a large, accurate model (Whisper Large V3).

The result is faster transcription while maintaining the quality of the large model.

---

## Table of Contents

1. [How It Works](#how-it-works)
2. [The Two Models](#the-two-models)
3. [How Audio Processing Works](#how-audio-processing-works)
4. [The Speculative Decoding Algorithm](#the-speculative-decoding-algorithm)
5. [Top-p Sampling Explained](#top-p-sampling-explained)
6. [Rejection Sampling Explained](#rejection-sampling-explained)
7. [Project Structure](#project-structure)
8. [Installation](#installation)
9. [How to Run](#how-to-run)
10. [API Reference](#api-reference)
11. [Configuration Options](#configuration-options)
12. [Performance Results](#performance-results)

---

## How It Works

Traditional speech recognition with Whisper Large V3 is slow because the model generates one token at a time, and each token requires a full forward pass through a 1.5 billion parameter model.

Speculative decoding speeds this up by:

1. Using a small model (Whisper Tiny, 39M parameters) to quickly generate a "draft" of several tokens
2. Having the large model verify all draft tokens in a single forward pass
3. Accepting the tokens that match, and correcting where they differ
4. Repeating until the transcription is complete

Since the small model is often correct (both models learned similar patterns), many draft tokens get accepted. This means fewer forward passes through the large model, resulting in faster transcription.

---

## The Two Models

### Whisper Tiny (Draft Model)

- Size: 39 million parameters
- Role: Quickly proposes candidate tokens
- Speed: Very fast, but less accurate
- Used for: Generating draft sequences

### Whisper Large V3 (Target Model)

- Size: 1.5 billion parameters  
- Role: Verifies and corrects the draft
- Speed: Slow, but highly accurate
- Used for: Final quality assurance

Both models are loaded from Hugging Face:

```python
# From whisper_speculative/models.py

target_model = AutoModelForSpeechSeq2Seq.from_pretrained(
    "openai/whisper-large-v3",
    torch_dtype=torch.float16,  # Use half precision on GPU
    low_cpu_mem_usage=True,
    use_safetensors=True,
).to(device).eval()

draft_model = AutoModelForSpeechSeq2Seq.from_pretrained(
    "openai/whisper-tiny",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    use_safetensors=True,
).to(device).eval()
```

The models run in evaluation mode (`eval()`) and use GPU with half precision (float16) for faster computation.

---

## How Audio Processing Works

### Why Two Processors?

Each Whisper model has its own processor (tokenizer + feature extractor). Although they share the same tokenizer vocabulary, they may have slightly different audio preprocessing. We use both to ensure each model receives input in its expected format.

```python
# Load both processors
target_processor = AutoProcessor.from_pretrained("openai/whisper-large-v3")
draft_processor = AutoProcessor.from_pretrained("openai/whisper-tiny")
```

### Audio to Features

When you provide an audio file, the processor:

1. Loads the audio and resamples to 16kHz (Whisper's expected sample rate)
2. Converts the waveform to a log-mel spectrogram (80 frequency bins)
3. Pads or truncates to 30 seconds
4. Returns a tensor ready for the model's encoder

```python
# Process the same audio for both models
target_inputs = target_processor(audio_array, sampling_rate=16000, return_tensors="pt")
draft_inputs = draft_processor(audio_array, sampling_rate=16000, return_tensors="pt")
```

### Encoding Audio

Before decoding, both models encode the audio into hidden representations:

```python
target_enc = target_model.get_encoder()(**target_inputs)
draft_enc = draft_model.get_encoder()(**draft_inputs)
```

These encoded representations are reused throughout the decoding process, so encoding only happens once per audio file.

### Decoder Prefix

Whisper uses special tokens to control language and task. These form the "prefix" that starts the decoding:

```python
# Get language and task tokens
forced_decoder_ids = target_processor.get_decoder_prompt_ids(
    language="en", task="transcribe"
)

# Build prefix: [<start>, <language>, <task>]
prefix_ids = [target_model.config.decoder_start_token_id] + \
             [fid[1] for fid in forced_decoder_ids]
```

---

## The Speculative Decoding Algorithm

### Greedy Mode (top_p = 0.0)

When `top_p` is set to 0.0, the algorithm uses greedy decoding (always pick the most likely token):

```
Step 1: Draft Phase
    - Start with the current sequence
    - Use draft model to generate K tokens (e.g., K=6)
    - Each token is the argmax of the model's output
    - Use KV cache to speed up sequential generation

Step 2: Verification Phase  
    - Feed the entire sequence (original + draft) to the target model
    - Get the target model's predictions for each position
    - Compare draft tokens with target predictions

Step 3: Accept/Reject
    - Find the first position where draft differs from target
    - Accept all tokens before that position
    - Use the target's token at the mismatch position
    - Discard remaining draft tokens

Step 4: Repeat
    - Continue from the accepted sequence
    - Stop when end-of-sequence token is generated
```

Here is the actual code:

```python
# From whisper_speculative/speculative.py

def speculative_decode_greedy(target_model, draft_model, target_enc, draft_enc,
                               prefix, max_new_tokens, draft_k, eos_token_id):
    tokens = prefix.clone()
    
    while tokens.shape[1] < prefix.shape[1] + max_new_tokens:
        # Step 1: Draft K tokens
        draft_tokens = tokens.clone()
        draft_cache = None
        
        for _ in range(draft_k):
            input_ids = draft_tokens[:, -1:] if draft_cache else draft_tokens
            out = draft_model(
                decoder_input_ids=input_ids,
                encoder_outputs=draft_enc_out,
                past_key_values=draft_cache,
                use_cache=True
            )
            draft_cache = out.past_key_values
            next_tok = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
            draft_tokens = torch.cat([draft_tokens, next_tok], dim=1)
        
        proposed = draft_tokens[:, tokens.shape[1]:]
        
        # Step 2: Target verification
        target_out = target_model(
            decoder_input_ids=draft_tokens,
            encoder_outputs=target_enc_out,
            use_cache=False
        )
        target_preds = torch.argmax(target_out.logits[:, tokens.shape[1]-1:-1, :], dim=-1)
        
        # Step 3: Find first mismatch
        matches = (proposed == target_preds)
        if matches.all():
            tokens = draft_tokens  # Accept all
        else:
            mismatch_idx = (~matches).nonzero(as_tuple=True)[1][0].item()
            accepted = proposed[:, :mismatch_idx]
            correction = target_preds[:, mismatch_idx:mismatch_idx+1]
            tokens = torch.cat([tokens, accepted, correction], dim=1)
        
        if tokens[0, -1].item() == eos_token_id:
            break
    
    return tokens
```

---

## Top-p Sampling Explained

Top-p sampling (also called nucleus sampling) is a way to add controlled randomness to text generation. Instead of always picking the most likely token (greedy), it samples from a subset of likely tokens.

### How It Works

1. Get probability distribution over all tokens
2. Sort tokens by probability (highest first)
3. Find the smallest set of tokens whose probabilities sum to at least `p` (e.g., 0.9)
4. Sample randomly from only those tokens

For example, with top_p=0.9:
- If "the" has 70% probability and "a" has 25% probability, only these two tokens are considered (95% > 90%)
- The model randomly picks between them based on their relative probabilities

### The Code

```python
# From whisper_speculative/sampling.py

def top_p_sample(logits, top_p, temperature=1.0):
    # Apply temperature (higher = more random)
    if temperature != 1.0:
        logits = logits / temperature
    
    # Convert to probabilities
    probs = torch.softmax(logits, dim=-1)
    
    # Sort by probability
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    
    # Compute cumulative sum
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    
    # Mask tokens beyond threshold
    mask = cumulative_probs > top_p
    mask[..., 1:] = mask[..., :-1].clone()
    mask[..., 0] = False  # Always keep at least one token
    
    # Zero out masked tokens and renormalize
    sorted_probs[mask] = 0.0
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
    
    # Sample from filtered distribution
    sampled_idx = torch.multinomial(sorted_probs, num_samples=1)
    return torch.gather(sorted_indices, dim=-1, index=sampled_idx)
```

### Why Use It?

- Greedy decoding can be repetitive or get stuck
- Full random sampling can produce nonsense
- Top-p balances quality and diversity
- Lower values (0.2) are more deterministic, higher values (0.9) are more random

---

## Rejection Sampling Explained

When using top-p sampling with speculative decoding, we cannot simply compare tokens because both models might produce different valid samples. Instead, we use rejection sampling to ensure the output distribution matches the target model.

### The Problem

If the draft model samples token A with probability 0.3, but the target model would sample A with probability 0.6, we should accept A more often. Conversely, if the draft model over-represents a token, we should sometimes reject it.

### The Solution

For each draft token, compute an acceptance probability:

```
acceptance_probability = min(1, target_probability / draft_probability)
```

- If target gives higher probability than draft: always accept (ratio > 1, clamped to 1)
- If target gives lower probability: accept with probability equal to the ratio

### The Code

```python
# From whisper_speculative/speculative.py

# Get probabilities from both models
draft_probs = ...  # Probability draft model assigned to its sampled tokens
target_probs = ... # Probability target model assigns to those same tokens

# Compute acceptance probability
accept_probs = torch.clamp(target_probs / (draft_probs + 1e-10), max=1.0)

# Random acceptance decision
accepted = torch.rand_like(accept_probs) < accept_probs

if accepted.all():
    # All tokens accepted - also sample one more from target
    tokens = torch.cat([draft_tokens, next_tok], dim=1)
else:
    # Find first rejection
    reject_idx = (~accepted).nonzero(as_tuple=True)[1][0].item()
    
    # Accept tokens before rejection
    accepted_tokens = proposed[:, :reject_idx]
    tokens = torch.cat([tokens, accepted_tokens], dim=1)
    
    # Sample replacement from target distribution
    next_tok = top_p_sample(target_logits[:, reject_idx, :], top_p, temperature)
    tokens = torch.cat([tokens, next_tok], dim=1)
```

This ensures the final output has the same distribution as if we had sampled directly from the target model, while still being faster.

---

## Project Structure

```
whisper_speculative/
    __init__.py          # Package exports
    config.py            # Configuration dataclass
    models.py            # Model loading
    sampling.py          # Top-p sampling implementation
    speculative.py       # Core decoding algorithms
    evaluation.py        # Benchmarking functions
    experiment.py        # Grid search experiments
    api.py               # Python API (SpeculativeWhisper class)
    server.py            # REST API (FastAPI)

main.py                  # Command-line interface
requirements.txt         # Dependencies
```

### Module Descriptions

**config.py**: Defines the `Config` dataclass with all settings (model names, draft_k, top_p, etc.)

**models.py**: Loads both Whisper models and their processors, handles device selection and precision

**sampling.py**: Implements top-p (nucleus) sampling for token selection

**speculative.py**: Contains the core speculative decoding algorithms (greedy and top-p variants)

**evaluation.py**: Runs baseline and speculative decoding on datasets, measures timing

**experiment.py**: Runs grid search over different parameter combinations

**api.py**: Provides the `SpeculativeWhisper` class for easy Python usage

**server.py**: FastAPI server with REST endpoints for transcription

---

## Installation

### Requirements

- Python 3.10 or higher
- CUDA-capable GPU (recommended) or CPU
- 8GB+ GPU memory for running both models

### Steps

```bash
# Clone the repository
git clone https://github.com/PhotonTJ/Speculative-Decoding-with-Whisper-Large-V3-and-Tiny.git
cd Speculative-Decoding-with-Whisper-Large-V3-and-Tiny

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## How to Run

### Option 1: Python API

The simplest way to transcribe audio files:

```python
from whisper_speculative import SpeculativeWhisper

# Initialize (loads both models)
sw = SpeculativeWhisper(
    draft_model="tiny",
    final_model="large-v3",
    device="cuda"  # or "cpu"
)

# Transcribe a single file
text = sw.transcribe("audio.wav")
print(text)

# Transcribe multiple files
texts = sw.transcribe(["audio1.wav", "audio2.wav"])
for text in texts:
    print(text)

# Get timing information
result = sw.transcribe("audio.wav", return_timing=True)
print(f"Text: {result['texts']}")
print(f"Time: {result['total_time']:.2f} seconds")
```

### Option 2: Command Line

Run a single evaluation:

```bash
python main.py --top-p 0.2 --draft-k 6 --max-samples 50
```

Run a grid search experiment:

```bash
python main.py --experiment --top-p-values 0.2,0.4,0.6 --draft-k-values 4,6,8
```

Results are saved to `results.json` (or specify with `--output`).

### Option 3: REST API

Start the server:

```bash
python -m whisper_speculative.server
```

The server runs at `http://localhost:8000`. Open `http://localhost:8000/docs` for interactive documentation.

Transcribe via curl:

```bash
# Single file
curl -X POST "http://localhost:8000/transcribe" -F "file=@audio.wav"

# Multiple files
curl -X POST "http://localhost:8000/transcribe/batch" \
    -F "files=@audio1.wav" \
    -F "files=@audio2.wav"
```

---

## API Reference

### Python API

```python
SpeculativeWhisper(
    draft_model="tiny",      # Draft model: tiny, base, small, medium
    final_model="large-v3",  # Target model: large, large-v2, large-v3
    device=None,             # Device: cuda, cpu, or None (auto-detect)
    draft_k=6,               # Tokens to draft per iteration
    top_p=0.0,               # Sampling: 0.0=greedy, >0=nucleus sampling
    temperature=1.0,         # Sampling temperature
    language="en",           # Target language
    task="transcribe"        # Task: transcribe or translate
)

.transcribe(
    audio,                   # File path, numpy array, or list of either
    max_tokens=128,          # Maximum tokens to generate
    return_timing=False      # If True, return dict with timing info
)
```

### REST API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| GET | `/health` | Detailed health status |
| POST | `/transcribe` | Transcribe single audio file |
| POST | `/transcribe/batch` | Transcribe multiple files |
| POST | `/reload` | Reload models with new configuration |

---

## Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `draft_model` | "tiny" | Small model for drafting (tiny, base, small) |
| `final_model` | "large-v3" | Large model for verification |
| `draft_k` | 6 | Number of tokens to draft per iteration |
| `top_p` | 0.0 | Top-p value (0.0 = greedy, 0.9 = diverse) |
| `temperature` | 1.0 | Sampling temperature (higher = more random) |
| `max_new_tokens` | 128 | Maximum tokens to generate |
| `device` | auto | Device selection (cuda/cpu) |
| `language` | "en" | Target language code |
| `task` | "transcribe" | Task type (transcribe/translate) |

### Choosing Parameters

For fastest speed with good quality:
```bash
--top-p 0.0 --draft-k 8
```

For balanced speed and diversity:
```bash
--top-p 0.2 --draft-k 6
```

For more diverse outputs:
```bash
--top-p 0.5 --draft-k 4
```

---

## Performance Results

Tested on LibriSpeech validation set (50 samples):

| Configuration | Speedup | Word Error Rate |
|--------------|---------|-----------------|
| Baseline (Large V3) | 1.0x | 0.138 |
| Speculative (top_p=0.0, k=6) | 1.75x | 0.140 |
| Speculative (top_p=0.2, k=8) | 1.92x | 0.147 |
| Speculative (top_p=0.4, k=4) | 1.61x | 0.127 |

Key findings:
- Lower top_p values give higher speedup (more deterministic = higher acceptance rate)
- Higher draft_k values generally improve speedup
- Quality (WER) remains comparable to baseline

---

## License

This project uses OpenAI's Whisper models through Hugging Face Transformers.
