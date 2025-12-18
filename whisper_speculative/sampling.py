"""Sampling utilities."""

import torch


def top_p_sample(logits, top_p, temperature=1.0):
    """
    Sample from top-p (nucleus) of the probability distribution.
    
    Args:
        logits: Raw logits (batch_size, vocab_size)
        top_p: Cumulative probability threshold
        temperature: Sampling temperature
    
    Returns:
        Sampled token indices (batch_size, 1)
    """
    if temperature != 1.0:
        logits = logits / temperature
    
    probs = torch.softmax(logits, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    
    # Mask tokens beyond top_p threshold
    mask = cumulative_probs > top_p
    mask[..., 1:] = mask[..., :-1].clone()
    mask[..., 0] = False
    
    sorted_probs[mask] = 0.0
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
    
    sampled_idx = torch.multinomial(sorted_probs, num_samples=1)
    return torch.gather(sorted_indices, dim=-1, index=sampled_idx)
