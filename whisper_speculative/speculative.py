"""Speculative decoding algorithms."""

import torch
from transformers.modeling_outputs import BaseModelOutput
from .sampling import top_p_sample


def speculative_decode_greedy(target_model, draft_model, target_enc, draft_enc,
                               prefix, max_new_tokens, draft_k, eos_token_id):
    """
    Greedy speculative decoding with KV cache.
    
    Algorithm:
        1. Draft model proposes K tokens greedily
        2. Target model verifies all K tokens in one pass
        3. Accept matching prefix + target's correction
        4. Repeat until EOS or max length
    """
    tokens = prefix.clone()
    target_enc_out = BaseModelOutput(last_hidden_state=target_enc.last_hidden_state)
    draft_enc_out = BaseModelOutput(last_hidden_state=draft_enc.last_hidden_state)
    
    while tokens.shape[1] < prefix.shape[1] + max_new_tokens:
        # Draft K tokens with KV cache
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
            if next_tok.item() == eos_token_id:
                break
        
        proposed = draft_tokens[:, tokens.shape[1]:]
        if proposed.shape[1] == 0:
            break
        
        # Target verification
        target_out = target_model(
            decoder_input_ids=draft_tokens,
            encoder_outputs=target_enc_out,
            use_cache=False
        )
        target_logits = target_out.logits[:, tokens.shape[1]-1:-1, :]
        target_preds = torch.argmax(target_logits, dim=-1)
        
        # Accept matching tokens
        matches = (proposed == target_preds)
        if matches.all():
            tokens = draft_tokens
        else:
            mismatch_idx = (~matches).nonzero(as_tuple=True)[1][0].item()
            accepted = proposed[:, :mismatch_idx]
            correction = target_preds[:, mismatch_idx:mismatch_idx+1]
            tokens = torch.cat([tokens, accepted, correction], dim=1)
        
        if tokens[0, -1].item() == eos_token_id:
            break
    
    return tokens


def speculative_decode_top_p(target_model, draft_model, target_enc, draft_enc,
                              prefix, max_new_tokens, draft_k, eos_token_id,
                              top_p, temperature=1.0):
    """
    Top-p speculative decoding with rejection sampling.
    
    Algorithm:
        1. Draft model proposes K tokens using top-p sampling
        2. Target model computes probabilities for all positions
        3. Accept with probability min(1, p_target / p_draft)
        4. Resample rejected position from target distribution
    """
    tokens = prefix.clone()
    target_enc_out = BaseModelOutput(last_hidden_state=target_enc.last_hidden_state)
    draft_enc_out = BaseModelOutput(last_hidden_state=draft_enc.last_hidden_state)
    
    while tokens.shape[1] < prefix.shape[1] + max_new_tokens:
        # Draft K tokens with top-p sampling
        draft_tokens = tokens.clone()
        draft_probs_list = []
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
            logits = out.logits[:, -1, :]
            
            next_tok = top_p_sample(logits, top_p, temperature)
            probs = torch.softmax(logits / temperature if temperature != 1.0 else logits, dim=-1)
            draft_probs_list.append(probs.gather(-1, next_tok))
            
            draft_tokens = torch.cat([draft_tokens, next_tok], dim=1)
            if next_tok.item() == eos_token_id:
                break
        
        proposed = draft_tokens[:, tokens.shape[1]:]
        if proposed.shape[1] == 0:
            break
        
        draft_probs = torch.cat(draft_probs_list, dim=1)
        
        # Target verification
        target_out = target_model(
            decoder_input_ids=draft_tokens,
            encoder_outputs=target_enc_out,
            use_cache=False
        )
        target_logits = target_out.logits[:, tokens.shape[1]-1:-1, :]
        if temperature != 1.0:
            target_logits = target_logits / temperature
        target_probs_full = torch.softmax(target_logits, dim=-1)
        target_probs = target_probs_full.gather(-1, proposed.unsqueeze(-1)).squeeze(-1)
        
        # Rejection sampling
        accept_probs = torch.clamp(target_probs / (draft_probs + 1e-10), max=1.0)
        accepted = torch.rand_like(accept_probs) < accept_probs
        
        if accepted.all():
            next_logits = target_out.logits[:, -1, :]
            next_tok = top_p_sample(next_logits, top_p, temperature)
            tokens = torch.cat([draft_tokens, next_tok], dim=1)
        else:
            reject_idx = (~accepted).nonzero(as_tuple=True)[1][0].item()
            accepted_tokens = proposed[:, :reject_idx]
            tokens = torch.cat([tokens, accepted_tokens], dim=1)
            next_tok = top_p_sample(target_logits[:, reject_idx, :], top_p, temperature)
            tokens = torch.cat([tokens, next_tok], dim=1)
        
        if tokens[0, -1].item() == eos_token_id:
            break
    
    return tokens


def speculative_decode(target_model, draft_model, target_enc, draft_enc,
                        prefix, max_new_tokens, draft_k, eos_token_id,
                        top_p=0.0, temperature=1.0):
    """Unified speculative decoding interface."""
    if top_p == 0.0:
        return speculative_decode_greedy(
            target_model, draft_model, target_enc, draft_enc,
            prefix, max_new_tokens, draft_k, eos_token_id
        )
    else:
        return speculative_decode_top_p(
            target_model, draft_model, target_enc, draft_enc,
            prefix, max_new_tokens, draft_k, eos_token_id,
            top_p, temperature
        )
