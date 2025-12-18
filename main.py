#!/usr/bin/env python3
"""
Usage:
    # Single run
    python main.py --top-p 0.2 --draft-k 6
    
    # Experiment mode (grid search)
    python main.py --experiment
    
    # Custom experiment values
    python main.py --experiment --top-p-values 0.1,0.2,0.3 --draft-k-values 4,6,8
"""

import argparse
import json
from datasets import load_dataset
from evaluate import load as load_metric

from whisper_speculative import (
    Config, load_models,
    run_baseline, run_speculative,
    run_experiment, save_results, print_results
)


def parse_args():
    parser = argparse.ArgumentParser(description="Whisper Speculative Decoding")
    
    # Model settings
    parser.add_argument("--target-model", default="openai/whisper-large-v3")
    parser.add_argument("--draft-model", default="openai/whisper-tiny")
    
    # Decoding settings
    parser.add_argument("--top-p", type=float, default=0.0, help="0.0=greedy, >0=nucleus sampling")
    parser.add_argument("--draft-k", type=int, default=6)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    
    # Dataset settings
    parser.add_argument("--dataset", default="hf-internal-testing/librispeech_asr_dummy")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--max-samples", type=int, default=50)
    
    # Experiment mode
    parser.add_argument("--experiment", action="store_true", help="Run grid search")
    parser.add_argument("--top-p-values", default="0.2,0.4,0.6,0.8")
    parser.add_argument("--draft-k-values", default="3,4,5,6,8")
    
    # Output
    parser.add_argument("--output", default="results.json", help="Output JSON file")
    
    return parser.parse_args()


def single_run(args, models, dataset, wer_metric):
    """Run single configuration and return results."""
    config = Config(
        top_p=args.top_p,
        draft_k=args.draft_k,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
    )
    
    mode = "greedy" if args.top_p == 0.0 else f"top_p={args.top_p}"
    print(f"\nRunning baseline ({mode})...")
    base = run_baseline(dataset, models, config)
    
    print(f"Running speculative (draft_k={args.draft_k}, {mode})...")
    spec = run_speculative(dataset, models, config)
    
    wer_base = wer_metric.compute(predictions=base["predictions"], references=base["references"])
    wer_spec = wer_metric.compute(predictions=spec["predictions"], references=base["references"])
    speedup = base["avg_time"] / spec["avg_time"]
    
    results = {
        "config": {
            "top_p": args.top_p,
            "draft_k": args.draft_k,
            "temperature": args.temperature,
            "max_new_tokens": args.max_new_tokens,
            "dataset_size": len(dataset)
        },
        "baseline": {
            "total_time": round(base["total_time"], 2),
            "avg_time": round(base["avg_time"], 4),
            "wer": round(wer_base, 4)
        },
        "speculative": {
            "total_time": round(spec["total_time"], 2),
            "avg_time": round(spec["avg_time"], 4),
            "wer": round(wer_spec, 4)
        },
        "speedup": round(speedup, 2),
        "samples": [
            {
                "reference": base["references"][i],
                "baseline": base["predictions"][i],
                "speculative": spec["predictions"][i]
            }
            for i in range(min(3, len(base["references"])))
        ]
    }
    
    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\nBaseline ({mode})")
    print(f"  Time: {base['avg_time']:.3f}s/sample | WER: {wer_base:.4f}")
    print(f"\nSpeculative (draft_k={args.draft_k})")
    print(f"  Time: {spec['avg_time']:.3f}s/sample | WER: {wer_spec:.4f}")
    print(f"\nSpeedup: {speedup:.2f}x")
    print("=" * 60)
    
    return results


def main():
    args = parse_args()
    
    # Setup
    print(f"Device: cuda/cpu auto-detect")
    print(f"Target: {args.target_model}")
    print(f"Draft:  {args.draft_model}")
    
    config = Config(
        target_model=args.target_model,
        draft_model=args.draft_model,
    )
    
    print("\nLoading models...")
    models = load_models(config)
    print(f"Device: {models['device']}")
    
    # Load dataset
    dataset = load_dataset(args.dataset, "clean", split=args.split)
    if args.max_samples:
        dataset = dataset.select(range(min(len(dataset), args.max_samples)))
    print(f"Dataset: {len(dataset)} samples")
    
    wer_metric = load_metric("wer")
    
    # Run experiment or single run
    if args.experiment:
        top_p_values = [float(x) for x in args.top_p_values.split(",")]
        draft_k_values = [int(x) for x in args.draft_k_values.split(",")]
        
        print(f"\nExperiment: top_p={top_p_values}, draft_k={draft_k_values}")
        
        results = run_experiment(
            models, dataset, wer_metric,
            top_p_values, draft_k_values, config
        )
        print_results(results)
    else:
        results = single_run(args, models, dataset, wer_metric)
    
    # Save results
    save_results(results, args.output)


if __name__ == "__main__":
    main()
