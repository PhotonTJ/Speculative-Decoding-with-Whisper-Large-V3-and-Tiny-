"""Experiment runner for grid search."""

import json
from dataclasses import asdict
from .config import Config
from .evaluation import run_baseline, run_speculative


def run_experiment(models, dataset, wer_metric, top_p_values, draft_k_values, config):
    """
    Run grid search over top_p and draft_k values.
    
    Returns:
        dict: Experiment results with all configurations
    """
    results = []
    baseline_cache = {}
    refs = None
    
    total = len(top_p_values) * len(draft_k_values)
    count = 0
    
    for top_p in top_p_values:
        # Cache baseline for each top_p
        if top_p not in baseline_cache:
            cfg = Config(
                top_p=top_p,
                temperature=config.temperature,
                max_new_tokens=config.max_new_tokens,
                language=config.language,
                task=config.task
            )
            print(f"Running baseline (top_p={top_p})...")
            base_result = run_baseline(dataset, models, cfg)
            wer = wer_metric.compute(
                predictions=base_result["predictions"],
                references=base_result["references"]
            )
            baseline_cache[top_p] = {
                "avg_time": base_result["avg_time"],
                "wer": wer
            }
            refs = base_result["references"]
        
        base = baseline_cache[top_p]
        
        for draft_k in draft_k_values:
            count += 1
            print(f"[{count}/{total}] top_p={top_p}, draft_k={draft_k}")
            
            cfg = Config(
                top_p=top_p,
                draft_k=draft_k,
                temperature=config.temperature,
                max_new_tokens=config.max_new_tokens,
                language=config.language,
                task=config.task
            )
            
            spec_result = run_speculative(dataset, models, cfg)
            wer = wer_metric.compute(
                predictions=spec_result["predictions"],
                references=refs
            )
            speedup = base["avg_time"] / spec_result["avg_time"]
            
            results.append({
                "top_p": top_p,
                "draft_k": draft_k,
                "baseline_avg_time": round(base["avg_time"], 4),
                "speculative_avg_time": round(spec_result["avg_time"], 4),
                "speedup": round(speedup, 2),
                "baseline_wer": round(base["wer"], 4),
                "speculative_wer": round(wer, 4)
            })
            
            print(f"    Speedup: {speedup:.2f}x | WER: {wer:.4f}")
    
    # Sort by speedup
    results_sorted = sorted(results, key=lambda x: x["speedup"], reverse=True)
    
    # Find best configurations
    best_speedup = max(results, key=lambda x: x["speedup"])
    best_wer = min(results, key=lambda x: x["speculative_wer"])
    closest_2x = min(results, key=lambda x: abs(x["speedup"] - 2.0))
    
    return {
        "config": {
            "top_p_values": top_p_values,
            "draft_k_values": draft_k_values,
            "dataset_size": len(dataset),
            "max_new_tokens": config.max_new_tokens,
            "temperature": config.temperature
        },
        "results": results_sorted,
        "best": {
            "speedup": best_speedup,
            "wer": best_wer,
            "closest_2x": closest_2x
        }
    }


def save_results(results, filepath):
    """Save experiment results to JSON."""
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {filepath}")


def print_results(results):
    """Print formatted results table."""
    print("\n" + "=" * 80)
    print("EXPERIMENT RESULTS")
    print("=" * 80)
    
    print(f"\n{'top_p':>8} | {'draft_k':>8} | {'Base(s)':>8} | {'Spec(s)':>8} | {'Speedup':>8} | {'WER':>8}")
    print("-" * 70)
    
    for r in results["results"]:
        print(f"{r['top_p']:>8.2f} | {r['draft_k']:>8d} | {r['baseline_avg_time']:>8.3f} | "
              f"{r['speculative_avg_time']:>8.3f} | {r['speedup']:>7.2f}x | {r['speculative_wer']:>8.4f}")
    
    print("\n" + "=" * 80)
    print("BEST CONFIGURATIONS")
    print("=" * 80)
    
    best = results["best"]
    print(f"\nBest Speedup:  top_p={best['speedup']['top_p']}, draft_k={best['speedup']['draft_k']} "
          f"-> {best['speedup']['speedup']}x (WER: {best['speedup']['speculative_wer']})")
    print(f"Best WER:      top_p={best['wer']['top_p']}, draft_k={best['wer']['draft_k']} "
          f"-> {best['wer']['speedup']}x (WER: {best['wer']['speculative_wer']})")
    print(f"Closest to 2x: top_p={best['closest_2x']['top_p']}, draft_k={best['closest_2x']['draft_k']} "
          f"-> {best['closest_2x']['speedup']}x")
