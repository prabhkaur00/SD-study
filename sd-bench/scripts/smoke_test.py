#!/usr/bin/env python3
"""
Smoke test: run each method once, print rich metrics for eyeballing.
Verify this passes before running the full sweep.

Usage: python scripts/smoke_test.py
"""
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.datasets import load_prompts, get_tokenizer
from src.runner import run_one

MODEL_ID = os.environ.get("HF_MODEL", "meta-llama/Llama-3.1-8B-Instruct")

SMOKE_DATASET = "sharegpt"
SMOKE_N_PROMPTS = 4
SMOKE_GAMMA = 3
SMOKE_BATCH = 2
SMOKE_MAX_OUTPUT = 64
SMOKE_MAX_MODEL_LEN = 640  # 512 input + 64 output + slack

METHODS = [
    ("none", None),
    ("ngram", SMOKE_GAMMA),
]


def _fmt(v, fmt=".1f", fallback="N/A"):
    if isinstance(v, float) and math.isnan(v):
        return fallback
    try:
        return format(v, fmt)
    except Exception:
        return str(v)


def main():
    print(f"{'='*60}")
    print("Speculative Decoding Smoke Test")
    print(f"Dataset: {SMOKE_DATASET}  |  Prompts: {SMOKE_N_PROMPTS}")
    print(f"Batch: {SMOKE_BATCH}  |  γ (ngram): {SMOKE_GAMMA}  |  max_out: {SMOKE_MAX_OUTPUT}")
    print(f"{'='*60}\n")

    print(f"Loading tokenizer from {MODEL_ID} ...")
    try:
        tok = get_tokenizer(MODEL_ID)
        print("  OK\n")
    except Exception as e:
        print(f"  Warning: {e}; using char-based estimate\n")
        tok = None

    prompts = load_prompts(
        SMOKE_DATASET,
        n_prompts=SMOKE_N_PROMPTS,
        max_input_tokens=512,
        tokenizer=tok,
    )
    print(f"Loaded {len(prompts)} prompts from {SMOKE_DATASET}\n")

    results = {}
    for method, gamma in METHODS:
        tag = f"{method} (γ={gamma})" if gamma is not None else method

        print(f"\n{'='*60}")
        print(f"=== METHOD: {tag} ===")
        print("=" * 60)

        run_id = f"smoke_{method}_g{gamma or 0}"
        try:
            row, outputs = run_one(
                run_id=run_id,
                method=method,
                gamma=gamma,
                batch_size=SMOKE_BATCH,
                dataset=SMOKE_DATASET,
                prompts=prompts,
                max_output_tokens=SMOKE_MAX_OUTPUT,
                temperature=0.0,
                model_id=MODEL_ID,
                max_model_len=SMOKE_MAX_MODEL_LEN,
            )
        except Exception as e:
            import traceback
            print(f"\n  ERROR during {tag}:")
            traceback.print_exc()
            continue

        # Per-prompt text preview
        for i, (p, out) in enumerate(zip(prompts, outputs), 1):
            prompt_preview = p["text"][:60].replace("\n", " ")
            gen_text = out.outputs[0].text[:80].replace("\n", " ") if out.outputs else ""
            print(f"\nPrompt {i}: \"{prompt_preview}...\"")
            print(f"  → Generated: \"{gen_text}...\"")

        # Speculative decoding stats (aggregate; per-request not exposed by vLLM offline API)
        drafted = row.get("total_drafted_tokens", math.nan)
        accepted = row.get("total_accepted_tokens", math.nan)
        ar = row.get("acceptance_rate", math.nan)

        print()
        if not math.isnan(drafted):
            ar_pct = f"{ar*100:.1f}%" if not math.isnan(ar) else "N/A"
            print(
                f"  Proposed: {int(drafted)} | "
                f"Accepted: {int(accepted)} | "
                f"Acceptance: {ar_pct}"
            )
        else:
            print(
                "  Proposed: N/A | Accepted: N/A | Acceptance: N/A"
                "  (SD stats not exposed via LLMEngine in this vLLM build)"
            )

        tps = row.get("throughput_tok_per_sec", math.nan)
        peak_kv = row.get("peak_kv_cache_usage_pct", math.nan)
        kv_str = f"{peak_kv:.0f}%" if not math.isnan(peak_kv) else "N/A"
        print(f"  Throughput: {_fmt(tps)} tok/s | Peak KV: {kv_str}")
        print(f"  Latency p50: {_fmt(row.get('p50_latency_sec', math.nan), '.3f')}s  "
              f"p95: {_fmt(row.get('p95_latency_sec', math.nan), '.3f')}s")
        print(f"  TTFT mean:   {_fmt(row.get('mean_ttft_sec', math.nan), '.3f')}s")
        print(f"  Total output tokens: {row.get('total_output_tokens', 'N/A')}")
        print(f"  Wall time: {_fmt(row.get('total_wall_time_sec', math.nan), '.2f')}s")

        results[method] = row

    # Cross-method speedup summary
    if "none" in results and results:
        print(f"\n{'='*60}")
        print("Summary")
        print("=" * 60)
        base_tps = results["none"].get("throughput_tok_per_sec", math.nan)
        for method, gamma in METHODS:
            row = results.get(method)
            if row is None:
                continue
            tps = row.get("throughput_tok_per_sec", math.nan)
            if not math.isnan(base_tps) and not math.isnan(tps) and base_tps > 0:
                speedup = tps / base_tps
                print(f"  {method:8s}: {tps:.1f} tok/s  ({speedup:.2f}x vs none)")
            else:
                print(f"  {method:8s}: {_fmt(tps)} tok/s")

    print(f"\n{'='*60}")
    print("[OK] Smoke test complete. Review output above before running sweep.")
    print("     Command: python scripts/run_sweep.py")


if __name__ == "__main__":
    main()
