#!/usr/bin/env python3
"""
Full sweep with resume, per-run CSV append with fsync, and post-hoc speedup.

Usage (from sd-bench/):  python scripts/run_sweep.py
                          python scripts/run_sweep.py --config configs/sweep.yaml
"""
import argparse
import csv
import math
import os
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset import load_prompts, get_tokenizer
from src.runner import run_one

# Paths relative to sd-bench/
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RUNS_CSV = RESULTS_DIR / "runs.csv"
RAW_DIR = RESULTS_DIR / "raw"

CSV_FIELDS = [
    "run_id", "method", "gamma", "batch_size", "dataset", "n_prompts",
    "timestamp", "vllm_version", "gpu_name",
    # Throughput
    "throughput_tok_per_sec", "baseline_throughput_tok_per_sec", "speedup",
    "total_output_tokens", "total_wall_time_sec",
    # Speculative decoding acceptance
    "total_drafted_tokens", "total_accepted_tokens", "acceptance_rate",
    "mean_accepted_length_per_step", "accepted_length_std",
    "accepted_length_p5", "accepted_length_p50", "accepted_length_p95",
    # SD time fractions (NaN in vLLM 0.10.x)
    "time_drafting_frac", "time_verification_frac",
    "time_sampling_frac", "time_overhead_frac",
    # KV cache (from KVPoller)
    "peak_kv_usage_pct", "mean_kv_usage_pct", "kv_n_samples", "num_preemptions",
    # Latency from histograms (TTFT / e2e / TPOT)
    "ttft_mean_sec", "ttft_p50_sec", "ttft_p95_sec", "ttft_p99_sec",
    "e2e_mean_sec", "e2e_p50_sec", "e2e_p95_sec", "e2e_p99_sec",
    "tpot_mean_sec", "tpot_p50_sec", "tpot_p95_sec", "tpot_p99_sec",
    # Per-request time breakdown
    "prefill_time_mean_sec", "decode_time_mean_sec",
    "queue_time_mean_sec", "inference_time_mean_sec",
    # Generation length
    "mean_generation_length", "std_generation_length",
    "status", "error_msg",
]


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def load_completed_runs(csv_path: Path) -> set:
    """Return set of run_ids already recorded with status=ok."""
    if not csv_path.exists():
        return set()
    completed = set()
    try:
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("status") == "ok":
                    completed.add(row["run_id"])
    except Exception as e:
        print(f"[csv] Warning: could not read existing CSV ({e}); treating as empty")
    return completed


def append_row(csv_path: Path, row: dict) -> None:
    """Append a single row; flush + fsync before returning (kill -9 safe)."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not csv_path.exists() or csv_path.stat().st_size == 0

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if needs_header:
            writer.writeheader()
        writer.writerow({k: _csv_val(row.get(k, "")) for k in CSV_FIELDS})
        f.flush()
        os.fsync(f.fileno())


def _csv_val(v):
    """Stringify NaN as empty string for clean CSV output."""
    if isinstance(v, float) and math.isnan(v):
        return ""
    return v


# ---------------------------------------------------------------------------
# Sweep grid construction
# ---------------------------------------------------------------------------

def build_sweep(cfg: dict) -> list:
    """Expand the YAML config into a flat list of run specs (dicts).

    Order: dataset → gamma → batch_size → method.
    The 'none' baseline (no gamma) is emitted once per (bs, ds) in the first
    gamma pass so it runs before the SD methods at that batch size.
    """
    # Collect numeric gammas in first-appearance order across all methods
    seen_g: set = set()
    ordered_gammas: list = []
    for method_cfg in cfg["methods"]:
        for g in (method_cfg.get("gammas") or [None]):
            if g is not None and g not in seen_g:
                ordered_gammas.append(g)
                seen_g.add(g)

    runs = []
    for ds in cfg["datasets"]:
        for g in ordered_gammas:
            for bs in cfg["batch_sizes"]:
                for method_cfg in cfg["methods"]:
                    name = method_cfg["name"]
                    method_gammas = method_cfg.get("gammas") or [None]
                    if name == "none":
                        # Emit baseline once per (bs, ds) in the first gamma pass
                        if g == ordered_gammas[0]:
                            runs.append({
                                "run_id": f"none_b{bs}_{ds}",
                                "method": name,
                                "gamma": None,
                                "batch_size": bs,
                                "dataset": ds,
                            })
                    elif g in method_gammas:
                        runs.append({
                            "run_id": f"{name}_g{g}_b{bs}_{ds}",
                            "method": name,
                            "gamma": g,
                            "batch_size": bs,
                            "dataset": ds,
                        })
    return runs


# ---------------------------------------------------------------------------
# Post-hoc speedup fill
# ---------------------------------------------------------------------------

def fill_speedups(csv_path: Path) -> None:
    """
    Re-read runs.csv, compute speedup vs. matched 'none' baseline, rewrite atomically.
    Match key: (batch_size, dataset).
    """
    if not csv_path.exists():
        return

    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    baselines: dict = {}
    for r in rows:
        if r.get("method") == "none" and r.get("status") == "ok":
            tps_str = r.get("throughput_tok_per_sec", "")
            if tps_str:
                try:
                    baselines[(r["batch_size"], r["dataset"])] = float(tps_str)
                except ValueError:
                    pass

    changed = False
    for r in rows:
        key = (r.get("batch_size", ""), r.get("dataset", ""))
        bl = baselines.get(key)
        tps_str = r.get("throughput_tok_per_sec", "")
        if bl and tps_str and r.get("method") != "none" and r.get("status") == "ok":
            try:
                tps = float(tps_str)
                r["baseline_throughput_tok_per_sec"] = f"{bl:.6f}"
                r["speedup"] = f"{tps / bl:.6f}"
                changed = True
            except ValueError:
                pass

    if not changed:
        return

    tmp = csv_path.with_suffix(".tmp")
    with open(tmp, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(csv_path)
    print(f"[csv] Speedups written to {csv_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parent.parent / "configs" / "sweep.yaml"),
    )
    args = parser.parse_args()

    cfg_path = Path(args.config)
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    runs = build_sweep(cfg)
    completed = load_completed_runs(RUNS_CSV)
    remaining = [r for r in runs if r["run_id"] not in completed]

    n_total = len(runs)
    n_done = len(completed)
    n_todo = len(remaining)
    print(f"Sweep: {n_total} total | {n_done} already done | {n_todo} to run\n")

    model_id = cfg["model"]
    print(f"Loading tokenizer for {model_id} ...")
    try:
        tok = get_tokenizer(model_id)
        print("  OK\n")
    except Exception as e:
        print(f"  Warning: {e}; using char estimate\n")
        tok = None

    # Cache all dataset prompts upfront (avoids re-downloading per run)
    prompt_cache: dict = {}
    for ds_name in cfg["datasets"]:
        print(f"Loading dataset: {ds_name}")
        prompt_cache[ds_name] = load_prompts(
            ds_name,
            n_prompts=cfg["n_prompts"],
            max_input_tokens=cfg["max_input_tokens"],
            tokenizer=tok,
        )
        print(f"  {len(prompt_cache[ds_name])} prompts ready")
    print()

    global_idx = n_done
    for run_spec in remaining:
        global_idx += 1
        run_id = run_spec["run_id"]
        method = run_spec["method"]
        gamma = run_spec["gamma"]
        batch_size = run_spec["batch_size"]
        dataset = run_spec["dataset"]
        prompts = prompt_cache[dataset][: cfg["n_prompts"]]

        label = f"[{global_idx}/{n_total}] {run_id}"
        print(f"\n{label} ...", flush=True)

        row: dict = {}
        try:
            row, _ = run_one(
                run_id=run_id,
                method=method,
                gamma=gamma,
                batch_size=batch_size,
                dataset=dataset,
                prompts=prompts,
                max_output_tokens=cfg["max_output_tokens"],
                temperature=cfg["temperature"],
                model_id=model_id,
                max_model_len=cfg["max_model_len"],
                gpu_util=cfg["gpu_memory_utilization"],
                raw_dir=RAW_DIR,
            )
            row["status"] = "ok"
            row["error_msg"] = ""

            ar = row.get("acceptance_rate", math.nan)
            tps = row.get("throughput_tok_per_sec", math.nan)
            peak_kv = row.get("peak_kv_cache_usage_pct", math.nan)

            ar_str = f"α={ar:.2f}" if not math.isnan(ar) else "α=N/A"
            kv_str = f"peak_kv={peak_kv:.0f}%" if not math.isnan(peak_kv) else "peak_kv=N/A"
            tps_str = f"{tps:.1f} tok/s" if not math.isnan(tps) else "tok/s=N/A"
            print(f"{label}: {ar_str}, {kv_str}, {tps_str}")

        except _cuda_oom_types() as exc:
            row = _error_row(run_id, method, gamma, batch_size, dataset, "oom", str(exc)[:500])
            print(f"{label}: OOM — {exc}")

        except Exception as exc:
            import traceback
            msg = str(exc)[:500]
            row = _error_row(run_id, method, gamma, batch_size, dataset, "failed", msg)
            print(f"{label}: FAILED — {msg}")
            traceback.print_exc()

        append_row(RUNS_CSV, row)

    # Fill speedups now that all baselines are present
    fill_speedups(RUNS_CSV)

    print(f"\n[Done] Results in {RUNS_CSV}")
    print(f"       Raw JSON:  {RAW_DIR}/")


def _cuda_oom_types():
    """Return the OOM exception type(s) to catch, gracefully if torch absent."""
    try:
        import torch
        return torch.cuda.OutOfMemoryError
    except (ImportError, AttributeError):
        return MemoryError


def _error_row(run_id, method, gamma, batch_size, dataset, status, error_msg):
    row = {k: "" for k in CSV_FIELDS}
    row.update({
        "run_id": run_id,
        "method": method,
        "gamma": gamma if gamma is not None else "",
        "batch_size": batch_size,
        "dataset": dataset,
        "status": status,
        "error_msg": error_msg,
    })
    return row


if __name__ == "__main__":
    main()
