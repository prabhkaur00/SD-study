"""Extract and aggregate metrics from vLLM outputs and engine internals."""
import math
import statistics
import time
from typing import Optional


_NAN = float("nan")


def _safe(fn, fallback=_NAN):
    try:
        v = fn()
        return v if v is not None else fallback
    except Exception:
        return fallback


# ---------------------------------------------------------------------------
# Per-request timing and generation length
# ---------------------------------------------------------------------------

def extract_request_metrics(outputs: list) -> dict:
    """
    Derive per-request latency, TTFT, and generation length from RequestOutput list.
    All timing is in seconds. Missing fields become NaN.
    """
    latencies, ttfts, gen_lens = [], [], []

    for out in outputs:
        m = getattr(out, "metrics", None)
        if m is not None:
            arrival = getattr(m, "arrival_time", None)
            finished = getattr(m, "finished_time", None)
            first_tok = getattr(m, "first_token_time", None)
            if arrival is not None and finished is not None:
                latencies.append(finished - arrival)
            if arrival is not None and first_tok is not None:
                ttfts.append(first_tok - arrival)

        gen_len = sum(len(o.token_ids) for o in out.outputs)
        gen_lens.append(gen_len)

    def _pct(data, p):
        if not data:
            return _NAN
        s = sorted(data)
        k = (len(s) - 1) * p / 100.0
        lo, hi = int(k), min(int(k) + 1, len(s) - 1)
        return s[lo] + (k - lo) * (s[hi] - s[lo])

    return {
        "mean_per_request_latency_sec": _safe(lambda: statistics.mean(latencies)),
        "p50_latency_sec": _pct(latencies, 50),
        "p95_latency_sec": _pct(latencies, 95),
        "p99_latency_sec": _pct(latencies, 99),
        "mean_ttft_sec": _safe(lambda: statistics.mean(ttfts)),
        "mean_generation_length": _safe(lambda: statistics.mean(gen_lens)),
        "std_generation_length": _safe(
            lambda: statistics.stdev(gen_lens) if len(gen_lens) > 1 else 0.0
        ),
        "total_output_tokens": sum(gen_lens),
    }


# ---------------------------------------------------------------------------
# KV cache usage — sampled once per engine.step() call
# ---------------------------------------------------------------------------

def extract_kv_usage_from_engine(engine) -> float:
    """
    Sample current GPU KV-cache utilization as a fraction in [0, 1].
    Returns NaN when the internal path is unavailable.
    """
    # Attempt 1: scheduler block_manager (most common path in 0.7-0.10)
    try:
        sched = engine.scheduler
        if isinstance(sched, list):
            sched = sched[0]
        bm = sched.block_manager
        allocator = getattr(bm, "gpu_allocator", None)
        if allocator is not None:
            n_free = allocator.get_num_free_blocks()
            n_total = _safe(lambda: bm.num_total_gpu_blocks)
            if not math.isnan(n_total) and n_total > 0:
                return 1.0 - n_free / n_total
    except Exception:
        pass

    # Attempt 2: scheduler.get_stats() if available
    try:
        sched = engine.scheduler
        if isinstance(sched, list):
            sched = sched[0]
        stats = sched.get_stats()
        usage = getattr(stats, "gpu_cache_usage_perc", None)
        if usage is not None:
            return float(usage) / 100.0
    except Exception:
        pass

    # Attempt 3: driver_worker cache_engine
    try:
        executor = engine.model_executor
        driver = getattr(executor, "driver_worker", None)
        if driver is None:
            workers = getattr(executor, "workers", [])
            if workers:
                driver = workers[0]
        if driver is not None:
            ce = getattr(driver, "cache_engine", None)
            if isinstance(ce, list):
                ce = ce[0]
            if ce is not None:
                n_free = ce.get_num_free_blocks("gpu") if hasattr(ce, "get_num_free_blocks") else None
                n_total = getattr(ce, "num_gpu_blocks", None)
                if n_free is not None and n_total:
                    return 1.0 - n_free / n_total
    except Exception:
        pass

    return _NAN


# ---------------------------------------------------------------------------
# Speculative-decode stats from engine internals
# ---------------------------------------------------------------------------

def extract_spec_decode_stats(engine, step_kv_usages: list) -> dict:
    """
    Best-effort extraction of speculative-decoding metrics.
    All unavailable fields are NaN — never silently dropped.
    """
    result = {
        "total_drafted_tokens": _NAN,
        "total_accepted_tokens": _NAN,
        "acceptance_rate": _NAN,
        "mean_accepted_length_per_step": _NAN,
        "accepted_length_std": _NAN,
        "accepted_length_p5": _NAN,
        "accepted_length_p50": _NAN,
        "accepted_length_p95": _NAN,
        "time_drafting_frac": _NAN,
        "time_verification_frac": _NAN,
        "time_sampling_frac": _NAN,
        "time_overhead_frac": _NAN,
    }

    # Attempt: stat_loggers (Prometheus-style counters collected per-step)
    try:
        loggers = getattr(engine, "stat_loggers", None) or {}
        for logger in loggers.values():
            raw = getattr(logger, "_metrics", None) or getattr(logger, "metrics", None) or {}
            for key, val in raw.items():
                k = key.lower()
                if "acceptance_rate" in k or "draft_acceptance" in k:
                    result["acceptance_rate"] = float(val)
                if "num_accepted" in k or "accepted_tokens" in k:
                    result["total_accepted_tokens"] = float(val)
                if "num_draft" in k or "draft_tokens" in k or "proposed" in k:
                    result["total_drafted_tokens"] = float(val)
    except Exception:
        pass

    # Attempt: spec_decode_worker accumulated counters
    try:
        executor = engine.model_executor
        driver = getattr(executor, "driver_worker", None)
        if driver is None:
            workers = getattr(executor, "workers", [])
            if workers:
                driver = workers[0]
        if driver is not None:
            sdw = None
            for attr in ("spec_decode_worker", "_spec_decode_worker"):
                sdw = getattr(driver, attr, None)
                if sdw is not None:
                    break
            if sdw is not None:
                metrics_obj = getattr(sdw, "metrics", None) or getattr(sdw, "_metrics", None)
                if isinstance(metrics_obj, dict):
                    result["total_drafted_tokens"] = float(
                        metrics_obj.get("num_spec_tokens_proposed", _NAN)
                    )
                    result["total_accepted_tokens"] = float(
                        metrics_obj.get("num_spec_tokens_accepted", _NAN)
                    )
                elif metrics_obj is not None:
                    for field in ("num_spec_tokens_proposed", "num_draft_tokens", "drafted"):
                        v = getattr(metrics_obj, field, None)
                        if v is not None:
                            result["total_drafted_tokens"] = float(v)
                            break
                    for field in ("num_spec_tokens_accepted", "accepted"):
                        v = getattr(metrics_obj, field, None)
                        if v is not None:
                            result["total_accepted_tokens"] = float(v)
                            break
    except Exception:
        pass

    # Derive acceptance_rate if we have both parts
    drafted = result["total_drafted_tokens"]
    accepted = result["total_accepted_tokens"]
    if not math.isnan(drafted) and not math.isnan(accepted) and drafted > 0:
        result["acceptance_rate"] = accepted / drafted

    # KV cache stats from the per-step samples collected in the run loop
    if step_kv_usages:
        pct_samples = [v * 100 for v in step_kv_usages if not math.isnan(v)]
        if pct_samples:
            result["peak_kv_cache_usage_pct"] = max(pct_samples)
            result["mean_kv_cache_usage_pct"] = sum(pct_samples) / len(pct_samples)
        else:
            result["peak_kv_cache_usage_pct"] = _NAN
            result["mean_kv_cache_usage_pct"] = _NAN
    else:
        result["peak_kv_cache_usage_pct"] = _NAN
        result["mean_kv_cache_usage_pct"] = _NAN

    return result


# ---------------------------------------------------------------------------
# Full CSV row assembly
# ---------------------------------------------------------------------------

def build_metrics_row(
    run_id: str,
    method: str,
    gamma: Optional[int],
    batch_size: int,
    dataset: str,
    n_prompts: int,
    wall_time: float,
    outputs: list,
    engine,
    step_kv_usages: list,
    num_preemptions: int = 0,
    status: str = "ok",
    error_msg: str = "",
) -> dict:
    """Assemble the complete metrics dict (one CSV row)."""
    import vllm

    try:
        import torch
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except Exception:
        gpu_name = "unknown"

    req_m = extract_request_metrics(outputs)
    sd_m = extract_spec_decode_stats(engine, step_kv_usages)

    total_out = req_m["total_output_tokens"]
    throughput = total_out / wall_time if wall_time > 0 else _NAN

    return {
        # Identifiers
        "run_id": run_id,
        "method": method,
        "gamma": gamma if gamma is not None else _NAN,
        "batch_size": batch_size,
        "dataset": dataset,
        "n_prompts": n_prompts,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "vllm_version": vllm.__version__,
        "gpu_name": gpu_name,
        # Throughput
        "throughput_tok_per_sec": throughput,
        "baseline_throughput_tok_per_sec": _NAN,  # filled post-hoc by sweep
        "speedup": _NAN,                           # filled post-hoc by sweep
        "total_output_tokens": total_out,
        "total_wall_time_sec": wall_time,
        # Acceptance
        "total_drafted_tokens": sd_m["total_drafted_tokens"],
        "total_accepted_tokens": sd_m["total_accepted_tokens"],
        "acceptance_rate": sd_m["acceptance_rate"],
        "mean_accepted_length_per_step": sd_m["mean_accepted_length_per_step"],
        "accepted_length_std": sd_m["accepted_length_std"],
        "accepted_length_p5": sd_m["accepted_length_p5"],
        "accepted_length_p50": sd_m["accepted_length_p50"],
        "accepted_length_p95": sd_m["accepted_length_p95"],
        # Time breakdown
        "time_drafting_frac": sd_m["time_drafting_frac"],
        "time_verification_frac": sd_m["time_verification_frac"],
        "time_sampling_frac": sd_m["time_sampling_frac"],
        "time_overhead_frac": sd_m["time_overhead_frac"],
        # Memory
        "peak_kv_cache_usage_pct": sd_m["peak_kv_cache_usage_pct"],
        "mean_kv_cache_usage_pct": sd_m["mean_kv_cache_usage_pct"],
        "num_preemptions": num_preemptions,
        # Latency
        "mean_per_request_latency_sec": req_m["mean_per_request_latency_sec"],
        "p50_latency_sec": req_m["p50_latency_sec"],
        "p95_latency_sec": req_m["p95_latency_sec"],
        "p99_latency_sec": req_m["p99_latency_sec"],
        "mean_ttft_sec": req_m["mean_ttft_sec"],
        # Generation length
        "mean_generation_length": req_m["mean_generation_length"],
        "std_generation_length": req_m["std_generation_length"],
        # Status
        "status": status,
        "error_msg": error_msg,
    }
