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
# KV cache usage
# ---------------------------------------------------------------------------

def extract_kv_usage_from_engine(engine) -> float:
    """
    Sample current GPU KV-cache utilization as a fraction in [0, 1].
    Returns NaN when the internal path is unavailable.
    """
    # Attempt 1: scheduler block_manager
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

    # Attempt 2: scheduler.get_stats()
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
# Speculative-decode stats — V1 engine path (vLLM 0.6+)
# ---------------------------------------------------------------------------

def extract_spec_decode_stats_v1(llm) -> dict:
    """
    Extract spec-decode counters from LLM.get_metrics() (vLLM V1 engine).

    vLLM V1 exposes cumulative Prometheus-style counters via get_metrics().
    Calling this after generate() and before deleting the LLM instance gives
    totals for the entire run (safe because we create a fresh LLM per config).

    Counter names used:
      vllm:spec_decode_num_drafts            — total draft rounds
      vllm:spec_decode_num_draft_tokens      — total tokens proposed
      vllm:spec_decode_num_accepted_tokens   — total tokens accepted
    """
    empty = {
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
        "peak_kv_cache_usage_pct": _NAN,
        "mean_kv_cache_usage_pct": _NAN,
    }

    try:
        raw_metrics = llm.get_metrics()
    except AttributeError:
        return empty
    if not raw_metrics:
        return empty

    # Build name -> scalar value dict.
    # vLLM metric objects vary across builds: dataclass with .name/.value,
    # or plain dicts. Handle both.
    by_name: dict = {}
    for m in raw_metrics:
        name = getattr(m, "name", None) or (m.get("name") if isinstance(m, dict) else None)
        if name is None:
            continue
        value = getattr(m, "value", None)
        if value is None and isinstance(m, dict):
            value = m.get("value")
        # Some builds use .labels_and_values for labeled counters — skip those for now
        if value is not None:
            by_name[name] = value

    def _get(key: str) -> float:
        v = by_name.get(key)
        if v is None:
            return _NAN
        try:
            f = float(v)
            return f if not math.isnan(f) else _NAN
        except (TypeError, ValueError):
            return _NAN

    num_drafts = _get("vllm:spec_decode_num_drafts")
    num_draft_tokens = _get("vllm:spec_decode_num_draft_tokens")
    num_accepted = _get("vllm:spec_decode_num_accepted_tokens")

    result = dict(empty)  # start from NaN defaults

    if not math.isnan(num_draft_tokens):
        result["total_drafted_tokens"] = num_draft_tokens
        result["total_accepted_tokens"] = num_accepted if not math.isnan(num_accepted) else _NAN
        if num_draft_tokens > 0 and not math.isnan(num_accepted):
            result["acceptance_rate"] = num_accepted / num_draft_tokens

    if not math.isnan(num_drafts) and num_drafts > 0 and not math.isnan(num_accepted):
        # +1: the target model always emits one bonus token per draft round
        result["mean_accepted_length_per_step"] = 1.0 + num_accepted / num_drafts

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
    sd_stats: dict,
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

    # Merge KV cache samples into sd_stats
    if step_kv_usages:
        pct = [v * 100 for v in step_kv_usages if not math.isnan(v)]
        sd_stats["peak_kv_cache_usage_pct"] = max(pct) if pct else _NAN
        sd_stats["mean_kv_cache_usage_pct"] = sum(pct) / len(pct) if pct else _NAN

    total_out = req_m["total_output_tokens"]
    throughput = total_out / wall_time if wall_time > 0 else _NAN

    return {
        "run_id": run_id,
        "method": method,
        "gamma": gamma if gamma is not None else _NAN,
        "batch_size": batch_size,
        "dataset": dataset,
        "n_prompts": n_prompts,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "vllm_version": vllm.__version__,
        "gpu_name": gpu_name,
        "throughput_tok_per_sec": throughput,
        "baseline_throughput_tok_per_sec": _NAN,
        "speedup": _NAN,
        "total_output_tokens": total_out,
        "total_wall_time_sec": wall_time,
        "total_drafted_tokens": sd_stats["total_drafted_tokens"],
        "total_accepted_tokens": sd_stats["total_accepted_tokens"],
        "acceptance_rate": sd_stats["acceptance_rate"],
        "mean_accepted_length_per_step": sd_stats["mean_accepted_length_per_step"],
        "accepted_length_std": sd_stats["accepted_length_std"],
        "accepted_length_p5": sd_stats["accepted_length_p5"],
        "accepted_length_p50": sd_stats["accepted_length_p50"],
        "accepted_length_p95": sd_stats["accepted_length_p95"],
        "time_drafting_frac": sd_stats["time_drafting_frac"],
        "time_verification_frac": sd_stats["time_verification_frac"],
        "time_sampling_frac": sd_stats["time_sampling_frac"],
        "time_overhead_frac": sd_stats["time_overhead_frac"],
        "peak_kv_cache_usage_pct": sd_stats.get("peak_kv_cache_usage_pct", _NAN),
        "mean_kv_cache_usage_pct": sd_stats.get("mean_kv_cache_usage_pct", _NAN),
        "num_preemptions": num_preemptions,
        "mean_per_request_latency_sec": req_m["mean_per_request_latency_sec"],
        "p50_latency_sec": req_m["p50_latency_sec"],
        "p95_latency_sec": req_m["p95_latency_sec"],
        "p99_latency_sec": req_m["p99_latency_sec"],
        "mean_ttft_sec": req_m["mean_ttft_sec"],
        "mean_generation_length": req_m["mean_generation_length"],
        "std_generation_length": req_m["std_generation_length"],
        "status": status,
        "error_msg": error_msg,
    }
