"""Extract and aggregate metrics from vLLM LLM.get_metrics() and RequestOutput."""
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
# Metric list helpers — work on the list returned by LLM.get_metrics()
# ---------------------------------------------------------------------------

def _find_metric(metrics: list, name: str):
    for m in metrics:
        if getattr(m, "name", None) == name:
            return m
    return None


def _hist_mean(m) -> float:
    if m is None or getattr(m, "count", 0) == 0:
        return _NAN
    return m.sum / m.count


def _hist_percentile(m, p: float) -> float:
    """
    Estimate percentile p (in [0,1]) from cumulative Prometheus histogram buckets.
    m.buckets is a dict of {upper_bound_str -> cumulative_count}, ordered ascending,
    with "+Inf" as the final key.
    """
    if m is None or getattr(m, "count", 0) == 0:
        return _NAN
    buckets = getattr(m, "buckets", None)
    if not buckets:
        return _NAN
    target = p * m.count
    prev_bound = 0.0
    prev_count = 0
    for bound_str, cum_count in buckets.items():
        if bound_str == "+Inf":
            return prev_bound  # all data is at or below the last finite bound
        bound = float(bound_str)
        if cum_count >= target:
            if cum_count == prev_count:
                return bound
            frac = (target - prev_count) / (cum_count - prev_count)
            return prev_bound + frac * (bound - prev_bound)
        prev_bound = bound
        prev_count = cum_count
    return prev_bound


# ---------------------------------------------------------------------------
# Generation length — still read from RequestOutput (V1 populates token_ids)
# ---------------------------------------------------------------------------

def extract_request_metrics(outputs: list) -> dict:
    """
    Compute generation-length stats from RequestOutput.
    In vLLM V1 offline mode RequestOutput.metrics is None, so per-request
    timing is not available here — use extract_latency_stats() instead.
    """
    gen_lens = [sum(len(o.token_ids) for o in out.outputs) for out in outputs]
    return {
        "mean_generation_length": _safe(lambda: statistics.mean(gen_lens)),
        "std_generation_length": _safe(
            lambda: statistics.stdev(gen_lens) if len(gen_lens) > 1 else 0.0
        ),
        "total_output_tokens": sum(gen_lens),
    }


# ---------------------------------------------------------------------------
# KV cache snapshot (post-generation gauge, mainly for debugging)
# ---------------------------------------------------------------------------

def extract_kv_snapshot(metrics: list) -> float:
    """
    Single-point KV usage from the gauge (fraction 0-1).
    Will be ~0 after generation finishes; use KVPoller in runner.py for
    peak/mean captured during generation.
    """
    m = _find_metric(metrics, "vllm:kv_cache_usage_perc")
    return float(getattr(m, "value", _NAN)) if m else _NAN


# ---------------------------------------------------------------------------
# Latency stats from histograms
# ---------------------------------------------------------------------------

def extract_latency_stats(metrics: list) -> dict:
    """Return TTFT, e2e latency, and TPOT means + p50/p95/p99 from histograms."""
    out: dict = {}
    for src_name, key in [
        ("vllm:time_to_first_token_seconds", "ttft"),
        ("vllm:e2e_request_latency_seconds", "e2e"),
        ("vllm:time_per_output_token_seconds", "tpot"),
    ]:
        m = _find_metric(metrics, src_name)
        out[f"{key}_mean_sec"] = _hist_mean(m)
        out[f"{key}_p50_sec"] = _hist_percentile(m, 0.50)
        out[f"{key}_p95_sec"] = _hist_percentile(m, 0.95)
        out[f"{key}_p99_sec"] = _hist_percentile(m, 0.99)
    return out


# ---------------------------------------------------------------------------
# Per-request time breakdown
# ---------------------------------------------------------------------------

def extract_time_breakdown(metrics: list) -> dict:
    """Return mean prefill/decode/queue/inference times in seconds (per request)."""
    out: dict = {}
    for src_name, key in [
        ("vllm:request_prefill_time_seconds", "prefill_time_mean_sec"),
        ("vllm:request_decode_time_seconds", "decode_time_mean_sec"),
        ("vllm:request_queue_time_seconds", "queue_time_mean_sec"),
        ("vllm:request_inference_time_seconds", "inference_time_mean_sec"),
    ]:
        out[key] = _hist_mean(_find_metric(metrics, src_name))
    return out


# ---------------------------------------------------------------------------
# Preemption count
# ---------------------------------------------------------------------------

def extract_preemption_count(metrics: list) -> int:
    m = _find_metric(metrics, "vllm:num_preemptions")
    return int(getattr(m, "value", 0)) if m else 0


# ---------------------------------------------------------------------------
# Speculative-decode stats — V1 engine (vLLM 0.6+)
# ---------------------------------------------------------------------------

def _hist_buckets_per_pos(m, num_positions):
    if m is None or num_positions is None:
        return []
    buckets = getattr(m, "buckets", {}) or {}
    out = []
    prev = 0
    for i in range(num_positions):
        cum = int(buckets.get(str(i), prev))
        out.append(cum - prev)
        prev = cum
    return out


def extract_spec_decode_stats_v1(metrics: list, gamma=None) -> dict:
    """
    Extract spec-decode counters from the metrics list.
    Takes the list directly so the caller can share one get_metrics() call
    across all extractors.

    Counter names (vLLM 0.10.x V1):
      vllm:spec_decode_num_drafts           — total draft rounds
      vllm:spec_decode_num_draft_tokens     — total tokens proposed
      vllm:spec_decode_num_accepted_tokens  — total tokens accepted
    """
    empty: dict = {
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

    if not metrics:
        return empty

    def _get(name: str) -> float:
        m = _find_metric(metrics, name)
        if m is None:
            return _NAN
        v = getattr(m, "value", None)
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

    result = dict(empty)

    if not math.isnan(num_draft_tokens):
        result["total_drafted_tokens"] = num_draft_tokens
        result["total_accepted_tokens"] = num_accepted if not math.isnan(num_accepted) else _NAN
        if num_draft_tokens > 0 and not math.isnan(num_accepted):
            result["acceptance_rate"] = num_accepted / num_draft_tokens

    if (not math.isnan(num_drafts) and num_drafts > 0
            and not math.isnan(num_accepted)):
        # +1: target model always emits one bonus token per draft round
        result["mean_accepted_length_per_step"] = 1.0 + num_accepted / num_drafts

    pos_hist = _find_metric(metrics, "vllm:spec_decode_num_accepted_tokens_per_pos")
    per_pos_count = _hist_buckets_per_pos(pos_hist, gamma)
    per_pos_acceptance_rate = [
        (c / num_drafts) if num_drafts else float("nan")
        for c in per_pos_count
    ]
    result["per_pos_count"] = per_pos_count
    result["per_pos_acceptance_rate"] = per_pos_acceptance_rate

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
    kv_summary: dict,
    latency_stats: dict,
    breakdown_stats: dict,
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
        "baseline_throughput_tok_per_sec": _NAN,
        "speedup": _NAN,
        "total_output_tokens": total_out,
        "total_wall_time_sec": wall_time,
        # Acceptance
        "total_drafted_tokens": sd_stats["total_drafted_tokens"],
        "total_accepted_tokens": sd_stats["total_accepted_tokens"],
        "acceptance_rate": sd_stats["acceptance_rate"],
        "mean_accepted_length_per_step": sd_stats["mean_accepted_length_per_step"],
        "accepted_length_std": sd_stats["accepted_length_std"],
        "accepted_length_p5": sd_stats["accepted_length_p5"],
        "accepted_length_p50": sd_stats["accepted_length_p50"],
        "accepted_length_p95": sd_stats["accepted_length_p95"],
        # SD time fractions (not exposed by vLLM 0.10.x)
        "time_drafting_frac": sd_stats["time_drafting_frac"],
        "time_verification_frac": sd_stats["time_verification_frac"],
        "time_sampling_frac": sd_stats["time_sampling_frac"],
        "time_overhead_frac": sd_stats["time_overhead_frac"],
        # KV cache (from KVPoller samples during generation)
        "peak_kv_usage_pct": kv_summary.get("peak_kv_usage_pct", _NAN),
        "mean_kv_usage_pct": kv_summary.get("mean_kv_usage_pct", _NAN),
        "kv_n_samples": kv_summary.get("kv_n_samples", 0),
        "num_preemptions": num_preemptions,
        # Latency from histograms
        "ttft_mean_sec": latency_stats.get("ttft_mean_sec", _NAN),
        "ttft_p50_sec": latency_stats.get("ttft_p50_sec", _NAN),
        "ttft_p95_sec": latency_stats.get("ttft_p95_sec", _NAN),
        "ttft_p99_sec": latency_stats.get("ttft_p99_sec", _NAN),
        "e2e_mean_sec": latency_stats.get("e2e_mean_sec", _NAN),
        "e2e_p50_sec": latency_stats.get("e2e_p50_sec", _NAN),
        "e2e_p95_sec": latency_stats.get("e2e_p95_sec", _NAN),
        "e2e_p99_sec": latency_stats.get("e2e_p99_sec", _NAN),
        "tpot_mean_sec": latency_stats.get("tpot_mean_sec", _NAN),
        "tpot_p50_sec": latency_stats.get("tpot_p50_sec", _NAN),
        "tpot_p95_sec": latency_stats.get("tpot_p95_sec", _NAN),
        "tpot_p99_sec": latency_stats.get("tpot_p99_sec", _NAN),
        # Per-request time breakdown
        "prefill_time_mean_sec": breakdown_stats.get("prefill_time_mean_sec", _NAN),
        "decode_time_mean_sec": breakdown_stats.get("decode_time_mean_sec", _NAN),
        "queue_time_mean_sec": breakdown_stats.get("queue_time_mean_sec", _NAN),
        "inference_time_mean_sec": breakdown_stats.get("inference_time_mean_sec", _NAN),
        # Generation length
        "mean_generation_length": req_m["mean_generation_length"],
        "std_generation_length": req_m["std_generation_length"],
        # Status
        "status": status,
        "error_msg": error_msg,
    }
