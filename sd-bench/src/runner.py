"""Build a vLLM LLM for one config, run prompts, return metrics + outputs."""
import gc
import json
import math
import os
import threading
import time
from pathlib import Path
from typing import Optional

MODEL_ID = os.environ.get("HF_MODEL", "meta-llama/Llama-3.1-8B-Instruct")


# ---------------------------------------------------------------------------
# KV cache poller — samples gpu_cache_usage_perc during generation
# ---------------------------------------------------------------------------

class KVPoller:
    """
    Polls vllm:gpu_cache_usage_perc in a background thread during generation
    to capture peak/mean KV usage.  The gauge drops to ~0 once generation
    finishes, so we must sample while the run is in progress.
    """
    def __init__(self, llm, interval_sec: float = 0.05):
        self.llm = llm
        self.interval = interval_sec
        self.samples: list = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _poll(self):
        while not self._stop.is_set():
            try:
                for m in self.llm.get_metrics():
                    if getattr(m, "name", None) == "vllm:gpu_cache_usage_perc":
                        v = getattr(m, "value", None)
                        if v is not None:
                            self.samples.append(float(v))
                        break
            except Exception:
                pass
            self._stop.wait(self.interval)

    def __enter__(self):
        self.samples = []
        self._stop.clear()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)

    def summary(self) -> dict:
        s = [v for v in self.samples if not math.isnan(v)]
        if not s:
            return {"peak_kv_usage_pct": float("nan"),
                    "mean_kv_usage_pct": float("nan"),
                    "kv_n_samples": 0,
                    "kv_samples": []}
        return {"peak_kv_usage_pct": max(s) * 100.0,
                "mean_kv_usage_pct": sum(s) / len(s) * 100.0,
                "kv_n_samples": len(s),
                "kv_samples": s}


# ---------------------------------------------------------------------------
# Engine builder
# ---------------------------------------------------------------------------

def _build_llm(
    method: str,
    gamma: Optional[int],
    model_id: str,
    max_model_len: int,
    gpu_util: float,
    max_num_seqs: int,
):
    """
    Create a vLLM LLM instance.  Uses the LLM class (not LLMEngine) so that
    get_metrics() is available for V1 spec-decode and latency stats.
    Probes the constructor signature before passing optional kwargs so that
    version-specific removals don't crash.
    """
    from vllm import LLM
    import inspect

    _params = set(inspect.signature(LLM.__init__).parameters)

    def _maybe(key, val):
        return {key: val} if key in _params else {}

    base = {
        "model": model_id,
        "max_model_len": max_model_len,
        "gpu_memory_utilization": gpu_util,
        "max_num_seqs": max_num_seqs,
        "trust_remote_code": False,
        "disable_log_stats": False,  # keep stats collection enabled
        **_maybe("dtype", "float16"),  # T4 prefers fp16 over bfloat16
    }

    if method == "ngram":
        spec_cfg = {
            "method": "ngram",
            "num_speculative_tokens": gamma,
            "prompt_lookup_max": 7,
            "prompt_lookup_min": 3,
        }
        try:
            return LLM(**base, speculative_config=spec_cfg)
        except TypeError:
            # Pre-0.6 flat API
            return LLM(
                **base,
                speculative_model="[ngram]",
                num_speculative_tokens=gamma,
                ngram_prompt_lookup_max=7,
                ngram_prompt_lookup_min=3,
            )
    elif method == "eagle3":
        spec_cfg = {
            "method": "eagle3",
            "model": "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B",
            "num_speculative_tokens": gamma,
        }
        return LLM(**base, speculative_config=spec_cfg)
    elif method == "draft_model":
        spec_cfg = {
            "method": "draft_model",
            "model": "meta-llama/Llama-3.2-1B-Instruct",
            "num_speculative_tokens": gamma,
        }
        return LLM(**base, speculative_config=spec_cfg)
    else:
        return LLM(**base)


# ---------------------------------------------------------------------------
# Main run function
# ---------------------------------------------------------------------------

def run_one(
    run_id: str,
    method: str,
    gamma: Optional[int],
    batch_size: int,
    dataset: str,
    prompts: list,
    max_output_tokens: int = 512,
    temperature: float = 0.0,
    model_id: str = MODEL_ID,
    max_model_len: int = 1088,
    gpu_util: float = 0.9,
    raw_dir: Optional[Path] = None,
) -> tuple:
    """
    Run a single benchmarking configuration.
    Returns (metrics_row: dict, outputs: list[RequestOutput]).
    """
    from vllm import SamplingParams
    try:
        from src.metrics import (
            build_metrics_row,
            extract_spec_decode_stats_v1,
            extract_latency_stats,
            extract_time_breakdown,
            extract_preemption_count,
        )
    except ModuleNotFoundError:
        from metrics import (  # type: ignore
            build_metrics_row,
            extract_spec_decode_stats_v1,
            extract_latency_stats,
            extract_time_breakdown,
            extract_preemption_count,
        )

    llm = _build_llm(method, gamma, model_id, max_model_len, gpu_util, batch_size)

    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_output_tokens,
        ignore_eos=False,
    )

    # LLM.generate() preserves input order → zip(prompts, outputs) is safe
    prompt_texts = [p["text"] for p in prompts]

    # Poll KV usage during generation (gauge drops to ~0 after run finishes)
    with KVPoller(llm, interval_sec=0.05) as kv_poller:
        t0 = time.perf_counter()
        outputs = llm.generate(prompt_texts, sampling_params)
        wall_time = time.perf_counter() - t0

    kv_summary = kv_poller.summary()

    # Single get_metrics() call — shared across all post-run extractors
    raw_metrics = llm.get_metrics()

    sd_stats = extract_spec_decode_stats_v1(raw_metrics, gamma=gamma)
    latency_stats = extract_latency_stats(raw_metrics)
    breakdown_stats = extract_time_breakdown(raw_metrics)
    num_preemptions = extract_preemption_count(raw_metrics)

    if raw_dir is not None:
        _write_raw(
            run_id, prompts, outputs, kv_poller.samples, raw_dir,
            per_pos_count=sd_stats.get("per_pos_count", []),
            per_pos_acceptance_rate=sd_stats.get("per_pos_acceptance_rate", []),
        )

    row = build_metrics_row(
        run_id=run_id,
        method=method,
        gamma=gamma,
        batch_size=batch_size,
        dataset=dataset,
        n_prompts=len(outputs),
        wall_time=wall_time,
        outputs=outputs,
        sd_stats=sd_stats,
        kv_summary=kv_summary,
        latency_stats=latency_stats,
        breakdown_stats=breakdown_stats,
        num_preemptions=num_preemptions,
    )

    del llm
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass

    return row, outputs


# ---------------------------------------------------------------------------
# Raw JSON writer
# ---------------------------------------------------------------------------

def _write_raw(
    run_id: str,
    prompts: list,
    outputs: list,
    kv_samples: list,
    raw_dir: Path,
    per_pos_count: Optional[list] = None,
    per_pos_acceptance_rate: Optional[list] = None,
) -> None:
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    per_request = []
    for p, out in zip(prompts, outputs):
        gen_len = sum(len(o.token_ids) for o in out.outputs)
        per_request.append({
            "prompt_id": p["prompt_id"],
            "input_len": p["input_len"],
            "output_len": gen_len,
            # RequestOutput.metrics is None in V1 offline mode;
            # per-request latency comes from engine-level histograms
            "accepted_length_per_step_list": [],
        })

    with open(raw_dir / f"{run_id}.json", "w") as f:
        json.dump({
            "run_id": run_id,
            "per_request": per_request,
            "kv_utilization_timeseries": kv_samples,
            "per_pos_count": per_pos_count or [],
            "per_pos_acceptance_rate": per_pos_acceptance_rate or [],
        }, f, indent=2)
