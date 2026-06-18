"""Build a vLLM LLM for one config, run prompts, return metrics + outputs."""
import gc
import json
import math
import os
import time
from pathlib import Path
from typing import Optional

MODEL_ID = os.environ.get("HF_MODEL", "meta-llama/Llama-3.1-8B-Instruct")


def _build_llm(
    method: str,
    gamma: Optional[int],
    model_id: str,
    max_model_len: int,
    gpu_util: float,
    max_num_seqs: int,
):
    """
    Create a vLLM LLM instance (not LLMEngine) so that get_metrics() is
    available for V1 spec-decode stats.  Probes the constructor signature
    before passing each optional kwarg so version-specific removals don't crash.
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
        **_maybe("dtype", "float16"),         # T4 prefers fp16 over bfloat16
        **_maybe("disable_log_stats", False),
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
    else:
        return LLM(**base)


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
        from src.metrics import build_metrics_row, extract_spec_decode_stats_v1, extract_kv_usage_from_engine
    except ModuleNotFoundError:
        from metrics import build_metrics_row, extract_spec_decode_stats_v1, extract_kv_usage_from_engine  # type: ignore

    llm = _build_llm(method, gamma, model_id, max_model_len, gpu_util, batch_size)

    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_output_tokens,
        ignore_eos=False,
    )

    # LLM.generate() preserves input order, so zip(prompts, outputs) is safe
    prompt_texts = [p["text"] for p in prompts]

    t0 = time.perf_counter()
    outputs = llm.generate(prompt_texts, sampling_params)
    wall_time = time.perf_counter() - t0

    # Pull spec-decode stats while the model is still loaded (cumulative since init)
    sd_stats = extract_spec_decode_stats_v1(llm)

    # Single KV cache sample post-generation via the underlying engine
    step_kv_usages = []
    try:
        engine = getattr(llm, "llm_engine", None)
        if engine is not None:
            kv = extract_kv_usage_from_engine(engine)
            if not math.isnan(kv):
                step_kv_usages = [kv]
    except Exception:
        pass

    if raw_dir is not None:
        _write_raw(run_id, prompts, outputs, step_kv_usages, raw_dir)

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
        step_kv_usages=step_kv_usages,
    )

    del llm
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass

    return row, outputs


def _write_raw(
    run_id: str,
    prompts: list,
    outputs: list,
    step_kv_usages: list,
    raw_dir: Path,
) -> None:
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    per_request = []
    for p, out in zip(prompts, outputs):
        m = getattr(out, "metrics", None)
        arrival = getattr(m, "arrival_time", None) if m else None
        finished = getattr(m, "finished_time", None) if m else None
        first_tok = getattr(m, "first_token_time", None) if m else None
        gen_len = sum(len(o.token_ids) for o in out.outputs)
        per_request.append({
            "prompt_id": p["prompt_id"],
            "input_len": p["input_len"],
            "output_len": gen_len,
            "latency": (finished - arrival) if (arrival and finished) else None,
            "ttft": (first_tok - arrival) if (arrival and first_tok) else None,
            "accepted_length_per_step_list": [],  # not exposed by vLLM offline API
        })

    with open(raw_dir / f"{run_id}.json", "w") as f:
        json.dump({"run_id": run_id, "per_request": per_request,
                   "kv_utilization_timeseries": step_kv_usages}, f, indent=2)
