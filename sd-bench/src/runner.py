"""Build a vLLM LLMEngine for one config, run prompts, return metrics + outputs."""
import gc
import json
import math
import os
import time
from pathlib import Path
from typing import Optional

MODEL_ID = os.environ.get("HF_MODEL", "meta-llama/Llama-3.1-8B-Instruct")


def _build_engine(
    method: str,
    gamma: Optional[int],
    model_id: str,
    max_model_len: int,
    gpu_util: float,
    max_num_seqs: int,
):
    """
    Create LLMEngine, using the vLLM 0.10.x speculative_config dict API.
    Falls back to the legacy flat-param API if speculative_config is rejected.
    """
    from vllm import LLMEngine
    from vllm.engine.arg_utils import EngineArgs

    base = dict(
        model=model_id,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_util,
        max_num_seqs=max_num_seqs,
        disable_log_stats=False,
        disable_log_requests=True,
        trust_remote_code=False,
    )

    if method == "ngram":
        spec_cfg = {
            "method": "ngram",
            "num_speculative_tokens": gamma,
            "prompt_lookup_max": 7,
            "prompt_lookup_min": 3,
        }
        try:
            ea = EngineArgs(**base, speculative_config=spec_cfg)
        except TypeError:
            # Pre-0.6 flat API
            ea = EngineArgs(
                **base,
                speculative_model="[ngram]",
                num_speculative_tokens=gamma,
                ngram_prompt_lookup_max=7,
                ngram_prompt_lookup_min=3,
            )
    else:
        ea = EngineArgs(**base)

    return LLMEngine.from_engine_args(ea)


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
    The metrics_row is ready to be written to runs.csv.
    baseline_throughput_tok_per_sec and speedup are left as NaN — the sweep
    script fills them post-hoc once the matching 'none' run is available.
    """
    from vllm import SamplingParams
    # Support both "python scripts/smoke_test.py" (sd-bench on sys.path)
    # and direct invocation from inside src/.
    try:
        from src.metrics import build_metrics_row, extract_kv_usage_from_engine
    except ModuleNotFoundError:
        from metrics import build_metrics_row, extract_kv_usage_from_engine  # type: ignore

    engine = _build_engine(method, gamma, model_id, max_model_len, gpu_util, batch_size)

    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_output_tokens,
        ignore_eos=False,
    )

    for p in prompts:
        engine.add_request(p["prompt_id"], p["text"], sampling_params)

    completed: dict = {}
    step_kv_usages: list = []
    num_preemptions = 0

    t0 = time.perf_counter()
    while engine.has_unfinished_requests():
        step_outputs = engine.step()

        kv = extract_kv_usage_from_engine(engine)
        if not math.isnan(kv):
            step_kv_usages.append(kv)

        for out in step_outputs:
            if out.finished:
                completed[out.request_id] = out

    wall_time = time.perf_counter() - t0

    # Restore original prompt order; warn if any prompt is missing
    outputs = []
    for p in prompts:
        out = completed.get(p["prompt_id"])
        if out is None:
            print(f"[runner] Warning: no output for prompt {p['prompt_id']}")
        else:
            outputs.append(out)

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
        engine=engine,
        step_kv_usages=step_kv_usages,
        num_preemptions=num_preemptions,
    )

    # Free GPU memory before returning
    del engine
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
    """Write per-run JSON with per-request detail + KV time-series."""
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
            # per-request per-step accepted lengths are not exposed by vLLM offline API
            "accepted_length_per_step_list": [],
        })

    data = {
        "run_id": run_id,
        "per_request": per_request,
        "kv_utilization_timeseries": step_kv_usages,
    }
    with open(raw_dir / f"{run_id}.json", "w") as f:
        json.dump(data, f, indent=2)
