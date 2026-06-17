# Speculative Decoding Benchmark

vLLM 0.10.x benchmarks comparing vanilla decoding vs. n-gram speculative decoding
on `meta-llama/Llama-3.1-8B-Instruct` (single A100-40 GB).

## Quick start

```bash
pip install -r requirements.txt
export HF_TOKEN=<your-huggingface-token>

# Step 1 — verify dataset loading (prints 3 sample prompts per dataset):
python src/datasets.py 3

# Step 2 — smoke test (eyeball output before sweeping):
python scripts/smoke_test.py

# Step 3 — full sweep with resume:
python scripts/run_sweep.py
```

## Repo layout

```
sd-bench/
├── requirements.txt
├── src/
│   ├── datasets.py   # deterministic prompt loading + cache
│   ├── metrics.py    # metric extraction from vLLM internals
│   └── runner.py     # LLMEngine wrapper, per-step KV sampling
├── scripts/
│   ├── smoke_test.py # 1 config per method, prints all metrics
│   └── run_sweep.py  # full sweep, resume from CSV
├── configs/
│   └── sweep.yaml    # grid: methods × γ × batch × dataset
└── results/
    ├── runs.csv       # one row per run, appended + fsynced
    └── raw/           # per-run JSON with per-request detail
```

## Sweep grid (~16 runs)

| Axis        | Values                              |
|-------------|-------------------------------------|
| Method      | `none`, `ngram`                     |
| γ           | 1, 3, 5 (ngram only)               |
| Batch size  | 1, 8 (`max_num_seqs` in vLLM)      |
| Dataset     | InstructCoder, ShareGPT             |
| Prompts     | 100 per dataset, ≤512 input tokens  |

## Metrics logged

Every run appends one row to `results/runs.csv`.  See `src/metrics.py` for
field definitions.  `speedup` and `baseline_throughput_tok_per_sec` are
backfilled by `run_sweep.py` once the matched `none` baseline is present.

### Fields that may be NaN

| Field | Reason |
|---|---|
| `total_drafted_tokens`, `total_accepted_tokens` | Not surfaced in `LLMEngine` offline API; requires reaching into `spec_decode_worker` internals which change across vLLM builds. |
| `acceptance_rate` | Derived from the two above; NaN when both are NaN. |
| `mean_accepted_length_per_step`, `accepted_length_{std,p5,p50,p95}` | Per-step accepted-length list is not available per-request in offline vLLM. |
| `time_{drafting,verification,sampling,overhead}_frac` | Internal draft/verify timing hooks are not exposed through `LLMEngine.step()`. |
| `mean_ttft_sec` | `RequestOutput.metrics.first_token_time` is `None` in some vLLM builds when using continuous batching. |
| `peak_kv_cache_usage_pct`, `mean_kv_cache_usage_pct` | Sampled via scheduler block manager; internal path varies across 0.10.x patch releases. |
| `accepted_length_per_step_list` (raw JSON) | Not exposed per-request by vLLM offline API. |

## Environment

- vLLM: 0.10.x
- GPU: A100-40 GB
- Model: `meta-llama/Llama-3.1-8B-Instruct` (gated — set `HF_TOKEN`)
- `HF_MODEL` env var overrides the default model ID
