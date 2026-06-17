"""Load, filter, and cache prompt datasets for speculative decoding benchmarks."""
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Optional

from tqdm import tqdm

CACHE_DIR = Path(__file__).parent.parent / ".cache" / "datasets"
MODEL_ID = os.environ.get("HF_MODEL", "meta-llama/Llama-3.1-8B-Instruct")

DATASET_REGISTRY = {
    "sharegpt": {
        "hf_name": "anon8231489123/ShareGPT_Vicuna_unfiltered",
        "split": "train",
        "data_files": "ShareGPT_V3_unfiltered_cleaned_split.json",
    },
    "instructcoder": {
        "hf_name": "likaixin/InstructCoder",
        "split": "train",
    },
}


def _extract_sharegpt_prompt(example: dict) -> Optional[str]:
    """Return the first human turn from a ShareGPT conversation."""
    convs = example.get("conversations") or []
    for turn in convs:
        role = turn.get("from") or turn.get("role") or ""
        if role in ("human", "user"):
            return (turn.get("value") or turn.get("content") or "").strip()
    return None


def _extract_instructcoder_prompt(example: dict) -> Optional[str]:
    """Return instruction (+input) from an InstructCoder row."""
    instruction = (example.get("instruction") or "").strip()
    inp = (example.get("input") or "").strip()
    if not instruction:
        return None
    return f"{instruction}\n\n{inp}".strip() if inp else instruction


_EXTRACTORS = {
    "sharegpt": _extract_sharegpt_prompt,
    "instructcoder": _extract_instructcoder_prompt,
}


def _cache_key(dataset_name: str, n_prompts: int, max_input_tokens: int, seed: int) -> str:
    raw = f"{dataset_name}:{n_prompts}:{max_input_tokens}:{seed}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def load_prompts(
    dataset_name: str,
    n_prompts: int = 100,
    max_input_tokens: int = 512,
    seed: int = 42,
    tokenizer=None,
    use_cache: bool = True,
) -> list:
    """
    Return a list of dicts {prompt_id, text, input_len}.
    Selection is fully deterministic given (dataset_name, n_prompts, max_input_tokens, seed).
    """
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset '{dataset_name}'. Available: {list(DATASET_REGISTRY)}")

    cache_file = (
        CACHE_DIR
        / f"{dataset_name}_{_cache_key(dataset_name, n_prompts, max_input_tokens, seed)}.json"
    )

    if use_cache and cache_file.exists():
        with open(cache_file) as f:
            return json.load(f)

    from datasets import load_dataset  # deferred — not needed for cache hits

    cfg = DATASET_REGISTRY[dataset_name]
    extractor = _EXTRACTORS[dataset_name]

    print(f"[datasets] Downloading {dataset_name} from {cfg['hf_name']} ...")
    load_kwargs: dict = {"split": cfg["split"], "trust_remote_code": True}
    if "data_files" in cfg:
        load_kwargs["data_files"] = cfg["data_files"]

    try:
        ds = load_dataset(cfg["hf_name"], **load_kwargs)
    except Exception:
        # Fall back without data_files specification
        load_kwargs.pop("data_files", None)
        ds = load_dataset(cfg["hf_name"], **load_kwargs)

    rng = random.Random(seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)

    result = []
    for idx in tqdm(indices, desc=f"Filtering {dataset_name}", ncols=90, leave=False):
        text = extractor(ds[idx])
        if not text or len(text) < 10:
            continue
        # Cheap gate: skip clearly oversized strings before tokenizing
        if len(text) > max_input_tokens * 12:
            continue

        if tokenizer is not None:
            n_toks = len(tokenizer.encode(text, add_special_tokens=False))
        else:
            n_toks = max(1, len(text) // 4)

        if n_toks > max_input_tokens:
            continue

        pid = f"{dataset_name}_{len(result):04d}"
        result.append({"prompt_id": pid, "text": text, "input_len": n_toks})
        if len(result) >= n_prompts:
            break

    if len(result) < n_prompts:
        print(
            f"[datasets] Warning: only found {len(result)} valid prompts "
            f"(wanted {n_prompts}) for {dataset_name}"
        )

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "w") as f:
        json.dump(result, f, indent=2)

    return result


def get_tokenizer(model_id: str = MODEL_ID):
    """Load the fast tokenizer (no model weights, quick download)."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_id, use_fast=True)


if __name__ == "__main__":
    import sys

    n_sample = int(sys.argv[1]) if len(sys.argv) > 1 else 3

    print(f"Model / tokenizer: {MODEL_ID}")
    try:
        tok = get_tokenizer()
        print("  Tokenizer loaded OK\n")
    except Exception as e:
        print(f"  Warning: could not load tokenizer ({e}); using char-based estimate\n")
        tok = None

    for ds_name in ("sharegpt", "instructcoder"):
        print(f"{'='*60}")
        print(f"Dataset: {ds_name}")
        print("=" * 60)
        prompts = load_prompts(
            ds_name,
            n_prompts=max(n_sample, 10),
            max_input_tokens=512,
            tokenizer=tok,
        )
        for p in prompts[:n_sample]:
            preview = p["text"][:120].replace("\n", " ")
            print(f"  [{p['prompt_id']}]  input_len={p['input_len']} tokens")
            print(f"    {preview!r}")
        print()

    print("[OK] datasets.py verification passed")
