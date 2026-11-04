"""Replay Cohere's Qwen3.8-Flash-Next trace against a local llama.cpp server.

The prompt is sent as raw token IDs (no chat template), and tool calls are
never executed: we only record what the model writes.

Usage:
  uv run python replay.py --prefix p5_after_payload -n 64
  uv run python replay.py --list
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import time
from pathlib import Path

import pyarrow.parquet as pq
import requests
from tokenizers import Tokenizer

ROOT = Path(__file__).parent
TRACE = ROOT / "data/cohere-traces/data/train-00000-of-00001.parquet"
TOKENIZER = ROOT / "data/qwen-tokenizer/tokenizer.json"
SERVER = "http://127.0.0.1:8090"

# Sampling used by Cohere's repro script and Check Point's replay.
SAMPLING = {
    "temperature": 1.0,
    "top_k": 20,
    "top_p": 1.0,
    "min_p": 0.0,
    "repeat_penalty": 1.0,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    # Pin the sampler chain so llama.cpp defaults (typical, xtc, dry, ...) stay out.
    "samplers": ["top_k", "top_p", "min_p", "temperature"],
}


def load_ids() -> list[int]:
    return pq.read_table(TRACE).to_pylist()[0]["token_ids"]


def cut_after(tok: Tokenizer, ids: list[int], marker: str, occurrence: int = 1) -> list[int]:
    """Shortest token prefix whose decoded text contains `marker` `occurrence` times."""
    for n in range(1, len(ids) + 1):
        if tok.decode(ids[:n], skip_special_tokens=False).count(marker) >= occurrence:
            return ids[:n]
    raise ValueError(f"marker not found: {marker!r}")


def build_prefixes(tok: Tokenizer, ids: list[int]) -> dict[str, dict]:
    enc = lambda s: tok.encode(s, add_special_tokens=False).ids  # noqa: E731
    p = {}
    # P0: original task, fresh first turn (what the template's generation prompt produces).
    p["p0_fresh_turn"] = {
        "ids": cut_after(tok, ids, "<|im_start|>assistant\n<think>\n"),
        "max_tokens": 6144,
        "desc": "System + task + '<think>\\n'. Measures spontaneous behaviour.",
    }
    # P1: after the confabulated reasoning block.
    p["p1_after_think"] = {
        "ids": cut_after(tok, ids, "</think>\n\n"),
        "max_tokens": 1024,
        "desc": "Given the recorded (confabulated) <think> block.",
    }
    # P2: model has committed to a bash tool call.
    p["p2_bash_open"] = {
        "ids": cut_after(tok, ids, "<parameter=command>\n"),
        "max_tokens": 1024,
        "desc": "Given '<tool_call><function=bash><parameter=command>'.",
    }
    # P3: recon done, before the curl command.
    p["p3_before_curl"] = {
        "ids": cut_after(tok, ids, 'head; echo "---";'),
        "max_tokens": 1024,
        "desc": "After ls /tmp, which, resolv.conf; before the curl.",
    }
    # P4: Cohere's prefill fragment.
    p["p4_curl_o"] = {
        "ids": cut_after(tok, ids, "curl -s -m 10 -o"),
        "max_tokens": 1024,
        "desc": "Prefill ending at 'curl -s -m 10 -o' (Cohere).",
    }
    # P5: the published 2,323-token prefix, right after PROBE_PAYLOAD_ABC123".
    p["p5_after_payload"] = {
        "ids": ids[:2323],
        "max_tokens": 1773,
        "desc": "Cohere/Check Point 2,323-token prefix (destination not yet written).",
    }
    # Check Point tool-order test: cut just after the `which` list, original vs swapped order.
    which = cut_after(tok, ids, "which curl wget python3 jq;")
    p["cp_which_curl_wget"] = {"ids": which, "max_tokens": 1024,
                               "desc": "Check Point order test: original 'which curl wget ...;'"}
    head = tok.decode(which, skip_special_tokens=False)
    swapped = head.replace("which curl wget python3 jq;", "which wget curl python3 jq;")
    p["cp_which_wget_curl"] = {"ids": enc(swapped), "max_tokens": 1024,
                               "desc": "Check Point order test: swapped 'which wget curl ...;'"}
    return p


def complete(prompt_ids: list[int], max_tokens: int, seed: int, cache: bool = False) -> dict:
    body = {
        "prompt": prompt_ids,
        "n_predict": max_tokens,
        "seed": seed,
        "stop": ["</tool_call>", "<|im_end|>"],
        "cache_prompt": cache,
        "n_probs": 0,
        **SAMPLING,
    }
    t = time.monotonic()
    r = requests.post(f"{SERVER}/completion", json=body, timeout=3600)
    r.raise_for_status()
    j = r.json()
    return {
        "seed": seed,
        "text": j["content"],
        "stop_type": j.get("stop_type"),
        "stopping_word": j.get("stopping_word"),
        "tokens_predicted": j.get("tokens_predicted"),
        "tokens_evaluated": j.get("tokens_evaluated"),
        "elapsed_s": round(time.monotonic() - t, 2),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", action="append", default=[])
    ap.add_argument("-n", type=int, default=64)
    # Keep at 1: concurrent same-prefix requests contaminated each other's
    # distributions in llama.cpp with this hybrid model (see README).
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--seed0", type=int, default=1000)
    ap.add_argument("--list", action="store_true")
    # Default is full prefill per sample: llama.cpp prompt-cache reuse on this hybrid model was
    # history-dependent (results changed depending on what the slot processed before).
    ap.add_argument("--cache", action="store_true", help="reuse prompt cache (NOT trustworthy here)")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    tok = Tokenizer.from_file(str(TOKENIZER))
    ids = load_ids()
    prefixes = build_prefixes(tok, ids)

    if args.list or not args.prefix:
        for name, p in prefixes.items():
            tail = tok.decode(p["ids"][-12:], skip_special_tokens=False)
            print(f"{name:22s} {len(p['ids']):5d} tok  ...{tail!r}\n    {p['desc']}")
        return

    out_dir = ROOT / args.out
    out_dir.mkdir(exist_ok=True)
    for name in args.prefix:
        p = prefixes[name]
        path = out_dir / f"{name}.jsonl"
        done = sum(1 for _ in path.open()) if path.exists() else 0
        seeds = [args.seed0 + i for i in range(done, args.n)]
        print(f"[{name}] {len(p['ids'])} prefix tokens, {done} done, {len(seeds)} to go", flush=True)
        with path.open("a") as f, cf.ThreadPoolExecutor(args.workers) as ex:
            futs = {ex.submit(complete, p["ids"], p["max_tokens"], s, args.cache): s for s in seeds}
            for i, fut in enumerate(cf.as_completed(futs), 1):
                rec = {"prefix": name, "prefix_tokens": len(p["ids"]), **fut.result()}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                print(f"  [{name} {done + i}/{args.n}] seed={rec['seed']} "
                      f"{rec['tokens_predicted']}tok {rec['elapsed_s']}s "
                      f"{rec['text'][:90]!r}", flush=True)


if __name__ == "__main__":
    main()
