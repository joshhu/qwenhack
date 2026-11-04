"""Per-token probability of the recorded assistant turn (teacher forcing).

For each position i in the recorded assistant output, ask the server for the
next-token distribution given ids[:i] and look up the recorded token ids[i].
Shows where the trace took low-probability branches.

Usage: uv run python teacher_forced.py [--start 2124] [--out results/teacher_forced.jsonl]
"""

from __future__ import annotations

import argparse
import json
import math

import requests
from tokenizers import Tokenizer

from replay import SERVER, TOKENIZER, load_ids

N_PROBS = 50


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=2124)  # first token after '<|im_start|>assistant\n<think>\n'
    ap.add_argument("--out", default="results/teacher_forced.jsonl")
    args = ap.parse_args()

    tok = Tokenizer.from_file(str(TOKENIZER))
    ids = load_ids()
    with open(args.out, "w") as f:
        for i in range(args.start, len(ids)):
            body = {"prompt": ids[:i], "n_predict": 1, "n_probs": N_PROBS, "temperature": 1.0,
                    # Full prefill every time: incremental cache reuse on this hybrid
                    # (recurrent + attention) model was not trustworthy in llama.cpp.
                    "samplers": ["temperature"], "post_sampling_probs": False, "cache_prompt": False}
            j = requests.post(f"{SERVER}/completion", json=body, timeout=600).json()
            top = j["completion_probabilities"][0]["top_logprobs"]
            target = tok.decode([ids[i]], skip_special_tokens=False)
            lp = next((t["logprob"] for t in top if t["id"] == ids[i]), None)
            rank = next((k for k, t in enumerate(top) if t["id"] == ids[i]), None)
            rec = {"pos": i, "token": target, "logprob": lp, "prob": math.exp(lp) if lp is not None else None,
                   "rank": rank, "top5": [(t["token"], round(math.exp(t["logprob"]), 4)) for t in top[:5]]}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            p = f"{rec['prob']:.4f}" if rec["prob"] is not None else f"<top{N_PROBS}"
            print(f"{i:5d} {p:>9s} r={rank} {target!r:24s} top={rec['top5'][:3]}", flush=True)


if __name__ == "__main__":
    main()
