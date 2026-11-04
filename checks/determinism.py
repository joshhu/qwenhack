"""Query the same next-token distribution repeatedly and print top tokens."""
import math, sys
import requests
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from replay import load_ids
ids = load_ids()
pos = int(sys.argv[1]) if len(sys.argv) > 1 else 2326
for cache in (False, False, False, True, True, True):
    j = requests.post("http://127.0.0.1:8090/completion", json={"prompt": ids[:pos], "n_predict": 1, "n_probs": 6,
        "samplers": ["temperature"], "temperature": 1.0, "post_sampling_probs": False, "cache_prompt": cache}).json()
    top = j["completion_probabilities"][0]["top_logprobs"]
    print("cache", cache, j.get("timings", {}).get("prompt_n"), " ".join(f"{t['token']!r}:{math.exp(t['logprob']):.3f}" for t in top))
