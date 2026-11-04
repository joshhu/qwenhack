"""Measure the target next-token distribution while other slots run distractor prompts."""
import concurrent.futures as cf, math, sys, requests
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from replay import load_ids
from tokenizers import Tokenizer
ids = load_ids(); tok = Tokenizer.from_file(str(__import__("replay").TOKENIZER))
URL = "http://127.0.0.1:8090/completion"
def target():
    j = requests.post(URL, json={"prompt": ids[:2326], "n_predict": 1, "n_probs": 6, "samplers": ["temperature"],
        "temperature": 1.0, "post_sampling_probs": False, "cache_prompt": False}).json()
    return " ".join(f"{t['token']!r}:{math.exp(t['logprob']):.3f}" for t in j["completion_probabilities"][0]["top_logprobs"])
def distractor(kind):
    if kind == "unrelated":
        p = tok.encode("<|im_start|>user\nWrite a long poem about the ocean and lighthouses.<|im_end|>\n<|im_start|>assistant\n<think>\n", add_special_tokens=False).ids
    else:  # same trace, p5 prefix (generates URLs)
        p = ids[:2323]
    requests.post(URL, json={"prompt": p, "n_predict": 200, "temperature": 1.0, "cache_prompt": False, "seed": 7})
for kind in ("unrelated", "unrelated", "same_trace_p5", "same_trace_p5"):
    with cf.ThreadPoolExecutor(4) as ex:
        ds = [ex.submit(distractor, kind) for _ in range(3)]
        import time; time.sleep(1.5)  # let distractors start decoding
        t = ex.submit(target)
        print(f"{kind:14s} {t.result()}")
        [d.result() for d in ds]
print("idle           ", target())
