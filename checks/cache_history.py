"""Does a cached (cache_prompt=true) sample depend on what the slot processed before?

Reference: seed 1000 on the p5 prefix with full prefill. Then put the slot into
different states and repeat the same request with cache reuse. On a correct
server every line prints same=True.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from replay import complete, load_ids  # noqa: E402

ids = load_ids()
ref = complete(ids[:2323], 40, 1000, False)["text"][:70]
print("reference (no cache):", repr(ref))
tests = {
    "after full-trace call (2371, n=1)": (ids[:2371], 1),
    "after p5 long gen (2323, n=300)": (ids[:2323], 300),
    "after p0 gen (2124, n=300)": (ids[:2124], 300),
    "after p3 gen (2294, n=300)": (ids[:2294], 300),
}
for name, (pre, n) in tests.items():
    complete(pre, n, 7, True)  # put the slot into that state
    t = complete(ids[:2323], 40, 1000, True)["text"][:70]
    print(f"{name:40s} same={t == ref} {t!r}")
