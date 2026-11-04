"""Headline numbers with 95% Wilson intervals (reads results/summary.json from analyze.py)."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def fisher_two_sided(a: int, b: int, c: int, d: int) -> float:
    """Fisher exact test for [[a, b], [c, d]]."""
    n1, n2, m1 = a + b, c + d, a + c
    n = n1 + n2
    lc = lambda n_, k_: math.lgamma(n_ + 1) - math.lgamma(k_ + 1) - math.lgamma(n_ - k_ + 1)  # noqa: E731
    p_of = lambda x: math.exp(lc(n1, x) + lc(n2, m1 - x) - lc(n, m1))  # noqa: E731
    p_obs = p_of(a)
    lo, hi = max(0, m1 - n2), min(n1, m1)
    return min(1.0, sum(p_of(x) for x in range(lo, hi + 1) if p_of(x) <= p_obs * (1 + 1e-9)))


def fmt(k: int, n: int) -> str:
    lo, hi = wilson(k, n)
    return f"{k}/{n} = {k / n:.1%} (95% CI {lo:.1%}–{hi:.1%})"


def main() -> None:
    d = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
    s = json.loads((d / "summary.json").read_text())
    g = lambda p, k: s[p]["stats"].get(k, 0)  # noqa: E731
    c = lambda p, k: s[p]["categories"].get(k, 0)  # noqa: E731
    n = lambda p: s[p]["n"]  # noqa: E731

    print("p0 first action = bash:", fmt(g("p0_fresh_turn", "first_fn:bash"), n("p0_fresh_turn")))
    print("p0 any network cmd:", fmt(g("p0_fresh_turn", "network_cmd"), n("p0_fresh_turn")))
    print("p0 confabulated previous turn:", fmt(g("p0_fresh_turn", "confab_prev_turn"), n("p0_fresh_turn")))
    print("p1 first action = bash:", fmt(g("p1_after_think", "first_fn:bash"), n("p1_after_think")))
    print("p2 network cmd:", fmt(g("p2_bash_open", "network_cmd"), n("p2_bash_open")))
    print("p3 curl network cmd:", fmt(g("p3_before_curl", "curl_net"), n("p3_before_curl")))
    print("p3 upload-like:", fmt(g("p3_before_curl", "upload_like"), n("p3_before_curl")))
    print("p4 URL written:", fmt(g("p4_curl_o", "has_url"), n("p4_curl_o")))
    print("p5 Alibaba Cloud host:", fmt(c("p5_after_payload", "Alibaba Cloud"), n("p5_after_payload")))
    print("p5 watchlist (aliyuncs/routify/sanmaohe):", fmt(g("p5_after_payload", "watchlist"), n("p5_after_payload")))
    print("p5 distinct hosts:", s["p5_after_payload"]["distinct_hosts"])
    a, na = g("cp_which_curl_wget", "curl_net"), n("cp_which_curl_wget")
    b, nb = g("cp_which_wget_curl", "curl_net"), n("cp_which_wget_curl")
    print("CP order 'curl wget':", fmt(a, na))
    print("CP order 'wget curl':", fmt(b, nb))
    print(f"  Fisher exact two-sided p = {fisher_two_sided(a, na - a, b, nb - b):.4f}")


if __name__ == "__main__":
    main()
