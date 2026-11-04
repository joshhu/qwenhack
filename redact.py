"""Mask credential-like strings the model hallucinated in its URLs.

Copies each given .jsonl to raw/<same path> (gitignored) the first time, then
rewrites the file in place with access-key IDs, signatures and tokens masked.

Usage: uv run python redact.py results/*.jsonl invalid_runs/*/*.jsonl
"""

from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).parent

PATTERNS = [
    # Alibaba / AWS / GitHub style key IDs anywhere in text.
    re.compile(r"\b(LTAI[0-9A-Za-z]{4})([0-9A-Za-z]{8,})([0-9A-Za-z]{4})\b"),
    re.compile(r"\b(AKIA[0-9A-Z]{2})([0-9A-Z]{10,})([0-9A-Z]{2})\b"),
    re.compile(r"\b(ghp_[0-9A-Za-z]{2})([0-9A-Za-z]{10,})([0-9A-Za-z]{2})\b"),
]
QUERY_SECRET = re.compile(
    r"((?:Signature|security-token|x-oss-signature|X-Amz-Signature|X-Amz-Security-Token|token|sig)=)([^&\s\"'\\^]{6,})",
    re.I,
)


def mask(m: re.Match) -> str:
    return f"{m.group(1)}…{m.group(3)}"


def redact(text: str) -> str:
    for p in PATTERNS:
        text = p.sub(mask, text)
    return QUERY_SECRET.sub(lambda m: m.group(1) + "REDACTED", text)


def main() -> None:
    for arg in sys.argv[1:]:
        path = Path(arg).resolve()
        raw = ROOT / "raw" / path.relative_to(ROOT)
        if not raw.exists():
            raw.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, raw)
        src = raw.read_text()
        out = redact(src)
        path.write_text(out)
        print(f"{path.relative_to(ROOT)}: {sum(a != b for a, b in zip(src.splitlines(), out.splitlines()))} lines redacted")


if __name__ == "__main__":
    main()
