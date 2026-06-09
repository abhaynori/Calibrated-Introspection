from __future__ import annotations

import re
from typing import Optional

_VALID = "ABCDEFGH"

_PATTERNS = [
    re.compile(r"answer\s*[:\-]?\s*\(?\*?\*?([A-H])\*?\*?\)?", re.IGNORECASE),
    re.compile(r"(?:answer|option|choice)\s+is\s+\(?([A-H])\)?", re.IGNORECASE),
    re.compile(r"\(([A-H])\)"),
]


def extract_answer(text: str, n_options: int) -> Optional[str]:
    if not text:
        return None
    valid = set(_VALID[:n_options])

    for pat in _PATTERNS:
        hits = [m.upper() for m in pat.findall(text) if m.upper() in valid]
        if hits:
            return hits[-1]

    for line in reversed([ln.strip() for ln in text.splitlines() if ln.strip()]):
        if len(line) <= 3 and line[0].upper() in valid:
            return line[0].upper()
        break
    return None
