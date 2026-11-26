"""Make diagnostic pinpoint-test rows unambiguous before they reach the model.

The manual's pinpoint tables extract as

    C4 CHECK THE COOLANT LEVEL | Yes INSTALL a new thermostat. ... No GO to Pinpoint Test A ...
    • NOTE: ... • Visually check the coolant level ... • Is the engine coolant level within specification? |

i.e. the Yes/No outcomes come *before* the checks and the question they answer. The model inverted such a
branch in testing. This rewrites each well-formed row as step → checks → question → "Yes:" / "No:" lines,
keeping every word of the manual (only order and labels change). Rows that don't match exactly are left as-is.
"""

from __future__ import annotations

import re

ROW_RE = re.compile(
    r"^(?P<step>[A-Z]{1,3}\d{1,2}) (?P<title>[A-Z0-9][^|]*?) \| Yes (?P<yes>.+?) No (?P<no>(?:[A-Z]{2,}|The |GO ).*)$"
)
QUESTION_LINE_RE = re.compile(r"^\s*•\s*(?P<body>.*\?)\s*\|?\s*$")


def restructure_pinpoint(text: str) -> str:
    if "| Yes " not in text:
        return text
    lines = text.split("\n")
    out: list[str] = []
    i = 0
    while i < len(lines):
        m = ROW_RE.match(lines[i].strip())
        q = QUESTION_LINE_RE.match(lines[i + 1]) if m and i + 1 < len(lines) else None
        if not (m and q):
            out.append(lines[i])
            i += 1
            continue
        bullets = [b.strip() for b in re.split(r"\s*•\s*", q.group("body")) if b.strip()]
        question = bullets.pop() if bullets and bullets[-1].endswith("?") else None
        if question is None:
            out.append(lines[i])
            i += 1
            continue
        out.append(f"{m.group('step')} {m.group('title').strip()}")
        out.extend(f"   • {b}" for b in bullets)
        out.append(f"   Question: {question}")
        out.append(f"   → Yes: {m.group('yes').strip()}")
        out.append(f"   → No: {m.group('no').strip()}")
        i += 2
    return "\n".join(out)
