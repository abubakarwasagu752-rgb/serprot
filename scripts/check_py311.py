#!/usr/bin/env python
"""Check every pipeline script parses under the TARGET Python (3.11 on the cluster).

Two modes, picked automatically:
  running on 3.11 or older  -> compile each file; the interpreter itself is the definitive check
  running on 3.12 or newer  -> compile, then additionally flag f-strings that nest the same quote
                               character inside their braces (legal from 3.12 via PEP 701,
                               a SyntaxError on 3.11), which compiling here would NOT catch

  python scripts/check_py311.py            # exits non-zero if anything would fail on 3.11
"""
import sys
import tokenize
from pathlib import Path

TARGET = (3, 11)
NEW_FSTRINGS = sys.version_info >= (3, 12)


def nested_quote_hits(path):
    """Only meaningful on 3.12+, where these tokenize successfully."""
    stack, hits = [], []
    with open(path, "rb") as fh:
        try:
            toks = list(tokenize.tokenize(fh.readline))
        except Exception as e:  # noqa: BLE001
            return [(0, f"tokenize failed: {e}")]
    for t in toks:
        if t.type == tokenize.FSTRING_START:
            stack.append(t.string[-1])
        elif t.type == tokenize.FSTRING_END:
            if stack:
                stack.pop()
        elif t.type == tokenize.STRING and stack:
            q = t.string.lstrip("rRbBfFuU")[:1]
            if q and q == stack[-1]:
                hits.append((t.start[0], t.line.strip()[:120]))
    return hits


def main():
    root = Path(__file__).resolve().parents[1]
    files = sorted(list((root / "scripts").glob("*.py")) + list((root / "src/serprot").glob("*.py")))
    mode = ("3.12+: compile + nested-quote f-string scan" if NEW_FSTRINGS
            else f"{sys.version_info.major}.{sys.version_info.minor}: compile only (this IS the target check)")
    print(f"checking {len(files)} files against Python {TARGET[0]}.{TARGET[1]}  [mode: {mode}]\n")

    problems = 0
    for f in files:
        src = f.read_text()
        try:
            compile(src, str(f), "exec")
        except SyntaxError as e:
            print(f"SYNTAX ERROR  {f.relative_to(root)}:{e.lineno}\n    {e.msg}\n    {(e.text or '').strip()[:120]}")
            problems += 1
            continue
        if NEW_FSTRINGS:
            for ln, line in nested_quote_hits(f):
                print(f"PY3.11-INCOMPATIBLE  {f.relative_to(root)}:{ln}"
                      f"\n    f-string nests its own quote character (needs 3.12+)\n    {line}")
                problems += 1

    print(f"\n{problems} problem(s)")
    if not NEW_FSTRINGS:
        print("note: run this on 3.12+ as well before shipping -- newer syntax compiles silently there,")
        print("      which is exactly how the last bug reached the cluster.")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
