"""Minimal reader for MySQL dump files (plain text parsing; the SQL is never executed)."""
import re

TOKEN = re.compile(r"'(?:[^'\\]|\\.|'')*'|\(|\)|,|;|[^,()';\s]+", re.S)
ESC = {"n": "\n", "t": "\t", "r": "\r", "0": "\0", "Z": "\x1a", "b": "\b"}


def _unq(tok: str) -> str:
    s = tok[1:-1]
    if "\\" in s or "''" in s:
        s = re.sub(r"\\(.)|''", lambda m: ESC.get(m.group(1), m.group(1)) if m.group(1) else "'", s)
    return s


def columns(path: str, table: str) -> list[str]:
    cols, inside = [], False
    with open(path, encoding="latin-1", errors="replace") as fh:
        for line in fh:
            if line.startswith(f"CREATE TABLE `{table}`"):
                inside = True
                continue
            if inside:
                m = re.match(r"\s+`([^`]+)`", line)
                if m:
                    cols.append(m.group(1))
                elif line.lstrip().startswith(("PRIMARY", "KEY", "UNIQUE", "FULLTEXT", ")")):
                    break
    return cols


def iter_rows(path: str, table: str):
    """Yield one tuple per row (strings, or None for NULL)."""
    prefix = f"INSERT INTO `{table}` VALUES"
    with open(path, encoding="latin-1", errors="replace") as fh:
        for line in fh:
            if not line.startswith(prefix):
                continue
            row = None
            for m in TOKEN.finditer(line, len(prefix)):
                t = m.group()
                if t == "(":
                    row = []
                elif t == ")":
                    if row is not None:
                        yield tuple(row)
                    row = None
                elif t in (",", ";") or row is None:
                    continue
                else:
                    row.append(None if t == "NULL" else (_unq(t) if t[0] == "'" else t))
