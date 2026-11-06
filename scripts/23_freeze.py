#!/usr/bin/env python
"""Freeze the project: a timestamped, checksummed, read-only snapshot of everything the paper rests on.

Two archives, because they have different jobs.

  CODE + RESULTS  scripts, src, slurm, tests, config, every analysis table, the figures, the
                  manuscript tables and the job logs. A few tens of MB, so it travels to your PC and
                  is what a repository deposit is built from.
  DERIVED DATA    the structures, the embedding cache and the descriptor shards -- gigabytes, and
                  expensive to regenerate (the ESMFold set alone cost GPU hours). Recorded by name,
                  size and modification time by default; archived in full with --include-derived.

Every file in the code archive is hashed, so a later `--verify` says whether anything has drifted.
With git available, the snapshot is also committed, which gives a second, independent record and
starts the repository the journal's availability requirement needs.

  python scripts/23_freeze.py --config config.yaml
  python scripts/23_freeze.py --config config.yaml --include-derived      # adds the big archive
  python scripts/23_freeze.py --config config.yaml --verify freeze/serprot_freeze_20261107_0510
"""
import hashlib
import os
import shutil
import subprocess
import sys
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from serprot.common import base_parser, load_config  # noqa: E402

# What the paper rests on: small, and worth keeping forever. Each entry is (path, suffixes to keep).
# tests/ is filtered to source only: the harnesses generate whole synthetic project roots inside it,
# which are test output rather than code and would otherwise dominate the archive.
CODE = [("scripts", None), ("src", None), ("slurm", None), ("tests", (".py", ".sh", ".md")),
        ("config.yaml", None), ("requirements.txt", None), ("README.md", None)]
RESULTS = [("runs/analysis", None), ("figures", None), ("tables", None)]
LOGS = [("logs", (".out",))]
# big, regenerable in principle but costly in practice
DERIVED = ["data/processed", "data/structures", "data/structures_esmfold", "data/raw"]
SKIP_DIRS = {"__pycache__", ".ipynb_checkpoints", "freeze", ".git"}


def sha256(p: Path, buf=1 << 20) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(buf), b""):
            h.update(chunk)
    return h.hexdigest()


def walk(root: Path, rel: str, only_suffix=None):
    src = root / rel
    if not src.exists():
        return
    if src.is_file():
        yield src
        return
    for dirpath, dirnames, filenames in os.walk(src):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in sorted(filenames):
            if only_suffix and not fn.endswith(only_suffix):
                continue
            yield Path(dirpath) / fn


def human(n: float) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}"
        n /= 1024


def tool_version(cmd):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return (r.stdout or r.stderr).strip().splitlines()[0]
    except Exception:  # noqa: BLE001
        return "not found"


def package_versions():
    import importlib
    out = [f"python {sys.version.split()[0]}"]
    for m in ["numpy", "pandas", "scipy", "sklearn", "torch", "transformers", "matplotlib", "openpyxl"]:
        try:
            out.append(f"{m} {importlib.import_module(m).__version__}")
        except Exception:  # noqa: BLE001
            out.append(f"{m} absent")
    return out


def verify(snapshot: Path) -> int:
    man = snapshot / "MANIFEST.sha256"
    if not man.exists():
        raise SystemExit(f"no MANIFEST.sha256 in {snapshot}")
    bad, missing, n = [], [], 0
    for line in man.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        digest, rel = line.split("  ", 1)
        p = snapshot / rel
        n += 1
        if not p.exists():
            missing.append(rel)
        elif sha256(p) != digest:
            bad.append(rel)
    print(f"checked {n} files in {snapshot}")
    print(f"  missing: {len(missing)}")
    print(f"  changed: {len(bad)}")
    for r in (missing + bad)[:20]:
        print(f"    {r}")
    if missing or bad:
        print("\nthe snapshot has drifted; it is no longer the thing the paper was written from")
        return 1
    print("\nintact: every file matches the hash recorded at freeze time")
    return 0


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--outdir", default=None, help="default: <root>/freeze")
    ap.add_argument("--include-derived", action="store_true",
                    help="also archive structures, embeddings and descriptor shards (gigabytes)")
    ap.add_argument("--no-git", action="store_true", help="skip the git commit")
    ap.add_argument("--no-readonly", action="store_true", help="leave the snapshot writable")
    ap.add_argument("--verify", default=None, help="check an existing snapshot against its manifest")
    args = ap.parse_args()

    if args.verify:
        raise SystemExit(verify(Path(args.verify).expanduser()))

    cfg = load_config(args.config)
    root = Path(cfg["paths"]["root"]).expanduser().resolve()
    base = Path(args.outdir).expanduser() if args.outdir else root / "freeze"
    stamp = datetime.now(timezone.utc).astimezone().strftime("%Y%m%d_%H%M")
    snap = base / f"serprot_freeze_{stamp}"
    if snap.exists():
        raise SystemExit(f"{snap} already exists; wait a minute or pass --outdir")
    snap.mkdir(parents=True)
    print(f"root     {root}")
    print(f"snapshot {snap}\n")

    # ---- copy the code and results
    copied, total = [], 0
    for group, rels in [("code", CODE), ("results", RESULTS), ("logs", LOGS)]:
        n0, b0 = len(copied), total
        for rel, suf in rels:
            for src in walk(root, rel, only_suffix=suf):
                r = src.relative_to(root)
                dst = snap / r
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                copied.append(r)
                total += src.stat().st_size
        print(f"{group:>8}: {len(copied) - n0} files, {human(total - b0)}")
    if total > 300 * 1024 ** 2:
        # bloat here is almost always generated data that slipped in; name it rather than hide it
        big = sorted(((snap / r).stat().st_size, r) for r in copied)[-5:]
        print(f"\nWARNING the code archive is {human(total)}, which is larger than code should be.")
        for sz, r in reversed(big):
            print(f"  {human(sz):>10}  {r}")

    # ---- hash everything copied, so drift is detectable later
    lines = [f"# serprot freeze {stamp}", f"# {len(copied)} files from {root}"]
    for r in copied:
        lines.append(f"{sha256(snap / r)}  {r.as_posix()}")
    (snap / "MANIFEST.sha256").write_text("\n".join(lines) + "\n")
    print(f"\nhashed {len(copied)} files -> MANIFEST.sha256")

    # ---- inventory the big derived data without copying it
    inv, dtotal, dcount = ["path\tbytes\tmodified"], 0, 0
    for rel in DERIVED:
        for src in walk(root, rel):
            st = src.stat()
            inv.append(f"{src.relative_to(root).as_posix()}\t{st.st_size}\t"
                       f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(st.st_mtime))}")
            dtotal += st.st_size
            dcount += 1
    (snap / "DERIVED_INVENTORY.tsv").write_text("\n".join(inv) + "\n")
    print(f"inventoried {dcount} derived files, {human(dtotal)} -> DERIVED_INVENTORY.tsv")

    # ---- provenance: the facts a reader needs to tell what this snapshot IS
    prov = [f"# serprot freeze {stamp}", "",
            f"Frozen at: {datetime.now(timezone.utc).astimezone().isoformat()}",
            f"Project root: {root}", f"Host: {os.uname().nodename}", "",
            "## Software", *package_versions(),
            f"mmseqs {tool_version(['mmseqs', 'version'])}", "",
            "## Data provenance",
            "UniProt: accessed 4 November 2026 (release 2026_04)",
            "AlphaFold DB v4: accessed 4-5 November 2026",
            "TEMPURA: see data/raw/ogt_table.csv",
            "ESMFold v1 and ESM-2 650M: facebook/esmfold_v1, facebook/esm2_t33_650M_UR50D", "",
            "## Contents",
            f"Code and results: {len(copied)} files, {human(total)} (hashed in MANIFEST.sha256)",
            f"Derived data: {dcount} files, {human(dtotal)} "
            f"({'archived alongside' if args.include_derived else 'listed only, not copied'})", "",
            "## Analysis tables in this snapshot"]
    adir = snap / "runs/analysis"
    if adir.exists():
        for p in sorted(adir.glob("*.tsv")):
            with open(p) as f:
                rows = sum(1 for _ in f) - 1
            prov.append(f"{p.name}\t{rows} rows")
    (snap / "PROVENANCE.md").write_text("\n".join(prov) + "\n")
    print("wrote PROVENANCE.md")

    # ---- a git commit is a second, independent record, and the start of the public deposit
    commit = None
    if not args.no_git and shutil.which("git"):
        try:
            subprocess.run(["git", "init", "-q"], cwd=snap, check=True)
            subprocess.run(["git", "add", "-A"], cwd=snap, check=True)
            env = {**os.environ, "GIT_AUTHOR_NAME": "serprot freeze",
                   "GIT_AUTHOR_EMAIL": "freeze@localhost",
                   "GIT_COMMITTER_NAME": "serprot freeze",
                   "GIT_COMMITTER_EMAIL": "freeze@localhost"}
            subprocess.run(["git", "commit", "-q", "-m", f"serprot freeze {stamp}"],
                           cwd=snap, check=True, env=env)
            commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=snap,
                                    capture_output=True, text=True).stdout.strip()
            print(f"git commit {commit[:12]}")
        except Exception as e:  # noqa: BLE001
            print(f"git snapshot skipped: {type(e).__name__}: {e}")

    # ---- the portable archive
    tarball = base / f"serprot_freeze_{stamp}.tar.gz"
    with tarfile.open(tarball, "w:gz") as tf:
        tf.add(snap, arcname=snap.name, filter=lambda ti: None if "/.git/" in ti.name else ti)
    print(f"\nwrote {tarball}  ({human(tarball.stat().st_size)})")

    if args.include_derived:
        big = base / f"serprot_derived_{stamp}.tar"
        print(f"archiving derived data to {big} \u2014 this is slow and large")
        with tarfile.open(big, "w") as tf:
            for rel in DERIVED:
                if (root / rel).exists():
                    tf.add(root / rel, arcname=rel)
        print(f"wrote {big}  ({human(big.stat().st_size)})")

    if not args.no_readonly:
        for dirpath, _dn, fns in os.walk(snap):
            if "/.git" in dirpath:
                continue
            for fn in fns:
                p = Path(dirpath) / fn
                p.chmod(p.stat().st_mode & ~0o222)
        print("snapshot set read-only")

    print("\nTo check later that nothing has drifted:")
    print(f"  python scripts/23_freeze.py --config {args.config} --verify {snap}")
    print("\nTo pull the portable archive to your PC, from PowerShell:")
    print(f"  scp 68090800301@cpe.kmutt.ac.th@ctl01.slurm.cpe.kmutt.ac.th:{tarball} "
          '"C:\\Users\\User\\Downloads\\"')
    if not args.include_derived:
        print("\nThe structures and embedding cache are NOT in this archive. They are listed in")
        print("DERIVED_INVENTORY.tsv; re-run with --include-derived to archive them in full.")


if __name__ == "__main__":
    main()
