#!/usr/bin/env python
"""Pre-flight check: can this machine reach the data sources, and are the family queries sane?

Run this first on the cluster (login node). It makes tiny requests only.
"""
import shutil
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from serprot.common import base_parser, cfg_file, load_config  # noqa: E402

UNIPROT = "https://rest.uniprot.org/uniprotkb/search"
AFDB = "https://alphafold.ebi.ac.uk/api/prediction/P00780"  # subtilisin-like protease, known entry
HF = "https://huggingface.co/facebook/esm2_t12_35M_UR50D/resolve/main/config.json"


def head(url, **kw):
    try:
        r = requests.get(url, timeout=30, **kw)
        return r.status_code, r
    except Exception as e:  # noqa: BLE001
        return f"ERR {type(e).__name__}", None


def main():
    ap = base_parser(__doc__)
    cfg = load_config(ap.parse_args().config)
    ok = True

    tax = " OR ".join(f"taxonomy_id:{t}" for t in cfg["taxa"])
    for fam, spec in cfg["families"].items():
        q = f"(xref:pfam-{spec['pfam']}) AND ({tax})"
        code, r = head(UNIPROT, params={"query": q, "format": "tsv", "fields": "accession", "size": 1})
        total = r.headers.get("x-total-results", "?") if r is not None else "?"
        print(f"[UniProt] {fam} {spec['pfam']}: HTTP {code}, total prokaryotic entries = {total}")
        ok &= code == 200

    code, _ = head(AFDB)
    print(f"[AlphaFold DB API]: HTTP {code}")
    ok &= code == 200
    code, _ = head(HF)
    print(f"[Hugging Face ESM-2 config]: HTTP {code}  (if this fails, download the model on a node with internet and set esm_model to the local folder)")

    print(f"[mmseqs on PATH]: {'yes' if shutil.which('mmseqs') else 'NO (a slow k-mer fallback will be used; install mmseqs2 for real runs)'}")

    for name, rel in [("OGT table", cfg["ogt"]["table"]), ("cleavage table", cfg["specificity"]["cleavages"]), ("Tm table (optional)", cfg["tm"]["table"])]:
        print(f"[{name}] {cfg_file(cfg, rel)}: {'found' if cfg_file(cfg, rel).exists() else 'MISSING'}")

    try:
        import torch

        print(f"[torch] {torch.__version__}, cuda available = {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            print(f"[GPU] {p.name}, {p.total_memory / 2**30:.0f} GiB")
    except ImportError:
        print("[torch] not installed")

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
