#!/usr/bin/env bash
#SBATCH --job-name=merops_dl
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.out
set -uo pipefail
BASE=https://ftp.ebi.ac.uk/pub/databases/merops/current_release
D=/shared/a2-public/abk/merops
mkdir -p "$D/raw" "$D/extracted"; cd "$D/raw"
curl -sS -L -m 120 "$BASE/" -o index.html || echo "directory listing failed"
grep -o 'meropsweb[0-9]*\.tar\.gz' index.html 2>/dev/null | sort -u > files.txt || true
echo "== archives listed:"; cat files.txt
F=$(sort -V files.txt | tail -n 1); F=${F:-meropsweb124.tar.gz}
echo "== downloading $F"
curl -L -C - --retry 5 --retry-delay 10 -o "$F" "$BASE/$F" || { echo "download failed"; exit 1; }
ls -lh "$F"
tar -xzf "$F" -C ../extracted || { echo "extract failed"; exit 1; }
cd ../extracted; du -sh .
echo "== tables mentioning substrate / cleav:"
grep -rn -i -E "CREATE TABLE.*(substrate|cleav)" . 2>/dev/null | head -30 || true
echo "== all table names:"
grep -rhoi -E "CREATE TABLE \`?[A-Za-z0-9_]+" . 2>/dev/null | sort -u | head -100 || true
