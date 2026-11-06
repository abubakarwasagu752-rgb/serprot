#!/usr/bin/env bash
# Read-only look at the cluster. Run on the login node and paste the whole output back:
#   bash slurm/preflight.sh 2>&1 | tee preflight.txt
echo "== whoami / host";   whoami; hostname
echo "== slurm partitions (name, gres, mem MB, cpus, time limit)"
sinfo -o "%P|%G|%m|%c|%l|%D|%t" 2>&1 | head -40
echo "== my accounts / qos"
sacctmgr -n -P show assoc user=$USER format=account,partition,qos,maxjobs,grptres 2>&1 | head -20
echo "== conda";  command -v conda && conda --version && conda env list
echo "== modules"; (module avail 2>&1 | head -40) || true
echo "== disk";   df -h "$HOME" 2>&1 | tail -n 2; quota -s 2>&1 | head -5
echo "== mmseqs"; command -v mmseqs || echo "mmseqs not installed yet"
echo "== internet from login node"
for u in https://rest.uniprot.org https://huggingface.co https://alphafold.ebi.ac.uk; do
  printf "%s -> " "$u"; curl -s -o /dev/null -m 15 -w "%{http_code}\n" "$u" || echo fail
done
