#!/usr/bin/env bash
#SBATCH --job-name=serprot_gpucheck
#SBATCH --partition=gpul40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x_%j.out
# Override partition if needed:  sbatch --partition=gpu4090 slurm/smoke_gpu.sbatch
mkdir -p logs
source "${SLURM_SUBMIT_DIR:-.}/slurm/env.sh"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python - <<'PY'
import time, torch
from transformers import AutoTokenizer, AutoModel
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0))
name = "facebook/esm2_t33_650M_UR50D"
tok = AutoTokenizer.from_pretrained(name); m = AutoModel.from_pretrained(name).cuda()
seq = "M" + "GDSGGPLVC" * 110            # ~1000 residues
enc = tok([seq] * 32, return_tensors="pt", truncation=True, max_length=1002).to("cuda")
for p in m.parameters(): p.requires_grad = False
for l in m.encoder.layer[-4:]:
    for p in l.parameters(): p.requires_grad = True
opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], 1e-5)
torch.cuda.reset_peak_memory_stats(); t = time.time()
for _ in range(3):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = m(**enc).last_hidden_state.float().pow(2).mean()
    loss.backward(); opt.step(); opt.zero_grad()
torch.cuda.synchronize()
print(f"650M, batch 32 x 1002 tokens, last 4 layers trainable: {(time.time()-t)/3:.2f} s/step, peak {torch.cuda.max_memory_allocated()/2**30:.1f} GiB")
PY
