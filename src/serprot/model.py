"""Dual-subspace ESM-2 model: a thermal latent z_T and a specificity latent z_S.

Variants
  joint   z_T == z_S (one shared latent); both heads read the same vector. No penalties. (reference)
  disent  separate projections + gradient-reversal adversaries + distance-correlation penalty
  noadv   separate projections + dCor penalty only
  nodcor  separate projections + adversaries only
  plain   separate projections, no penalties (capacity-matched control)
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

VARIANTS = ("joint", "disent", "noadv", "nodcor", "plain")


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


def grl(x, lam):
    return GradReverse.apply(x, lam)


def mlp(i, h, o, drop):
    return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Dropout(drop), nn.Linear(h, o))


def distance_correlation_sq(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Squared sample distance correlation (V-statistic). Differentiable; needs batch >= ~16."""

    def centred(a):
        d = torch.sqrt(((a[:, None, :] - a[None, :, :]) ** 2).sum(-1) + 1e-9)
        return d - d.mean(0, keepdim=True) - d.mean(1, keepdim=True) + d.mean()

    A, B = centred(x.float()), centred(y.float())
    dcov2 = (A * B).mean()
    return dcov2.clamp(min=0) / (torch.sqrt((A * A).mean() * (B * B).mean()) + eps)


def breadth_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """logits [B,P,20] -> normalised entropy averaged over positions [B]; differentiable."""
    logp = F.log_softmax(logits.float(), dim=-1)
    h = -(logp.exp() * logp).sum(-1) / math.log(20)
    return h.mean(-1)


class DisentangledESM(nn.Module):
    def __init__(self, esm_path: str, n_pos: int, n_phyla: int, variant: str = "disent", latent: int = 64,
                 hidden: int = 256, dropout: float = 0.1, unfreeze_last: int = 4):
        super().__init__()
        from transformers import EsmModel

        assert variant in VARIANTS, variant
        self.variant, self.n_pos, self.n_phyla = variant, n_pos, n_phyla
        self.backbone = EsmModel.from_pretrained(esm_path, add_pooling_layer=False)
        for p in self.backbone.parameters():
            p.requires_grad = False
        layers = self.backbone.encoder.layer
        for layer in layers[max(len(layers) - unfreeze_last, 0):]:
            for p in layer.parameters():
                p.requires_grad = True
        ln = getattr(self.backbone.encoder, "emb_layer_norm_after", None)
        if ln is not None:
            for p in ln.parameters():
                p.requires_grad = True
        d = self.backbone.config.hidden_size

        self.proj_T = mlp(d, hidden, latent, dropout)
        self.proj_S = self.proj_T if variant == "joint" else mlp(d, hidden, latent, dropout)
        self.head_ogt = mlp(latent, hidden, 1, dropout)
        self.head_tm = mlp(latent, hidden, 1, dropout)
        self.head_spec = mlp(latent, hidden, n_pos * 20, dropout)
        # adversaries (read gradient-reversed latents)
        self.adv_ogt = mlp(latent, hidden, 1, dropout)          # tries to read OGT from z_S
        self.adv_spec = mlp(latent, hidden, n_pos * 20, dropout)  # tries to read specificity from z_T
        self.adv_phy_T = nn.Linear(latent, max(n_phyla, 1))
        self.adv_phy_S = nn.Linear(latent, max(n_phyla, 1))
        self._emb_out = None

    # -------------------------------------------------------------- backbone pooling
    def encode(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        m = attention_mask.clone()
        m[:, 0] = 0                                              # <cls>
        lengths = attention_mask.sum(1)
        m[torch.arange(m.size(0), device=m.device), lengths - 1] = 0  # <eos>
        m = m.unsqueeze(-1).to(out.dtype)
        return (out * m).sum(1) / m.sum(1).clamp(min=1)

    def forward(self, input_ids, attention_mask, lam_adv: float = 0.0):
        h = self.encode(input_ids, attention_mask)
        zT, zS = self.proj_T(h), self.proj_S(h)
        res = {
            "zT": zT, "zS": zS,
            "ogt": self.head_ogt(zT).squeeze(-1),
            "tm": self.head_tm(zT).squeeze(-1),
            "spec": self.head_spec(zS).view(-1, self.n_pos, 20),
        }
        if lam_adv > 0 and self.variant in ("disent", "nodcor"):
            res["adv_ogt"] = self.adv_ogt(grl(zS, lam_adv)).squeeze(-1)
            res["adv_spec"] = self.adv_spec(grl(zT, lam_adv)).view(-1, self.n_pos, 20)
            if self.n_phyla > 1:
                res["adv_phy_T"] = self.adv_phy_T(grl(zT, lam_adv))
                res["adv_phy_S"] = self.adv_phy_S(grl(zS, lam_adv))
        return res

    # -------------------------------------------------------------- attribution hook
    def enable_embedding_grad(self):
        """Make the embedding-layer output a graph leaf we can read gradients from (for saliency)."""
        def hook(_, __, output):
            if not output.requires_grad:
                output.requires_grad_(True)
            output.retain_grad()
            self._emb_out = output
        return self.backbone.embeddings.register_forward_hook(hook)

    def trainable_state(self):
        keep = {n for n, p in self.named_parameters() if p.requires_grad}
        return {k: v.detach().cpu() for k, v in self.state_dict().items() if k in keep}


def load_trained(ckpt_path, device):
    """Rebuild a trained DisentangledESM from the checkpoint written by 05_train_disentangled.py."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    m = DisentangledESM(ck["esm_model"], ck["n_pos"], ck["n_phyla"], ck["variant"], ck["latent"], ck["hidden"],
                        ck["dropout"], ck["unfreeze_last"])
    m.load_state_dict(ck["state"], strict=False)
    return m.to(device).eval(), ck
