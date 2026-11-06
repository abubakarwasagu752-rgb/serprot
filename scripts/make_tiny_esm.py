#!/usr/bin/env python
"""Create a tiny random ESM-2-shaped model + tokenizer on disk. For offline smoke tests only."""
import argparse
import tempfile
from pathlib import Path

VOCAB = ["<cls>", "<pad>", "<eos>", "<unk>", "L", "A", "G", "V", "S", "E", "R", "T", "I", "D", "P", "K", "Q", "N", "F",
         "Y", "M", "H", "W", "C", "X", "B", "U", "Z", "O", ".", "-", "<null_1>", "<mask>"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=3)
    a = ap.parse_args()
    from transformers import EsmConfig, EsmModel, EsmTokenizer

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    vf = out / "vocab.txt"
    vf.write_text("\n".join(VOCAB) + "\n")
    tok = EsmTokenizer(vocab_file=str(vf))
    tok.save_pretrained(out)
    cfg = EsmConfig(vocab_size=len(VOCAB), hidden_size=a.hidden, num_hidden_layers=a.layers, num_attention_heads=4,
                    intermediate_size=a.hidden * 2, max_position_embeddings=1026, pad_token_id=1, mask_token_id=32,
                    position_embedding_type="rotary", token_dropout=True, emb_layer_norm_before=False)
    EsmModel(cfg, add_pooling_layer=False).save_pretrained(out)
    print(f"tiny ESM saved to {out}")


if __name__ == "__main__":
    main()
