"""PLAN_BINDER B1.3 (docs/PLAN_BINDER.md section 13): frozen features of the GENERATED frames (results/bind/b1_3/units.json).
openpi venv, GPU.  Exactly s1_feats.py / b1_feats.py's encoders:
  siglip   pi05_l90c@5000's own image tokens (embed_prefix(...)[:, :256], 2048-d) -> results/bind/b1_3/feats/siglip_{train,test}.npy
  dino     DINOv2-small patch tokens (local weights)                               -> results/bind/b1_3/feats/dino_{train,test}.npy
  phrases  the policy's own word embeddings of every generated phrase + S1's phrases (the holistic comparator's input)
           -> results/bind/b1_3/feats/phrases.npz

    BACKBONE_CONFIG=pi05_l90c BACKBONE_CKPT=... DINO_CACHE=... .venv/bin/python experiments/b1_3_feats.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
B13 = os.path.join(ROOT, "results", "bind", "b1_3")
FE = os.path.join(B13, "feats")
LMAX, BATCH = 16, 16


def main():
    import torch
    from lifelong_ood.policy.backbone import PolicyBackbone
    U = json.load(open(os.path.join(B13, "units.json"), encoding="utf-8"))
    lab = json.load(open(os.path.join(ROOT, "results", "bind", "s1", "labels.json"), encoding="utf-8"))
    os.makedirs(FE, exist_ok=True)
    sets = {s: sorted({u["frame"] for u in U[s]}) for s in ("train", "test")}
    bk = PolicyBackbone(os.environ.get("BACKBONE_CONFIG") or "pi05_l90c", checkpoint_dir=os.environ.get("BACKBONE_CKPT") or None)
    m = bk.model
    for s, frames in sets.items():
        out = np.zeros((len(frames), 256, 2048), np.float16)
        for g in range(0, len(frames), BATCH):
            obs = []
            for f in frames[g:g + BATCH]:
                z = np.load(os.path.join(ROOT, f), allow_pickle=True)
                obs.append({"observation/image": z["img"], "observation/wrist_image": z["wimg"],
                            "observation/state": z["state8"], "prompt": "pick up the object"})
            ob = bk.observations(obs)
            with torch.no_grad():
                images, img_masks, lt, lm, _ = m._preprocess_observation(ob, train=False)
                embs, _, _ = m.embed_prefix(images, img_masks, lt, lm)
            out[g:g + len(obs)] = embs[:, :256].float().cpu().numpy().astype(np.float16)
        np.save(os.path.join(FE, f"siglip_{s}.npy"), out)
        json.dump(frames, open(os.path.join(FE, f"siglip_{s}_frames.json"), "w"))
        print(f"[b1.3 feats] siglip {s}: {len(frames)} frames", flush=True)
    from openpi.models import tokenizer as _tok
    sp = _tok.PaligemmaTokenizer(200)._tokenizer
    P = sorted({u["phrase"] for s in ("train", "test") for u in U[s]} | set(lab["phrases"]))
    emb = np.zeros((len(P), LMAX, 2048), np.float16)
    ln = np.zeros(len(P), np.int64)
    for i, p in enumerate(P):
        ids = sp.encode(p, add_bos=False)[:LMAX]
        with torch.no_grad():
            e = m.paligemma_with_expert.embed_language_tokens(torch.tensor([ids], device=bk.device))
        emb[i, :len(ids)] = e[0].float().cpu().numpy().astype(np.float16)
        ln[i] = len(ids)
    np.savez(os.path.join(FE, "phrases.npz"), emb=emb, len=ln, phrases=np.array(P))
    print(f"[b1.3 feats] {len(P)} phrases, max len {ln.max()}", flush=True)
    del bk, m
    torch.cuda.empty_cache()
    from transformers import AutoModel
    dm = AutoModel.from_pretrained("facebook/dinov2-small", cache_dir=os.environ["DINO_CACHE"], local_files_only=True).cuda().eval()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()
    for s, frames in sets.items():
        out = np.zeros((len(frames), 256, 384), np.float16)
        for g in range(0, len(frames), 32):
            im = np.stack([np.load(os.path.join(ROOT, f), allow_pickle=True)["img"] for f in frames[g:g + 32]])
            x = (torch.tensor(im, dtype=torch.float32).permute(0, 3, 1, 2).cuda() / 255.0 - mean) / std
            with torch.no_grad():
                h = dm(pixel_values=x).last_hidden_state[:, 1:]
            assert h.shape[1] == 256, h.shape
            out[g:g + len(im)] = h.float().cpu().numpy().astype(np.float16)
        np.save(os.path.join(FE, f"dino_{s}.npy"), out)
        json.dump(frames, open(os.path.join(FE, f"dino_{s}_frames.json"), "w"))
        print(f"[b1.3 feats] dino {s}: {len(frames)} frames", flush=True)


if __name__ == "__main__":
    main()
