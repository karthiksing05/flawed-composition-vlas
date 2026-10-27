"""PLAN_BINDER B1 (docs/PLAN_BINDER.md section 7): frozen patch features for the binder's regions (openpi venv, GPU).

  policy  pi05_l90c@5000's own image tokens of the TRAINING demos' frame-0 images (results/bind/b1/demos/*.npz, b1_demos.py), as
          s1_feats.py computed them for S1's frames (embed_prefix(...)[:, :256], SigLIP + projector, 2048-d)
          -> results/bind/b1/feats/demos_img.npy + demos_frames.json (keys "<tid>#<k>").  S1's train / dev tokens are reused.
  dino    the comparison arm: DINOv2-small (facebook/dinov2-small, weights already on disk, local_files_only) patch tokens of the
          same 224 images (ImageNet normalisation, no resize: 16 x 16 patches of 14 px, the policy's grid)
          -> results/bind/b1/feats/dino_{train,dev,demos}.npy [N, 256, 384] fp16 + frame lists.

    DINO_CACHE=... .venv/bin/python experiments/b1_feats.py
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
FE = os.path.join(ROOT, "results", "bind", "b1", "feats")
S1F = os.path.join(ROOT, "results", "bind", "s1", "feats")
BATCH = 32


def demo_images():
    keys, ims = [], []
    for f in sorted(glob.glob(os.path.join(ROOT, "results", "bind", "b1", "demos", "*.npz"))):
        z = np.load(f, allow_pickle=True)
        tid = str(z["tid"])
        for k, im in enumerate(z["img"]):
            keys.append(f"{tid}#{k}")
            ims.append(im)
    return keys, ims


def main():
    import torch
    os.makedirs(FE, exist_ok=True)
    keys, ims = demo_images()
    if not os.path.exists(os.path.join(FE, "demos_img.npy")):
        from lifelong_ood.policy.backbone import PolicyBackbone
        bk = PolicyBackbone(os.environ.get("BACKBONE_CONFIG") or "pi05_l90c", checkpoint_dir=os.environ.get("BACKBONE_CKPT") or None)
        m = bk.model
        out = np.lib.format.open_memmap(os.path.join(FE, "demos_img.tmp.npy"), mode="w+", dtype=np.float16, shape=(len(ims), 256, 2048))
        for g in range(0, len(ims), BATCH):
            obs = [{"observation/image": im, "observation/wrist_image": im, "observation/state": np.zeros(8, np.float32),
                    "prompt": "pick up the object"} for im in ims[g:g + BATCH]]
            ob = bk.observations(obs)
            with torch.no_grad():
                images, img_masks, lt, lm, _ = m._preprocess_observation(ob, train=False)
                embs, _, _ = m.embed_prefix(images, img_masks, lt, lm)
            out[g:g + len(obs)] = embs[:, :256].float().cpu().numpy().astype(np.float16)
        out.flush()
        del out
        os.replace(os.path.join(FE, "demos_img.tmp.npy"), os.path.join(FE, "demos_img.npy"))
        json.dump(keys, open(os.path.join(FE, "demos_frames.json"), "w"))
        print(f"[b1 feats] policy tokens: {len(keys)} demo frames", flush=True)
        del bk, m
        torch.cuda.empty_cache()
    from transformers import AutoModel
    dm = AutoModel.from_pretrained("facebook/dinov2-small", cache_dir=os.environ["DINO_CACHE"], local_files_only=True).cuda().eval()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()
    sets = {"train": json.load(open(os.path.join(S1F, "train_frames.json"))), "dev": json.load(open(os.path.join(S1F, "dev_frames.json"))),
            "demos": keys}
    for nm, frames in sets.items():
        if os.path.exists(os.path.join(FE, f"dino_{nm}.npy")):
            continue
        out = np.zeros((len(frames), 256, 384), np.float16)
        for g in range(0, len(frames), BATCH):
            if nm == "demos":
                im = np.stack(ims[g:g + BATCH])
            else:
                im = np.stack([np.load(os.path.join(ROOT, f), allow_pickle=True)["img"] for f in frames[g:g + BATCH]])
            x = (torch.tensor(im, dtype=torch.float32).permute(0, 3, 1, 2).cuda() / 255.0 - mean) / std
            with torch.no_grad():
                h = dm(pixel_values=x).last_hidden_state[:, 1:]
            assert h.shape[1] == 256, h.shape
            out[g:g + len(im)] = h.float().cpu().numpy().astype(np.float16)
        np.save(os.path.join(FE, f"dino_{nm}.npy"), out)
        json.dump(frames, open(os.path.join(FE, f"dino_{nm}_frames.json"), "w"))
        print(f"[b1 feats] dino {nm}: {len(frames)} frames", flush=True)


if __name__ == "__main__":
    main()
