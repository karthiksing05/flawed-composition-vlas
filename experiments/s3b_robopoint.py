"""PLAN_SOLVE S3b (docs/PLAN_S3.md): RoboPoint (wentao-yuan/robopoint-v1-vicuna-v1.5-13b, github wentaoyuan/RoboPoint @ 206cb1e) as an
off-the-shelf point-to-mask binder on S1's 240 dev units (results/bind/s1/labels.json, the same frames S1 / S1b / S1c scored).

Run with the S3b venv in ISOLATED mode (the downloads live in /coc/flash10/ksingara3/s3b_ext; nothing is imported from there except the
installed robopoint package):
    /coc/flash10/ksingara3/s3b_ext/venv/bin/python -I experiments/s3b_robopoint.py [--max N]  -> results/bind/s3b/pred_dev.npz

Query (pre-registered; RoboPoint's own answer-format suffix):
  object role: "Locate several points on the {phrase}."            e.g. "... on the black bowl on the left."
  goal role:   "Locate several points {phrase}."                   e.g. "... on the plate." / "... in the top drawer of the cabinet."
  (a goal phrase that does not start with on / in / into / onto / inside / under / to gets "on the " prepended)
Greedy decoding, max 256 new tokens.  Points (normalized x, y) -> pixels on the 224 x 224 frame -> S1's 16 x 16 tokens (s1b_train.tok_of).
peak = the token holding the most points (ties: the first point's); mask = tokens holding >= 1 point; no point -> peak -1 (a miss).
"""
import argparse
import json
import os
import re
import sys
import time

import numpy as np

ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
EXT = "/coc/flash10/ksingara3/s3b_ext"
MP = os.path.join(EXT, "weights", "robopoint-v1-vicuna-v1.5-13b")
OD = os.path.join(ROOT, "results", "bind", "s3b")
SUFFIX = (" Your answer should be formatted as a list of tuples, i.e. [(x1, y1), (x2, y2), ...], where each tuple contains the x and y "
          "coordinates of a point satisfying the conditions above. The coordinates should be between 0 and 1, indicating the "
          "normalized pixel locations of the points in the image.")
PREP = ("on ", "in ", "into ", "onto ", "inside ", "under ", "to ")


def tok_of(px):                      # = s1b_train.tok_of (copied: the repo's modules are not importable in this venv)
    x, y = np.clip(px[..., 0], 0, 223), np.clip(px[..., 1], 0, 223)
    return (y // 14).astype(int) * 16 + (x // 14).astype(int)


def query(role, phrase):
    if role == "object":
        return f"Locate several points on the {phrase}." + SUFFIX
    p = phrase if phrase.startswith(PREP) else "on the " + phrase
    return f"Locate several points {p}." + SUFFIX


def parse(txt):
    pts = [(float(x), float(y)) for x, y in re.findall(r"\(\s*([0-9]*\.?[0-9]+)\s*,\s*([0-9]*\.?[0-9]+)\s*\)", txt)]
    return [(x, y) for x, y in pts if 0 <= x <= 1 and 0 <= y <= 1]


def main():
    import torch
    from PIL import Image
    from robopoint.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
    from robopoint.conversation import conv_templates
    from robopoint.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
    from robopoint.model.builder import load_pretrained_model
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=0)     # smoke: the first N TRAINING units (no dev unit is seen before ps3b)
    ap.add_argument("--conv", default="llava_v1")
    a = ap.parse_args()
    tok, model, proc, _ = load_pretrained_model(MP, None, get_model_name_from_path(MP))
    model.eval()
    lab = json.load(open(os.path.join(ROOT, "results", "bind", "s1", "labels.json"), encoding="utf-8"))["train" if a.max else "dev"]
    if a.max:
        lab = lab[:a.max]
    os.makedirs(OD, exist_ok=True)
    keys, peaks, masks, metas = [], [], [], []
    t0 = time.time()
    for i, e in enumerate(lab):
        img = np.load(os.path.join(ROOT, e["frame"]), allow_pickle=True)["img"]
        im = Image.fromarray(img).convert("RGB")
        q = query(e["role"], e["phrase"])
        conv = conv_templates[a.conv].copy()
        conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + q)
        conv.append_message(conv.roles[1], None)
        ids = tokenizer_image_token(conv.get_prompt(), tok, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)
        x = process_images([im], proc, model.config).to(model.device, dtype=torch.float16)
        with torch.inference_mode():
            try:
                out = model.generate(ids, images=x, image_sizes=[im.size], do_sample=False, max_new_tokens=256, use_cache=True)
            except TypeError:
                out = model.generate(ids, images=x, do_sample=False, max_new_tokens=256, use_cache=True)
        txt = tok.batch_decode(out, skip_special_tokens=True)[0].strip()
        pts = parse(txt)
        msk = np.zeros(256, bool)
        pk = -1
        if pts:
            tk = tok_of(np.array([[x_ * img.shape[1], y_ * img.shape[0]] for x_, y_ in pts]))
            msk[tk] = True
            cnt = np.bincount(tk, minlength=256)
            best = cnt.max()
            pk = int(next(t for t in tk if cnt[t] == best))
        keys.append(f"{e['role']}|{e['frame']}")
        peaks.append(pk)
        masks.append(msk)
        metas.append(json.dumps(dict(query=q, answer=txt[:400], n_points=len(pts))))
        print(f"[s3b robopoint] {i + 1}/{len(lab)} {e['role']} {e['phrase']!r}: {len(pts)} pts, peak {pk} ({time.time() - t0:.0f}s) | {txt[:120]!r}",
              flush=True)
    np.savez(os.path.join(OD, "pred_dev.npz" if not a.max else "pred_dev_smoke.npz"), keys=np.array(keys), peak=np.array(peaks),
             mask=np.stack(masks), meta=np.array(metas))
    print(f"[s3b robopoint] done {len(keys)} units, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
