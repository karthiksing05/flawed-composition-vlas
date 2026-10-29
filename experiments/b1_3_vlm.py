"""PLAN_BINDER B1.3 (docs/PLAN_BINDER.md section 13): Qwen2.5-VL-7B (bbox) and Molmo-7B-D (point) zero-shot on the GENERATED test
units (results/bind/b1_3/units.json "test"), with B0's primary prompts, decoding and conversion (experiments/b0_vlm.py: the
"phrase" config, no instruction context).  Run with the isolated B0 venv:
    /coc/flash10/ksingara3/b0_ext/venv/bin/python -I experiments/b1_3_vlm.py --arm qwen|molmo [--max N]
-> results/bind/b1_3/vlm/<arm>.npz.  --max N: the first N generated TRAINING units only (a smoke; no test unit).
The read scores each arm twice: under LIBERO's (mirrored) left / right convention, i.e. the generated labels, and under the
literal image-side convention (left / right instance and compartment targets swapped to the image-side referent).
"""
import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path.insert(0, os.path.join(ROOT, "experiments"))
import b0_vlm as B0  # noqa: E402

B13 = os.path.join(ROOT, "results", "bind", "b1_3")


def main():
    import torch
    from PIL import Image
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["qwen", "molmo"])
    ap.add_argument("--max", type=int, default=0)
    a = ap.parse_args()
    assert torch.cuda.is_available()
    U = json.load(open(os.path.join(B13, "units.json"), encoding="utf-8"))
    if a.max:
        units = U["train"][:a.max]
    else:
        sys.path.insert(0, ROOT)
        rows = []
        for f in sorted(os.listdir(os.path.join(ROOT, "ledger", "m3_prereg"))):
            if "pb1_3" in f:
                rows.append(f)
        assert rows, "pre-register first (experiments/prereg/pb1_3.py)"
        units = U["test"]
    cfg = "bbox_phrase" if a.arm == "qwen" else "point_phrase"
    t0 = time.time()
    model, proc, info = B0.build(a.arm, torch.device("cuda"))
    keys, peaks, masks, metas = [], [], [], []
    for i, e in enumerate(units):
        im = Image.fromarray(np.load(os.path.join(ROOT, e["frame"]), allow_pickle=True)["img"]).convert("RGB")
        q = B0.prompt(a.arm, cfg, e["phrase"], "")
        pk, msk, meta = B0.run_one(a.arm, cfg, model, proc, im, q, None)
        keys.append(f"{e['role']}|{e['frame']}")
        peaks.append(pk)
        masks.append(msk)
        metas.append(json.dumps(meta))
        if i < 6 or i % 50 == 0:
            print(f"[b1.3 vlm] {a.arm} {i + 1}/{len(units)} {e['phrase']!r}: peak {pk} | {meta.get('answer', '')[:120]!r} ({time.time() - t0:.0f}s)", flush=True)
    od = os.path.join(B13, "vlm", "smoke" if a.max else "")
    os.makedirs(od, exist_ok=True)
    np.savez(os.path.join(od, f"{a.arm}.npz"), keys=np.array(keys), peak=np.array(peaks), mask=np.stack(masks), meta=np.array(metas),
             info=np.array(json.dumps(info)))
    print(f"[b1.3 vlm] {a.arm} done: {len(keys)} units, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
