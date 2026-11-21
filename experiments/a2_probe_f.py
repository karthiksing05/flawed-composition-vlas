"""RAA-F probe (docs/notes/raaf_plan.md 6.1; docs/notes/raaf_probe_runbook.md): a2_probe.py's method on the servo-free state-token models.  GPU, openpi venv.

a2_probe.py is imported UNCHANGED (Patcher, the row machinery, the observation selection, the registered rows).  This file only
  * extends the model registry (--model: pi05, raasl, A = pi05_g8v_raac_lf_ae, A0 = pi05_g8v_raac_l_ae, B = pi05_g8v_raac_lb_ae, Aplus = --config / --ckpt; any label with --config);
  * loads the recorded `crop` (a2f_capture.py) as observation/crop_image;
  * adds the input-patch groups.  Each row patches a part of the model INPUT from the counterfactual run into the clean run (the servo is not an input of these models: servo_k = 0, no in.servo / in.chan rows):
      in.prefix         images + prompt from the counterfactual run, state token clean (a2_probe's definition; for B this includes the crop image)
      in.tok            the 4-dim state token from the counterfactual run, everything else clean
      in.crop (B)       the crop image (the third image block of the prefix, positions 512-767) from the counterfactual run, everything else clean
      in.tokcrop (B)    in.tok and in.crop patched together: the learned where-input of Route B (the plan's `in.tok + in.crop`)
      in.prefix_nocrop (B)  in.prefix with the crop image kept clean (descriptive: the visual share without the where-crop)
    For raasl the original in.prefix / in.tok / in.servo / in.chan rows are the same as a2_probe.py's.
  * --rows inputs: only `total` and the in.* rows (1-2 batches per observation instead of 13).  Used for the SR 83 reproduction check; the selection run uses --rows full
    (the 237 registered rows, so that top1 / PRn of the 72 writes and the token-ablation drop exist).

    python experiments/a2_probe_f.py --model A|B|A0|Aplus|raasl|pi05 --cap DIR[+DIR..] --out DIR --sets N+K [--rows full|inputs] [--limit N] [--config NAME --ckpt DIR] [--ident]
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT, HERE]
import a2_probe as AP                                                      # noqa: E402

PT = "/coc/flash10/ksingara3/robot/checkpoints_l90c/pytorch"
TAG = "main6000"
CROPK = "observation/crop_image"
ROUTES = {                                                                 # label -> (config, checkpoint dir, has a crop image)
    "pi05": ("pi05_libero", None, False),
    "raasl": ("pi05_g8v_raas_l_ae", AP.CK, False),
    "A": ("pi05_g8v_raac_lf_ae", "%s/pi05_g8v_raac_lf_ae_%s" % (PT, TAG), False),
    "A0": ("pi05_g8v_raac_l_ae", "%s/pi05_g8v_raac_l_ae_%s" % (PT, TAG), False),          # conditional route (config added when A0 is trained)
    "B": ("pi05_g8v_raac_lb_ae", "%s/pi05_g8v_raac_lb_ae_%s" % (PT, TAG), True),
    "Aplus": (None, None, False),                                            # hook: pass --config (the A+ config name) and --ckpt
}


def load_npz_f(path):
    z = np.load(path, allow_pickle=False)
    d = {"observation/image": z["image"], "observation/wrist_image": z["wrist"], "observation/state": z["state"], "prompt": str(z["prompt"])}
    if "crop" in z.files:
        d[CROPK] = z["crop"]
    return d, int(z["seed"]), z["actions"]


def build_rows_f(label, clean, cf, ident, rows, servo, crop):
    """-> [(name, row)].  rows = 'full': a2_probe.build_rows's registered rows for the writes / bands plus the in.* rows below; 'inputs': total + in.*."""
    if rows == "full":
        R = [(n, r) for n, r in AP.build_rows("pi05", clean, cf, ident)]
    else:
        R = [("total", dict(pre=cf, tok=cf, srv=cf, comps=[]))]

    def add(name, pre=clean, tok=clean, srv=clean):
        R.append((name, dict(pre=pre, tok=tok, srv=srv, comps=[])))
    add("in.prefix", pre=cf)
    add("in.tok", tok=cf)
    if servo:
        add("in.servo", srv=cf)
        add("in.chan", tok=cf, srv=cf)
    if crop:
        pc = dict(clean)
        pc[CROPK] = cf[CROPK]
        pn = dict(cf)
        pn[CROPK] = clean[CROPK]
        add("in.crop", pre=pc)
        add("in.tokcrop", pre=pc, tok=cf)
        add("in.prefix_nocrop", pre=pn)
    if rows == "full":
        z = dict(clean)
        z["observation/state"] = np.zeros_like(np.asarray(clean["observation/state"]))
        add("abl.tok", tok=z)
        if servo:
            add("abl.servo", srv=z)
            add("abl.chan", tok=z, srv=z)
    return R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="a registered label (%s) or any label with --config" % ",".join(ROUTES))
    ap.add_argument("--config", default="")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--crop", type=int, default=-1, help="1 / 0 forces the crop groups on / off (default: the registry; Aplus: on if the capture has crops)")
    ap.add_argument("--cap", required=True, help="capture dirs (+-separated) holding <label>_nk_<tid>_cap.jsonl / obs/ (a2f_capture.sh names them by the label)")
    ap.add_argument("--geom", default="", help="pi05 F7 capture dirs (the eef of row 0; only with --sets F7)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sets", default="N+K")
    ap.add_argument("--rows", default="full", choices=("full", "inputs"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=20)
    ap.add_argument("--ident", action="store_true")
    a = ap.parse_args()
    cfg, ck, crop = ROUTES.get(a.model, (None, None, False))
    cfg, ck = a.config or cfg, a.ckpt or ck
    assert cfg, "unknown label %s: pass --config" % a.model
    if a.crop >= 0:
        crop = bool(a.crop)
    os.makedirs(a.out, exist_ok=True)
    AP.load_npz = load_npz_f
    from lifelong_ood.policy.backbone import PolicyBackbone
    bk = PolicyBackbone(cfg, checkpoint_dir=ck, device="cuda") if ck else PolicyBackbone(cfg, device="cuda")
    print("[a2f] %s label %s crop %s" % (bk.describe(), a.model, crop), flush=True)
    P = AP.Patcher(bk)
    state_token = a.model != "pi05"
    caps = [c for c in a.cap.split("+") if c]
    geom = [c for c in a.geom.split("+") if c]
    todo, skipped = [], {}
    for s in a.sets.split("+"):
        o, sk = AP.obs_f7(P, caps, a.model, geom or caps) if s == "F7" else AP.obs_nk(P, caps, a.model, s)
        todo += o
        skipped[s] = sk
        print("[a2f] set %s: %d observations, %d skipped %s" % (s, len(o), len(sk), sk[:6]), flush=True)
    if a.limit:
        todo = todo[:a.limit]
    if crop:
        for ob in todo:
            for k in ("clean", "cf"):
                assert CROPK in ob[k], "no crop in the capture of %s (%s): was it taken with a2f_capture.py + raa_b.py?" % (ob["id"], k)
    json.dump(dict(skipped={k: [list(map(str, x)) for x in v] for k, v in skipped.items()}, n=len(todo)), open(os.path.join(a.out, "selection_%s.json" % a.sets), "w"))
    for oi, ob in enumerate(todo):
        path = os.path.join(a.out, ob["id"] + ".npz")
        if os.path.exists(path):
            continue
        t0 = time.time()
        P.cf, P.cl = {}, {}
        rec = {}
        P.run([dict(pre=ob["cf"], tok=ob["cf"], srv=ob["cf"], comps=[])], ob["seed"], rec=rec)
        P.cf = rec
        if a.ident:
            rec2 = {}
            P.run([dict(pre=ob["clean"], tok=ob["clean"], srv=ob["clean"], comps=[])], ob["seed"], rec=rec2)
            P.cl = rec2
        if state_token:
            R = build_rows_f(a.model, ob["clean"], ob["cf"], a.ident, a.rows, P.servo, crop)
        else:
            R = [(n, r) for n, r in AP.build_rows("pi05", ob["clean"], ob["cf"], a.ident)]
        names, chunks, ref = [], [], []
        per = a.batch - 1
        for i in range(0, len(R), per):
            part = R[i:i + per]
            rows = [dict(pre=ob["clean"], tok=ob["clean"], srv=ob["clean"], comps=[])] + [r for _, r in part]
            ch = P.run(rows, ob["seed"])
            r0 = len(names)
            names += ["ref"] + [n for n, _ in part]
            chunks += list(ch)
            ref += [r0] * len(rows)
        np.savez(path, names=np.array(names), chunks=np.stack(chunks).astype(np.float32), ref=np.array(ref), cap_actions=ob["cap_actions"],
                 meta=np.array(json.dumps({k: ob[k] for k in ("id", "set", "task", "trial", "cluster", "seed", "eef", "p_cor", "p_cf", "cand", "phase")})))
        c0 = np.stack(chunks)[0]
        print("[a2f] %d/%d %s rows %d %.0fs replay-vs-server max|diff| %.4g" % (oi + 1, len(todo), ob["id"], len(names), time.time() - t0,
                                                                              float(np.abs(c0[:, :7] - ob["cap_actions"][:, :7]).max())), flush=True)
    print("[a2f] done", flush=True)


if __name__ == "__main__":
    main()
