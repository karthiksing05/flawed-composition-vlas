"""FLAWED_COMPOSITION D1x (section 4.7): the interchange test at the EXPERT-READ level for a BindVLA checkpoint (main2, C1).  No
training.  SmolVLA venv, one GPU.

For a compound C with demos, either an in-support training task (its held-out episodes, the D1 IN frames) or a dev task (the D1
DEV frames; scored only, never trained on, never executed), choose training compounds A and B (the 72 in-support tasks):
  A: another task whose step-0 object phrase equals C's and whose goal phrase differs
  B: another task whose step-0 goal phrase equals C's and whose object phrase differs
  (same BDDL scene first, then any; ties by a fixed hash).
On each of C's frames (its own images and state):
  true   the prefix of C's own instruction
  A, B   the prefixes of A's and of B's instructions
  rec    B's prefix key / value states at every layer, with the step-0 OBJECT slot position replaced by A's: the expert reads A's
         object message and B's goal, verb, global and state messages (joint input never formed by any one instruction)
Every sample is the model's 10-step Euler sample from the D1 noise draws (fc_d1_debt.hseed); action error = D1's (first 10 steps,
7 dims, MEAN_STD units).  Agreement (interchange-intervention accuracy): rec is nearer (output MSE) to true than to A and to B.
-> results/flawed/d1x_<name>.json

    python experiments/fc_d1x.py --ckpt results/bindvla/main2/ckpt/step_060000.pt --name main2 [--limit N --out DIR]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "experiments")]
OUT = os.path.join(ROOT, "results", "flawed")
C1_HELD = ("libero_90:37", "libero_object:9", "libero_spatial:8", "libero_spatial:9")
OBJ_POS = 1                                   # prefix position of the step-0 object slot (toks = [verb0, obj0, goal0, verb1, ...])


def splice(kvA, kvB, pos):
    out = {}
    for l in kvB:
        k, v = kvB[l]["key_states"].clone(), kvB[l]["value_states"].clone()
        k[:, pos], v[:, pos] = kvA[l]["key_states"][:, pos], kvA[l]["value_states"][:, pos]
        out[l] = {"key_states": k, "value_states": v}
    return out


@torch.no_grad()
def sample_kv(M, kv, pad, noises):
    outs = []
    for nz in noises:
        x = nz.clone()
        dt = -1.0 / 10
        for s in range(10):
            t = torch.full((x.shape[0],), 1.0 + s * dt, device="cuda")
            x = x + dt * M.bv.flow.denoise_step(prefix_pad_masks=pad, past_key_values=kv, x_t=x, timestep=t)
        outs.append(x[:, :, :7].float())
    return torch.stack(outs, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    t0 = time.time()
    import bv_train as BT
    import fc_d1_debt as D
    from lifelong_ood import l90c
    from lifelong_ood.policy import factors as F
    import c_split as CS
    tasks = {t.tid: t for t in F.load(CS.SUITES)}
    store = BT.Store()
    M = D.Model(a.ckpt)
    support = [t for t in l90c.load_split()["train"] if t not in C1_HELD]
    text_of = {}
    for e in store.eps.values():
        text_of.setdefault(e["tid"], e["text"])
    info = []
    for t in support:
        o, g = D.step0_phrases(text_of[t])
        info.append(dict(tid=t, text=text_of[t], scene=tasks[t].scene, o=o, g=g))

    def choose(tid, text, scene):
        o, g = D.step0_phrases(text)
        if not o or not g:
            return None
        key = lambda r: (r["scene"] != scene, D.hseed("d1x", tid, r["tid"]))      # noqa: E731
        As = sorted([r for r in info if r["tid"] != tid and r["o"] == o and r["g"] and r["g"] != g], key=key)
        Bs = sorted([r for r in info if r["tid"] != tid and r["g"] == g and r["o"] and r["o"] != o], key=key)
        return (As[0], Bs[0]) if As and Bs else None

    frames = []
    for r in D.in_frames(store):
        if r["tid"] in support:
            p = choose(r["tid"], r["text"], r["scene"])
            if p:
                it = store.item(r["i"])
                j = np.arange(r["i"], r["i"] + 50)
                frames.append(dict(set="IN", tid=r["tid"], ep=r["ep"], frame=r["frame"], phase=r["phase"], text=r["text"], A=p[0], B=p[1],
                                   img=it["img"], wrist=it["wrist"], state=store.state[r["i"]],
                                   acts=store.actions[np.minimum(j, store.end[r["i"]] - 1)], pad=j >= store.end[r["i"]]))
    z0 = np.load(os.path.join(D.CACHE, "d1_dev.npz"))
    z = {k: z0[k] for k in z0.files}
    lang = json.load(open(os.path.join(OUT, "d1_frames.json"), encoding="utf-8"))["per_task"]
    for j in range(len(z["tid"])):
        tid = str(z["tid"][j])
        l90c.check_task(tid)
        p = choose(tid, lang[tid]["language"], tasks[tid].scene)
        if p:
            frames.append(dict(set="DEV", tid=tid, ep=int(z["ep"][j]), frame=int(z["frame"][j]), phase=str(z["phase"][j]),
                               text=lang[tid]["language"], A=p[0], B=p[1], img=z["agent"][j], wrist=z["wrist"][j], state=z["state"][j],
                               acts=z["actions"][j], pad=z["pad"][j]))
    if a.limit:
        frames = [f for f in frames if f["set"] == "IN"][: a.limit] + [f for f in frames if f["set"] == "DEV"][: a.limit]
    rec = []
    BS = 16
    for b in range(0, len(frames), BS):
        fs = frames[b:b + BS]
        with torch.no_grad():
            V = M.vision([f["img"] for f in fs], [f["wrist"] for f in fs])
            st = M.state([f["state"] for f in fs])
            tgt = (torch.from_numpy(np.stack([f["acts"] for f in fs])).cuda() - M.a_mu) / (M.a_sd + 1e-8)
            pad = torch.from_numpy(np.stack([f["pad"] for f in fs])).cuda()
            nz = []
            for d in range(D.DRAWS):
                nz.append(torch.stack([torch.from_numpy(np.random.default_rng(D.hseed("d1-noise", f["tid"], f["ep"], f["frame"], d)).standard_normal(
                    (M.cfg.chunk, M.cfg.max_action_dim)).astype(np.float32)) for f in fs]).cuda())
            P = {}
            for k, tx in (("true", [f["text"] for f in fs]), ("A", [f["A"]["text"] for f in fs]), ("B", [f["B"]["text"] for f in fs])):
                embs, pp = M.prefix(M.roles(V, tx), V, st)
                P[k] = (M.kv(embs, pp), pp)
            out = {k: sample_kv(M, kv, pp, nz) for k, (kv, pp) in P.items()}
            out["rec"] = sample_kv(M, splice(P["A"][0], P["B"][0], OBJ_POS), P["B"][1], nz)
            m = (~pad[:, :D.HE]).float()

            def err(p):
                d = ((p[:, :, :D.HE] - tgt[None, :, :D.HE]) ** 2).mean(-1)
                return ((d * m).sum(-1) / m.sum(-1).clamp(min=1)).mean(0)

            def dist(p, q):
                return ((p[:, :, :D.HE] - q[:, :, :D.HE]) ** 2).mean((-1, -2)).mean(0)
            E = {k: err(v) for k, v in out.items()}
            dT, dA, dB = dist(out["rec"], out["true"]), dist(out["rec"], out["A"]), dist(out["rec"], out["B"])
        for j, f in enumerate(fs):
            rec.append(dict(set=f["set"], tid=f["tid"], ep=f["ep"], frame=f["frame"], phase=f["phase"], A=f["A"]["tid"], B=f["B"]["tid"],
                            err={k: float(v[j]) for k, v in E.items()}, d_rec_true=float(dT[j]), d_rec_A=float(dA[j]), d_rec_B=float(dB[j]),
                            agree=bool(dT[j] < dA[j] and dT[j] < dB[j])))
        print(f"[d1x] {b + len(fs)}/{len(frames)} {time.time() - t0:.0f}s", flush=True)
    os.makedirs(a.out, exist_ok=True)
    fn = os.path.join(a.out, f"d1x_{a.name}{'_smoke' if a.limit else ''}.json")
    with open(fn, "w", encoding="utf-8") as fh:
        json.dump(dict(name=a.name, ckpt=a.ckpt, records=rec, n_in=sum(r["set"] == "IN" for r in rec), n_dev=sum(r["set"] == "DEV" for r in rec),
                       pairs={r["tid"]: [r["A"], r["B"]] for r in rec}, secs=time.time() - t0), fh)
    print(f"[d1x] wrote {fn} ({len(rec)} records)")


if __name__ == "__main__":
    main()
