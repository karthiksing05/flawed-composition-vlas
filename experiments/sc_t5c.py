"""T5c (docs/SCAFFOLD_RESULTS.md 15; m3_prereg sc_t5c): T5b's rule with room to work.  LIBERO venv, CPU, no training.  Both arms get 600 steps, new layout only.
Ur600: U with T5b's retry.  U600: U alone, so late holds count for it too.

    python experiments/sc_t5c.py roll --index 0
    python experiments/sc_t5c.py read       -> results/flawed/sc_t5c_read.json
SC_SMOKE=1: 2 smoke states and 2 combos, results under sc_t5c_smoke.
"""
import argparse
import itertools
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT]
import sc_t5b as TB                                                         # noqa: E402

OUT, COMPS, SEEDS, OBJECTS, task, tasks = TB.OUT, TB.COMPS, TB.SEEDS, TB.OBJECTS, TB.task, TB.tasks
SMOKE = bool(os.environ.get("SC_SMOKE"))
OD = os.path.join(OUT, "sc_t5c_smoke" if SMOKE else "sc_t5c")
ARMS = ("Ur600", "U600")
STEPS, B, GAIN = 600, 2000, 0.10


def combos():
    n2 = tasks()
    if SMOKE:
        return [(a, 10, task(n2, "K6", "new")["tid"]) for a in ARMS]
    return [(a, s, task(n2, c, "new")["tid"]) for a, s, c in itertools.product(ARMS, SEEDS, COMPS)]


def roll(arm, seed, tid):
    import torch
    import fa1_common as FC
    import fa1h_roll
    from pp_p_roll import load_any
    fa1h_roll.MAX_STEPS = STEPS
    fn = os.path.join(OD, f"{arm}_s{seed}_{tid}.json")
    if os.path.exists(fn):
        print("exists", fn, flush=True)
        return
    torch.set_num_threads(1)
    n2 = tasks()
    t = n2["tasks"][tid]
    assert t["tid"] == tid
    m, _ = load_any(os.path.join(FC.DATA, "ckpt", f"pp_sc_U_s{seed}", "P.npz"))
    T = TB.timeout_of(m)
    S = np.load(os.path.join(OUT, "n2_states", f"n2_goal_{tid}.npy"))
    br = json.load(open(os.path.join(OUT, "n2_branches_smoke.json" if SMOKE else "n2_branches_main.json"), encoding="utf-8"))
    if SMOKE:
        br = br[:2]
    eps = []
    for b in br:
        pol = TB.Retry(m, T) if arm == "Ur600" else m
        e = fa1h_roll.episode(pol, os.path.join(ROOT, t["bddl"]), S[int(b["trial"])], t["language"], b["seed_base"])
        eps.append(dict(trial=int(b["trial"]), success=bool(e["success"]), error=e["error"], held_step=e["held_step"], held_segments=e["held_segments"], steps=e["steps"], bound=e["bound"],
                        attempts=getattr(pol, "attempts", 1) if arm == "Ur600" else 1, starts=pol.starts if arm == "Ur600" else [0], T=T))
        print("[t5c]", arm, seed, t["compound"], t["layout"], b["trial"], json.dumps(eps[-1]), flush=True)
    os.makedirs(OD, exist_ok=True)
    with open(fn + ".tmp", "w", encoding="utf-8") as fh:
        json.dump(dict(arm=arm, seed=seed, compound=t["compound"], layout=t["layout"], tid=tid, host=os.uname().nodename, eps=eps), fh, indent=1)
    os.replace(fn + ".tmp", fn)


def read():
    n2 = tasks()
    ks = n2["states"]["main"]

    def eps(arm, s, c):
        p = os.path.join(OD, f"{arm}_s{s}_{task(n2, c, 'new')['tid']}.json")
        E = json.load(open(p, encoding="utf-8"))["eps"]
        for e in E:
            assert e.get("error") is None, (p, e.get("error"))
        d = {int(e["trial"]): e for e in E}
        return [d[k] for k in ks]

    def arr(arm, comps):
        return np.array([[[float(e["success"]) for e in eps(arm, s, c)] for s in SEEDS] for c in comps])

    def diff(a, b):
        rng = np.random.default_rng(0)
        v = []
        for _ in range(B):
            t = rng.choice(a.shape[0], a.shape[0])
            s = rng.choice(a.shape[2], a.shape[2])
            v.append(float((a[t][:, :, s] - b[t][:, :, s]).mean()))
        return dict(a=float(a.mean()), b=float(b.mean()), delta=float((a - b).mean()), ci=[float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))])

    G = dict(known=tuple(c for c in COMPS if c[0] == "K"), new_pairings=tuple(c for c in COMPS if c[0] == "N"))
    res = dict(V={g: diff(arr("Ur600", cs), arr("U600", cs)) for g, cs in G.items()}, objects={n: diff(arr("Ur600", cs), arr("U600", cs)) for n, cs in OBJECTS.items()})
    res["label"] = "RETRY PAYS" if any(x["delta"] >= GAIN and x["ci"][0] > 0.0 for x in res["V"].values()) else "RETRY DOES NOT PAY"
    res["per_task"] = {a: {c: float(arr(a, (c,)).mean()) for c in COMPS} for a in ARMS}
    A = {}
    for n, cs in OBJECTS.items():
        E = [e for s in SEEDS for c in cs for e in eps("Ur600", s, c)]
        A[n] = {}
        for k in (1, 2, 3, 4, 5):
            Ek = [e for e in E if e["attempts"] == k]
            A[n][str(k)] = dict(share=len(Ek) / len(E), success=float(np.mean([e["success"] for e in Ek])) if Ek else None,
                                hold_after_retry=float(np.mean([e["held_step"] is not None and e["held_step"] >= e["starts"][1] - 10 for e in Ek])) if Ek and k > 1 else None)
        U = [e for s in SEEDS for c in cs for e in eps("U600", s, c)]
        A[n]["U600_hold_by_T"] = float(np.mean([e["held_step"] is not None and e["held_step"] <= e["T"] for e in U]))
        A[n]["U600_hold_late"] = float(np.mean([e["held_step"] is not None and e["held_step"] > e["T"] for e in U]))
        A[n]["U600_success_late"] = float(np.mean([e["success"] and e["held_step"] is not None and e["held_step"] > e["T"] for e in U]))
    res["attempts"] = A
    print("[t5c]", json.dumps(res), flush=True)
    with open(os.path.join(OUT, "sc_t5c_read.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("roll", "read", "count"))
    ap.add_argument("--index", type=int)
    a = ap.parse_args()
    if a.what == "count":
        print(len(combos()))
    elif a.what == "roll":
        arm_, s_, tid_ = combos()[a.index]
        roll(arm_, s_, tid_)
    else:
        read()
