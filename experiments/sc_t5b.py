"""T5b (docs/SCAFFOLD_RESULTS.md 14; m3_prereg sc_t5b): a real sequential retry in every scene.  LIBERO venv, CPU, no training, the same 300 steps.
U's policy unchanged.  If nothing is held for T steps, T = the 99th percentile of the demos' first held step under the library's own event, the attempt ends: the gripper opens, the hand rises straight up to its start height, returns to its start pose, and the policy runs again on fresh noise.
Reads proprioception only: no object position, no entity, no scene constant.

    python experiments/sc_t5b.py roll --index 0
    python experiments/sc_t5b.py read       -> results/flawed/sc_t5b_read.json
SC_SMOKE=1: 2 smoke states and 4 combos, results under sc_t5b_smoke.
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
OUT = os.path.join(ROOT, "results", "flawed")
SMOKE = bool(os.environ.get("SC_SMOKE"))
OD = os.path.join(OUT, "sc_t5b_smoke" if SMOKE else "sc_t5b")
SEEDS = (10, 11, 12)
COMPS = tuple(f"K{i}" for i in range(1, 7)) + tuple(f"N{i}" for i in range(1, 7))
OBJECTS = dict(bowl=("K1", "K2", "K3"), cheese=("K6", "N1", "N2", "N3"), bottle=("K4", "K5", "N4", "N5", "N6"))
REPLAN, RET_MAX, NEAR, B, MARGIN, GAIN = 5, 60, 0.02, 2000, 0.10, 0.10


def tasks():
    return json.load(open(os.path.join(OUT, "n2_tasks.json"), encoding="utf-8"))


def task(n2, c, l):
    return next(t for t in n2["tasks"] if t["compound"] == c and t["layout"] == l)


def combos():
    n2 = tasks()
    if SMOKE:
        return [(10, task(n2, c, l)["tid"]) for c, l in (("K1", "new"), ("K4", "new"), ("K6", "new"), ("K4", "familiar"))]
    return [(s, task(n2, c, l)["tid"]) for s, c, l in itertools.product(SEEDS, COMPS, ("familiar", "new"))]


def timeout_of(m):
    """T from the demos alone: the 99th percentile over known demos of the step of the library event's first row."""
    import fa1_common as FC
    from lifelong_ood.policy import parts_flow as PF
    D = np.load(os.path.join(FC.DATA, "demos.npz"))
    key = D["task"].astype(np.int64) * 1000 + D["demo"]
    h = np.asarray(PF.pgate_rows(key, D["grip"], D["prev_grip"], D["ee_pos"][:, 2], m.pgate["tw"], m.pgate["lift"])) > 0.5
    first = [int(D["step"][key == k][h[key == k]].min()) for k in np.unique(key) if h[key == k].any()]
    return int(np.percentile(first, 99))


def rotvec(R):
    c = np.clip((np.trace(R) - 1) / 2, -1, 1)
    th = float(np.arccos(c))
    if th < 1e-6:
        return np.zeros(3)
    return th / (2 * np.sin(th)) * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])


class Retry:
    def __init__(self, m, T):
        self._m, self.T = m, T
        self.ee0, self.R0 = None, None
        self.since, self.ret, self.ret_steps, self.attempts = 0, False, 0, 1
        self.starts = [0]
        self.clock = 0

    def __getattr__(self, k):
        return getattr(self._m, k)

    def _servo(self, ee, R):
        import torch
        wp = np.array([ee[0], ee[1], self.ee0[2]]) if ee[2] < self.ee0[2] - NEAR else self.ee0     # up first, then across
        a = np.zeros(7)
        a[:3] = np.clip((wp - ee) / REPLAN / 0.02, -1, 1)
        a[3:6] = np.clip(rotvec(self.R0 @ R.T) / REPLAN / 0.2, -1, 1)
        a[6] = -1.0
        return torch.tensor(np.tile(a, (1, self._m.H, 1)), dtype=torch.float32)

    def sample(self, b, **kw):
        ee, R = b["ee_pos"][0].numpy().astype(np.float64), b["ee_rot"][0].numpy().astype(np.float64)
        if self.ee0 is None:
            self.ee0, self.R0 = ee.copy(), R.copy()
        self.clock += REPLAN
        if self.ret:
            self.ret_steps += REPLAN
            if np.linalg.norm(ee - self.ee0) >= NEAR and self.ret_steps < RET_MAX:
                return self._servo(ee, R)
            self.ret, self.since = False, 0
            self.attempts += 1
            self.starts.append(self.clock)
        if float(b["held"][0]) > 0.5:
            self.since = 0
        else:
            self.since += REPLAN
            if self.since > self.T:
                self.ret, self.ret_steps = True, 0
                return self._servo(ee, R)
        return self._m.sample(b, **kw)


def roll(seed, tid):
    import torch
    import fa1_common as FC
    from fa1h_roll import episode
    from pp_p_roll import load_any
    fn = os.path.join(OD, f"Ur_s{seed}_{tid}.json")
    if os.path.exists(fn):
        print("exists", fn, flush=True)
        return
    torch.set_num_threads(1)
    n2 = tasks()
    t = n2["tasks"][tid]
    assert t["tid"] == tid
    m, _ = load_any(os.path.join(FC.DATA, "ckpt", f"pp_sc_U_s{seed}", "P.npz"))
    T = timeout_of(m)
    print("[t5b] T", T, flush=True)
    S = np.load(os.path.join(OUT, "n2_states", f"n2_goal_{tid}.npy"))
    br = json.load(open(os.path.join(OUT, "n2_branches_smoke.json" if SMOKE else "n2_branches_main.json"), encoding="utf-8"))
    if SMOKE:
        br = br[:2]
    eps = []
    for b in br:
        pol = Retry(m, T)
        e = episode(pol, os.path.join(ROOT, t["bddl"]), S[int(b["trial"])], t["language"], b["seed_base"])
        eps.append(dict(trial=int(b["trial"]), success=bool(e["success"]), error=e["error"], held_step=e["held_step"], held_segments=e["held_segments"], steps=e["steps"], bound=e["bound"], attempts=pol.attempts, starts=pol.starts, T=T))
        print("[t5b]", seed, t["compound"], t["layout"], b["trial"], json.dumps(eps[-1]), flush=True)
    os.makedirs(OD, exist_ok=True)
    with open(fn + ".tmp", "w", encoding="utf-8") as fh:
        json.dump(dict(arm="Ur", seed=seed, compound=t["compound"], layout=t["layout"], tid=tid, host=os.uname().nodename, eps=eps), fh, indent=1)
    os.replace(fn + ".tmp", fn)


def read():
    n2 = tasks()
    ks = n2["states"]["main"]

    def eps(arm, s, c, l):
        tid = task(n2, c, l)["tid"]
        p = os.path.join(OUT, "sc_t4", f"U_s{s}_{tid}.json") if arm == "U" else os.path.join(OD, f"Ur_s{s}_{tid}.json")
        E = json.load(open(p, encoding="utf-8"))["eps"]
        for e in E:
            assert e.get("error") is None, (p, e.get("error"))
        d = {int(e["trial"]): e for e in E}
        return [d[k] for k in ks]

    def arr(arm, comps, l):
        return np.array([[[float(e["success"]) for e in eps(arm, s, c, l)] for s in SEEDS] for c in comps])

    def diff(a, b):
        rng = np.random.default_rng(0)
        v = []
        for _ in range(B):
            t = rng.choice(a.shape[0], a.shape[0])
            s = rng.choice(a.shape[2], a.shape[2])
            v.append(float((a[t][:, :, s] - b[t][:, :, s]).mean()))
        return dict(a=float(a.mean()), b=float(b.mean()), delta=float((a - b).mean()), ci=[float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))])

    G = dict(known=tuple(c for c in COMPS if c[0] == "K"), new_pairings=tuple(c for c in COMPS if c[0] == "N"))
    res = dict(V={f"{g}|{l}": diff(arr("Ur", cs, l), arr("U", cs, l)) for g, cs in G.items() for l in ("familiar", "new")})
    res["objects"] = {f"{n}|{l}": diff(arr("Ur", cs, l), arr("U", cs, l)) for n, cs in OBJECTS.items() for l in ("familiar", "new")}
    fam, new = [res["V"][f"{g}|familiar"] for g in G], [res["V"][f"{g}|new"] for g in G]
    res["label"] = "HARMS" if any(x["ci"][1] < -MARGIN for x in fam) else ("PAYS" if all(x["ci"][0] >= -MARGIN for x in fam) and any(x["delta"] >= GAIN and x["ci"][0] > 0.0 for x in new) else "NOT YET")
    res["per_task"] = {a: {c: {l: float(arr(a, (c,), l).mean()) for l in ("familiar", "new")} for c in COMPS} for a in ("U", "Ur")}
    res["attempts"] = {f"{n}|{l}": {str(k): float(np.mean([e["attempts"] == k for s in SEEDS for c in cs for e in eps("Ur", s, c, l)])) for k in (1, 2, 3, 4)} for n, cs in OBJECTS.items() for l in ("familiar", "new")}
    res["success_by_attempt"] = {f"{n}|{l}": {str(k): float(np.mean([e["success"] for s in SEEDS for c in cs for e in eps("Ur", s, c, l) if e["attempts"] == k] or [np.nan])) for k in (1, 2, 3, 4)} for n, cs in OBJECTS.items() for l in ("familiar", "new")}
    res["T"] = sorted({e["T"] for s in SEEDS for e in eps("Ur", s, "K1", "new")})
    print("[t5b]", json.dumps(res), flush=True)
    with open(os.path.join(OUT, "sc_t5b_read.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("roll", "read", "count"))
    ap.add_argument("--index", type=int)
    a = ap.parse_args()
    if a.what == "count":
        print(len(combos()))
    elif a.what == "roll":
        s_, tid_ = combos()[a.index]
        roll(s_, tid_)
    else:
        read()
