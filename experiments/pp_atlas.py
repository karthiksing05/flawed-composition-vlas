"""The break atlas (docs/BREAK_ATLAS.md; m3_prereg pp_atlas).  LIBERO venv, CPU.  Results in results/flawed/pp_atlas/.
  selftest                      the stage rule on synthetic episodes
  pmin --seed 10 --tid 16       P_min's episodes, tracked   -> pmin_s<seed>_<tid>.json, _track.npz
  servo --tid 16                the v2 servo (planner seed 10), tracked -> servo_s10_<tid>.json, _track.npz
  read                          -> results/flawed/pp_atlas_read.json
The VLAs run through experiments/pp_atlas_client.py and pp_atlas_roll.sh -> <model>_<tid>.json, _track.npz.
"""
import argparse
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT]
OUT = os.path.join(ROOT, "results", "flawed")
OD = os.path.join(OUT, "pp_atlas")
WAIT, ARRIVE, SEAT, HALF = 10, 0.10, 0.03, 0.5
STAGES = ("success", "GRASP", "HANDOFF", "TRANSPORT", "RELEASE", "PHYSICS")
GROUP = dict(within=("GRASP", "RELEASE"), cross=("HANDOFF", "TRANSPORT"))
NEW = ("N1", "N2", "N3", "N4", "N5", "N6")
PAIR = dict(N1=("K6", "K1"), N2=("K6", "K2"), N3=("K6", "K3"), N4=("K4", "K1"), N5=("K4", "K2"), N6=("K4", "K6"))
KNOWN = ("K1", "K2", "K3", "K4", "K6")
B = 2000


class Rec:
    """Wraps an env: reads the simulator after each step from the WAIT-th step on, with its own Held.  Nothing a policy sees changes."""

    def __init__(self, env, oe, te):
        self._e, self._oe, self._te, self.n = env, oe, te, 0
        self._con, self._dj, self._hm, self._last = None, None, None, None
        self.rows = dict(held=[], obj=[], tgt=[], ee=[], quat=[], a6=[])

    def __getattr__(self, k):
        return getattr(self._e, k)

    def step(self, a):
        import fa1_common as FC
        import fa1b_common as FB
        r = self._e.step(a)
        self.n += 1
        if self.n < WAIT:
            return r
        if self._con is None:
            self._con, self._dj, self._hm = FB.Contact(self._e, [self._oe]), FC.drawer_joint(self._e), FB.Held()
        h, _ = self._hm.step(*self._con.touching(self._oe), self._con.pos(self._oe), r[0]["robot0_eef_pos"])
        pos = FC.scene(self._e, r[0], self._dj)["pos"]
        E = self._e.env
        R = self.rows
        R["held"].append(bool(h))
        R["obj"].append(pos[FC.ENT.index(self._oe)].copy())
        R["tgt"].append(pos[FC.ENT.index(self._te)].copy())
        R["ee"].append(np.array(r[0]["robot0_eef_pos"], np.float64))
        R["quat"].append(np.array(E.sim.data.body_xquat[E.obj_body_id[self._oe]], np.float64))
        R["a6"].append(float(a[6]))
        self._last = pos.copy()
        return r

    def data(self):
        R = self.rows
        d = dict(held=np.array(R["held"], bool), a6=np.array(R["a6"], np.float32))
        for k in ("obj", "tgt", "ee", "quat"):
            d[k] = np.array(R[k], np.float32).reshape(-1, 4 if k == "quat" else 3)
        d["final"] = np.array(self._last if self._last is not None else np.zeros((9, 3)), np.float32)
        return d


def save(path, recs):
    a = dict(n=np.array(len(recs)))
    for i, r in enumerate(recs):
        a.update({f"e{i}_{k}": v for k, v in r.items()})
    np.savez_compressed(path, **a)


def load_track(path):
    z = np.load(path)
    return [{k: z[f"e{i}_{k}"] for k in ("held", "obj", "tgt", "ee", "quat", "a6", "final")} for i in range(int(z["n"]))]


def classify(held, d, success, bound, oe):
    """-> (stage, subtype, last held step or None).  docs/BREAK_ATLAS.md 2."""
    idx = np.flatnonzero(held)
    if success:
        return "success", "", (int(idx[-1]) if len(idx) else None)
    if not len(idx):
        return "GRASP", ("other_object" if bound not in (None, oe) else "none"), None
    dmin, d0, e = float(d[idx].min()), float(d[idx[0]]), int(idx[-1])
    if dmin > ARRIVE:
        return ("HANDOFF" if 1.0 - dmin / d0 < HALF else "TRANSPORT"), "", e
    if e == len(d) - 1:
        return "RELEASE", "never_let_go", e
    if float(d[e]) > SEAT:
        return "RELEASE", "away", e
    return "PHYSICS", "", e


def selftest():
    T = 20
    on = np.zeros(T, bool)
    on[5:15] = True
    far = np.full(T, 0.40)
    near = np.linspace(0.40, 0.02, T)
    cases = [
        (dict(held=on, d=near, success=True, bound="o"), "success", ""),
        (dict(held=np.zeros(T, bool), d=far, success=False, bound="x"), "GRASP", "other_object"),
        (dict(held=np.zeros(T, bool), d=far, success=False, bound=None), "GRASP", "none"),
        (dict(held=on, d=far, success=False, bound="o"), "HANDOFF", ""),
        (dict(held=on, d=np.where(np.arange(T) < 10, 0.40, 0.15), success=False, bound="o"), "TRANSPORT", ""),
        (dict(held=np.arange(T) >= 5, d=near, success=False, bound="o"), "RELEASE", "never_let_go"),
        (dict(held=on, d=np.where(np.arange(T) < 10, 0.40, 0.05), success=False, bound="o"), "RELEASE", "away"),
        (dict(held=on, d=np.where(np.arange(T) < 10, 0.40, 0.02), success=False, bound="o"), "PHYSICS", ""),
    ]
    for c, st, sub in cases:
        got = classify(c["held"], c["d"], c["success"], c["bound"], "o")
        assert got[:2] == (st, sub), (st, sub, got)
    print("selftest ok", len(cases), flush=True)


def tasks():
    return json.load(open(os.path.join(OUT, "n2_tasks.json"), encoding="utf-8"))


def branches():
    return json.load(open(os.path.join(OUT, "n2_branches_main.json"), encoding="utf-8"))


def run_pmin(a):
    import torch
    import fa1_common as FC
    import fa1h_roll as RH
    from pp_p_roll import load_any
    fn = os.path.join(OD, f"pmin_s{a.seed}_{a.tid}")
    if os.path.exists(fn + ".json"):
        return
    torch.set_num_threads(1)
    t = tasks()["tasks"][a.tid]
    assert t["tid"] == a.tid
    m, _ = load_any(os.path.join(FC.DATA, "ckpt", f"pp_full_naf_s{a.seed}", "P.npz"))
    S = np.load(os.path.join(OUT, "n2_states", f"n2_goal_{a.tid}.npy"))
    orig, recs, eps = FC.fresh_env, [], []
    for b in branches():
        cur = {}

        def mk(bddl, seed=7):
            cur["r"] = Rec(orig(bddl, seed), t["object"], t["target"])
            return cur["r"]

        FC.fresh_env = mk
        try:
            e = RH.episode(m, os.path.join(ROOT, t["bddl"]), S[int(b["trial"])], t["language"], b["seed_base"])
        finally:
            FC.fresh_env = orig
        recs.append(cur["r"].data())
        eps.append(dict(trial=int(b["trial"]), success=bool(e["success"]), error=e["error"], bound=e["bound"]))
        print("[pmin]", a.seed, a.tid, b["trial"], e["success"], flush=True)
    save(fn + "_track.npz", recs)
    json.dump(dict(eps=eps), open(fn + ".json", "w", encoding="utf-8"), indent=1)


def run_servo(a):
    import fa1_common as FC
    import pp_n4d as N
    from pp_l1_floor import kp_for
    fn = os.path.join(OD, f"servo_s10_{a.tid}")
    if os.path.exists(fn + ".json"):
        return
    t = tasks()["tasks"][a.tid]
    assert t["tid"] == a.tid
    kp = kp_for(t["compound"])
    S = np.load(os.path.join(OUT, "n2_states", f"n2_goal_{a.tid}.npy"))
    orig, recs, eps = FC.fresh_env, [], []
    for b in branches():
        cur = {}

        def mk(bddl, seed=7):
            cur["r"] = Rec(orig(bddl, seed), t["object"], t["target"])
            return cur["r"]

        FC.fresh_env = mk
        try:
            _, info = N.plan_episode(os.path.join(ROOT, t["bddl"]), S[int(b["trial"])], kp, t["object"], t["target"], np.random.default_rng(int(b["seed_base"]) + 10), servo=True)
        finally:
            FC.fresh_env = orig
        recs.append(cur["r"].data())
        eps.append(dict(trial=int(b["trial"]), success=bool(info["success"]), error=None, bound=None))
        print("[servo]", a.tid, b["trial"], info["success"], flush=True)
    save(fn + "_track.npz", recs)
    json.dump(dict(eps=eps), open(fn + ".json", "w", encoding="utf-8"), indent=1)


def tilt_deg(q):
    w, x, y, z = [float(v) for v in q]
    return float(math.degrees(math.acos(max(-1.0, min(1.0, 1.0 - 2.0 * (x * x + y * y))))))


def stage_table(stem, t, ks):
    """-> [states] of stage names, subtypes, and release geometry rows."""
    R = json.load(open(stem + ".json", encoding="utf-8"))
    E = R["eps"] if "eps" in R else [e for v in R["tasks"].values() for e in v["episodes"]]
    Tr = load_track(stem + "_track.npz")
    assert len(E) == len(ks) and len(Tr) == len(ks), (stem, len(E), len(Tr))
    st, sub, geo, succ = [], [], [], []
    for k, e, r in zip(ks, E, Tr):
        assert int(e["trial"]) == k and e.get("error") is None, (stem, k, e.get("error"))
        d = np.linalg.norm((r["obj"] - r["tgt"])[:, :2], axis=1)
        s, x, last = classify(r["held"], d, bool(e["success"]), e.get("bound"), t["object"])
        st.append(s)
        sub.append(x)
        succ.append(bool(e["success"]))
        if last is not None:
            geo.append(dict(stage=s, ee=(r["ee"][last] - r["tgt"][last])[:2].tolist(), obj=(r["obj"][last] - r["tgt"][last])[:2].tolist(), tilt=tilt_deg(r["quat"][last])))
    return st, sub, geo, succ


def read():
    n2 = tasks()
    ks = n2["states"]["main"]
    tid = {(t["compound"], t["layout"]): t for t in n2["tasks"]}
    res = {}
    for sysname, stems in (("pi05", ["pi05_{tid}"]), ("groot", ["groot_{tid}"]), ("pmin", ["pmin_s10_{tid}", "pmin_s11_{tid}", "pmin_s12_{tid}"]), ("servo", ["servo_s10_{tid}"])):
        res[sysname] = {}
        for lay in ("familiar", "new"):
            RATE, per, subs, geos, rec_agree = {}, {}, {}, {}, []
            for c in NEW + KNOWN:
                t = tid[(c, lay)]
                A = {s: np.zeros(len(ks)) for s in STAGES}
                subs[c], geos[c] = {}, []
                for st in stems:
                    S, X, G, SU = stage_table(os.path.join(OD, st.format(tid=t["tid"])), t, ks)
                    for j, s in enumerate(S):
                        A[s][j] += 1.0 / len(stems)
                    for x in X:
                        if x:
                            subs[c][x] = subs[c].get(x, 0) + 1
                    geos[c] += G
                    if sysname in ("pi05", "groot"):
                        f = os.path.join(OUT, "n2_roll", f"{sysname}_main_{t['tid']}.json")
                        Er = {int(e["trial"]): bool(e["success"]) for v in json.load(open(f, encoding="utf-8"))["tasks"].values() for e in v["episodes"]}
                        rec_agree += [float(Er[k] == s) for k, s in zip(ks, SU) if k in Er]
                RATE[c] = A
                per[c] = {s: float(A[s].mean()) for s in STAGES}

            def excess(names, ti, si):
                v = []
                for i in ti:
                    c = NEW[i]
                    ko, kt = PAIR[c]
                    v.append(sum(RATE[c][n][si] - 0.5 * (RATE[ko][n][si] + RATE[kt][n][si]) for n in names).mean())
                return float(np.mean(v))

            def est(names):
                rng = np.random.default_rng(0)
                e = excess(names, np.arange(len(NEW)), np.arange(len(ks)))
                v = [excess(names, rng.choice(len(NEW), len(NEW)), rng.choice(len(ks), len(ks))) for _ in range(B)]
                lo, hi = float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))
                return dict(E=e, ci=[lo, hi], breaks=bool(lo > 0.0 and e > 0.10))

            r = dict(stages={s: est((s,)) for s in STAGES[1:]}, groups={g: est(n) for g, n in GROUP.items()}, per_task=per, subtypes=subs)
            r["verdict"] = {(True, False): "WITHIN-CHUNK", (False, True): "CROSS-CHUNK", (True, True): "BOTH", (False, False): "NEITHER"}[(r["groups"]["within"]["breaks"], r["groups"]["cross"]["breaks"])]
            order = STAGES[1:]
            reached = {}
            for grp, cs in (("new", NEW), ("known", KNOWN)):
                left = float(len(cs))
                reached[grp] = {}
                for s in ("GRASP", "HANDOFF", "TRANSPORT"):
                    x = sum(per[c][s] for c in cs)
                    reached[grp][s] = x / left if left > 0 else None
                    left -= x
                x = sum(per[c]["RELEASE"] + per[c]["PHYSICS"] + per[c]["success"] for c in cs)
                reached[grp]["arrived_or_success"] = x / len(cs)
                y = sum(per[c]["RELEASE"] for c in cs)
                reached[grp]["RELEASE"] = y / x if x > 0 else None
            r["reached"] = reached
            r["release_geometry"] = {}
            for c in NEW + KNOWN:
                G = [g for g in geos[c] if g["stage"] in ("RELEASE", "PHYSICS", "success")]
                if G:
                    r["release_geometry"][c] = dict(n=len(G), ee=np.median([g["ee"] for g in G], 0).tolist(), obj=np.median([g["obj"] for g in G], 0).tolist(), tilt=float(np.median([g["tilt"] for g in G])))
            if rec_agree:
                r["agrees_with_recorded"] = float(np.mean(rec_agree))
            res[sysname][lay] = r
            print(sysname, lay, r["verdict"], json.dumps({k: r[k] for k in ("stages", "groups", "reached")}), flush=True)
            print(sysname, lay, "per_task", json.dumps(per), "agree", r.get("agrees_with_recorded"), flush=True)
    json.dump(res, open(os.path.join(OUT, "pp_atlas_read.json"), "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("selftest", "pmin", "servo", "read"))
    ap.add_argument("--seed", type=int)
    ap.add_argument("--tid", type=int)
    A = ap.parse_args()
    if A.mode == "selftest":
        selftest()
    else:
        os.makedirs(OD, exist_ok=True)
        {"pmin": lambda: run_pmin(A), "servo": lambda: run_servo(A), "read": read}[A.mode]()
