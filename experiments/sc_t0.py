"""T0 (docs/SCAFFOLD_RESULTS.md 1; m3_prereg sc_t03): recorded atlas tracks only, no rollouts.  CPU.
  selftest    the event and decomposition rules on synthetic episodes; reads no recorded outcome
  read        -> results/flawed/sc_t0.json
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pp_atlas as PA                                                       # noqa: E402

REPLAN, B, REACH, REACH_ALT, MIN_N = 5, 2000, 0.05, 0.07, 10
STEMS = dict(pi05=["pi05_{tid}"], groot=["groot_{tid}"], pmin=["pmin_s10_{tid}", "pmin_s11_{tid}", "pmin_s12_{tid}"], servo=["servo_s10_{tid}"])
FATAL = ("HANDOFF", "TRANSPORT", "RELEASE")
GATES = ("reach", "hold", "arrive", "place")
LONG = ("reach", "arrive")
SOLID = (4, 5, 6)                                                           # stove cook region, cabinet top side, wine rack (FC.ENT order)
OBJECTS = dict(cheese=("N1", "N2", "N3", "K6"), bottle=("N4", "N5", "N6", "K4"), bowl=("K1", "K2", "K3"))


def phase_of(row):
    """Rows: row 0 is the last settle step, row i >= 1 is policy action i - 1, and a query precedes actions 0, 5, 10, ..."""
    return (int(row) - 1) % REPLAN


def first_open_flip(a6, held):
    """The first row after the hold's onset whose gripper command goes from closed (> 0) to open (< 0), or None."""
    idx = np.flatnonzero(held)
    if not len(idx):
        return None
    for i in range(int(idx[0]) + 1, len(a6)):
        if a6[i - 1] > 0 and a6[i] < 0:
            return i
    return None


def gates(r, success, goff, reach=REACH):
    """-> {gate: passed}, cumulative, in order.  A later gate passing implies the earlier ones (a hold counts as a reach; success as all)."""
    held = np.asarray(r["held"], bool)
    idx = np.flatnonzero(held)
    upto = int(idx[0]) + 1 if len(idx) else len(held)
    a = np.linalg.norm(r["ee"][:upto] - (r["obj"][:upto] + goff), axis=1)
    d = np.linalg.norm((r["obj"] - r["tgt"])[:, :2], axis=1)
    arr = bool(len(idx) and d[idx].min() <= PA.ARRIVE)
    h = bool(len(idx))
    return dict(reach=bool(success or h or a.min() <= reach), hold=bool(success or h), arrive=bool(success or arr), place=bool(success))


def rate(k, n):
    return (float(k) + 0.5) / (float(n) + 1.0)


def decompose(fam, new):
    """fam, new: pass counts [..., 4 gates] with the entry counts [...]: (n, counts).  -> (log ratio per gate, total, long share)."""
    def rates(x):
        n, c = x
        prev = n
        out = []
        for i in range(len(GATES)):
            out.append(rate(c[i], prev))
            prev = c[i]
        return np.array(out)
    L = np.log(rates(new)) - np.log(rates(fam))
    tot = float(L.sum())
    return L, tot, (float(sum(L[GATES.index(g)] for g in LONG) / tot) if abs(tot) > 0.02 else None)


def selftest():
    assert [phase_of(i) for i in (1, 5, 6, 11)] == [0, 4, 0, 0]
    held = np.array([0, 0, 1, 1, 1, 1, 0, 0], bool)
    a6 = np.array([-1, -1, 1, 1, 1, 1, -1, -1], float)
    assert first_open_flip(a6, held) == 6
    assert first_open_flip(a6, np.zeros(8, bool)) is None
    assert first_open_flip(np.ones(8), held) is None
    T = 20
    r = dict(held=np.zeros(T, bool), ee=np.zeros((T, 3)), obj=np.zeros((T, 3)) + [1.0, 0, 0], tgt=np.zeros((T, 3)))
    assert gates(r, False, np.zeros(3)) == dict(reach=False, hold=False, arrive=False, place=False)
    r2 = dict(r, obj=np.zeros((T, 3)) + [0.04, 0, 0], tgt=np.zeros((T, 3)) + [1.0, 0, 0])
    assert gates(r2, False, np.zeros(3)) == dict(reach=True, hold=False, arrive=False, place=False)
    h = np.zeros(T, bool)
    h[5:] = True
    r3 = dict(r2, held=h, tgt=np.zeros((T, 3)) + [0.09, 0, 0])
    assert gates(r3, False, np.zeros(3)) == dict(reach=True, hold=True, arrive=True, place=False)
    assert gates(r, True, np.zeros(3))["reach"]
    L, tot, sh = decompose((100, np.array([100, 100, 100, 100])), (100, np.array([50, 50, 50, 50])))
    assert abs(L[0] - math.log(50.5 / 101) + math.log(100.5 / 101)) < 1e-9 and sh is not None and sh > 0.9
    print("selftest ok", flush=True)


def load(stem):
    p = os.path.join(PA.OD, stem)
    R = json.load(open(p + ".json", encoding="utf-8"))
    E = R["eps"] if "eps" in R else [e for v in R["tasks"].values() for e in v["episodes"]]
    return E, PA.load_track(p + "_track.npz")


def read():
    from pp_l1_floor import kp_for
    n2 = PA.tasks()
    ks = n2["states"]["main"]
    tid = {(t["compound"], t["layout"]): t for t in n2["tasks"]}
    TASKS = PA.NEW + PA.KNOWN
    Data = {}
    for m, stems in STEMS.items():
        for lay in ("familiar", "new"):
            for c in TASKS:
                t = tid[(c, lay)]
                for si, st in enumerate(stems):
                    E, Tr = load(st.format(tid=t["tid"]))
                    assert len(E) == len(ks) == len(Tr)
                    Data[(m, lay, c, si)] = (E, Tr, t)
    res = dict(A={}, B={}, C={})

    # ---- A: do fatal gripper events line up with query boundaries?
    for m in STEMS:
        res["A"][m] = {}
        for lay in ("familiar", "new"):
            fat, ctl, ends, jerk = [], [], [], {"free": np.zeros((REPLAN, 2)), "held": np.zeros((REPLAN, 2))}
            for c in TASKS:
                for si in range(len(STEMS[m])):
                    E, Tr, t = Data[(m, lay, c, si)]
                    for j, (e, r) in enumerate(zip(E, Tr)):
                        d = np.linalg.norm((r["obj"] - r["tgt"])[:, :2], axis=1)
                        s, _, last = PA.classify(r["held"], d, bool(e["success"]), e.get("bound"), t["object"])
                        fl = first_open_flip(r["a6"], r["held"])
                        if fl is not None:
                            if c in PA.NEW and s in FATAL:
                                fat.append((j, phase_of(fl)))
                            if s == "success":
                                ctl.append((j, phase_of(fl)))
                        if c in PA.NEW and s in FATAL and last is not None and last + 1 < len(d):
                            ends.append(phase_of(last + 1))
                        v = np.diff(r["ee"], axis=0)
                        for i in range(1, len(v)):                         # v[i] is the motion of row i + 1, jerk against the row before
                            k = "held" if r["held"][i + 1] else "free"
                            jerk[k][phase_of(i + 1)] += [float(np.linalg.norm(v[i] - v[i - 1])), 1.0]
            fa, ca = np.array(fat, int).reshape(-1, 2), np.array(ctl, int).reshape(-1, 2)

            def share(a, w):
                x = [np.mean(a[a[:, 0] == q, 1] == 0) * (a[:, 0] == q).sum() for q in w]
                n = sum((a[:, 0] == q).sum() for q in w)
                return float(np.nansum(x) / n) if n else float("nan")
            w0 = np.arange(len(ks))
            dlt = share(fa, w0) - share(ca, w0) if len(fa) and len(ca) else None
            rng = np.random.default_rng(0)
            if dlt is not None:
                v = np.array([share(fa, w) - share(ca, w) for w in (rng.choice(len(ks), len(ks)) for _ in range(B))])
                ci = [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
            else:
                ci = None
            lab = "TOO FEW" if len(fa) < MIN_N or len(ca) < MIN_N else ("BOUNDARY" if ci[0] > 0.0 and dlt > 0.10 else "NOT AT BOUNDARIES")
            res["A"][m][lay] = dict(label=lab, n_fatal=len(fa), n_control=len(ca), share0_fatal=share(fa, w0) if len(fa) else None, share0_control=share(ca, w0) if len(ca) else None,
                                    delta=dlt, ci=ci, hist_fatal=np.bincount(fa[:, 1], minlength=REPLAN).tolist() if len(fa) else [0] * REPLAN,
                                    hist_control=np.bincount(ca[:, 1], minlength=REPLAN).tolist() if len(ca) else [0] * REPLAN,
                                    hist_hold_end=np.bincount(np.array(ends, int), minlength=REPLAN).tolist() if ends else [0] * REPLAN,
                                    jerk={k: (x[:, 0] / np.maximum(x[:, 1], 1.0)).tolist() for k, x in jerk.items()})
            print("A", m, lay, json.dumps(res["A"][m][lay]), flush=True)

    # ---- B: new layout's loss across gates, log-ratio decomposition
    goff = {c: np.array(kp_for(c)["grasp_off"]) for c in TASKS}

    def table(m, lay, reach):
        """[tasks, seeds, states, gates] bool, the entries for one system and layout."""
        X = np.zeros((len(TASKS), len(STEMS[m]), len(ks), len(GATES)), bool)
        for ci_, c in enumerate(TASKS):
            for si in range(len(STEMS[m])):
                E, Tr, _ = Data[(m, lay, c, si)]
                for j, (e, r) in enumerate(zip(E, Tr)):
                    g = gates(r, bool(e["success"]), goff[c], reach)
                    X[ci_, si, j] = [g[n] for n in GATES]
        return X

    def pool(X, ti, sj):
        Y = X[ti][:, :, sj]
        return float(np.prod(Y.shape[:3])), Y.sum((0, 1, 2)).astype(float)

    for m in STEMS:
        F, N = table(m, "familiar", REACH), table(m, "new", REACH)
        tall, sall = np.arange(len(TASKS)), np.arange(len(ks))
        L, tot, sh = decompose(pool(F, tall, sall), pool(N, tall, sall))
        rng = np.random.default_rng(0)
        v = []
        for _ in range(B):
            ti = rng.choice(len(TASKS), len(TASKS))
            v.append(decompose(pool(F, ti, rng.choice(len(ks), len(ks))), pool(N, ti, rng.choice(len(ks), len(ks))))[2])
        v = np.array([np.nan if x is None else x for x in v], float)
        ci = [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
        lab = "LONG" if ci[0] >= 0.5 and sh is not None and sh >= 2 / 3 else ("SHORT" if ci[1] <= 0.5 and sh is not None and sh <= 1 / 3 else "MIXED")
        F2, N2 = table(m, "familiar", REACH_ALT), table(m, "new", REACH_ALT)
        _, _, sh2 = decompose(pool(F2, tall, sall), pool(N2, tall, sall))
        per = {}
        for grp, cs in (("new_pairings", PA.NEW), ("known", PA.KNOWN)):
            ti = np.array([TASKS.index(c) for c in cs])
            L2, t2, s2 = decompose(pool(F, ti, sall), pool(N, ti, sall))
            per[grp] = dict(log_ratio=dict(zip(GATES, L2.tolist())), total=t2, long_share=s2)
        fp, np_ = pool(F, tall, sall), pool(N, tall, sall)
        res["B"][m] = dict(label=lab, long_share=sh, ci=ci, n_nan=int(np.isnan(v).sum()), log_ratio=dict(zip(GATES, L.tolist())), total=tot, long_share_reach07=sh2, by_group=per,
                           pass_familiar=(fp[1] / fp[0]).tolist(), pass_new=(np_[1] / np_[0]).tolist())
        print("B", m, json.dumps(res["B"][m]), flush=True)

    # ---- C: new layout, success and hold by start clearance to a solid fixture (descriptive)
    for m in STEMS:
        res["C"][m] = {}
        for ob, cs in OBJECTS.items():
            bins = {"<.10": [0, 0, 0], ".10-.20": [0, 0, 0], ">=.20": [0, 0, 0]}
            for c in cs:
                for si in range(len(STEMS[m])):
                    E, Tr, _ = Data[(m, "new", c, si)]
                    for e, r in zip(E, Tr):
                        cl = min(float(np.linalg.norm(r["obj"][0, :2] - r["final"][f, :2])) for f in SOLID)
                        k = "<.10" if cl < 0.10 else (".10-.20" if cl < 0.20 else ">=.20")
                        bins[k][0] += 1
                        bins[k][1] += int(bool(np.any(r["held"])) or bool(e["success"]))
                        bins[k][2] += int(bool(e["success"]))
            res["C"][m][ob] = {k: dict(n=v[0], hold=(v[1] / v[0] if v[0] else None), success=(v[2] / v[0] if v[0] else None)) for k, v in bins.items()}
        print("C", m, json.dumps(res["C"][m]), flush=True)
    with open(os.path.join(PA.OUT, "sc_t0.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    {"selftest": selftest, "read": read}[sys.argv[1]]()
