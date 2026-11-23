"""RVA1 reader (docs/notes/plan_review_atlas.md 2; prereg experiments/prereg/rva1_pred.txt).  CPU, numpy only; run through sbatch.

Failure-mode split of the held-out-pairing (N1-N6) failures from EXISTING rollouts (episode json + server channel log chan_<name>_<job>.jsonl):
  G  grasp failure: the bound object is not the instructed one (chan-log side: or no held segment at all, flag never_carried)
  H  held to the end: the last query has role target and the episode ran to the step cap
  else the release query = the last query of the last held (role target) segment; d = xy |true_cm| there (target minus held object, cm):
  W  d >= 10 and the held object lies within 8 cm (xy) of a known-partner target of the instructed object  (wrong target)
  O  d >= 10 and not W (dropped in transit / released elsewhere)
  M  d < 10, bins M<3, M3-6, M6-10 (centimetre miss)
  J  join failure (the channel log has no complete query block of `calls` queries); E  episode error.
Registered quantities: per-arm shares of failures (clustered bootstrap, tasks then states, B = 2000, rng 0), by pairing (cheese N1-N3, bottle N4-N6),
servo-gain decomposition per class (arm off - arm kept, paired by task and state), rules P / R / MIXED, the validity gate against raa_instr.

    python experiments/rva1_read.py selftest
    python experiments/rva1_read.py gate   --out results/flawed/rva1/rva1_gate.json
    python experiments/rva1_read.py arm    --arm pzA0 [--set dev|w5] --out results/flawed/rva1/rva1_pzA0.json
    python experiments/rva1_read.py decomp --a pzA0 --b pzsL [--set dev] --out results/flawed/rva1/rva1_decomp_pzA0_pzsL.json
Outputs are opened with "x" (never overwritten).
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT]
OUT = os.path.join(ROOT, "results", "flawed")
B, CAP = 2000, 300
D_FAR, D_PARTNER = 10.0, 8.0
CLASSES = ("S", "G", "W", "O", "H", "M", "J", "E")
FAIL = ("G", "W", "O", "H", "M", "J", "E")
NT = ["N1", "N2", "N3", "N4", "N5", "N6"]
NTID = {c: 16 + 2 * i for i, c in enumerate(NT)}                           # familiar-layout tids of N1..N6
PARTNERS = {"cream_cheese_1": ["akita_black_bowl_1"], "wine_bottle_1": ["wooden_cabinet_1_top_side", "wine_rack_1_top_region"],
            "akita_black_bowl_1": ["plate_1", "flat_stove_1_cook_region", "wooden_cabinet_1_top_side"]}
OBJ = {"N1": "cream_cheese_1", "N2": "cream_cheese_1", "N3": "cream_cheese_1", "N4": "wine_bottle_1", "N5": "wine_bottle_1", "N6": "wine_bottle_1"}
DEVT, RXT = list(range(4, 24)), list(range(24, 64))
RAA = os.path.join(OUT, "raa")
RAAF = os.path.join(OUT, "raaf")
# arm -> list of (directory, stem, chan prefix) per phase file group
ARMS = {
    "pzsL": (os.path.join(RAAF, "dev_pzsL"), "raafpzsL", "raafpzsL"), "pzA0": (os.path.join(RAAF, "dev_pzA0"), "raafpzA0", "raafpzA0"),
    "raasl": (RAA, "raasl", "raasl"), "A": (os.path.join(RAAF, "dev_A"), "raafA", "raafA"), "A0": (os.path.join(RAAF, "dev_A0"), "raafA0", "raafA0"),
    "raasl43": (RAA, "raasl43", "raasl43"), "Aplus0": (os.path.join(RAAF, "dev_Aplus0"), "raafAplus0", "raafAplus0"),
    "B": (os.path.join(RAAF, "dev_B"), "raafB", "raafB"),
}


# ------------------------------------------------------------------ classifier
def blocks(recs):
    """Split one (text, ep) record list at t == 0 and return the LAST block (a requeued job appends a repeat of the whole episode)."""
    idx = [i for i, r in enumerate(recs) if r["t"] == 0]
    return recs[idx[-1]:] if idx else recs


def classify(ep, recs, obj, geom, tgt_ent=None, partners=None):
    """-> dict(cls, d, never_carried, bin) for one episode json `ep` and its chan-log records `recs` (list, file order)."""
    if ep.get("error"):
        return dict(cls="E", d=None)
    if ep["success"]:
        return dict(cls="S", d=None)
    recs = blocks(recs) if recs else []
    if len(recs) != int(ep["calls"]):
        return dict(cls="J", d=None, n_chan=len(recs), calls=int(ep["calls"]))
    if ep.get("bound") != obj:
        return dict(cls="G", d=None, never_carried=False)
    held = [i for i, r in enumerate(recs) if r["role"] == "target"]
    if not held:
        return dict(cls="G", d=None, never_carried=True)
    if recs[-1]["role"] == "target" and int(ep["steps"]) >= CAP:
        return dict(cls="H", d=None)
    r = recs[held[-1]]                                                       # release query = the last query of the last held segment
    tx, ty = r["true_cm"][0], r["true_cm"][1]
    d = float(np.hypot(tx, ty))
    out = dict(d=d, t=int(r["t"]))
    if d >= D_FAR:
        te = r["roles"][1] if r.get("roles") else tgt_ent
        near = False
        if geom is not None and te in geom:
            for p in (partners if partners is not None else PARTNERS.get(obj, [])):
                if p not in geom or p == te:
                    continue
                rel = [(geom[p][0] - geom[te][0]) * 100.0 + tx, (geom[p][1] - geom[te][1]) * 100.0 + ty]       # partner minus object (cm)
                if np.hypot(*rel) <= D_PARTNER:
                    near = True
        out["cls"] = "W" if near else "O"
    else:
        out.update(cls="M", bin=0 if d < 3 else (1 if d < 6 else 2))
    return out


# ------------------------------------------------------------------ loading
def load_chan(d, prefix):
    by = {}
    for f in sorted(glob.glob(os.path.join(d, "chan_%s_*.jsonl" % prefix))):
        for ln in open(f, encoding="utf-8"):
            if ln.strip():
                r = json.loads(ln)
                by.setdefault((r["text"], int(r["ep"])), []).append(r)
    return by


def episodes(path, tid):
    E = json.load(open(path, encoding="utf-8"))["tasks"][str(tid)]["episodes"]
    return {int(e.get("trial", e.get("init_state", -1))): e for e in E}


def arm_grid(arm, sset="dev"):
    """-> dict task -> {trial: classification}."""
    meta = json.load(open(os.path.join(OUT, "n2_tasks.json"), encoding="utf-8"))
    lang = {t["tid"]: t["language"] for t in meta["tasks"]}
    if sset == "dev":
        geom = json.load(open(os.path.join(OUT, "rva1", "geom_dev.json"), encoding="utf-8"))
        d, stem, pre = ARMS[arm]
        phases = (("main", DEVT, d), ("rx", RXT, d))
        chan = load_chan(d, pre)
    else:
        geom = json.load(open(os.path.join(OUT, "rva1", "geom_w5.json"), encoding="utf-8"))
        d, stem, pre = W5ARMS[arm]
        phases = (("main", list(range(60)), d),)
        chan = load_chan(d, pre)
    res = {}
    for c in NT:
        tid = NTID[c]
        row = {}
        for ph, trials, dd in phases:
            eps = episodes(os.path.join(dd, "%s_%s_%d.json" % (stem, ph, tid)), tid)
            for t in trials:
                recs = chan.get((lang[tid], t), [])
                row[t] = classify(eps[t], recs, OBJ[c], geom[str(t)])
        res[c] = row
    return res


# W5ARMS: arm -> (directory, stem, chan prefix) on the 5th fresh set; filled by name (results/flawed/rvw5/a_<arm>)
W5ARMS = {a: (os.path.join(OUT, "rvw5", "a_" + a), a, a) for a in ("pzsL", "pzA0", "raasl", "A0", "A", "w5", "now", "H", "raasl43")}


# ------------------------------------------------------------------ statistics
def to_arrays(grid, trials):
    """-> dict class -> [6, S] indicator array (S = success, plus M bins m0 m1 m2, and never_carried nc)."""
    A = {k: np.zeros((6, len(trials))) for k in CLASSES + ("m0", "m1", "m2", "nc")}
    for i, c in enumerate(NT):
        for j, t in enumerate(trials):
            r = grid[c][t]
            A[r["cls"]][i, j] = 1
            if r["cls"] == "M":
                A["m%d" % r["bin"]][i, j] = 1
            if r.get("never_carried"):
                A["nc"][i, j] = 1
    return A


def boot(fn, A, tasks, seed=0):
    """clustered bootstrap: tasks, then states (raa_read.diff's scheme).  fn(sub) with sub = dict class -> [tasks, states] over the draw."""
    rng = np.random.default_rng(seed)
    nt, ns = len(tasks), A["S"].shape[1]
    sel = {k: v[tasks] for k, v in A.items()}
    pt = fn(sel)
    v = []
    for _ in range(B):
        t = rng.choice(nt, nt)
        s = rng.choice(ns, ns)
        v.append(fn({k: x[t][:, s] for k, x in sel.items()}))
    v = np.array(v, float)
    v = v[np.isfinite(v)]
    return [float(pt), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def share(c):
    def f(S):
        nf = sum(S[k].sum() for k in FAIL)
        return S[c].sum() / nf if nf > 0 else np.nan
    return f


def arm_summary(grid, trials):
    A = to_arrays(grid, trials)
    res = dict(n_episodes=int(A["S"].size), success=float(A["S"].mean()), counts={k: int(A[k].sum()) for k in A})
    nf = sum(A[k].sum() for k in FAIL)
    res["n_fail"] = int(nf)
    groups = {"all": [0, 1, 2, 3, 4, 5], "cheese": [0, 1, 2], "bottle": [3, 4, 5]}
    res["share"] = {}
    for g, tk in groups.items():
        for c in ("G", "W", "O", "H", "M", "J", "E", "m0", "m1", "m2", "nc"):
            if c in ("m0", "m1", "m2", "nc"):
                fn = (lambda cc: (lambda S: S[cc].sum() / max(sum(S[k].sum() for k in FAIL), 1e-9)))(c)
            else:
                fn = share(c)
            res["share"]["%s|%s" % (g, c)] = boot(fn, A, tk)
    res["join_fail_frac"] = float(A["J"].sum() / A["S"].size)
    return res, A


def decomp(Aa, Ab, trials):
    """arm a = servo-off, b = servo-kept.  Delta_c = rate_c(a) - rate_c(b), per episode on the same (task, state); rate over ALL episodes."""
    rng = np.random.default_rng(0)
    nt, ns = 6, Aa["S"].shape[1]
    keys = ("G", "W", "O", "H", "M", "J", "E", "m0", "m1", "m2")

    def stat(ia, ib, t, s):
        d = {k: float(Aa[k][t][:, s].mean() - Ab[k][t][:, s].mean()) for k in keys}
        d["total"] = sum(d[k] for k in FAIL)
        d["success_diff_a_minus_b"] = float(Aa["S"][t][:, s].mean() - Ab["S"][t][:, s].mean())
        return d
    t0, s0 = np.arange(nt), np.arange(ns)
    pt = stat(Aa, Ab, t0, s0)
    vs = {k: [] for k in pt}
    for _ in range(B):
        t, s = rng.choice(nt, nt), rng.choice(ns, ns)
        d = stat(Aa, Ab, t, s)
        for k in pt:
            vs[k].append(d[k])
    res = {k: [pt[k], float(np.percentile(vs[k], 2.5)), float(np.percentile(vs[k], 97.5))] for k in pt}
    dM, dW, tot = res["M"], res["W"], res["total"]
    P = dM[1] > 0 and dM[0] >= 0.5 * tot[0] and dW[2] < 0.10
    R = dW[1] > 0 and dW[0] >= 0.5 * tot[0]
    res["rule"] = "P (read but imprecise)" if P else ("R (routing)" if R else "MIXED")
    res["rule_inputs"] = dict(dM=dM, dW=dW, total=tot, P=bool(P), R=bool(R))
    return res


# ------------------------------------------------------------------ gate against raa_instr
def gate(out):
    import raa_instr_read as RI
    meta = json.load(open(os.path.join(OUT, "n2_tasks.json"), encoding="utf-8"))
    lang = {t["tid"]: t["language"] for t in meta["tasks"]}
    geom = json.load(open(os.path.join(OUT, "rva1", "geom_dev.json"), encoding="utf-8"))
    d, stem, pre = ARMS["raasl"]
    chan = load_chan(d, pre)
    rows, dd = [], []
    for c, tid in (("N1", 16), ("N4", 22), ("N6", 26)):
        eps = episodes(os.path.join(d, "%s_rx_%d.json" % (stem, tid)), tid)
        for t in range(24, 44):
            cl = classify(eps[t], chan.get((lang[tid], t), []), OBJ[c], geom[str(t)])
            f = os.path.join(OUT, "raa_instr", "raasl", "instr_raac_%d_t%d.npz" % (tid, t))
            z = RI.load(f)
            n = len(z["a"])
            succ = bool(z["done"][-1] > 0)
            R = RI.carries(z)
            if succ:
                ic, di = "S", None
            elif not R:
                ic, di = "G", None
            elif R[-1][1] == n - 1:
                ic, di = "H", None
            else:
                p = R[-1][1]
                di = float(100 * np.linalg.norm(z["obj"][p, :2] - z["tgt"][p, :2]))
                ic = "near" if di < D_FAR else "far"
            cc = cl["cls"]
            cco = {"S": "S", "G": "G", "H": "H", "J": "J", "E": "E", "M": "near", "W": "far", "O": "far"}[cc]
            rows.append(dict(tid=tid, trial=t, chan=cc, chan_coarse=cco, instr=ic, d_chan=cl.get("d"), d_instr=di, agree=cco == ic))
            if cl.get("d") is not None and di is not None:
                dd.append(abs(cl["d"] - di))
    agree = float(np.mean([r["agree"] for r in rows]))
    med = float(np.median(dd)) if dd else None
    res = dict(n=len(rows), agree=agree, median_abs_d_diff_cm=med, n_d=len(dd), passes=bool(agree >= 0.90 and med is not None and med <= 2.0), rows=rows)
    with open(out, "x", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print("[rva1 gate] n %d agree %.3f median |d_chan - d_instr| %s cm (n %d) PASS %s" % (len(rows), agree, med, len(dd), res["passes"]), flush=True)
    return res


# ------------------------------------------------------------------ self test
def selftest():
    def rec(t, role, true, roles=("wine_bottle_1", "plate_1", True)):
        return dict(ep=1, t=t, text="x", role=role, true_cm=list(true), roles=list(roles))
    geom = {"plate_1": [0.0, 0.0, 0.9], "wooden_cabinet_1_top_side": [0.3, 0.0, 1.0], "wine_rack_1_top_region": [-0.4, 0.0, 1.0], "wine_bottle_1": [0.1, 0.1, 0.9]}
    ok = dict(success=False, error=None, bound="wine_bottle_1", steps=300)

    def run(recs, **kw):
        e = dict(ok, calls=len(recs)); e.update(kw)
        return classify(e, recs, "wine_bottle_1", geom)
    objq = lambda n: [rec(5 * i, "object", (10, 10, -20)) for i in range(n)]       # noqa: E731
    # M: released with the object 4 cm from the target
    r = run(objq(2) + [rec(10, "target", (30, 0, 0)), rec(15, "target", (3, -3, 0)), rec(20, "object", (3, -3, 0)), rec(25, "object", (3, -3, 0))])
    assert r["cls"] == "M" and r["bin"] == 1, r
    r = run(objq(1) + [rec(5, "target", (1, 1, 0)), rec(10, "object", (1, 1, 0))])
    assert r["cls"] == "M" and r["bin"] == 0, r
    r = run(objq(1) + [rec(5, "target", (7, 2, 0)), rec(10, "object", (7, 2, 0))])
    assert r["cls"] == "M" and r["bin"] == 2, r
    # W: object released 30 cm from plate, in the direction of the cabinet partner (target - object = -30 cm in x -> object at x = +30 cm = cabinet)
    r = run(objq(1) + [rec(5, "target", (-30, 0, 0)), rec(10, "object", (-30, 0, 0))])
    assert r["cls"] == "W", r
    # O: 30 cm away in y, no partner there
    r = run(objq(1) + [rec(5, "target", (0, -30, 0)), rec(10, "object", (0, -30, 0))])
    assert r["cls"] == "O", r
    # G: bound is another object, and bound None, and no held segment
    assert run(objq(3), bound="cream_cheese_1")["cls"] == "G"
    assert run(objq(3), bound=None)["cls"] == "G"
    assert run(objq(3))["cls"] == "G" and run(objq(3)).get("never_carried") is True
    # H: held to the end at the cap
    assert run(objq(1) + [rec(5, "target", (2, 2, 0)), rec(10, "target", (2, 2, 0))])["cls"] == "H"
    # J: join failure; E; S
    e = dict(ok, calls=5)
    assert classify(e, objq(2), "wine_bottle_1", geom)["cls"] == "J"
    assert classify(dict(ok, error="x", calls=1), objq(1), "wine_bottle_1", geom)["cls"] == "E"
    assert classify(dict(ok, success=True, calls=1), objq(1), "wine_bottle_1", geom)["cls"] == "S"
    # requeue duplicate: an incomplete earlier block then a full repeat
    rr = objq(2) + [rec(5, "target", (2, 0, 0))] + objq(1) + [rec(5, "target", (2, 0, 0)), rec(10, "object", (2, 0, 0))]
    e = dict(ok, calls=3)
    assert classify(e, rr, "wine_bottle_1", geom)["cls"] == "M"
    # bootstrap / decomposition on synthetic grids
    def mk(cls_by):
        g = {c: {t: dict(cls=cls_by(i, t)) for t in range(4, 14)} for i, c in enumerate(NT)}
        for c in NT:
            for t in g[c]:
                if g[c][t]["cls"] == "M":
                    g[c][t]["bin"] = 1
        return g
    ga = mk(lambda i, t: "M" if (t % 2) else "S")
    gb = mk(lambda i, t: "S")
    Aa, Ab = to_arrays(ga, list(range(4, 14))), to_arrays(gb, list(range(4, 14)))
    d = decomp(Aa, Ab, None)
    assert d["rule"].startswith("P"), d["rule"]
    ga = mk(lambda i, t: "O" if (t % 2) else "S")
    d = decomp(to_arrays(ga, list(range(4, 14))), Ab, None)
    assert d["rule"] == "MIXED", d["rule"]
    print("[rva1 selftest] OK", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("selftest", "gate", "arm", "decomp"))
    ap.add_argument("--arm")
    ap.add_argument("--a")
    ap.add_argument("--b")
    ap.add_argument("--set", default="dev")
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.cmd == "selftest":
        return selftest()
    out = os.path.abspath(a.out)
    assert not os.path.exists(out), "never overwrite: " + out
    os.makedirs(os.path.dirname(out), exist_ok=True)
    trials = DEVT + RXT if a.set == "dev" else list(range(60))
    if a.cmd == "gate":
        return gate(out)
    if a.cmd == "arm":
        grid = arm_grid(a.arm, a.set)
        res, _ = arm_summary(grid, trials)
        res.update(arm=a.arm, set=a.set, B=B)
        res["episodes"] = {c: {str(t): grid[c][t] for t in trials} for c in NT}
    else:
        ga, gb = arm_grid(a.a, a.set), arm_grid(a.b, a.set)
        res = decomp(to_arrays(ga, trials), to_arrays(gb, trials), trials)
        res.update(a=a.a, b=a.b, set=a.set, B=B)
    with open(out, "x", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print("[rva1] -> %s" % out, flush=True)
    print(json.dumps({k: v for k, v in res.items() if k != "episodes"})[:3000], flush=True)


if __name__ == "__main__":
    main()
