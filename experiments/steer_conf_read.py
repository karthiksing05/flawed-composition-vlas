"""STEER-CONF: the confirmatory read of the exploratory predictor k x a on Rrel and RAA (CPU, openpi venv, numpy only).
Registered in experiments/prereg/steer_conf.py; predictions in docs/notes/steer_conf_predictions.md.

k x a = trained-task success x g_a(o) x p_a(t), exactly as steer_v2_explore.py defines it ("k*a"; no refitting, no entity normalisation):
  a(policy, phase, swapped-in entity) = clip((mean steer(cf) - mean steer(true)) / (1 - floor), 0, 1) over the paired queries,
  floor = the policy's phase mean of steer(true);  cell score(policy, task) = a_grasp(o) * a_place(t) * K(policy),
  K = the share of successful episodes in the probe's own K1-K4, K6 rollouts of the policy (the SAME definition as v2).
The score of a policy depends on that policy's own records only, so the old 90 cells keep their v2 values exactly.

Cells: Rrel, RAA x N1-N6 (scene 1) = 12 new cells.  Label = success >= .5, success = mean over the 60 states of SR 73 (dev trials 4-23 + new
trials 24-63; raa_read.grid, familiar layout).  Old cells: results/flawed/steer/targets.json, as in SR 74.
  C1  AUC (ties .5) of k x a on the 12 new cells >= .75.  Descriptive: the SR 74 joint bootstrap (B = 2000, rng 0).  NOT TESTABLE if fewer
      than 12 cells are scoreable or one class is missing (an undefined AUC is never a pass).
  C2  pooled AUC over the 90 old + 12 new cells, score frozen; PASS iff the lower bound of the SR 74 joint bootstrap (steer_read.stat_block:
      B = 2000, numpy default_rng(0), policies and tasks jointly) is > .5.
  C3  sign test: sign(mean k x a over the 6 cells, Rrel - RAA) == sign(mean measured N1-N6 success, Rrel - RAA); either difference zero
      -> NOT TESTABLE.  One pair: chance level is .5, so C3 alone cannot carry the claim.

    python experiments/steer_conf_read.py selftest            reproduces steer_v2_explore.json's k*a row on the old 90 cells
    python experiments/steer_conf_read.py read [--out F]      -> results/flawed/steer_conf_read.json (refuses to overwrite)
"""
import glob
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE]
import steer_read as SR                                                      # noqa: E402

ROOT, FL = SR.ROOT, SR.FL
FROZEN = {"experiments/steer_v2_explore.py": "37047092cba3bf45f8ccbdaa70adc8b6795e08b51f2c27ce7e5d6ac4d465f638",
          "experiments/steer_read.py": "541d7713151779a8dfc71b9a6d43458113f9fbb88aaee92e9978591aabba66f7"}
NEWP = ("rrel", "raa")                                                       # directories results/flawed/steer/s1_rrel, s1_raa
TASKS1 = [t for t in SR.NEWT if SR.NEWT[t][0] == 1]
nan = float("nan")


def sha(p):
    with open(os.path.join(ROOT, p), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def check_frozen():
    for p, h in FROZEN.items():
        assert sha(p) == h, "frozen file changed: " + p


def load(dirs):
    """-> (R, ks): probe records and K-episode successes per policy id.  Old policies: SR.pid_of(file), as steer_v2_explore.py."""
    R, ks = {}, {}
    for f in dirs:
        for g in sorted(glob.glob(os.path.join(FL, "steer", f[0], "*_steer.jsonl"))):
            pid = f[1] if f[1] else SR.pid_of(g)
            R.setdefault(pid, []).extend(json.loads(x) for x in open(g, encoding="utf-8") if x.strip())
            jf = g.replace("_steer.jsonl", ".json")
            if os.path.exists(jf):
                ks.setdefault(pid, []).extend(bool(e["success"]) for v in json.load(open(jf, encoding="utf-8"))["tasks"].values() for e in v["episodes"])
    return R, {p: float(np.mean(v)) for p, v in ks.items()}


def ka(R, KS, pols):
    """The v2 k*a score.  The statements inside are steer_v2_explore.py's own lines (raw, floor, A, cells' 'k*a' branch), unchanged apart from
    being wrapped in a function.  -> {(policy, task): score}"""
    def raw(pid):
        out = {"grasp": {}, "place": {}}
        for r in R[pid]:
            c, t = SR.score(r, "cf"), SR.score(r, "true")
            if np.isnan(c):
                continue
            d = out[r["phase"]].setdefault(r["cand"], dict(cf=[], tr=[], cfp=[]))
            d["cfp"].append(c)                      # v1 uses all cf-valid queries
            if not np.isnan(t):
                d["cf"].append(c); d["tr"].append(t)
        return out
    RAW = {p: raw(p) for p in pols}
    floor = {}
    for p in pols:
        for ph in ("grasp", "place"):
            v = [x for d in RAW[p][ph].values() for x in d["tr"]]
            floor[(p, ph)] = float(np.mean(v)) if v else nan
    A = {}
    for p in pols:
        for ph in ("grasp", "place"):
            for cand, d in RAW[p][ph].items():
                if d["cf"]:
                    e = (np.mean(d["cf"]) - np.mean(d["tr"])) / (1 - floor[(p, ph)])
                    A[(p, ph, cand)] = float(min(1, max(0, e)))
    out = {}
    for p in pols:
        for t, (s_, o, tt) in SR.NEWT.items():
            if s_ != 1:
                continue
            g, q = A.get((p, "grasp", o)), A.get((p, "place", tt))
            if g is None or q is None:
                continue
            v = g * q * KS.get(p, nan)
            if np.isnan(v):
                continue
            out[(p, t)] = v
    return out


def old_cells():
    T = json.load(open(os.path.join(FL, "steer", "targets.json"), encoding="utf-8"))
    tgt = {(c["policy"], c["task"]): c["success"] for c in T["cells"] if c["scene"] == 1}
    dirs = [(os.path.basename(d), None) for d in sorted(glob.glob(os.path.join(FL, "steer", "s1_*"))) if os.path.basename(d)[3:] not in NEWP]
    R, KS = load(dirs)
    pols = sorted(p for p in R if p in {q for q, _ in tgt})
    sc = ka(R, KS, pols)
    return {k: (v, tgt[k]) for k, v in sc.items() if k in tgt}, pols, KS


def new_cells():
    import raa_read as RR
    meta = RR.n2()
    R, KS = load([("s1_" + p, p) for p in NEWP])
    pols = [p for p in NEWP if p in R]
    sc = ka(R, KS, pols)
    lab = {}
    for p in NEWP:
        for t in TASKS1:
            lab[(p, t)] = float(RR.grid(meta, p, [t], RR.DEV + RR.NEWS, "succ").mean())
    return {k: (v, lab[k]) for k, v in sc.items()}, lab, KS


def auc(cells):
    s = np.array([cells[k][0] for k in sorted(cells)]); y = np.array([cells[k][1] for k in sorted(cells)])
    return SR.auc_(s, y)


def selftest():
    check_frozen()
    cells, pols, KS = old_cells()
    ref = json.load(open(os.path.join(FL, "steer_v2_explore.json"), encoding="utf-8"))["table"]["k*a"]["all90"]
    sb = SR.stat_block(cells)
    print("[conf] selftest: n_cells %d auc %.6f ci %s rho %.6f | v2 json: n %d auc %.6f ci %s rho %.6f" % (
        sb["n_cells"], sb["auc"], sb["auc_ci"], sb["rho"], ref["n_cells"], ref["auc"], ref["auc_ci"], ref["rho"]), flush=True)
    assert sb["n_cells"] == ref["n_cells"] == 90 and abs(sb["auc"] - ref["auc"]) < 1e-9 and abs(sb["rho"] - ref["rho"]) < 1e-9
    assert np.allclose(sb["auc_ci"], ref["auc_ci"]), "bootstrap differs"
    print("[conf] selftest ok", flush=True)


def read(out):
    assert not os.path.exists(out), "never overwrite " + out
    check_frozen()
    oc, pols, KSold = old_cells()
    nc, lab, KSnew = new_cells()
    res = dict(frozen=dict(FROZEN), old_policies=pols, n_old_cells=len(oc), n_new_cells=len(nc), K_success=dict(KSnew),
               new_cells={"%s|%s" % k: dict(score=v[0], success=v[1], label=bool(v[1] >= .5)) for k, v in sorted(nc.items())},
               labels_all12={"%s|%s" % k: v for k, v in sorted(lab.items())})
    # C1
    y = np.array([nc[k][1] >= .5 for k in sorted(nc)])
    if len(nc) < 12 or y.all() or (~y).all():
        res["C1"] = dict(verdict="NOT TESTABLE", why="%d of 12 cells scoreable, %d positive" % (len(nc), int(y.sum())))
    else:
        a = auc(nc)
        ci, dr = SR.boot_cells(nc, lambda s, b: SR.auc_(s, b))
        res["C1"] = dict(auc=a, descriptive_ci=ci, ci_draws_dropped=dr, n_pos=int(y.sum()), verdict="PASS" if a >= .75 else "FAIL")
    # C2
    both = dict(oc); both.update(nc)
    sb = SR.stat_block(both)
    res["C2"] = dict(auc=sb["auc"], ci=sb["auc_ci"], dropped=sb["auc_dropped"], n_cells=sb["n_cells"], n_policies=sb["n_policies"], n_pos=sb["n_pos"],
                     rho=sb["rho"], rho_ci=sb["rho_ci"],
                     verdict="NOT TESTABLE" if np.isnan(sb["auc"]) else ("PASS" if sb["auc_ci"][0] > .5 else "FAIL"))
    # C3
    ms = {p: float(np.mean([nc[(p, t)][0] for t in TASKS1])) for p in NEWP if all((p, t) in nc for t in TASKS1)}
    my = {p: float(np.mean([lab[(p, t)] for t in TASKS1])) for p in NEWP}
    if len(ms) < 2 or ms["rrel"] == ms["raa"] or my["rrel"] == my["raa"]:
        res["C3"] = dict(mean_score=ms, mean_success=my, verdict="NOT TESTABLE")
    else:
        ok = np.sign(ms["rrel"] - ms["raa"]) == np.sign(my["rrel"] - my["raa"])
        res["C3"] = dict(mean_score=ms, mean_success=my, verdict="PASS" if ok else "FAIL")
    # descriptive only: trained-task success alone on the same 12 cells, and the old-90 value
    try:
        kt = {k: (KSnew[k[0]], nc[k][1]) for k in nc}
        res["descriptive"] = dict(ktrain_auc_new12=auc(kt), old90_auc=SR.stat_block(oc)["auc"])
    except Exception as e:                                                   # noqa: BLE001
        res["descriptive"] = dict(error=str(e))
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, default=float)
    print(json.dumps({k: res[k] for k in ("C1", "C2", "C3", "descriptive")}, indent=1, default=float))
    print("[conf] wrote", out, flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "selftest":
        selftest()
    elif sys.argv[1] == "read":
        read(sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else os.path.join(FL, "steer_conf_read.json"))
