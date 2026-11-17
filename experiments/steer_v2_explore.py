"""EXPLORATORY (not confirmatory): predictor-v2 variants for the SR 74 steerability probe, from the already-collected per-query probe outputs only
(results/flawed/steer/s1_*/*_steer.jsonl).  Reuses the SR 74 read's statistics (steer_read.py): same cells, same joint bootstrap (B=2000, seed 0).
Variants (per policy x swapped-in word, then s_hat = g * p; no fitted weights):
  v1   registered: mean steer(cf).  Already per-task matched: g_s(o) uses only grasp queries whose swapped-in entity is o, p_s(t) only place queries into t (V-d = v1).
  a    excess steering: (mean steer(cf) - mean steer(true) on the same queries) / (1 - floor), floor = policy x phase mean steer(true); clipped to [0,1].
  b    entity-normalised, rank: within each swapped-in entity, the policy's percentile rank among the scene-1 probed policies with targets (0..1).
  bc   entity-normalised, centred: g - mean over policies of g for that entity + grand mean of g, clipped to [0,1].
  c    both: a, then rank within entity (c) ; and a, then centred (cc).
Combined (no fitting): trained success x s_hat.  Baselines: trained success (ktrain), nonsense-word sensitivity."""
import glob, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [HERE]
import steer_read as SR
ROOT = SR.ROOT; FL = SR.FL
OUT = os.path.join(FL, "steer_v2_explore.json")
assert not os.path.exists(OUT)
T = json.load(open(os.path.join(FL, "steer", "targets.json"), encoding="utf-8"))
tgt = {(c["policy"], c["task"] if c["scene"] == 1 else "S2" + c["task"]): c["success"] for c in T["cells"]}
R, ks = {}, {}
for f in sorted(glob.glob(os.path.join(FL, "steer", "s1_*", "*_steer.jsonl"))):
    pid = SR.pid_of(f)
    R.setdefault(pid, []).extend(json.loads(x) for x in open(f, encoding="utf-8") if x.strip())
    jf = f.replace("_steer.jsonl", ".json")
    if os.path.exists(jf):
        ks.setdefault(pid, []).extend(bool(e["success"]) for v in json.load(open(jf, encoding="utf-8"))["tasks"].values() for e in v["episodes"])
KS = {p: float(np.mean(v)) for p, v in ks.items()}
pols1 = sorted(p for p in R if p in {q for q, _ in tgt})
nan = float("nan")

def raw(pid):
    """-> {phase: {cand: dict(cf=[...], tr=[...])}} paired queries only"""
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
RAW = {p: raw(p) for p in pols1}
floor = {}
for p in pols1:
    for ph in ("grasp", "place"):
        v = [x for d in RAW[p][ph].values() for x in d["tr"]]
        floor[(p, ph)] = float(np.mean(v)) if v else nan
V1, A = {}, {}
nclip = [0, 0]
for p in pols1:
    for ph in ("grasp", "place"):
        for cand, d in RAW[p][ph].items():
            V1[(p, ph, cand)] = float(np.mean(d["cfp"]))
            if d["cf"]:
                e = (np.mean(d["cf"]) - np.mean(d["tr"])) / (1 - floor[(p, ph)])
                nclip[0] += 1; nclip[1] += e <= 0
                A[(p, ph, cand)] = float(min(1, max(0, e)))

def ent_norm(M, mode):
    out = {}
    keys = {(ph, c) for (_, ph, c) in M}
    gm = {ph: np.mean([v for (_, h, _), v in M.items() if h == ph]) for ph in ("grasp", "place")}
    for ph, c in keys:
        ps = [p for p in pols1 if (p, ph, c) in M]
        x = np.array([M[(p, ph, c)] for p in ps])
        if mode == "rank":
            z = SR.rank(x) / max(len(x) - 1, 1) if len(x) > 1 else np.array([.5])
        else:
            z = np.clip(x - x.mean() + gm[ph], 0, 1)
        for p, zz in zip(ps, z):
            out[(p, ph, c)] = float(zz)
    return out
VAR = {"v1": V1, "a": A, "b": ent_norm(V1, "rank"), "bc": ent_norm(V1, "cen")}
VAR["c"] = ent_norm(A, "rank"); VAR["cc"] = ent_norm(A, "cen")
SENS = {}
for p in pols1:
    ns = [SR.ang(np.asarray(r["d5"]["nonsense"]), np.asarray(r["d5"]["true"])) for r in R[p]]
    SENS[p] = float(np.mean([float(a > np.pi / 6) for a in ns if not np.isnan(a)]))

def cells(pols, name):
    sc, tg = {}, {}
    for p in pols:
        for t, (s_, o, tt) in SR.NEWT.items():
            if s_ != 1 or (p, t) not in tgt:
                continue
            if name == "ktrain": v = KS.get(p, nan)
            elif name == "nonsense": v = SENS[p]
            else:
                kt = name.startswith("k*")
                M = VAR[name[2:] if kt else name]
                g, q = M.get((p, "grasp", o)), M.get((p, "place", tt))
                if g is None or q is None: continue
                v = g * q * (KS.get(p, nan) if kt else 1)
            if np.isnan(v): continue
            sc[(p, t)], tg[(p, t)] = v, tgt[(p, t)]
    return sc, tg

def within_task_rho(c):
    """mean over tasks of Spearman across policies (ties -> nan skipped); descriptive"""
    rs = []
    for t in sorted({k[1] for k in c}):
        ks_ = [k for k in c if k[1] == t]
        r = SR.spearman([c[k][0] for k in ks_], [c[k][1] for k in ks_])
        if not np.isnan(r): rs.append(r)
    return float(np.mean(rs)) if rs else nan
tasks = [t for t in SR.NEWT if SR.NEWT[t][0] == 1]
WFP = [p for p in pols1 if p in SR.WF]
res = dict(note="EXPLORATORY, not confirmatory; variants chosen after seeing SR 74", policies=pols1, wf=WFP, floors={"%s|%s" % k: v for k, v in floor.items()},
           excess_clipped_to_zero=[int(nclip[1]), int(nclip[0])], ktrain=KS, table={})
names = ["v1", "a", "b", "bc", "c", "cc"]
for nm in ["ktrain", "nonsense"] + names + ["k*" + n for n in names]:
    row = {}
    for sname, pols in (("wf", WFP), ("all90", pols1)):
        sc, tg = cells(pols, nm)
        c = {k: (sc[k], tg[k]) for k in sc}
        row[sname] = SR.stat_block(c); row[sname]["within_task_rho_mean"] = within_task_rho(c)
        full = [p for p in pols if all((p, t) in sc for t in tasks)]
        row["T1_" + sname] = SR.t1(sc, tg, full, tasks) if len(full) >= 3 else None
    res["table"][nm] = row
    f = lambda x: "%.3f" % x
    print("%-9s wf AUC %s [%s,%s] rho %s | all AUC %s [%s,%s] rho %s [%s,%s] wtrho %s | T1all %s [%s,%s] T1wf %s" % (
        nm, f(row["wf"]["auc"]), *map(f, row["wf"]["auc_ci"]), f(row["wf"]["rho"]), f(row["all90"]["auc"]), *map(f, row["all90"]["auc_ci"]),
        f(row["all90"]["rho"]), *map(f, row["all90"]["rho_ci"]), f(row["all90"]["within_task_rho_mean"]),
        f(row["T1_all90"]["rho"]), *map(f, row["T1_all90"]["ci"]), f(row["T1_wf"]["rho"]) if row["T1_wf"] else "NA"), flush=True)
json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1, default=float)
print("wrote", OUT)
