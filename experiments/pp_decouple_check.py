"""PLAN_PARTS round 10 (docs/PARTS_RESULTS.md section 35; m3_prereg pp_r10): is the roles' training decoupled?  CPU.

ckpt   bitwise comparisons of checkpoints under DATA/ckpt.  --pairs "<tagA>:<tagB>:<scope>+..." with scope obj (the object term:
       parameters obj.* and the object-role buffers rclip, rclip_p, home) or all (the whole state); every seed of --seeds.
hold   per paired S2 episode, the hold flag (held_step set) of two arms' S2 rollouts (--a <dir>:<name>, --b <dir>:<name>), every
       task of both layouts, per seed; per seed, layout and object atom the hold rates and the number of disagreeing episodes.
var    the per-object factors of fa1g_read.three_factor's estimator (h_o: hold; a_o: arrival given hold, over the object's known
       tasks; q_t: place on arrival, over the target's known tasks), per seed and layout, for each arm of --arms
       "<label>=<dir>:<name>+...", with the across-seed range and SD per arm and the between-arm range of the per-seed values.
--verdict  with ckpt + hold results present, the pre-registered verdict of section 35.
-> results/flawed/pp_<tag>_decouple.json (one key per mode; modes run in one call are merged into the file)

    python experiments/pp_decouple_check.py ckpt --tag r10 --seeds 10+11+12 --pairs full_pdc:full_tdc:obj
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT]
import fa1_common as FC                                                     # noqa: E402
from fa1g_read import radius                                                 # noqa: E402

OUT = os.path.join(ROOT, "results", "flawed")
OBJ_KEYS = ("rclip", "rclip_p", "home")


def load_state(tag, seed):
    z = np.load(os.path.join(FC.DATA, "ckpt", f"pp_{tag}_s{seed}", "P.npz"), allow_pickle=False)
    return {k: z[k] for k in z.files if k != "__meta__"}, json.loads(str(z["__meta__"]))


def cmp_ckpt(a, b, scope, seed):
    A, ma = load_state(a, seed)
    B, mb = load_state(b, seed)
    keys = sorted(k for k in A if scope == "all" or k.startswith("obj.") or k in OBJ_KEYS)
    assert set(keys) <= set(B), ("missing keys", sorted(set(keys) - set(B)))
    diff = {k: float(np.abs(A[k].astype(np.float64) - B[k].astype(np.float64)).max()) for k in keys
            if A[k].shape != B[k].shape or not np.array_equal(A[k], B[k])}
    return dict(a=a, b=b, scope=scope, seed=seed, n_tensors=len(keys), bitwise_equal=not diff, max_abs_diff=max(diff.values(), default=0.0),
                differing=sorted(diff), decouple=[ma.get("decouple"), mb.get("decouple")], reuse=[ma.get("reuse"), mb.get("reuse")])


def episodes(d, name, seed, tid):
    f = os.path.join(OUT, d, f"{name}_s{seed}_main_{tid}.json")
    assert os.path.exists(f), f
    return {int(e["trial"]): e for v in json.load(open(f, encoding="utf-8"))["tasks"].values() for e in v["episodes"]}


def hold(a, b, seeds):
    n2 = json.load(open(os.path.join(OUT, "n2_tasks.json"), encoding="utf-8"))
    (da, na), (db, nb) = a.split(":"), b.split(":")
    rows, total_dis, total, detail = [], 0, 0, []
    for sd in seeds:
        for lay in ("familiar", "new"):
            tasks = [t for t in n2["tasks"] if t["layout"] == lay]
            for o in sorted({t["object"] for t in tasks}):
                ha, hb, dis = [], [], 0
                for t in [t for t in tasks if t["object"] == o]:
                    Ea, Eb = episodes(da, na, sd, t["tid"]), episodes(db, nb, sd, t["tid"])
                    assert sorted(Ea) == sorted(Eb)
                    for k in Ea:
                        x, y = Ea[k]["held_step"] is not None, Eb[k]["held_step"] is not None
                        ha.append(x)
                        hb.append(y)
                        dis += int(x != y)
                        if x != y:                                           # where and how a paired hold disagrees
                            hosts = [json.load(open(os.path.join(OUT, dd, f"{nn}_s{sd}_main_{t['tid']}.json"), encoding="utf-8"))["host"]
                                     for dd, nn in ((da, na), (db, nb))]
                            detail.append(dict(seed=sd, tid=t["tid"], compound=t["compound"], layout=lay, trial=k, hosts=hosts,
                                               **{f"{f}_{s}": E[k][f] for s, E in (("a", Ea), ("b", Eb))
                                                  for f in ("held_step", "success", "bound", "bound_step", "steps", "calls", "error")}))
                rows.append(dict(seed=sd, layout=lay, object=o, n=len(ha), hold_a=float(np.mean(ha)), hold_b=float(np.mean(hb)),
                                 delta=float(np.mean(ha) - np.mean(hb)), disagree=dis))
                total_dis += dis
                total += len(ha)
    return dict(a=a, b=b, seeds=list(seeds), cells=rows, episodes=total, disagreeing=total_dis,
                max_abs_delta=max(abs(r["delta"]) for r in rows), disagreeing_episodes=detail)


def factors(d, name, seed, lay):
    n2 = json.load(open(os.path.join(OUT, "n2_tasks.json"), encoding="utf-8"))
    cal = json.load(open(os.path.join(OUT, "a1_calib.json"), encoding="utf-8"))["pairs"]
    known = [t for t in n2["tasks"] if t["layout"] == lay and t["compound"][0] == "K" and int(t["compound"][1:]) <= 6]
    H, A, S = {}, {}, {}
    for t in known:
        rad = radius(t["object"], t["target"], cal)
        E = episodes(d, name, seed, t["tid"]).values()
        H[t["compound"]] = np.array([e["held_step"] is not None for e in E], float)
        A[t["compound"]] = np.array([e["min_dxy_held"] is not None and e["min_dxy_held"] <= rad for e in E], float)
        S[t["compound"]] = np.array([bool(e["success"]) for e in E], float)
    out = {}
    for o in sorted({t["object"] for t in known}):
        ks = [t["compound"] for t in known if t["object"] == o]
        h, ar = np.concatenate([H[k] for k in ks]), np.concatenate([A[k] for k in ks])
        out[f"h_{o}"] = float(h.mean())
        out[f"a_{o}"] = float((ar * h).sum() / max(h.sum(), 1))
    for tg in sorted({t["target"] for t in known}):
        ks = [t["compound"] for t in known if t["target"] == tg]
        ar, s = np.concatenate([A[k] for k in ks]), np.concatenate([S[k] for k in ks])
        out[f"q_{tg}"] = float((s * ar).sum() / max(ar.sum(), 1))
    return out


def var(arms, seeds):
    res = {}
    for spec in arms.split("+"):
        lab, dn = spec.split("=")
        d, name = dn.split(":")
        res[lab] = {lay: {sd: factors(d, name, sd, lay) for sd in seeds} for lay in ("familiar", "new")}
    summ = {}
    for lay in ("familiar", "new"):
        keys = sorted(next(iter(res.values()))[lay][seeds[0]])
        for k in keys:
            row = {}
            for lab in res:
                v = np.array([res[lab][lay][sd][k] for sd in seeds])
                row[lab] = dict(per_seed=v.tolist(), range=float(v.max() - v.min()), sd=float(v.std(ddof=1)) if len(v) > 1 else 0.0)
            per_seed_between = [float(max(res[lab][lay][sd][k] for lab in res) - min(res[lab][lay][sd][k] for lab in res)) for sd in seeds]
            row["between_arm_range_per_seed"] = per_seed_between
            summ[f"{lay}:{k}"] = row
    return dict(arms=arms, seeds=list(seeds), factors=summ)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("ckpt", "hold", "var", "verdict"))
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seeds", default="10+11+12")
    ap.add_argument("--pairs", default="")
    ap.add_argument("--a", default="")
    ap.add_argument("--b", default="")
    ap.add_argument("--arms", default="")
    ap.add_argument("--key", default="")                                     # the result key (default: the mode)
    a = ap.parse_args()
    seeds = [int(x) for x in a.seeds.split("+")]
    fn = os.path.join(OUT, f"pp_{a.tag}_decouple.json")
    out = json.load(open(fn, encoding="utf-8")) if os.path.exists(fn) else {}
    key = a.key or a.mode
    if a.mode == "ckpt":
        out[key] = [cmp_ckpt(x, y, sc, sd) for p in a.pairs.split("+") for x, y, sc in [p.split(":")] for sd in seeds]
        for r in out[key]:
            print("[dc] ckpt", r["a"], r["b"], r["scope"], r["seed"], "equal", r["bitwise_equal"], "max", r["max_abs_diff"],
                  "n", r["n_tensors"], r["differing"][:6], flush=True)
    elif a.mode == "hold":
        out[key] = hold(a.a, a.b, seeds)
        print("[dc] hold episodes", out[key]["episodes"], "disagree", out[key]["disagreeing"], "max |delta|", out[key]["max_abs_delta"], flush=True)
        for d in out[key]["disagreeing_episodes"]:
            print("[dc] hold disagree", json.dumps(d), flush=True)
    elif a.mode == "var":
        out[key] = var(a.arms, seeds)
        for k, row in out[key]["factors"].items():
            print("[dc] var", k, {lab: [round(x, 3) for x in v["per_seed"]] for lab, v in row.items() if lab != "between_arm_range_per_seed"},
                  "between", [round(x, 3) for x in row["between_arm_range_per_seed"]], flush=True)
    else:                                                                    # section 35's pre-registered mapping
        ta = all(r["bitwise_equal"] for r in out["ckpt_a"])
        tb = out["hold"]["disagreeing"] == 0
        out["verdict"] = dict(a_training_decoupled=ta, b_per_object_decoupled=tb,
                              verdict="decoupled" if ta and tb else ("decoupled in training only" if ta else "not decoupled"))
        print("[dc] VERDICT", json.dumps(out["verdict"]), flush=True)
    with open(fn, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print(f"[dc] wrote {fn}", flush=True)


if __name__ == "__main__":
    main()
