"""The registered read of S1-CROSS (m3_prereg tag s1_cross; docs/SCAFFOLD_RESULTS.md 65): does the scene-1 front end (tvpb, SR 55)
carry to a third model?  Frozen pi0 and frozen SmolVLA, familiar layout, the new fresh batch k = 236-275 (harness trials 24-63),
N1-N6 and K1-K4, K6.  LIBERO venv, CPU.

Arms (all new on these states, results/flawed/s1_cross/<model>_<arm>_<tid>, run by experiments/s1_cross_roll.sh, clients unchanged):
    I     intact (pp_atlas_client.py)
    tv    silhouette role views (tv_frozen.py client --arm tv; SR 46 / 49)
    tvpb  role views + role phrases + the held object painted as its bounding box in the target phase (f1m_cue.py client --arm sb; SR 55)
    rgp   ReGuide RG-perc (rg.py client --arm rgp; SR 52 / 55)
Registered mapping, per model: success matrices [task, state]; paired bootstrap B = 2000, numpy default_rng(0), tasks and states
drawn jointly (tv_frozen.boot2, as SR 52 / 55); CI 2.5 / 97.5.
    F   testability floor (SR 50 style): TESTABLE iff I's success on K1-K4, K6 >= .70; else NOT TESTABLE (labels still reported,
        not counted in the aggregate).
    V1  d = tvpb - I on N1-N6:      FRONT END HELPS iff CI lo > 0; else NO CLEAR DIFFERENCE.
    V2  d = tvpb - rgp on N1-N6:    FRONT END WINS iff CI lo > 0; RG WINS iff CI hi < 0; EQUIVALENT iff -.10 <= lo and hi <= .10;
                                    else NO CLEAR DIFFERENCE.  A win label takes precedence; the equivalence flag is reported beside it.
    V3  d = tvpb - I on K1-K4, K6 (trained-task cost):  COST iff CI hi < 0; NO COST iff CI lo >= -.05; else UNRESOLVED.
    AGG over testable models: CARRIES iff V1 = FRONT END HELPS and V3 != COST on every testable model; PARTIAL iff on at least one but
        not all; DOES NOT CARRY iff on none; NOT TESTED iff no model is testable.
Descriptive (no verdict): tv - I, tvpb - tv, rgp - I, tv - rgp on N1-N6; tvpb - tv and tvpb - rgp on N4 + N6; per-task success; K
success per arm and arm - I; the atlas's within / cross-chunk excess per arm (tv_frozen.atlas_est); stage shares (incl. GRASP, F2);
high-carry share on N4 / N6 / K4 (n46_phase lift_held > .22 m among grasped); detector stats (tv, tvpb); ReGuide diagnostics.
Checks: every arm's client args equal I's except out / port / videos, and per-episode trial / branch / variant / at equal I's
(tv_fresh.harness); every episode's trial is the registered one; no episode errors; state files and code match the s1_cross row;
ReGuide's files match the reguide row, f1m_cue.py the f1m_cue row, tv_frozen.py the tvpb_rg row.

    python experiments/s1_cross_read.py [--out results/flawed/s1_cross_read.json]
    python experiments/s1_cross_read.py --dry --dir SMOKE_DIR --trials 0,1 --out OUT_OUTSIDE_RESULTS     plumbing check, no ledger
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT]
import tv_frozen as TF                                                      # noqa: E402

MODELS = ("pi0", "smolvla")
READ_ARMS = ("I", "tv", "tvpb", "rgp")
DIR = os.path.join(TF.OUT, "s1_cross")
READ_OUT = os.path.join(TF.OUT, "s1_cross_read.json")
TRIALS = list(range(24, 64))
STATES = os.path.join(TF.OUT, "n2_states_s1x")
BR = os.path.join(TF.OUT, "n2_branches_s1x.json")
TIDS = (16, 18, 20, 22, 24, 26, 0, 2, 4, 6, 10)
EQ = 0.10
FLOOR = 0.70
NI = 0.05
CUT = 0.22
RG_FILES = ("experiments/rg.py", "experiments/rg_stats.py", "experiments/sc_pu.py", "results/flawed/rg_stats.json",
            "results/flawed/sc_pu_calib.json")
MAPPING = ("per model (pi0, smolvla), familiar layout, trials 24-63 (fresh k 236-275); success matrices [task, state]; paired bootstrap "
           "B=2000 rng 0, tasks and states drawn jointly (tv_frozen.boot2); CI 2.5/97.5. F: TESTABLE iff I success on K1-K4,K6 >= .70. "
           "V1: tvpb - I on N1-N6: FRONT END HELPS iff lo > 0, else NO CLEAR DIFFERENCE. V2: tvpb - rgp on N1-N6: FRONT END WINS iff "
           "lo > 0; RG WINS iff hi < 0; EQUIVALENT iff -.10 <= lo and hi <= .10; else NO CLEAR DIFFERENCE (win labels take precedence; "
           "equivalence flag reported). V3: tvpb - I on K1-K4,K6: COST iff hi < 0; NO COST iff lo >= -.05; else UNRESOLVED. AGG over "
           "testable models: CARRIES iff V1 HELPS and V3 != COST on all; PARTIAL iff on some; DOES NOT CARRY iff on none; NOT TESTED "
           "iff none testable. Descriptive: other differences, K per arm, atlas excess, stages, high-carry share, detector and RG "
           "diagnostics.")


def v1(d):
    return "FRONT END HELPS" if d[1] > 0 else "NO CLEAR DIFFERENCE"


def v2(d):
    eq = bool(d[1] >= -EQ and d[2] <= EQ)
    if d[1] > 0:
        return "FRONT END WINS", eq
    if d[2] < 0:
        return "RG WINS", eq
    return ("EQUIVALENT" if eq else "NO CLEAR DIFFERENCE"), eq


def v3(d):
    if d[2] < 0:
        return "COST"
    if d[1] >= -NI:
        return "NO COST"
    return "UNRESOLVED"


def agg(per):
    """per: {model: dict(testable, V1, V3)} -> aggregate label over testable models."""
    T = [m for m, v in per.items() if v["testable"]]
    if not T:
        return "NOT TESTED", T
    ok = [m for m in T if per[m]["V1"] == "FRONT END HELPS" and per[m]["V3"] != "COST"]
    if len(ok) == len(T):
        return "CARRIES", T
    return ("PARTIAL" if ok else "DOES NOT CARRY"), T


def sha(p):
    with open(p if os.path.isabs(p) else os.path.join(ROOT, p), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def state_files():
    return [os.path.join(STATES, "states.npy")] + [os.path.join(STATES, "n2_goal_%d.npy" % t) for t in TIDS] + [BR]


def code_checks():
    from lifelong_ood import ledger
    tags = ("s1_cross", "reguide", "f1m_cue", "tvpb_rg")
    rows = {t: [r for r in ledger.load("m3_prereg") if r.get("tag") == t] for t in tags}
    assert all(len(v) == 1 for v in rows.values()), {k: len(v) for k, v in rows.items()}
    reg = {k: v[0]["metrics"]["files_sha256"] for k, v in rows.items()}
    return dict(states_match_s1_cross_row={os.path.relpath(p, ROOT): reg["s1_cross"].get(os.path.relpath(p, ROOT)) == sha(p)
                                           for p in state_files()},
                rg_files_match_reguide_row={p: reg["reguide"].get(p) == sha(p) for p in RG_FILES},
                f1m_cue_matches_f1m_cue_row={"experiments/f1m_cue.py": reg["f1m_cue"].get("experiments/f1m_cue.py") == sha("experiments/f1m_cue.py")},
                tv_frozen_matches_tvpb_rg_row={"experiments/tv_frozen.py": reg["tvpb_rg"].get("experiments/tv_frozen.py") == sha("experiments/tv_frozen.py")},
                code_matches_s1_cross_row={k: sha(k) == v for k, v in reg["s1_cross"].items()
                                           if (k.startswith("experiments/") or k.startswith("lifelong_ood/") or k.startswith("/"))
                                           and os.path.exists(k if os.path.isabs(k) else os.path.join(ROOT, k))})


def read_model(m, D, ks, T, PA, TFR, NP, RR):
    NEW, KNOWN, PAIR = PA.NEW, PA.KNOWN, PA.PAIR
    comps = NEW + KNOWN

    def stem(arm, c):
        return os.path.join(D, "%s_%s_%d" % (m, arm, T[c]["tid"]))
    A, H = {}, {}
    for arm in READ_ARMS:
        files = [stem(arm, c) for c in comps]
        miss = [f for f in files if not (os.path.exists(f + ".json") and os.path.exists(f + "_track.npz"))]
        if miss:
            A[arm] = "MISSING %d of %d tasks" % (len(miss), len(files))
            continue
        if arm != "I" and not isinstance(A.get("I"), str):
            H[arm] = {}
            for c, f in zip(comps, files):
                diff, ep_ok = TFR.harness(m, stem("I", c), f)
                H[arm][c] = dict(args_diff=diff, episodes_match=ep_ok)
        RATE, succ, per, errs = {}, {}, {}, 0
        for c, f in zip(comps, files):
            R = json.load(open(f + ".json", encoding="utf-8"))
            errs += sum(e.get("error") is not None for v in R["tasks"].values() for e in v["episodes"])
            S, X, G, SU = PA.stage_table(f, T[c], ks)                     # asserts every episode's trial and no error
            RATE[c] = {s: np.array([float(x == s) for x in S]) for s in PA.STAGES}
            succ[c] = np.array(SU, float)
            per[c] = {s: float(RATE[c][s].mean()) for s in PA.STAGES}
        atl = TF.atlas_est(RATE, NEW, PAIR, ks, PA)
        new_m = np.stack([succ[c] for c in NEW])
        kn_m = np.stack([succ[c] for c in KNOWN])
        n46 = np.stack([succ["N4"], succ["N6"]])
        carry = {}
        for c in ("N4", "N6", "K4"):
            rows = NP.load(stem(arm, c), T[c], ks)
            g = [r for r in rows if r.get("lift_held") is not None]
            hi = [r for r in g if r["lift_held"] > CUT]
            lo = [r for r in g if r["lift_held"] <= CUT]
            carry[c] = dict(n=len(rows), grasped=len(g), high=len(hi), high_share=(len(hi) / float(len(g))) if g else None,
                            success_high=(sum(r["success"] for r in hi) / float(len(hi))) if hi else None,
                            success_low=(sum(r["success"] for r in lo) / float(len(lo))) if lo else None)
        diag = None
        if arm == "rgp":
            diag = dict(new=RR.rg_diag([stem(arm, c) + "_rg.json" for c in NEW]),
                        known=RR.rg_diag([stem(arm, c) + "_rg.json" for c in KNOWN]))
        side = None
        if arm in ("tv", "tvpb"):
            side = TF.side_summary([stem(arm, c) + "_tv.json" for c in comps])
        A[arm] = dict(new=new_m, known=kn_m, n46=n46, row=dict(
            success_new=TF.boot1(new_m), success_known=TF.boot1(kn_m), success_n4n6=TF.boot1(n46),
            per_task_success={c: float(succ[c].mean()) for c in comps},
            stage_shares_new={s: float(np.mean([per[c][s] for c in NEW])) for s in PA.STAGES},
            cross_stage_share_new=float(np.mean([per[c]["HANDOFF"] + per[c]["TRANSPORT"] for c in NEW])), per_task_stages=per,
            atlas=atl, episode_errors=errs, carry=carry, rg=diag, side=side))
    row = {arm: (v if isinstance(v, str) else v["row"]) for arm, v in A.items()}

    def dl(x, y, key="new"):
        if isinstance(A.get(x), dict) and isinstance(A.get(y), dict):
            return TF.boot2(A[x][key], A[y][key])
        return None
    Dl = dict(V1_tvpb_minus_I=dl("tvpb", "I"), V2_tvpb_minus_rgp=dl("tvpb", "rgp"), V3_tvpb_minus_I_known=dl("tvpb", "I", "known"),
              tv_minus_I=dl("tv", "I"), tvpb_minus_tv=dl("tvpb", "tv"), rgp_minus_I=dl("rgp", "I"), tv_minus_rgp=dl("tv", "rgp"),
              tvpb_minus_tv_N4N6=dl("tvpb", "tv", "n46"), tvpb_minus_rgp_N4N6=dl("tvpb", "rgp", "n46"))
    Dl["known_minus_I"] = {x: dl(x, "I", "known") for x in READ_ARMS[1:]}
    kI = row["I"]["success_known"][0] if isinstance(row.get("I"), dict) else None
    V = dict(floor_I_known=kI, testable=None if kI is None else bool(kI >= FLOOR),
             V1=None if Dl["V1_tvpb_minus_I"] is None else v1(Dl["V1_tvpb_minus_I"]),
             V2=None if Dl["V2_tvpb_minus_rgp"] is None else v2(Dl["V2_tvpb_minus_rgp"]),
             V3=None if Dl["V3_tvpb_minus_I_known"] is None else v3(Dl["V3_tvpb_minus_I_known"]))
    hok = all(not h["args_diff"] and h["episodes_match"] for hh in H.values() for h in hh.values()) if H else None
    return dict(arms=row, deltas=Dl, verdicts=V, harness=H, harness_matches_I=hok)


def read(a):
    import pp_atlas as PA
    import tv_fresh as TFR
    import n46_phase as NP
    import rg_read as RR
    out = os.path.abspath(a.out)
    assert not os.path.exists(out), "never overwrite a result file: %s" % out
    if a.dry:
        assert not out.startswith(os.path.join(ROOT, "results")), "a dry read never writes into results/"
    ks = [int(x) for x in a.trials.split(",")] if a.trials else TRIALS
    if not a.dry:
        br = json.load(open(BR, encoding="utf-8"))
        assert [int(b["trial"]) for b in br] == ks, "branches do not address trials 24-63"
        assert [int(b["seed_base"]) for b in br] == [13000000 + 1000 * (t + 212) for t in ks], "branch seeds are not k = trial + 212"
    checks = None if a.dry else code_checks()
    n2 = PA.tasks()
    T = {t["compound"]: t for t in n2["tasks"] if t["layout"] == "familiar"}
    D = os.path.abspath(a.dir)
    models = a.models.split("+")
    res = dict(models={}, states=ks, dir=D, dry=bool(a.dry), B=PA.B, mapping=MAPPING, checks=checks, floor=FLOOR, eq=EQ, ni=NI)
    for m in models:
        res["models"][m] = read_model(m, D, ks, T, PA, TFR, NP, RR)
    per = {m: r["verdicts"] for m, r in res["models"].items() if r["verdicts"]["testable"] is not None}
    res["aggregate"], res["testable_models"] = agg(per)
    print("== s1_cross read%s: AGGREGATE %s over testable %s" % (" (DRY)" if a.dry else "", res["aggregate"], res["testable_models"]),
          flush=True)
    if checks:
        for k, v in checks.items():
            bad = [x for x, ok in v.items() if not ok]
            print("   check %s: %d of %d match%s" % (k, len(v) - len(bad), len(v), (" | MISMATCH %s" % bad) if bad else ""), flush=True)
    for m, r in res["models"].items():
        print("== %s: verdicts %s | harness matches I: %s" % (m, json.dumps(r["verdicts"]), r["harness_matches_I"]), flush=True)
        if r["harness"] and not r["harness_matches_I"]:
            for arm, hh in r["harness"].items():
                for c, h in hh.items():
                    if h["args_diff"] or not h["episodes_match"]:
                        print("   HARNESS %s %s %s" % (arm, c, h), flush=True)
        for arm in READ_ARMS:
            x = r["arms"].get(arm)
            if isinstance(x, str) or x is None:
                print("   %-4s %s" % (arm, x), flush=True)
                continue
            g = x["atlas"]["groups"]
            print("   %-4s new %s known %s N4+N6 %s | cross-stage %.3f | within E %s cross E %s -> %s | errors %d" % (
                arm, TF._f(x["success_new"]), TF._f(x["success_known"]), TF._f(x["success_n4n6"]), x["cross_stage_share_new"],
                TF._f([g["within"]["E"]] + g["within"]["ci"]), TF._f([g["cross"]["E"]] + g["cross"]["ci"]), x["atlas"]["verdict"],
                x["episode_errors"]), flush=True)
            print("        per task %s" % json.dumps({c: round(v, 3) for c, v in x["per_task_success"].items()}), flush=True)
            print("        stages new %s" % json.dumps({s: round(v, 3) for s, v in x["stage_shares_new"].items()}), flush=True)
            print("        carry %s" % json.dumps({c: {k: (round(v, 3) if isinstance(v, float) else v) for k, v in y.items()}
                                                   for c, y in x["carry"].items()}), flush=True)
            if x.get("side"):
                print("        side %s" % json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in x["side"].items()
                                                      if k not in ("segmenter_iou", "paint")}), flush=True)
                if x["side"].get("paint"):
                    print("        paint %s" % json.dumps(x["side"]["paint"]), flush=True)
            if x.get("rg"):
                for k in ("new", "known"):
                    print("        rg %s %s" % (k, json.dumps(x["rg"][k])), flush=True)
        for k, v in r["deltas"].items():
            if k != "known_minus_I":
                print("   %s %s" % (k, TF._f(v)), flush=True)
        print("   known - I %s" % {k: TF._f(v) for k, v in r["deltas"]["known_minus_I"].items()}, flush=True)
    tmp = out + ".tmp%d" % os.getpid()
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, default=float)
    os.replace(tmp, out)
    print("wrote", out, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=DIR)
    ap.add_argument("--out", default=READ_OUT)
    ap.add_argument("--trials", default="")
    ap.add_argument("--models", default="+".join(MODELS))
    ap.add_argument("--dry", action="store_true")
    read(ap.parse_args())


if __name__ == "__main__":
    main()
