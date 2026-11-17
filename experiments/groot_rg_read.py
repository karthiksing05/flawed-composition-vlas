"""The registered read of GROOT-RG (m3_prereg tag groot_rg; docs/SCAFFOLD_RESULTS.md 71): ReGuide transport (rg.py, SR 52 / 55 / 65) on
FROZEN GR00T N1.5, scene 1, familiar layout, the new fresh batch k = 500-539 (harness trials 24-63), N1-N6 and K1-K4, K6.  LIBERO venv, CPU.

Arms (all new on these states, results/flawed/groot_rg/groot_<arm>_<tid>, run by experiments/groot_rg_roll.sh, clients unchanged):
    I     intact (pp_atlas_client.py)
    rgp   ReGuide RG-perc  (rg.py client --arm rgp)   PRIMARY
    rgo   ReGuide RG-oracle (rg.py client --arm rgo)   reference (simulator poses, privileged)
Registered mapping: success matrices [task, state]; paired bootstrap B = 2000, numpy default_rng(0), tasks and states drawn jointly
(tv_frozen.boot2, as SR 52 / 55 / 65); CI 2.5 / 97.5.
    F   testability floor (SR 50): TESTABLE iff I's success on K1-K4, K6 >= .70; else NOT TESTABLE (labels still reported).
    V1  dN = rgp - I on N1-N6:   REGUIDE HELPS iff CI lo > 0; else NO CLEAR DIFFERENCE.
    V2  dK = rgp - I on K1-K4, K6:   NON-INFERIOR iff lo > -.05; COST iff hi < 0; else NOT SHOWN (COST takes precedence).
    V3  FIX label (SR 62-69's rule): FIX WORKS iff dN lo > 0 AND dK lo > -.05; PARTIAL iff exactly one holds; FAILS iff neither.
    V4  BAR (SR 54): MEETS BAR iff rgp's N1-N6 success >= .50 AND dN lo > 0; else BELOW BAR.
    V5  carry (F1m), on N4 + N6: dN46 = rgp - I.  CARRY CLEARED iff rgp's N4+N6 success >= .50 AND dN46 lo > 0; CARRY REMAINS iff
        rgp's N4+N6 success < .50; else UNRESOLVED.  (Descriptive beside it: high-carry share, n46_phase lift_held > .22 m among
        grasped episodes, per arm.)
    V6  perception cost, rgp - rgo on N1-N6: PERCEPTION COSTS iff hi < 0; RGP WINS iff lo > 0; EQUIVALENT iff the CI lies within
        +-.10; else NO CLEAR DIFFERENCE (a win/cost label takes precedence).  Descriptive, no verdict weight.
The same V1-V5 labels are reported for rgo (SECONDARY).
Descriptive (no verdict): per-task success; K success per arm and arm - I; the atlas's within / cross-chunk excess per arm
(tv_frozen.atlas_est); stage shares (incl. GRASP, F2); high-carry share on N4 / N6 / N5 / K4 and success after high / low carries;
ReGuide diagnostics (rg_read.rg_diag).
Checks: every arm's client args equal I's except out / port / videos, and per-episode trial / branch / variant / at equal I's
(tv_fresh.harness); every episode's trial is the registered one; no episode errors; state files and code match the groot_rg row; ReGuide's
files match the reguide row.

    python experiments/groot_rg_read.py [--out results/flawed/groot_rg_read.json]
    python experiments/groot_rg_read.py --dry --dir SMOKE_DIR --trials 0,1 --out OUT_OUTSIDE_RESULTS     plumbing check, no ledger
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

MODEL = "groot"
READ_ARMS = ("I", "rgp", "rgo")
DIR = os.path.join(TF.OUT, "groot_rg")
READ_OUT = os.path.join(TF.OUT, "groot_rg_read.json")
TRIALS = list(range(24, 64))
STATES = os.path.join(TF.OUT, "n2_states_gx")
BR = os.path.join(TF.OUT, "n2_branches_gx.json")
TIDS = (16, 18, 20, 22, 24, 26, 0, 2, 4, 6, 10)
EQ = 0.10
FLOOR = 0.70
NI = 0.05
BAR = 0.50
CUT = 0.22
RG_FILES = ("experiments/rg.py", "experiments/rg_stats.py", "experiments/sc_pu.py", "results/flawed/rg_stats.json",
            "results/flawed/sc_pu_calib.json")
MAPPING = ("GR00T N1.5, familiar layout, trials 24-63 (fresh k 500-539); success matrices [task, state]; paired bootstrap B=2000 rng 0, "
           "tasks and states drawn jointly (tv_frozen.boot2); CI 2.5/97.5. F: TESTABLE iff I success on K1-K4,K6 >= .70. V1: rgp - I on "
           "N1-N6: REGUIDE HELPS iff lo > 0, else NO CLEAR DIFFERENCE. V2: rgp - I on K: NON-INFERIOR iff lo > -.05; COST iff hi < 0; "
           "else NOT SHOWN. V3: FIX WORKS iff V1 lo > 0 AND dK lo > -.05; PARTIAL iff exactly one; FAILS iff neither. V4: MEETS BAR iff "
           "rgp N1-N6 >= .50 AND dN lo > 0. V5 (N4+N6): CARRY CLEARED iff rgp N4+N6 >= .50 AND rgp - I lo > 0; CARRY REMAINS iff rgp "
           "N4+N6 < .50; else UNRESOLVED. V6: rgp - rgo on N1-N6: PERCEPTION COSTS iff hi < 0; RGP WINS iff lo > 0; EQUIVALENT iff CI "
           "within +-.10; else NO CLEAR DIFFERENCE. Same V1-V5 for rgo. Descriptive: K per arm, atlas excess, stages, high-carry share, "
           "ReGuide diagnostics.")


def v1(d):
    return "REGUIDE HELPS" if d[1] > 0 else "NO CLEAR DIFFERENCE"


def v2(d):
    if d[2] < 0:
        return "COST"
    return "NON-INFERIOR" if d[1] > -NI else "NOT SHOWN"


def v3(dn, dk):
    h1, h2 = dn[1] > 0, dk[1] > -NI
    return "FIX WORKS" if (h1 and h2) else ("PARTIAL" if (h1 or h2) else "FAILS")


def v4(rate_new, dn):
    return "MEETS BAR" if (rate_new >= BAR and dn[1] > 0) else "BELOW BAR"


def v5(rate46, d46):
    if rate46 >= BAR and d46[1] > 0:
        return "CARRY CLEARED"
    return "CARRY REMAINS" if rate46 < BAR else "UNRESOLVED"


def v6(d):
    if d[2] < 0:
        return "PERCEPTION COSTS"
    if d[1] > 0:
        return "RGP WINS"
    return "EQUIVALENT" if (d[1] >= -EQ and d[2] <= EQ) else "NO CLEAR DIFFERENCE"


def sha(p):
    with open(p if os.path.isabs(p) else os.path.join(ROOT, p), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def state_files():
    return [os.path.join(STATES, "states.npy")] + [os.path.join(STATES, "n2_goal_%d.npy" % t) for t in TIDS] + [BR]


def code_checks():
    from lifelong_ood import ledger
    rows = {t: [r for r in ledger.load("m3_prereg") if r.get("tag") == t] for t in ("groot_rg", "reguide")}
    assert all(len(v) == 1 for v in rows.values()), {k: len(v) for k, v in rows.items()}
    reg = {k: v[0]["metrics"]["files_sha256"] for k, v in rows.items()}
    return dict(states_match_groot_rg_row={os.path.relpath(p, ROOT): reg["groot_rg"].get(os.path.relpath(p, ROOT)) == sha(p)
                                           for p in state_files()},
                rg_files_match_reguide_row={p: reg["reguide"].get(p) == sha(p) for p in RG_FILES},
                code_matches_groot_rg_row={k: sha(k) == v for k, v in reg["groot_rg"].items()
                                           if (k.startswith("experiments/") or k.startswith("lifelong_ood/") or k.startswith("/"))
                                           and os.path.exists(k if os.path.isabs(k) else os.path.join(ROOT, k))})


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
        assert [int(b["seed_base"]) for b in br] == [13000000 + 1000 * (t + 476) for t in ks], "branch seeds are not k = trial + 476"
    checks = None if a.dry else code_checks()
    n2 = PA.tasks()
    T = {t["compound"]: t for t in n2["tasks"] if t["layout"] == "familiar"}
    NEW, KNOWN, PAIR = PA.NEW, PA.KNOWN, PA.PAIR
    comps = NEW + KNOWN
    D = os.path.abspath(a.dir)

    def stem(arm, c):
        return os.path.join(D, "%s_%s_%d" % (MODEL, arm, T[c]["tid"]))
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
                diff, ep_ok = TFR.harness(MODEL, stem("I", c), f)
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
        for c in ("N4", "N6", "N5", "K4"):
            rows = NP.load(stem(arm, c), T[c], ks)
            g = [r for r in rows if r.get("lift_held") is not None]
            hi = [r for r in g if r["lift_held"] > CUT]
            lo = [r for r in g if r["lift_held"] <= CUT]
            carry[c] = dict(n=len(rows), grasped=len(g), high=len(hi), high_share=(len(hi) / float(len(g))) if g else None,
                            success_high=(sum(r["success"] for r in hi) / float(len(hi))) if hi else None,
                            success_low=(sum(r["success"] for r in lo) / float(len(lo))) if lo else None,
                            lift_med=float(np.median([r["lift_held"] for r in g])) if g else None)
        g46 = [carry["N4"], carry["N6"]]
        gr, hi = sum(x["grasped"] for x in g46), sum(x["high"] for x in g46)
        carry["N4+N6"] = dict(grasped=gr, high=hi, high_share=(hi / float(gr)) if gr else None)
        diag = None
        if arm in ("rgp", "rgo"):
            diag = dict(new=RR.rg_diag([stem(arm, c) + "_rg.json" for c in NEW]),
                        known=RR.rg_diag([stem(arm, c) + "_rg.json" for c in KNOWN]))
        A[arm] = dict(new=new_m, known=kn_m, n46=n46, row=dict(
            success_new=TF.boot1(new_m), success_known=TF.boot1(kn_m), success_n4n6=TF.boot1(n46),
            per_task_success={c: float(succ[c].mean()) for c in comps},
            stage_shares_new={s: float(np.mean([per[c][s] for c in NEW])) for s in PA.STAGES},
            cross_stage_share_new=float(np.mean([per[c]["HANDOFF"] + per[c]["TRANSPORT"] for c in NEW])), per_task_stages=per,
            atlas=atl, episode_errors=errs, carry=carry, rg=diag))
    row = {arm: (v if isinstance(v, str) else v["row"]) for arm, v in A.items()}

    def dl(x, y, key="new"):
        if isinstance(A.get(x), dict) and isinstance(A.get(y), dict):
            return TF.boot2(A[x][key], A[y][key])
        return None
    Dl = dict(V1_rgp_minus_I=dl("rgp", "I"), V2_rgp_minus_I_known=dl("rgp", "I", "known"), V5_rgp_minus_I_N4N6=dl("rgp", "I", "n46"),
              V6_rgp_minus_rgo=dl("rgp", "rgo"), rgo_minus_I=dl("rgo", "I"), rgo_minus_I_known=dl("rgo", "I", "known"),
              rgo_minus_I_N4N6=dl("rgo", "I", "n46"), rgp_minus_rgo_known=dl("rgp", "rgo", "known"),
              rgp_minus_rgo_N4N6=dl("rgp", "rgo", "n46"))
    kI = row["I"]["success_known"][0] if isinstance(row.get("I"), dict) else None
    V = dict(floor_I_known=kI, testable=None if kI is None else bool(kI >= FLOOR))
    for arm in ("rgp", "rgo"):
        if isinstance(A.get(arm), dict) and isinstance(A.get("I"), dict):
            dn, dk, d46 = dl(arm, "I"), dl(arm, "I", "known"), dl(arm, "I", "n46")
            V[arm] = dict(V1=v1(dn), V2=v2(dk), V3=v3(dn, dk), V4=v4(A[arm]["row"]["success_new"][0], dn),
                          V5=v5(A[arm]["row"]["success_n4n6"][0], d46))
    V["V6"] = None if Dl["V6_rgp_minus_rgo"] is None else v6(Dl["V6_rgp_minus_rgo"])
    hok = all(not h["args_diff"] and h["episodes_match"] for hh in H.values() for h in hh.values()) if H else None
    res = dict(model=MODEL, states=ks, dir=D, dry=bool(a.dry), B=PA.B, mapping=MAPPING, arms=row, deltas=Dl, verdicts=V, harness=H,
               harness_matches_I=hok, checks=checks)
    print("== groot_rg read%s: verdicts %s | harness matches I: %s" % (" (DRY)" if a.dry else "", json.dumps(V), hok), flush=True)
    if checks:
        for k, v in checks.items():
            bad = [x for x, ok in v.items() if not ok]
            print("   check %s: %d of %d match%s" % (k, len(v) - len(bad), len(v), (" | MISMATCH %s" % bad) if bad else ""), flush=True)
    if H and not hok:
        for arm, hh in H.items():
            for c, h in hh.items():
                if h["args_diff"] or not h["episodes_match"]:
                    print("   HARNESS %s %s %s" % (arm, c, h), flush=True)
    for arm in READ_ARMS:
        x = row.get(arm)
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
        if x.get("rg"):
            for k in ("new", "known"):
                print("        rg %s %s" % (k, json.dumps(x["rg"][k])), flush=True)
    for k, v in Dl.items():
        print("   %s %s" % (k, TF._f(v)), flush=True)
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
    ap.add_argument("--dry", action="store_true")
    read(ap.parse_args())


if __name__ == "__main__":
    main()
