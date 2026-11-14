"""The registered read of ReGuide on frozen pi0.5 (m3_prereg tag reguide; docs/SCAFFOLD_RESULTS.md 52).  LIBERO venv, CPU.

Arms, pi0.5, familiar layout, fresh states k = 36-55 (harness trials 24-43), N1-N6 and K1-K4, K6:
    I     results/flawed/pp_atlas_fresh/pi05_<tid>            (reused; the atlas replication's intact rollouts)
    tvp   results/flawed/tv_fresh/pi05_tvp_<tid>             (reused; our frozen front end TV+P, SR 47)
    rgo   results/flawed/rg/pi05_rgo_<tid>                   (new; ReGuide with simulator poses: PRIVILEGED)
    rgp   results/flawed/rg/pi05_rgp_<tid>                   (new; ReGuide with learned-segmenter + depth poses)
Registered mapping: success matrices [task, state]; paired bootstrap B = 2000, numpy default_rng(0), tasks and states drawn jointly
(tv_frozen.boot2); CI 2.5 / 97.5; on N1-N6:
    A  d = rgp - I:    RG HELPS iff CI lo > 0;  RG HURTS iff CI hi < 0;  else NO CLEAR DIFFERENCE.
    B  d = tvp - rgp:  FRONT END WINS iff CI lo > 0;  RG WINS iff CI hi < 0;  else NO CLEAR DIFFERENCE.
Descriptive (no verdict): rgo - I, tvp - rgo, rgo - rgp on N1-N6; K1-K4, K6 success per arm and arm - I (trained-task cost); the
atlas's within / cross-chunk excess per arm (pp_atlas estimator via tv_frozen.atlas_est); stage shares; ReGuide's own diagnostics
(transports, triggers, releases, clamped steps, registration error of rgp).
Checks (reported): every arm's client args equal I's except out / port / videos, and per-episode trial / branch / variant / at equal
I's (tv_fresh.harness); I reproduces pp_atlas_fresh_read.json; tvp reproduces tv_fresh_read.json (success and cross E); the state
files match tv_fresh's prereg row; the reused rollouts and rg_stats.json match the hashes in the reguide row.

    python experiments/rg_read.py [--out results/flawed/rg_read.json]
"""
import argparse
import collections
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT]
import tv_frozen as TF                                                      # noqa: E402

MODEL = "pi05"
READ_ARMS = ("I", "tvp", "rgo", "rgp")
RG_DIR = os.path.join(TF.OUT, "rg")


def lab_a(d):
    return "RG HELPS" if d[1] > 0 else ("RG HURTS" if d[2] < 0 else "NO CLEAR DIFFERENCE")


def lab_b(d):
    return "FRONT END WINS" if d[1] > 0 else ("RG WINS" if d[2] < 0 else "NO CLEAR DIFFERENCE")


MAPPING = ("pi0.5, familiar layout, trials 24-43 (fresh k 36-55); success matrices [task, state]; paired bootstrap B=2000 rng 0, tasks "
           "and states drawn jointly (tv_frozen.boot2); CI 2.5/97.5; on N1-N6. A: d = rgp - I: RG HELPS iff lo > 0; RG HURTS iff "
           "hi < 0; else NO CLEAR DIFFERENCE. B: d = tvp - rgp: FRONT END WINS iff lo > 0; RG WINS iff hi < 0; else NO CLEAR "
           "DIFFERENCE. Descriptive: rgo - I, tvp - rgo, rgo - rgp; K cost per arm; atlas excess per arm; RG diagnostics.")


def rg_diag(files):
    eps = []
    for f in files:
        if os.path.exists(f):
            eps += json.load(open(f, encoding="utf-8"))["eps"]
    if not eps:
        return None
    why = collections.Counter()
    ends = collections.Counter()
    for e in eps:
        for v in e["events"]:
            if v["kind"] == "start":
                why[str(v.get("why")).split(" ->")[0].split(" %")[0]] += 1
            if v["kind"] == "end":
                ends[str(v.get("why"))] += 1
    out = dict(episodes=len(eps), any_transport=float(np.mean([e["transports"] > 0 for e in eps])),
               transports_mean=float(np.mean([e["transports"] for e in eps])), releases_mean=float(np.mean([e["releases"] for e in eps])),
               clamped_mean=float(np.mean([e["clamped"] for e in eps])), rg_steps_mean=float(np.mean([e["steps_rg"] for e in eps])),
               starts=dict(why), ends=dict(ends), chunk_len=sorted({c for e in eps for c in e["chunk_len"]}))
    regs = [e["reg"] for e in eps if e.get("reg")]
    if regs:
        names = list(regs[0]["err_cm"])
        out["reg_err_cm_median"] = {n: float(np.median([r["err_cm"][n] for r in regs])) for n in names}
        out["reg_err_cm_p90"] = {n: float(np.percentile([r["err_cm"][n] for r in regs], 90)) for n in names}
        out["reg_src"] = dict(collections.Counter(s for r in regs for s in r["src"].values()))
    return out


def read(a):
    import pp_atlas as PA
    import tv_fresh as TFR
    from lifelong_ood import ledger
    out = os.path.abspath(a.out)
    assert not os.path.exists(out), "never overwrite a result file: %s" % out
    br = json.load(open(TFR.BR, encoding="utf-8"))
    assert [int(b["trial"]) for b in br] == TFR.FRESH_TRIALS
    rows = {t: [r for r in ledger.load("m3_prereg") if r.get("tag") == t] for t in ("tv_fresh", "reguide")}
    assert len(rows["tv_fresh"]) == 1 and len(rows["reguide"]) == 1, {k: len(v) for k, v in rows.items()}
    reg_f = rows["tv_fresh"][0]["metrics"]["files_sha256"]
    reg_r = rows["reguide"][0]["metrics"]["files_sha256"]
    files_ok = {TFR.rel(p): reg_f.get(TFR.rel(p)) == TFR.sha(p) for p in TFR.state_files()}
    hashed_ok = {k: TFR.sha(os.path.join(ROOT, k)) == v for k, v in reg_r.items()
                 if k.startswith("results/flawed/tv_fresh/") or k.startswith("results/flawed/pp_atlas_fresh/") or k.endswith("rg_stats.json")}
    n2 = PA.tasks()
    ks = TFR.FRESH_TRIALS
    T = {t["compound"]: t for t in n2["tasks"] if t["layout"] == "familiar"}
    NEW, KNOWN, PAIR = PA.NEW, PA.KNOWN, PA.PAIR
    comps = NEW + KNOWN
    D_RG = os.path.abspath(a.dir)
    where = dict(I=(TFR.I_DIR, MODEL + "_{tid}"), tvp=(TFR.DIR, MODEL + "_tvp_{tid}"),
                 rgo=(D_RG, MODEL + "_rgo_{tid}"), rgp=(D_RG, MODEL + "_rgp_{tid}"))
    A, H = {}, {}
    for arm in READ_ARMS:
        d, st = where[arm]
        files = [os.path.join(d, st.format(tid=T[c]["tid"])) for c in comps]
        miss = [f for f in files if not (os.path.exists(f + ".json") and os.path.exists(f + "_track.npz"))]
        if miss:
            A[arm] = "MISSING %d of %d tasks" % (len(miss), len(files))
            continue
        if arm != "I":
            H[arm] = {}
            for c, f in zip(comps, files):
                diff, ep_ok = TFR.harness(MODEL, os.path.join(TFR.I_DIR, "%s_%d" % (MODEL, T[c]["tid"])), f)
                H[arm][c] = dict(args_diff=diff, episodes_match=ep_ok)
        RATE, succ, per, errs = {}, {}, {}, 0
        for c, f in zip(comps, files):
            R = json.load(open(f + ".json", encoding="utf-8"))
            errs += sum(e.get("error") is not None for v in R["tasks"].values() for e in v["episodes"])
            S, X, G, SU = PA.stage_table(f, T[c], ks)
            RATE[c] = {s: np.array([float(x == s) for x in S]) for s in PA.STAGES}
            succ[c] = np.array(SU, float)
            per[c] = {s: float(RATE[c][s].mean()) for s in PA.STAGES}
        atl = TF.atlas_est(RATE, NEW, PAIR, ks, PA)
        new_m = np.stack([succ[c] for c in NEW])
        kn_m = np.stack([succ[c] for c in KNOWN])
        diag = None
        if arm in ("rgo", "rgp"):
            diag = rg_diag([f + "_rg.json" for f in files])
            diag_new = rg_diag([os.path.join(d, st.format(tid=T[c]["tid"])) + "_rg.json" for c in NEW])
            diag_known = rg_diag([os.path.join(d, st.format(tid=T[c]["tid"])) + "_rg.json" for c in KNOWN])
            diag = dict(all=diag, new=diag_new, known=diag_known)
        A[arm] = dict(new=new_m, known=kn_m, row=dict(
            success_new=TF.boot1(new_m), success_known=TF.boot1(kn_m), per_task_success={c: float(succ[c].mean()) for c in comps},
            stage_shares_new={s: float(np.mean([per[c][s] for c in NEW])) for s in PA.STAGES},
            grasp_ok_new=1.0 - float(np.mean([per[c]["GRASP"] for c in NEW])),
            cross_stage_share_new=float(np.mean([per[c]["HANDOFF"] + per[c]["TRANSPORT"] for c in NEW])), per_task_stages=per, atlas=atl,
            episode_errors=errs, rg=diag))
    row = {arm: (v if isinstance(v, str) else v["row"]) for arm, v in A.items()}

    def dl(x, y, key="new"):
        if isinstance(A.get(x), dict) and isinstance(A.get(y), dict):
            return TF.boot2(A[x][key], A[y][key])
        return None
    D = dict(rgp_minus_I=dl("rgp", "I"), tvp_minus_rgp=dl("tvp", "rgp"), rgo_minus_I=dl("rgo", "I"), tvp_minus_rgo=dl("tvp", "rgo"),
             rgo_minus_rgp=dl("rgo", "rgp"), tvp_minus_I=dl("tvp", "I"))
    D["known"] = {"%s_minus_I" % x: dl(x, "I", "known") for x in READ_ARMS[1:]}
    V = dict(A=None if D["rgp_minus_I"] is None else lab_a(D["rgp_minus_I"]),
             B=None if D["tvp_minus_rgp"] is None else lab_b(D["tvp_minus_rgp"]))
    ref = json.load(open(TFR.ATLAS_READ, encoding="utf-8"))
    chk = None
    if isinstance(A.get("I"), dict):
        g0, g1 = ref[MODEL]["familiar"]["groups"], A["I"]["row"]["atlas"]["groups"]
        chk = all(abs(g0[g]["E"] - g1[g]["E"]) < 1e-9 and all(abs(u - v) < 1e-9 for u, v in zip(g0[g]["ci"], g1[g]["ci"])) for g in g0)
    fr = json.load(open(TFR.READ, encoding="utf-8"))["models"][MODEL]["arms"]
    rep = {}
    for arm in ("I", "tvp"):
        if isinstance(row.get(arm), dict) and isinstance(fr.get(arm), dict):
            rep[arm] = (abs(row[arm]["success_new"][0] - fr[arm]["success_new"][0]) < 1e-9
                        and abs(row[arm]["success_known"][0] - fr[arm]["success_known"][0]) < 1e-9
                        and abs(row[arm]["atlas"]["groups"]["cross"]["E"] - fr[arm]["atlas"]["groups"]["cross"]["E"]) < 1e-9)
    hok = all(not h["args_diff"] and h["episodes_match"] for hh in H.values() for h in hh.values()) if H else None
    res = dict(model=MODEL, states=ks, dir=D_RG, B=PA.B, mapping=MAPPING, arms=row, deltas=D, verdicts=V, harness=H,
               harness_matches_I=hok, I_reproduces_pp_atlas_fresh_read=chk, tvp_reproduces_tv_fresh_read=rep,
               state_files_match_tv_fresh_prereg=files_ok, hashed_files_match_reguide_prereg=hashed_ok)
    print("== %s: verdicts %s | harness matches I: %s | I reproduces pp_atlas_fresh_read: %s | tvp reproduces tv_fresh_read: %s" % (
        MODEL, V, hok, chk, rep), flush=True)
    print("   state files match tv_fresh prereg: %s | hashed files match reguide prereg: %s of %s" % (
        all(files_ok.values()), sum(hashed_ok.values()), len(hashed_ok)), flush=True)
    if H and not hok:
        for arm, hh in H.items():
            for c, h in hh.items():
                if h["args_diff"] or not h["episodes_match"]:
                    print("   HARNESS %s %s %s" % (arm, c, h), flush=True)
    for arm in READ_ARMS:
        r = row.get(arm)
        if isinstance(r, str) or r is None:
            print("   %-4s %s" % (arm, r), flush=True)
            continue
        g = r["atlas"]["groups"]
        print("   %-4s new %s known %s | grasp ok %.3f cross-stage %.3f | within E %s cross E %s -> %s | errors %d" % (
            arm, TF._f(r["success_new"]), TF._f(r["success_known"]), r["grasp_ok_new"], r["cross_stage_share_new"],
            TF._f([g["within"]["E"]] + g["within"]["ci"]), TF._f([g["cross"]["E"]] + g["cross"]["ci"]), r["atlas"]["verdict"],
            r["episode_errors"]), flush=True)
        print("        per task %s" % json.dumps({c: round(v, 3) for c, v in r["per_task_success"].items()}), flush=True)
        print("        stages new %s" % json.dumps({s: round(v, 3) for s, v in r["stage_shares_new"].items()}), flush=True)
        if r.get("rg"):
            for k in ("new", "known"):
                print("        rg %s %s" % (k, json.dumps(r["rg"][k])), flush=True)
    for k, v in D.items():
        if k != "known":
            print("   %s %s" % (k, TF._f(v)), flush=True)
    print("   known %s" % {k: TF._f(v) for k, v in D["known"].items()}, flush=True)
    tmp = out + ".tmp%d" % os.getpid()
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    os.replace(tmp, out)
    print("wrote", out, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=RG_DIR)
    ap.add_argument("--out", default=os.path.join(TF.OUT, "rg_read.json"))
    read(ap.parse_args())


if __name__ == "__main__":
    main()
