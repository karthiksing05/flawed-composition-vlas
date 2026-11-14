"""TV-G: do test-time silhouette role views (tv) help FROZEN GR00T N1 on new pairings N1-N6, and do they beat an AC-VLA-style
whole-wrist grey (w) at a matched prompt (both arms keep the task's own instruction; no role phrases)?  m3_prereg tv_g;
docs/SCAFFOLD_RESULTS.md 49.

SR 46 (main states): GR00T I .167, tv .400 (TV WINS), w .158.  SR 47 (20 fresh states): tv - I +.183 [-.008, +.425], UNRESOLVED.
This read runs all three arms on a second, larger fresh batch: 60 familiar-layout states k = 56-115 (experiments/fc_n2_fresh2.py,
results/flawed/n2_states_fresh2, results/flawed/n2_branches_fresh2.json; harness trials 24-83, k = trial + 32), never used by any
earlier read.

Rollouts: experiments/tv_g_roll.sh -> results/flawed/tv_g/groot_<arm>_<tid>.  I = experiments/pp_atlas_client.py (the atlas's
intact client, unchanged); tv, w = experiments/tv_frozen.py client (unchanged).  Same server and client flags as SR 47.
This file is the read only.  tv_frozen.py's estimators (boot1, boot2, atlas_est, side_summary), tv_fresh.py's harness check and
pp_atlas.py's stage_table are imported, not copied; nothing in any of them is edited.

Registered mapping (m3_prereg tag tv_g): GR00T, familiar layout, trials 24-83; success matrices [task, state]; paired bootstrap
B = 2000, numpy default_rng(0), tasks and states drawn jointly (tv_frozen.boot2); CI = 2.5 / 97.5 percentiles; on N1-N6:
    PRIMARY    d = tv - I:  TV WINS iff lo > 0;  else NO CLEAR DIFFERENCE.
    SECONDARY  d = tv - w:  SEPARATED iff lo > 0;  W EQUIVALENT iff lo > -.10 and hi < .10;  else NOT SEPARATED.
Descriptive (no verdict): every arm's success; w - I on N1-N6; K1-K4, K6 success per arm and arm - I there (known-task cost); the
atlas's within / cross-chunk excess per arm (pp_atlas estimator); stage shares; detector / segmenter / paint diagnostics; tv - I
pooled over SR 47's 20 fresh states and these 60 (80 states; SR 47's I and tv rollouts, unchanged).
Checks (reported): every arm's client args equal I's except out / port / videos, and per-episode trial / branch / variant / at
equal I's (tv_fresh.harness); branches address trials 24-83 with seed_base 13000000 + 1000 (trial + 32); the state / branch files
match the hashes in the tv_g prereg row; SR 47's reused rollouts match the hashes in the row and reproduce tv_fresh_read.json.

    LIBERO venv, CPU:  python experiments/tv_g.py read [--out results/flawed/tv_g_read.json]
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT]
import pp_atlas as PA                                                       # noqa: E402
import tv_frozen as TF                                                      # noqa: E402
import tv_fresh as TFR                                                      # noqa: E402

MODEL = "groot"
TRIALS = list(range(24, 84))
SHIFT = 32
EPS = 0.10
OUT = PA.OUT
DIR = os.path.join(OUT, "tv_g")
READ = os.path.join(OUT, "tv_g_read.json")
STATES = os.path.join(OUT, "n2_states_fresh2")
BR = os.path.join(OUT, "n2_branches_fresh2.json")
TIDS = TFR.TIDS
ARMS = ("I", "tv", "w")
MAPPING = ("GR00T, familiar layout, trials 24-83 (fresh k 56-115); success matrices [task, state]; paired bootstrap B=2000 rng 0, "
           "tasks and states drawn jointly (tv_frozen.boot2); CI 2.5/97.5; on N1-N6. PRIMARY d = tv - I: TV WINS iff lo > 0; else NO "
           "CLEAR DIFFERENCE. SECONDARY d = tv - w: SEPARATED iff lo > 0; W EQUIVALENT iff lo > -.10 and hi < .10; else NOT SEPARATED. "
           "Descriptive: arm success, w - I, K cost per arm, atlas excess per arm, stage shares, diagnostics, tv - I pooled with SR 47.")


def primary(d):
    return "TV WINS" if d[1] > 0 else "NO CLEAR DIFFERENCE"


def secondary(d):
    if d[1] > 0:
        return "SEPARATED"
    if d[1] > -EPS and d[2] < EPS:
        return "W EQUIVALENT"
    return "NOT SEPARATED"


def state_files():
    return [os.path.join(STATES, "states.npy")] + [os.path.join(STATES, "n2_goal_%d.npy" % t) for t in TIDS] + [BR]


def reused_files():
    out = []
    for t in TIDS:
        for suf in (".json", "_track.npz"):
            out.append(os.path.join(TFR.I_DIR, "%s_%d%s" % (MODEL, t, suf)))
            out.append(os.path.join(TFR.DIR, "%s_tv_%d%s" % (MODEL, t, suf)))
    return out


def arm_tables(files, comps, T, ks):
    RATE, succ, per = {}, {}, {}
    for c, f in zip(comps, files):
        S, X, G, SU = PA.stage_table(f, T[c], ks)
        RATE[c] = {s: np.array([float(x == s) for x in S]) for s in PA.STAGES}
        succ[c] = np.array(SU, float)
        per[c] = {s: float(RATE[c][s].mean()) for s in PA.STAGES}
    return RATE, succ, per


def read(a):
    from lifelong_ood import ledger
    out = os.path.abspath(a.out)
    assert not os.path.exists(out), "never overwrite a result file: %s" % out
    br = json.load(open(BR, encoding="utf-8"))
    br_ok = ([int(b["trial"]) for b in br] == TRIALS
             and [int(b["seed_base"]) for b in br] == [13000000 + 1000 * (t + SHIFT) for t in TRIALS])
    assert br_ok, "branches do not address trials 24-83 with the seed rule"
    rows = [r for r in ledger.load("m3_prereg") if r.get("tag") == "tv_g"]
    assert len(rows) == 1, "expected exactly one tv_g prereg row, found %d" % len(rows)
    reg = rows[0]["metrics"]["files_sha256"]
    files_ok = {TFR.rel(p): reg.get(TFR.rel(p)) == TFR.sha(p) for p in state_files()}
    reused_ok = {TFR.rel(p): reg.get(TFR.rel(p)) == TFR.sha(p) for p in reused_files()}
    n2 = PA.tasks()
    T = {t["compound"]: t for t in n2["tasks"] if t["layout"] == "familiar"}
    NEW, KNOWN, PAIR = PA.NEW, PA.KNOWN, PA.PAIR
    comps = NEW + KNOWN
    assert sorted(T[c]["tid"] for c in comps) == sorted(TIDS)
    d_out = os.path.abspath(a.dir)
    A, H = {}, {}
    for arm in ARMS:
        files = [os.path.join(d_out, "%s_%s_%d" % (MODEL, arm, T[c]["tid"])) for c in comps]
        miss = [f for f in files if not (os.path.exists(f + ".json") and os.path.exists(f + "_track.npz"))]
        if miss:
            A[arm] = "MISSING %d of %d tasks" % (len(miss), len(files))
            continue
        if arm != "I":
            H[arm] = {}
            for c, f in zip(comps, files):
                diff, ep_ok = TFR.harness(MODEL, os.path.join(d_out, "%s_I_%d" % (MODEL, T[c]["tid"])), f)
                H[arm][c] = dict(args_diff=diff, episodes_match=ep_ok)
        RATE, succ, per = arm_tables(files, comps, T, TRIALS)
        atl = TF.atlas_est(RATE, NEW, PAIR, TRIALS, PA)
        new_m = np.stack([succ[c] for c in NEW])
        kn_m = np.stack([succ[c] for c in KNOWN])
        A[arm] = dict(new=new_m, known=kn_m, row=dict(
            success_new=TF.boot1(new_m), success_known=TF.boot1(kn_m), per_task_success={c: float(succ[c].mean()) for c in comps},
            stage_shares_new={s: float(np.mean([per[c][s] for c in NEW])) for s in PA.STAGES},
            grasp_ok_new=1.0 - float(np.mean([per[c]["GRASP"] for c in NEW])),
            cross_stage_share_new=float(np.mean([per[c]["HANDOFF"] + per[c]["TRANSPORT"] for c in NEW])), per_task_stages=per, atlas=atl,
            side=None if arm == "I" else TF.side_summary([f + "_tv.json" for f in files])))
    row = {arm: (v if isinstance(v, str) else v["row"]) for arm, v in A.items()}

    def dl(x, y, key="new"):
        if isinstance(A.get(x), dict) and isinstance(A.get(y), dict):
            return TF.boot2(A[x][key], A[y][key])
        return None
    D = dict(tv_minus_I=dl("tv", "I"), tv_minus_w=dl("tv", "w"), w_minus_I=dl("w", "I"))
    D["known"] = {"%s_minus_I" % x: dl(x, "I", "known") for x in ("tv", "w")}
    V = dict(primary_tv_minus_I=None if D["tv_minus_I"] is None else primary(D["tv_minus_I"]),
             secondary_tv_minus_w=None if D["tv_minus_w"] is None else secondary(D["tv_minus_w"]))
    # SR 47's fresh GR00T I and tv (trials 24-43), reused: reproduce tv_fresh_read, then pool with this batch (descriptive)
    ks47 = TFR.FRESH_TRIALS
    P47 = {}
    for arm, (d, st) in dict(I=(TFR.I_DIR, MODEL + "_{tid}"), tv=(TFR.DIR, MODEL + "_tv_{tid}")).items():
        files = [os.path.join(d, st.format(tid=T[c]["tid"])) for c in comps]
        _, succ, _ = arm_tables(files, comps, T, ks47)
        P47[arm] = dict(new=np.stack([succ[c] for c in NEW]), known=np.stack([succ[c] for c in KNOWN]))
    fr = json.load(open(TFR.READ, encoding="utf-8"))["models"][MODEL]["arms"]
    rep47 = {arm: (abs(float(P47[arm]["new"].mean()) - fr[arm]["success_new"][0]) < 1e-9
                   and abs(float(P47[arm]["known"].mean()) - fr[arm]["success_known"][0]) < 1e-9) for arm in P47}
    pooled = None
    if isinstance(A.get("I"), dict) and isinstance(A.get("tv"), dict):
        pI = np.concatenate([P47["I"]["new"], A["I"]["new"]], 1)
        pT = np.concatenate([P47["tv"]["new"], A["tv"]["new"]], 1)
        pooled = dict(states=pI.shape[1], I=TF.boot1(pI), tv=TF.boot1(pT), tv_minus_I=TF.boot2(pT, pI))
    hok = all(not h["args_diff"] and h["episodes_match"] for hh in H.values() for h in hh.values()) if H else None
    res = dict(model=MODEL, states=TRIALS, k=[t + SHIFT for t in TRIALS], dir=d_out, B=PA.B, eps=EPS, mapping=MAPPING, arms=row,
               deltas=D, verdicts=V, harness=H, harness_matches_I=hok, branches_ok=br_ok, state_files_match_prereg=files_ok,
               sr47_rollouts_match_prereg=reused_ok, sr47_reproduces_tv_fresh_read=rep47, pooled_with_sr47_descriptive=pooled)
    print("== %s: verdicts %s | harness matches I: %s | branches ok: %s" % (MODEL, V, hok, br_ok), flush=True)
    print("   state files match prereg: %s | SR 47 rollouts match prereg: %d of %d | SR 47 reproduces tv_fresh_read: %s" % (
        all(files_ok.values()), sum(reused_ok.values()), len(reused_ok), rep47), flush=True)
    if H and not hok:
        for arm, hh in H.items():
            for c, h in hh.items():
                if h["args_diff"] or not h["episodes_match"]:
                    print("   HARNESS %s %s %s" % (arm, c, h), flush=True)
    for arm in ARMS:
        r = row.get(arm)
        if isinstance(r, str) or r is None:
            print("   %-3s %s" % (arm, r), flush=True)
            continue
        g = r["atlas"]["groups"]
        print("   %-3s new %s known %s | grasp ok %.3f cross-stage %.3f | within E %s cross E %s -> %s" % (
            arm, TF._f(r["success_new"]), TF._f(r["success_known"]), r["grasp_ok_new"], r["cross_stage_share_new"],
            TF._f([g["within"]["E"]] + g["within"]["ci"]), TF._f([g["cross"]["E"]] + g["cross"]["ci"]), r["atlas"]["verdict"]), flush=True)
        print("        per task %s" % json.dumps({c: round(v, 3) for c, v in r["per_task_success"].items()}), flush=True)
        print("        stages new %s" % json.dumps({s: round(v, 3) for s, v in r["stage_shares_new"].items()}), flush=True)
        if r.get("side"):
            print("        side %s" % json.dumps({k: (TF._r(v) if isinstance(v, float) else v) for k, v in r["side"].items()
                                                  if k not in ("segmenter_iou", "paint")}), flush=True)
            if r["side"].get("paint"):
                print("        paint %s seg %s" % (json.dumps(r["side"]["paint"]), json.dumps(r["side"]["segmenter_iou"])), flush=True)
    for k, v in D.items():
        if k != "known":
            print("   %s %s" % (k, TF._f(v)), flush=True)
    print("   known %s" % {k: TF._f(v) for k, v in D["known"].items()}, flush=True)
    print("   pooled with SR 47 (descriptive): %s" % (None if pooled is None else {k: (v if k == "states" else TF._f(v))
                                                                                 for k, v in pooled.items()}), flush=True)
    tmp = out + ".tmp%d" % os.getpid()
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    os.replace(tmp, out)
    print("wrote", out, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("read",))
    ap.add_argument("--dir", default=DIR)
    ap.add_argument("--out", default=READ)
    read(ap.parse_args())


if __name__ == "__main__":
    main()
