"""TV-F on FRESH states: the confirmatory read of test-time role views (+ R1's role phrases) on frozen VLAs (m3_prereg tv_fresh;
docs/SCAFFOLD_RESULTS.md 47).  SR 46 (experiments/tv_frozen.py, main states 4-23) found pi0.5 tvp .925 against intact .50 with no
registered verdict; this read registers it on the 20 fresh familiar-layout states of the atlas replication (k = 36-55 = harness
trials 24-43; experiments/fc_n2_fresh.py, results/flawed/n2_states_fresh, results/flawed/n2_branches_fresh.json).

Rollouts: experiments/tv_frozen.py client, unchanged, launched by experiments/tv_fresh_roll.sh with --fresh n2_states_fresh and
--branches n2_branches_fresh.json (the only change against tv_roll.sh, and the same move pp_atlas_fresh_roll.sh made for I).
Arms: pi05 tvp, pi05 tv, groot tv -> results/flawed/tv_fresh/<model>_<arm>_<tid>.  I = results/flawed/pp_atlas_fresh/<model>_<tid>
(the atlas replication's intact rollouts: same states, branches, client seed, server command).
This file is the read only.  tv_frozen.py's estimators (boot1, boot2, atlas_est, side_summary) and pp_atlas.py's stage_table are
imported, not copied; nothing in either file is edited.

Registered mapping (per model, familiar layout, trials 24-43; success matrices [task, state]; paired bootstrap B = 2000, numpy
default_rng(0), tasks and states drawn jointly with replacement; CI = 2.5 / 97.5 percentiles; eps = .10):
    PRIMARY    pi05  tvp - I on N1-N6:  CONFIRMED iff d > .10 and lo > 0;  REFUTED iff hi < .10;  else UNRESOLVED.
    GUARD      pi05  tvp - I on K1-K4, K6:  NO COST iff lo > -.10;  else COST NOT EXCLUDED.
    SECONDARY  groot tv - I on N1-N6 and pi05 tv - I on N1-N6:  the primary's rule.
Descriptive (no verdict): every arm's success; pi05 tvp - tv; tv - I on K tasks; the atlas's within / cross-chunk excess per arm
(pp_atlas estimator, partners and stage rule unchanged); stage shares; segmenter / paint diagnostics; pi05 tvp against R2-retrain's
main-state N1-N6 success (.425, SR 37.1, results/flawed/sc_t37_read_r012.json; R2 never ran on fresh states, so unpaired);
the main-state SR 46 numbers side by side.
Harness checks (reported): each arm's client args equal I's except out / port / videos; per-episode trial, branch, variant, at equal
I's; branches address trials 24-43 with pp_atlas_fresh_read's seed rule; I's familiar excess reproduces pp_atlas_fresh_read.json; the
hashed state / branch files still match the prereg row.

    LIBERO venv, CPU:  python experiments/tv_fresh.py read [--out results/flawed/tv_fresh_read.json]
"""
import argparse
import glob
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT]
import pp_atlas as PA                                                       # noqa: E402
import tv_frozen as TF                                                      # noqa: E402

FRESH_TRIALS = list(range(24, 44))
EPS = 0.10
OUT = PA.OUT
I_DIR = os.path.join(OUT, "pp_atlas_fresh")
DIR = os.path.join(OUT, "tv_fresh")
READ = os.path.join(OUT, "tv_fresh_read.json")
ATLAS_READ = os.path.join(OUT, "pp_atlas_fresh_read.json")
MAIN_READ = os.path.join(OUT, "tv_frozen_read.json")
BR = os.path.join(OUT, "n2_branches_fresh.json")
STATES = os.path.join(OUT, "n2_states_fresh")
RUNS = dict(pi05=("I", "tvp", "tv"), groot=("I", "tv"))
TIDS = (16, 18, 20, 22, 24, 26, 0, 2, 4, 6, 10)
R2_MAIN_NEW = 0.425                     # SR 37.1 (sc_t37_read_r012.json): R2 retrained, N1-N6 familiar, main states
IGNORE_ARGS = ("out", "port", "videos")
EP_KEYS = ("trial", "branch", "variant", "at")
MAPPING = ("per model, familiar layout, trials 24-43 (fresh k 36-55); success matrices [task, state]; paired bootstrap B=2000 rng 0, "
           "tasks and states drawn jointly; CI 2.5/97.5. PRIMARY pi05 tvp - I on N1-N6: CONFIRMED iff d > .10 and lo > 0; REFUTED iff "
           "hi < .10; else UNRESOLVED. GUARD pi05 tvp - I on K1-K4,K6: NO COST iff lo > -.10. SECONDARY groot tv - I and pi05 tv - I on "
           "N1-N6: the primary's rule. Descriptive: atlas excess per arm, tvp - tv, K cost of tv, tvp vs R2-retrain main-state .425.")


def label(d):
    if d[0] > EPS and d[1] > 0:
        return "CONFIRMED"
    if d[2] < EPS:
        return "REFUTED"
    return "UNRESOLVED"


def guard(d):
    return "NO COST" if d[1] > -EPS else "COST NOT EXCLUDED"


def sha(p):
    with open(p, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def state_files():
    return [os.path.join(STATES, "states.npy")] + [os.path.join(STATES, "n2_goal_%d.npy" % t) for t in TIDS] + [BR]


def rel(p):
    return os.path.relpath(p, ROOT)


def harness(m, stem_i, stem_a):
    """Client args and per-episode branch addressing of an arm's rollout against I's, for one task."""
    Ri = json.load(open(stem_i + ".json", encoding="utf-8"))
    Ra = json.load(open(stem_a + ".json", encoding="utf-8"))
    ai, aa = Ri["args"], Ra["args"]
    diff = {k: [ai.get(k), aa.get(k)] for k in sorted(set(ai) | set(aa)) if k not in IGNORE_ARGS and ai.get(k) != aa.get(k)}
    Ei = [e for v in Ri["tasks"].values() for e in v["episodes"]]
    Ea = [e for v in Ra["tasks"].values() for e in v["episodes"]]
    ep_ok = len(Ei) == len(Ea) and all(all(x.get(k) == y.get(k) for k in EP_KEYS) for x, y in zip(Ei, Ea))
    return diff, ep_ok


def read(a):
    out = os.path.abspath(a.out)
    assert not os.path.exists(out), "never overwrite a result file: %s" % out
    br = json.load(open(BR, encoding="utf-8"))
    assert [int(b["trial"]) for b in br] == FRESH_TRIALS, "branches do not address the fresh trials"
    assert [int(b["seed_base"]) for b in br] == [13000000 + 1000 * (t + 12) for t in FRESH_TRIALS]
    rows = []
    for f in glob.glob(os.path.join(ROOT, "ledger", "m3_prereg", "*.json")):
        r = json.load(open(f, encoding="utf-8"))
        if r.get("tag") == "tv_fresh":
            rows.append(r)
    assert len(rows) == 1, "expected exactly one tv_fresh prereg row, found %d" % len(rows)
    reg = rows[0]["metrics"]["files_sha256"]
    files_ok = {rel(p): reg.get(rel(p)) == sha(p) for p in state_files()}
    n2 = PA.tasks()
    ks = FRESH_TRIALS
    T = {t["compound"]: t for t in n2["tasks"] if t["layout"] == "familiar"}
    NEW, KNOWN, PAIR = PA.NEW, PA.KNOWN, PA.PAIR
    comps = NEW + KNOWN
    assert sorted(T[c]["tid"] for c in comps) == sorted(TIDS)
    ref = json.load(open(ATLAS_READ, encoding="utf-8"))
    main = json.load(open(MAIN_READ, encoding="utf-8")) if os.path.exists(MAIN_READ) else None
    res = dict(states=ks, dir=DIR, i_dir=I_DIR, B=PA.B, eps=EPS, mapping=MAPPING, state_files_match_prereg=files_ok, models={})
    for m, arms in RUNS.items():
        A, H = {}, {}
        for arm in arms:
            d, st = (I_DIR, m + "_{tid}") if arm == "I" else (DIR, "%s_%s_{tid}" % (m, arm))
            files = [os.path.join(d, st.format(tid=T[c]["tid"])) for c in comps]
            miss = [f for f in files if not (os.path.exists(f + ".json") and os.path.exists(f + "_track.npz"))]
            if miss:
                A[arm] = "MISSING %d of %d tasks" % (len(miss), len(files))
                continue
            if arm != "I":
                H[arm] = {}
                for c, f in zip(comps, files):
                    diff, ep_ok = harness(m, os.path.join(I_DIR, "%s_%d" % (m, T[c]["tid"])), f)
                    H[arm][c] = dict(args_diff=diff, episodes_match=ep_ok)
            RATE, succ, per = {}, {}, {}
            for c, f in zip(comps, files):
                S, X, G, SU = PA.stage_table(f, T[c], ks)
                RATE[c] = {s: np.array([float(x == s) for x in S]) for s in PA.STAGES}
                succ[c] = np.array(SU, float)
                per[c] = {s: float(RATE[c][s].mean()) for s in PA.STAGES}
            atl = TF.atlas_est(RATE, NEW, PAIR, ks, PA)
            new_m = np.stack([succ[c] for c in NEW])
            kn_m = np.stack([succ[c] for c in KNOWN])
            A[arm] = dict(new=new_m, known=kn_m, row=dict(
                success_new=TF.boot1(new_m), success_known=TF.boot1(kn_m), per_task_success={c: float(succ[c].mean()) for c in comps},
                stage_shares_new={s: float(np.mean([per[c][s] for c in NEW])) for s in PA.STAGES},
                grasp_ok_new=1.0 - float(np.mean([per[c]["GRASP"] for c in NEW])),
                cross_stage_share_new=float(np.mean([per[c]["HANDOFF"] + per[c]["TRANSPORT"] for c in NEW])), per_task_stages=per, atlas=atl,
                side=None if arm == "I" else TF.side_summary([os.path.join(d, st.format(tid=T[c]["tid"])) + "_tv.json" for c in comps])))
        row = {arm: (v if isinstance(v, str) else v["row"]) for arm, v in A.items()}

        def dl(x, y, key="new"):
            if isinstance(A.get(x), dict) and isinstance(A.get(y), dict):
                return TF.boot2(A[x][key], A[y][key])
            return None
        D, V = {}, {}
        if m == "pi05":
            D["tvp_minus_I"], D["tvp_minus_I_known"], D["tv_minus_I"] = dl("tvp", "I"), dl("tvp", "I", "known"), dl("tv", "I")
            D["tvp_minus_tv"], D["tv_minus_I_known"] = dl("tvp", "tv"), dl("tv", "I", "known")
            V["primary_tvp_minus_I"] = None if D["tvp_minus_I"] is None else label(D["tvp_minus_I"])
            V["guard_tvp_known"] = None if D["tvp_minus_I_known"] is None else guard(D["tvp_minus_I_known"])
            V["secondary_tv_minus_I"] = None if D["tv_minus_I"] is None else label(D["tv_minus_I"])
            if isinstance(row.get("tvp"), dict):
                s = row["tvp"]["success_new"]
                D["tvp_minus_R2_main_descriptive"] = [s[0] - R2_MAIN_NEW, s[1] - R2_MAIN_NEW, s[2] - R2_MAIN_NEW]
        else:
            D["tv_minus_I"], D["tv_minus_I_known"] = dl("tv", "I"), dl("tv", "I", "known")
            V["secondary_tv_minus_I"] = None if D["tv_minus_I"] is None else label(D["tv_minus_I"])
        chk = None
        if isinstance(A.get("I"), dict) and m in ref:
            g0, g1 = ref[m]["familiar"]["groups"], A["I"]["row"]["atlas"]["groups"]
            chk = all(abs(g0[g]["E"] - g1[g]["E"]) < 1e-9 and all(abs(u - v) < 1e-9 for u, v in zip(g0[g]["ci"], g1[g]["ci"])) for g in g0)
        hok = all(not h["args_diff"] and h["episodes_match"] for hh in H.values() for h in hh.values()) if H else None
        mm = None
        if main and m in main.get("models", {}):
            mm = {arm: (r["success_new"][0], r["success_known"][0], r["atlas"]["groups"]["cross"]["E"])
                  for arm, r in main["models"][m]["arms"].items() if isinstance(r, dict)}
        res["models"][m] = dict(arms=row, deltas=D, verdicts=V, harness=H, harness_matches_I=hok, I_reproduces_pp_atlas_fresh_read=chk,
                                main_states_new_known_cross=mm)
        print("== %s: verdicts %s | harness matches I: %s | I reproduces pp_atlas_fresh_read: %s" % (m, V, hok, chk), flush=True)
        if H and not hok:
            for arm, hh in H.items():
                for c, h in hh.items():
                    if h["args_diff"] or not h["episodes_match"]:
                        print("   HARNESS %s %s %s" % (arm, c, h), flush=True)
        for arm in arms:
            r = row.get(arm)
            if isinstance(r, str) or r is None:
                print("   %-4s %s" % (arm, r), flush=True)
                continue
            g = r["atlas"]["groups"]
            print("   %-4s new %s known %s | grasp ok %.3f cross-stage %.3f | within E %s cross E %s -> %s" % (
                arm, TF._f(r["success_new"]), TF._f(r["success_known"]), r["grasp_ok_new"], r["cross_stage_share_new"],
                TF._f([g["within"]["E"]] + g["within"]["ci"]), TF._f([g["cross"]["E"]] + g["cross"]["ci"]), r["atlas"]["verdict"]), flush=True)
            print("        per task %s" % json.dumps({c: round(v, 3) for c, v in r["per_task_success"].items()}), flush=True)
            if r.get("side"):
                print("        side %s" % json.dumps({k: (TF._r(v) if isinstance(v, float) else v) for k, v in r["side"].items()
                                                      if k not in ("segmenter_iou", "paint")}), flush=True)
                if r["side"].get("paint"):
                    print("        paint %s seg %s" % (json.dumps(r["side"]["paint"]), json.dumps(r["side"]["segmenter_iou"])), flush=True)
        for k, v in D.items():
            print("   %s %s" % (k, TF._f(v)), flush=True)
        print("   main states (new, known, cross E): %s" % mm, flush=True)
    print("state files match prereg: %s" % files_ok, flush=True)
    tmp = out + ".tmp%d" % os.getpid()
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    os.replace(tmp, out)
    print("wrote", out, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("read",))
    ap.add_argument("--out", default=READ)
    read(ap.parse_args())


if __name__ == "__main__":
    main()
