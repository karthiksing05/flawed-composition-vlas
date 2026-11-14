"""The registered read of GROOT-BOX (m3_prereg tag groot_box; docs/SCAFFOLD_RESULTS.md 56): the refined front end (role views +
held-object box) and a proprioceptive cue swap on FROZEN GR00T, no phrases, familiar layout, the fourth fresh batch k = 156-195
(harness trials 24-63), N1-N6 and K1-K4, K6.  LIBERO venv, CPU.

Arms (all new on these states, results/flawed/groot_box/groot_<arm>_<tid>, run by experiments/groot_box_roll.sh):
    I     intact (pp_atlas_client.py)
    tv    role views (tv_frozen.py client --arm tv; SR 46 / 49)
    tvb   tv + the held object painted as its bounding box in the target phase (groot_box.py client --arm tvb = f1m_cue's sb, no phrases)
    tvg   tv + finger qpos replaced by GR00T's K6 cheese hold while the detector is on (groot_box.py client --arm tvg)
    tvbg  both (descriptive only)
Registered mapping: success matrices [task, state]; paired bootstrap B = 2000, numpy default_rng(0), tasks and states drawn jointly
(tv_frozen.boot2, as SR 52 / 55); CI 2.5 / 97.5.
    V1  d = tvb - I on N1-N6:       FRONT END HELPS iff CI lo > 0; else NO CLEAR DIFFERENCE.
    V2  d = tvb - tv on N4 + N6:    BOX ADDS iff CI lo > 0; NOT THIS CUE iff CI hi < .10; else UNRESOLVED.
    V3  d = tvg - tv on N4 + N6:    CUE FOUND iff CI lo > 0; NOT THIS CUE iff CI hi < .10; else UNRESOLVED.
    (For V2 / V3 the lo > 0 label takes precedence.)
Descriptive (no verdict): every other difference (tv - I, tvg - I, tvbg - I on N1-N6; tvbg - tv, tvbg - tvb, tvbg - tvg on N4 + N6;
tvb - tv on N1-N6), per-task success, K1-K4, K6 success per arm and arm - I (trained-task cost), the atlas's within / cross-chunk
excess per arm (tv_frozen.atlas_est), stage shares, high-carry share on N4 / N6 / N5 / K4 (n46_phase lift_held > .22 m among grasped)
and success after high / low carries, detector stats, box sizes, the finger entries actually sent at held queries (tvg, tvbg).
Checks: every arm's client args equal I's except out / port / videos, and per-episode trial / branch / variant / at equal I's
(tv_fresh.harness, the SR 48 / 52 / 55 identity checks); every episode's trial is the registered one; no episode errors; state
files and code match the groot_box row; f1m_cue.py matches its f1m_cue row; tv_frozen.py matches its tvpb_rg row.

    python experiments/groot_box_read.py [--out results/flawed/groot_box_read.json]
    python experiments/groot_box_read.py --dry --dir SMOKE_DIR --trials 0,1 --out OUT_OUTSIDE_RESULTS     plumbing check, no ledger
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
READ_ARMS = ("I", "tv", "tvb", "tvg", "tvbg")
SIDE_ARMS = ("tv", "tvb", "tvg", "tvbg")
DIR = os.path.join(TF.OUT, "groot_box")
READ_OUT = os.path.join(TF.OUT, "groot_box_read.json")
TRIALS = list(range(24, 64))
STATES = os.path.join(TF.OUT, "n2_states_fresh4")
BR = os.path.join(TF.OUT, "n2_branches_fresh4.json")
TIDS = (16, 18, 20, 22, 24, 26, 0, 2, 4, 6, 10)
EPS = 0.10
CUT = 0.22
MAPPING = ("GR00T, no phrases, familiar layout, trials 24-63 (fresh k 156-195); success matrices [task, state]; paired bootstrap "
           "B=2000 rng 0, tasks and states drawn jointly (tv_frozen.boot2); CI 2.5/97.5. V1: tvb - I on N1-N6: FRONT END HELPS iff "
           "lo > 0, else NO CLEAR DIFFERENCE. V2: tvb - tv on N4+N6: BOX ADDS iff lo > 0; NOT THIS CUE iff hi < .10; else UNRESOLVED. "
           "V3: tvg - tv on N4+N6: CUE FOUND iff lo > 0; NOT THIS CUE iff hi < .10; else UNRESOLVED. Descriptive: other differences, "
           "K cost per arm, atlas excess, high-carry share, detector and sent-state diagnostics.")


def v1(d):
    return "FRONT END HELPS" if d[1] > 0 else "NO CLEAR DIFFERENCE"


def cue(d, yes):
    if d[1] > 0:
        return yes
    if d[2] < EPS:
        return "NOT THIS CUE"
    return "UNRESOLVED"


def sha(p):
    with open(p if os.path.isabs(p) else os.path.join(ROOT, p), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def state_files():
    return [os.path.join(STATES, "states.npy")] + [os.path.join(STATES, "n2_goal_%d.npy" % t) for t in TIDS] + [BR]


def code_checks():
    from lifelong_ood import ledger
    rows = {t: [r for r in ledger.load("m3_prereg") if r.get("tag") == t] for t in ("groot_box", "f1m_cue", "tvpb_rg")}
    assert all(len(v) == 1 for v in rows.values()), {k: len(v) for k, v in rows.items()}
    reg = {k: v[0]["metrics"]["files_sha256"] for k, v in rows.items()}
    return dict(states_match_groot_box_row={os.path.relpath(p, ROOT): reg["groot_box"].get(os.path.relpath(p, ROOT)) == sha(p)
                                            for p in state_files()},
                f1m_cue_matches_f1m_cue_row={"experiments/f1m_cue.py": reg["f1m_cue"].get("experiments/f1m_cue.py") == sha("experiments/f1m_cue.py")},
                tv_frozen_matches_tvpb_rg_row={"experiments/tv_frozen.py": reg["tvpb_rg"].get("experiments/tv_frozen.py") == sha("experiments/tv_frozen.py")},
                code_matches_groot_box_row={k: sha(k) == v for k, v in reg["groot_box"].items()
                                            if (k.startswith("experiments/") or k.startswith("/")) and os.path.exists(k if k.startswith("/") else os.path.join(ROOT, k))})


def sent_check(files, g):
    """The finger entries of observation/state sent at held queries: share equal to g (tvg / tvbg) or the real qpos (others)."""
    s, q = [], []
    for f in files:
        if os.path.exists(f):
            for e in json.load(open(f, encoding="utf-8"))["eps"]:
                s += e.get("sent_held") or []
                q += e.get("qpos_held") or []
    out = dict(n_sent=len(s), n_qpos=len(q))
    if s:
        S = np.asarray(s, float)
        out.update(sent_mean=S.mean(0).tolist(), sent_eq_g=None if g is None else float(np.mean(np.all(np.abs(S - np.asarray(g)) < 1e-6, 1))))
    if q:
        out.update(qpos_held_mean=np.asarray(q, float).mean(0).tolist())
    return out


def read(a):
    import pp_atlas as PA
    import tv_fresh as TFR
    import n46_phase as NP
    out = os.path.abspath(a.out)
    assert not os.path.exists(out), "never overwrite a result file: %s" % out
    if a.dry:
        assert not out.startswith(os.path.join(ROOT, "results")), "a dry read never writes into results/"
    ks = [int(x) for x in a.trials.split(",")] if a.trials else TRIALS
    if not a.dry:
        br = json.load(open(BR, encoding="utf-8"))
        assert [int(b["trial"]) for b in br] == ks, "branches do not address trials 24-63"
        assert [int(b["seed_base"]) for b in br] == [13000000 + 1000 * (t + 132) for t in ks], "seed_base rule"
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
        g46 = [x for c in ("N4", "N6") for x in [carry[c]]]
        gr, hi = sum(x["grasped"] for x in g46), sum(x["high"] for x in g46)
        carry["N4+N6"] = dict(grasped=gr, high=hi, high_share=(hi / float(gr)) if gr else None)
        side, sent = None, None
        if arm in SIDE_ARMS:
            side = TF.side_summary([stem(arm, c) + "_tv.json" for c in comps])
            gval = None
            if arm in ("tvg", "tvbg"):
                gval = json.load(open(stem(arm, "N4") + "_tv.json", encoding="utf-8")).get("g")
            sent = dict(N4N6=sent_check([stem(arm, c) + "_tv.json" for c in ("N4", "N6")], gval),
                        K6=sent_check([stem(arm, "K6") + "_tv.json"], gval), all=sent_check([stem(arm, c) + "_tv.json" for c in comps], gval))
        A[arm] = dict(new=new_m, known=kn_m, n46=n46, row=dict(
            success_new=TF.boot1(new_m), success_known=TF.boot1(kn_m), success_n4n6=TF.boot1(n46),
            per_task_success={c: float(succ[c].mean()) for c in comps},
            stage_shares_new={s: float(np.mean([per[c][s] for c in NEW])) for s in PA.STAGES},
            cross_stage_share_new=float(np.mean([per[c]["HANDOFF"] + per[c]["TRANSPORT"] for c in NEW])), per_task_stages=per,
            atlas=atl, episode_errors=errs, carry=carry, side=side, sent=sent))
    row = {arm: (v if isinstance(v, str) else v["row"]) for arm, v in A.items()}

    def dl(x, y, key="new"):
        if isinstance(A.get(x), dict) and isinstance(A.get(y), dict):
            return TF.boot2(A[x][key], A[y][key])
        return None
    Dl = dict(V1_tvb_minus_I=dl("tvb", "I"), V2_tvb_minus_tv_N4N6=dl("tvb", "tv", "n46"), V3_tvg_minus_tv_N4N6=dl("tvg", "tv", "n46"),
              tv_minus_I=dl("tv", "I"), tvg_minus_I=dl("tvg", "I"), tvbg_minus_I=dl("tvbg", "I"), tvb_minus_tv_new=dl("tvb", "tv"),
              tvg_minus_tv_new=dl("tvg", "tv"), tvbg_minus_tv_N4N6=dl("tvbg", "tv", "n46"), tvbg_minus_tvb_N4N6=dl("tvbg", "tvb", "n46"),
              tvbg_minus_tvg_N4N6=dl("tvbg", "tvg", "n46"), tvb_minus_I_N4N6=dl("tvb", "I", "n46"), tv_minus_I_N4N6=dl("tv", "I", "n46"))
    Dl["known_minus_I"] = {x: dl(x, "I", "known") for x in READ_ARMS[1:]}
    V = dict(V1=None if Dl["V1_tvb_minus_I"] is None else v1(Dl["V1_tvb_minus_I"]),
             V2=None if Dl["V2_tvb_minus_tv_N4N6"] is None else cue(Dl["V2_tvb_minus_tv_N4N6"], "BOX ADDS"),
             V3=None if Dl["V3_tvg_minus_tv_N4N6"] is None else cue(Dl["V3_tvg_minus_tv_N4N6"], "CUE FOUND"))
    hok = all(not h["args_diff"] and h["episodes_match"] for hh in H.values() for h in hh.values()) if H else None
    res = dict(model=MODEL, states=ks, dir=D, dry=bool(a.dry), B=PA.B, mapping=MAPPING, arms=row, deltas=Dl, verdicts=V, harness=H,
               harness_matches_I=hok, checks=checks)
    print("== groot_box read%s: verdicts %s | harness matches I: %s" % (" (DRY)" if a.dry else "", V, hok), flush=True)
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
        r = row.get(arm)
        if isinstance(r, str) or r is None:
            print("   %-4s %s" % (arm, r), flush=True)
            continue
        g = r["atlas"]["groups"]
        print("   %-4s new %s known %s N4+N6 %s | cross-stage %.3f | within E %s cross E %s -> %s | errors %d" % (
            arm, TF._f(r["success_new"]), TF._f(r["success_known"]), TF._f(r["success_n4n6"]), r["cross_stage_share_new"],
            TF._f([g["within"]["E"]] + g["within"]["ci"]), TF._f([g["cross"]["E"]] + g["cross"]["ci"]), r["atlas"]["verdict"],
            r["episode_errors"]), flush=True)
        print("        per task %s" % json.dumps({c: round(v, 3) for c, v in r["per_task_success"].items()}), flush=True)
        print("        stages new %s" % json.dumps({s: round(v, 3) for s, v in r["stage_shares_new"].items()}), flush=True)
        print("        carry %s" % json.dumps({c: {k: (round(v, 3) if isinstance(v, float) else v) for k, v in x.items()}
                                               for c, x in r["carry"].items()}), flush=True)
        if r.get("side"):
            print("        side %s" % json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r["side"].items()
                                                  if k not in ("segmenter_iou", "paint")}), flush=True)
            if r["side"].get("paint"):
                print("        paint %s" % json.dumps(r["side"]["paint"]), flush=True)
        if r.get("sent"):
            print("        sent %s" % json.dumps(r["sent"], default=float), flush=True)
    for k, v in Dl.items():
        if k != "known_minus_I":
            print("   %s %s" % (k, TF._f(v)), flush=True)
    print("   known - I %s" % {k: TF._f(v) for k, v in Dl["known_minus_I"].items()}, flush=True)
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
