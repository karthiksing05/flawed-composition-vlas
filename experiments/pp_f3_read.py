"""PLAN_PARTS F3 closed-loop read (docs/PARTS_RESULTS.md section 1; m3_prereg tags pp_f123 and pp_f3).  CPU.

Data: results/flawed/pp_f3_roll/<model>_<arm>_<tid>.json, arms intact / purify, N1-N6, both layouts, N2 states 4-13 and N2's seeds.
Per model and layout: pooled s, g and p = sum s / sum g over the 6 tasks x 10 states, per arm; deltas purified - intact.
CI: 2,000-draw percentile bootstrap (seed 0) resampling tasks and states jointly (both arms see the same draw: paired).
F3 TIES (familiar layout, primary) iff delta p > 0 with its CI excluding 0.
Also: the intact re-run against N2's recorded episodes of the same states (agreement of s and g per episode).
With --prep: write results/flawed/pp_f3_branches.json (N2's main branches restricted to states 4-13) and exit.
-> results/flawed/pp_f3_read.json

    python experiments/pp_f3_read.py [--prep]
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "flawed")
STATES = list(range(4, 14))
NEW = ["N1", "N2", "N3", "N4", "N5", "N6"]
B, SEED = 2000, 0


def episodes(f, obj):
    eps = [e for v in json.load(open(f, encoding="utf-8"))["tasks"].values() for e in v["episodes"]]
    return {int(e["trial"]): (float(bool(e["success"])), float(bool(e["success"]) or e.get("bound") == obj), e.get("error")) for e in eps}


def main():
    if "--prep" in sys.argv:
        br = json.load(open(os.path.join(OUT, "n2_branches_main.json"), encoding="utf-8"))
        sub = [b for b in br if int(b["trial"]) in STATES]
        assert len(sub) == len(STATES)
        json.dump(sub, open(os.path.join(OUT, "pp_f3_branches.json"), "w", encoding="utf-8"), indent=1)
        print("wrote pp_f3_branches.json", len(sub))
        return
    n2 = json.load(open(os.path.join(OUT, "n2_tasks.json"), encoding="utf-8"))
    tid = {(t["compound"], t["layout"]): t for t in n2["tasks"]}
    out = {}
    for model in ("pi05", "groot"):
        res = {}
        for lay in ("familiar", "new"):
            S = {arm: np.full((len(NEW), len(STATES)), np.nan) for arm in ("intact", "purify", "recorded")}
            G = {arm: np.full((len(NEW), len(STATES)), np.nan) for arm in S}
            errs = 0
            ok = True
            for i, c in enumerate(NEW):
                t = tid[(c, lay)]
                for arm in S:
                    f = (os.path.join(OUT, "n2_roll", f"{model}_main_{t['tid']}.json") if arm == "recorded"
                         else os.path.join(OUT, "pp_f3_roll", f"{model}_{arm}_{t['tid']}.json"))
                    if not os.path.exists(f):
                        ok = False
                        continue
                    E = episodes(f, t["object"])
                    for j, k in enumerate(STATES):
                        if k in E:
                            S[arm][i, j], G[arm][i, j] = E[k][0], E[k][1]
                            errs += (E[k][2] is not None) and arm != "recorded"
            if not ok:
                res[lay] = dict(missing=True)
                continue

            def stats(ti, si):
                r = {}
                for arm in S:
                    s, g = S[arm][np.ix_(ti, si)], G[arm][np.ix_(ti, si)]
                    r[arm] = dict(s=float(np.nanmean(s)), g=float(np.nanmean(g)), p=float(np.nansum(s) / max(np.nansum(g), 1e-9)))
                return r
            allt, alls = np.arange(len(NEW)), np.arange(len(STATES))
            pt = stats(allt, alls)
            rng = np.random.default_rng(SEED)
            bs = {k: [] for k in ("p", "s", "g")}
            for _ in range(B):
                ti, si = rng.integers(0, len(NEW), len(NEW)), rng.integers(0, len(STATES), len(STATES))
                r = stats(ti, si)
                for k in bs:
                    bs[k].append(r["purify"][k] - r["intact"][k])
            d = {k: [pt["purify"][k] - pt["intact"][k], float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in bs.items()}
            agree = dict(s=float(np.nanmean(S["intact"] == S["recorded"])), g=float(np.nanmean(G["intact"] == G["recorded"])))
            per = {c: {arm: dict(s=float(np.nanmean(S[arm][i])), g=float(np.nanmean(G[arm][i]))) for arm in S} for i, c in enumerate(NEW)}
            res[lay] = dict(arms=pt, delta=d, rerun_vs_recorded_agreement=agree, per_task=per, episode_errors=int(errs))
        fam = res.get("familiar", {})
        v = None
        if "delta" in fam:
            dp = fam["delta"]["p"]
            v = "ties" if (dp[0] > 0 and dp[1] > 0) else "does not tie"
        out[model] = dict(layouts=res, F3=v)
        print("F3", model, v, json.dumps(fam.get("delta")), flush=True)
    json.dump(out, open(os.path.join(OUT, "pp_f3_read.json"), "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()
