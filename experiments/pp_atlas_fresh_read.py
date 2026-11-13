"""The break atlas's registered read on the FRESH familiar-layout states (m3_prereg pp_atlas_fresh).  experiments/pp_atlas_models_read.py
(itself pp_atlas.py's read for any list of models) is imported and run unchanged except for three substitutions, all made here and
never in those files:
  1. PA.OD = results/flawed/pp_atlas_fresh (the fresh rollouts of experiments/pp_atlas_fresh_roll.sh);
  2. PA.tasks() reports states.main = the fresh trials 24-43 (= k 36-55; experiments/fc_n2_fresh.py), so stage_table checks every
     episode's trial index against them;
  3. the layout loop reads only the familiar layout (the fresh replication rolled no new-layout task).  This is a one-token source
     substitution in read_models, asserted to occur exactly once.
Stage rule, partners (PAIR), estimator, bootstrap (B = 2000, rng seed 0) and labels are pp_atlas.py's own.  Then the registered
replication verdicts against the development read results/flawed/pp_atlas_more_read.json (familiar layout):
  F1 REPLICATES iff the cross-chunk group breaks (CI lower > 0 and E > .10) on >= 3 of pi05, groot, pi0, xvla;
  F2 REPLICATES iff the GRASP stage breaks (same bar) on both pi0 and smolvla;
  verdict agreement = number of the five models whose familiar verdict equals the development read's (bar: >= 4).

    LIBERO venv, CPU:  python experiments/pp_atlas_fresh_read.py [--out results/flawed/pp_atlas_fresh_read.json]
"""
import argparse
import copy
import inspect
import json
import os
import sys
import textwrap

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pp_atlas as PA                                                       # noqa: E402
import pp_atlas_models_read as PMR                                          # noqa: E402

MODELS = ["pi05", "groot", "pi0", "xvla", "smolvla"]
F1_MODELS = ["pi05", "groot", "pi0", "xvla"]
F2_MODELS = ["pi0", "smolvla"]
FRESH_TRIALS = list(range(24, 44))
DEV = os.path.join(PA.OUT, "pp_atlas_more_read.json")
LOOP = 'for lay in ("familiar", "new"):'


def patch():
    PA.OD = os.path.join(PA.OUT, "pp_atlas_fresh")
    orig = PA.tasks
    br = json.load(open(os.path.join(PA.OUT, "n2_branches_fresh.json"), encoding="utf-8"))
    assert [int(b["trial"]) for b in br] == FRESH_TRIALS, "branches do not address the fresh trials"
    assert [int(b["seed_base"]) for b in br] == [13000000 + 1000 * (t + 12) for t in FRESH_TRIALS]

    def tasks():
        n2 = copy.deepcopy(orig())
        n2["states"]["main"] = list(FRESH_TRIALS)
        return n2

    PA.tasks = tasks
    src = textwrap.dedent(inspect.getsource(PMR.read_models))
    assert src.count(LOOP) == 1, "read_models changed: layout loop not found exactly once"
    ns = PMR.__dict__
    exec(compile(src.replace(LOOP, 'for lay in ("familiar",):'), PMR.__file__, "exec"), ns)
    return ns["read_models"]


def replication(res, dev):
    def brk(r, kind, name):
        return bool(r["familiar"][kind][name]["breaks"])

    f1 = {m: brk(res[m], "groups", "cross") for m in F1_MODELS}
    f1_dev = {m: brk(dev[m], "groups", "cross") for m in F1_MODELS}
    f2 = {m: brk(res[m], "stages", "GRASP") for m in F2_MODELS}
    f2_dev = {m: brk(dev[m], "stages", "GRASP") for m in F2_MODELS}
    ver = {m: dict(fresh=res[m]["familiar"]["verdict"], development=dev[m]["familiar"]["verdict"]) for m in MODELS}
    agree = sum(v["fresh"] == v["development"] for v in ver.values())
    table = {m: dict(within=dict(E=res[m]["familiar"]["groups"]["within"]["E"], ci=res[m]["familiar"]["groups"]["within"]["ci"]),
                     cross=dict(E=res[m]["familiar"]["groups"]["cross"]["E"], ci=res[m]["familiar"]["groups"]["cross"]["ci"]),
                     GRASP=dict(E=res[m]["familiar"]["stages"]["GRASP"]["E"], ci=res[m]["familiar"]["stages"]["GRASP"]["ci"]),
                     verdict=res[m]["familiar"]["verdict"], development_verdict=dev[m]["familiar"]["verdict"]) for m in MODELS}
    out = dict(F1=dict(cross_breaks=f1, cross_breaks_development=f1_dev, n=sum(f1.values()), bar=3,
                       verdict="F1 REPLICATES" if sum(f1.values()) >= 3 else "F1 DOES NOT REPLICATE"),
               F2=dict(grasp_breaks=f2, grasp_breaks_development=f2_dev,
                       verdict="F2 REPLICATES" if all(f2.values()) else "F2 DOES NOT REPLICATE"),
               verdicts=ver, agree=agree, agree_bar=4, table=table)
    out["prediction_held"] = bool(out["F1"]["verdict"] == "F1 REPLICATES" and out["F2"]["verdict"] == "F2 REPLICATES" and agree >= 4)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(PA.OUT, "pp_atlas_fresh_read.json"))
    a = ap.parse_args()
    out = os.path.abspath(a.out)
    assert os.path.basename(out) not in ("pp_atlas_read.json", "pp_atlas_more_read.json"), "never overwrite a registered read"
    assert not os.path.exists(out), "exists: " + out
    read_models = patch()
    read_models(MODELS, out)
    res = json.load(open(out, encoding="utf-8"))
    rep = replication(res, json.load(open(DEV, encoding="utf-8")))
    res["replication"] = rep
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    for m, r in rep["table"].items():
        print("TABLE %-8s within %+.3f [%+.2f, %+.2f]  cross %+.3f [%+.2f, %+.2f]  GRASP %+.3f [%+.2f, %+.2f]  %s (dev %s)" % (
            m, r["within"]["E"], *r["within"]["ci"], r["cross"]["E"], *r["cross"]["ci"], r["GRASP"]["E"], *r["GRASP"]["ci"],
            r["verdict"], r["development_verdict"]), flush=True)
    print("REPLICATION", json.dumps({k: rep[k] for k in ("F1", "F2", "agree", "prediction_held")}), flush=True)
    print("wrote", out, flush=True)
