"""SR 83 descriptive sensitivity read (NOT registered): a2_read's registered analysis on a subset of the observations.  a2_read.py is imported unchanged.

    python experiments/a2_read_sens.py --model pi05|raasl --dirs DIR[+DIR..] --drop K4_fam[,..] --out results/flawed/a2/a2_sens_<tag>_<model>.json
The drop list removes whole tasks (meta.task); used to put both models on the same K tasks (K4 has no equal-token swap candidate for pi0.5's whole-sentence prompts).
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import a2_read as R                                                          # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--dirs", required=True)
ap.add_argument("--drop", default="")
ap.add_argument("--out", required=True)
a = ap.parse_args()
drop = set(x for x in a.drop.split(",") if x)
obs = [o for o in R.load(a.dirs.split("+")) if o["meta"]["task"] not in drop]
doc = R.analyze(a.model, obs)
doc["dropped_tasks"] = sorted(drop)
if os.path.exists(a.out):
    raise SystemExit("refusing to overwrite %s" % a.out)
with open(a.out, "w", encoding="utf-8") as fh:
    json.dump(doc, fh, indent=1, default=float)
R.show(doc, a.model)
