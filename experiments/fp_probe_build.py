"""Assemble results/flawed/fp_dev/probe_lang.csv from the probe jsonl files (experiments/fp_probe_client.py).  CPU, numpy only; via sbatch.
Per (model, scene, set, task, trial): mean over alternative instructions of the chunk distance (normalised by the base chunk's own norm) for the full
10-step chunk, the 5 executed steps and the xyz translation of those 5 steps; the same-language different-seed noise floor; ratios; determinism check."""
import csv, json, os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "results", "flawed")
RAW = os.path.join(R, "fp_dev", "probe_raw")
OUT = os.path.join(R, "fp_dev", "probe_lang.csv")
assert not os.path.exists(OUT), OUT
n2 = {t["tid"]: t["compound"] for t in json.load(open(os.path.join(R, "n2_tasks.json")))["tasks"]}
s2 = {0: "K1", 1: "K2", 2: "N1", 3: "N2"}
rows = []
for m in ("pi05", "pi0", "groot", "xvla", "smolvla"):
    for tag, scene, st, tmap in (("s1_dev", 1, "dev", n2), ("s1_fresh", 1, "fresh", n2), ("s2_std", 2, "std", s2)):
        p = os.path.join(RAW, "%s_%s.jsonl" % (m, tag))
        if not os.path.exists(p):
            continue
        for line in open(p, encoding="utf-8"):
            j = json.loads(line)
            r = dict(model=m, scene=scene, set=st, task=tmap[j["task_id"]], trial=j["trial"], n_alts=len(j["alts"]), rep_maxabs=j["rep_maxabs"])
            for nm in ("full", "exec5", "xyz5"):
                d = j["d_" + nm]
                fl = j["floor_" + nm]
                r["d_" + nm] = float(np.mean(d)) if d else ""
                r["floor_" + nm] = fl
                r["ratio_" + nm] = float(np.mean(d) / max(fl, 1e-6)) if d else ""
            rows.append(r)
cols = ["model", "scene", "set", "task", "trial", "n_alts", "d_full", "floor_full", "ratio_full", "d_exec5", "floor_exec5", "ratio_exec5", "d_xyz5", "floor_xyz5", "ratio_xyz5", "rep_maxabs"]
with open(OUT, "x", encoding="utf-8", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=cols)
    w.writeheader()
    for r in rows:
        w.writerow(r)
from collections import Counter
print(len(rows), Counter((r["model"], r["scene"], r["set"]) for r in rows))
print("max rep_maxabs by model", {m: max([r["rep_maxabs"] for r in rows if r["model"] == m] or [0]) for m in ("pi05", "pi0", "groot", "xvla", "smolvla")})
