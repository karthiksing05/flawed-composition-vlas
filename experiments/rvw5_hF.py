"""RVW5 hF (registered, reported): H - RAA-sL on F6_b (tid 8) of the 5th set, wa_read.bootm.  H - A0 is not available (A0 has no Phase B rollouts on the 5th set).  Output opened with "x"."""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import wa_read as WR
R = os.path.join(os.path.dirname(HERE), "results", "flawed")
rows6 = json.load(open(os.path.join(R, "rg_comp_5", "rg_comp_5_tasks.json"), encoding="utf-8"))["states"]["f6"]["main"]
X = {}
for arm in ("H", "raasl", "now", "w5"):
    d, _ = WR.load(os.path.join(R, "rvw5", "b_" + arm, "full_8.json"), rows6)
    X[arm] = np.array([d[k] for k in rows6], float)
res = {}
for a, b in (("H", "raasl"), ("H", "now"), ("w5", "H")):
    ci = WR.bootm(lambda I, a=a, b=b: float(X[a][I[0]].mean() - X[b][I[0]].mean()), [len(rows6)])
    res["%s - %s | F6_b" % (a, b)] = dict(delta=ci[0], ci=ci[1:], a=float(X[a].mean()), b=float(X[b].mean()))
with open(os.path.join(R, "rvw5", "rvw5_hF.json"), "x", encoding="utf-8") as fh:
    json.dump(res, fh, indent=1)
print(json.dumps(res))
