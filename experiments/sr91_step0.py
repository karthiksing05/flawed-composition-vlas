"""SR 91 Step 0 (EXPLORATORY, seen before registration): paired bootstrap (B = 2000, rng 0, tasks and states resampled, raa_read.diff) of W5 - rgp and W5 - RAA-sL
on N1-N6 and K1-K6 from the existing SR 90 rollouts (results/flawed/raasw6, third fresh set k = 640-699).  Numpy only; compute node.
Writes results/flawed/sr91_step0.json (refuses to overwrite).  Reports whether W5 removal arms exist in raasw6/b_w5."""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import raa_read as RR                                                        # noqa: E402
import raasw6_read as R6                                                     # noqa: E402

OUTF = os.path.join(RR.OUT, "sr91_step0.json")
assert not os.path.exists(OUTF), OUTF
meta = RR.n2()
G = {}
for arm in ("w5", "rgp", "raasl"):
    for grp, comps in (("K", RR.KNOWN), ("N", RR.NEW)):
        G[(arm, grp)] = (R6.grid(meta, arm, comps, "succ"), R6.grid(meta, arm, comps, "bound"))
res = dict(label="EXPLORATORY, seen before registration (SR 90 data, third fresh set)", B=RR.B, contrasts={}, means={})
for arm in ("w5", "rgp", "raasl"):
    for grp in ("N", "K"):
        res["means"]["%s|%s" % (arm, grp)] = float(G[(arm, grp)][0].mean())
for b in ("rgp", "raasl"):
    for grp in ("N", "K"):
        d = RR.diff(G[("w5", grp)], G[(b, grp)])
        d["noninf_m10"] = bool(d["ci"][0] > -0.10)
        d["noninf_m05"] = bool(d["ci"][0] > -0.05)
        res["contrasts"]["W5 - %s, %s1-%s6" % (b, grp, grp)] = d
bw = os.path.join(R6.BASE, "b_w5")
res["b_w5_files"] = sorted(set(os.path.basename(p).split("_")[0] for p in glob.glob(os.path.join(bw, "*.json"))))
res["w5_removal_arms_exist"] = any(os.path.basename(p).startswith(("noR", "noM")) for p in glob.glob(os.path.join(bw, "*")))
print("[sr91-step0]", json.dumps(res, indent=1), flush=True)
with open(OUTF, "w", encoding="utf-8") as fh:
    json.dump(res, fh, indent=1)
