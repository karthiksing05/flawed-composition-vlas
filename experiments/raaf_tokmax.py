"""RAA-F: normalised max |value| of the state token dims 0-2 (openpi quantile Normalize, no clip), originals vs shifted copies, rows h0..e-1.  numpy, sbatch only."""
import json, os, pickle, sys
import numpy as np
prep, delta, out = sys.argv[1:4]
assert not os.path.exists(out)
P = pickle.load(open(prep, "rb")); D = json.load(open(delta))
ns = json.load(open("/coc/flash10/ksingara3/robot/checkpoints_l90c/assets/pi05_g8v_raac/lol/libero_g8v_raac/norm_stats.json"))["norm_stats"]["state"]
q01, q99 = np.asarray(ns["q01"][:3]), np.asarray(ns["q99"][:3])
nz = lambda x: (x - q01) / (q99 - q01 + 1e-6) * 2 - 1
by = {(d["task"], d["demo"]): d for d in P["demos"]}
o, c = [], []
for dd in D["demos"]:
    d = by[(dd["task"], dd["demo"])]
    for cp in dd["copies"]:
        t = d["rel"][dd["h0"]:dd["e"]]
        o.append(nz(t)); c.append(nz(t + np.asarray(cp["delta"])[None]))
o, c = np.concatenate(o), np.concatenate(c)
r = dict(orig_max_abs=np.abs(o).max(0).tolist(), copy_max_abs=np.abs(c).max(0).tolist(), orig_p999=np.percentile(np.abs(o), 99.9, 0).tolist(), copy_p999=np.percentile(np.abs(c), 99.9, 0).tolist())
json.dump(r, open(out, "w"), indent=1); print(json.dumps(r))
