"""SR 91: the fourth fresh states are disjoint from SR 70-90's: no state row of rg_comp_z is np.allclose to any row of rg_comp / rg_comp_x / rg_comp_y (same rgc tid),
and no n2 state of n2_states_wz equals any row of every other n2_states_* set (incl. rx, wx, wy); prints the F7 / F6 accepted-k ranges of rg_comp_z.  CPU, numpy only.  Exit 1 if any row is equal."""
import glob
import json
import os
import sys

import numpy as np

R = "/coc/flash10/ksingara3/flawed-composition-vlas/results/flawed"
bad = 0
for f in sorted(glob.glob(R + "/rg_comp_z/states/rgc_*.npy")):
    y = np.load(f)
    for old in ("rg_comp", "rg_comp_x", "rg_comp_y"):
        g = f.replace("/rg_comp_z/", "/" + old + "/")
        if not os.path.exists(g):
            print("[disj] missing", g)
            continue
        o = np.load(g)
        n = 0
        if o.shape[1:] == y.shape[1:]:
            for i in range(len(y)):
                if any(np.allclose(y[i], o[j]) for j in range(len(o))):
                    n += 1
        else:
            print("[disj] shape differs", f, y.shape, o.shape)
        bad += n
        print("[disj] %s vs %s: %d rows %d old rows, %d equal" % (os.path.basename(f), old, len(y), len(o), n), flush=True)
nz = np.load(R + "/n2_states_wz/states.npy")
for d in sorted(glob.glob(R + "/n2_states_*/states.npy")):
    if d.endswith("_wz/states.npy"):
        continue
    o = np.load(d)
    n = sum(any(np.allclose(nz[i], o[j]) for j in range(len(o))) for i in range(len(nz))) if o.shape[1:] == nz.shape[1:] else -1
    print("[disj] n2 wz vs %s: %d equal (-1 = other state width)" % (os.path.basename(os.path.dirname(d)), n), flush=True)
    bad += max(n, 0)
meta = json.load(open(R + "/rg_comp_z/rg_comp_z_tasks.json", encoding="utf-8"))
print("[disj] rg_comp_z tasks json keys:", list(meta.keys()), "states keys:", list(meta["states"].keys()), flush=True)
for fam in ("f7", "f6"):
    st = meta["states"][fam]
    print("[disj] %s: %s" % (fam, {k: (v if not isinstance(v, list) or len(v) < 8 else "list(%d) first %s last %s" % (len(v), v[0], v[-1])) for k, v in st.items()}), flush=True)
print("[disj] TOTAL EQUAL ROWS %d" % bad)
sys.exit(1 if bad else 0)
