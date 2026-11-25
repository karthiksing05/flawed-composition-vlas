"""NPH (copy of rvw5_disjoint.py, N/K part only; docs/notes/plan_necessity_power.md 3.2 step 7, kill k3): no n2 state of n2_states_w8 equals any row of every other n2_states_* set (incl. w5).
CPU, numpy only.  Exit 1 if any row is equal."""
import glob
import os
import sys

import numpy as np

R = "/coc/flash10/ksingara3/flawed-composition-vlas/results/flawed"
nz = np.load(R + "/n2_states_w8/states.npy")
bad = 0
for d in sorted(glob.glob(R + "/n2_states_*/states.npy")):
    if d.endswith("_w8/states.npy"):
        continue
    o = np.load(d)
    n = sum(any(np.allclose(nz[i], o[j]) for j in range(len(o))) for i in range(len(nz))) if o.shape[1:] == nz.shape[1:] else -1
    print("[disj] n2 w8 vs %s: %d rows %d old rows, %d equal (-1 = other state width)" % (os.path.basename(os.path.dirname(d)), len(nz), len(o), n), flush=True)
    bad += max(n, 0)
print("[disj] TOTAL EQUAL ROWS %d" % bad)
sys.exit(1 if bad else 0)
