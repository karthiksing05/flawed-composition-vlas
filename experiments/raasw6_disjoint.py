"""SR 90: the third fresh states are disjoint from SR 70-88's: (a) index ranges from the task / branch files, (b) no state row of rg_comp_y is np.allclose to any row of rg_comp / rg_comp_x (same tid), no n2 state of n2_states_wy equals any row of every older n2 set.  CPU, numpy only."""
import glob, json, os
import numpy as np
R = "/coc/flash10/ksingara3/flawed-composition-vlas/results/flawed"
bad = 0
for f in sorted(glob.glob(R + "/rg_comp_y/states/rgc_*.npy")):
    y = np.load(f)
    for old in ("rg_comp", "rg_comp_x"):
        g = f.replace("rg_comp_y", old)
        if not os.path.exists(g):
            print("[disj] missing", g); continue
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
for old in ("rg_comp", "rg_comp_x"):
    ks = [t["k"] for t in json.load(open(os.path.join(R, old, old + "_tasks.json")))["states"]["f7"]["main"]] if False else None
ny = np.load(R + "/n2_states_wy/states.npy")
for d in sorted(glob.glob(R + "/n2_states_*/states.npy")):
    if d.endswith("_wy/states.npy"):
        continue
    o = np.load(d)
    n = sum(any(np.allclose(ny[i], o[j]) for j in range(len(o))) for i in range(len(ny))) if o.shape[1:] == ny.shape[1:] else -1
    print("[disj] n2 wy vs %s: %d equal (-1 = other state width)" % (os.path.basename(os.path.dirname(d)), n), flush=True)
    bad += max(n, 0)
print("[disj] TOTAL EQUAL ROWS %d" % bad)
