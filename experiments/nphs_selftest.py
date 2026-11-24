"""NPHS smoke (CPU): imports and a synthetic hierarchical-bootstrap run; no results are read."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nphs_read as N
import nphs_e1  # noqa: F401
rng = np.random.default_rng(1)
A = {s: (rng.random((6, 60)) < 0.6).astype(float) for s in (42, 43, 44)}
Bg = {s: (rng.random((6, 60)) < 0.5).astype(float) for s in (42, 43, 44)}
r = N.hboot(A, Bg, [42, 43, 44], [42, 43, 44])
print("[selftest]", r["delta"], r["ci"], r["p_le0"], N.V.holm(dict(a=r["p_le0"], b=r["p_ge0"])))
assert r["ci"][0] < r["delta"] < r["ci"][1]
