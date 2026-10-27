"""PLAN_BINDER B1 diagnostic (exploratory, after the read): how much absolute token position the binder's APPEARANCE channel carries.
Linear R^2 of the token's (row, col) from its 32-d PCA appearance feature, on TRAINING frames only (S1's training frames), per arm.
CPU, chunks env.  -> results/bind/b1/posdiag.json"""
import json
import os
import pickle
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "experiments")]
import b1_bind as BB  # noqa: E402

sys.modules["__main__"].Lexicon = BB.Lexicon
sys.modules["__main__"].PCA = BB.PCA
out = {}
rr, cc = np.divmod(np.arange(256), 16)
for arm in ("siglip", "dino"):
    M = pickle.load(open(os.path.join(ROOT, "results", "bind", "b1", arm, "model.pkl"), "rb"))
    idx, feats = BB.load_feats(arm)
    X = np.asarray(feats["train"][::8], np.float32)
    Z = M["pca"](X.reshape(-1, X.shape[-1]))
    Y = np.stack([np.tile(rr, len(X)), np.tile(cc, len(X))], 1).astype(np.float64)
    n = len(Z) // 2
    A = np.column_stack([np.ones(len(Z)), Z])
    b, *_ = np.linalg.lstsq(A[:n], Y[:n], rcond=None)
    pred = A[n:] @ b
    r2 = 1 - ((Y[n:] - pred) ** 2).sum(0) / ((Y[n:] - Y[n:].mean(0)) ** 2).sum(0)
    out[arm] = dict(r2_row=float(r2[0]), r2_col=float(r2[1]), n_tokens=int(len(Z)))
    print(f"[b1 posdiag] {arm}: held-out-token linear R^2 of (row, col) from the 32-d appearance PCA: {r2[0]:.3f}, {r2[1]:.3f}", flush=True)
json.dump(out, open(os.path.join(ROOT, "results", "bind", "b1", "posdiag.json"), "w"), indent=1)
