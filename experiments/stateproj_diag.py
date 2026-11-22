"""EXPLORATORY, DESCRIPTIVE, NOT REGISTERED: what did state_proj (Linear(32,1024), random init) learn in each trained arm?
Params only, no model build.  Output results/flawed/raaf/stateproj_diag.json (refuses to overwrite).
state = [dx,dy,dz,role] quantile-normalised, zero-padded to 32: input dims 0-3 are live, dims 4-31 are always 0, so their
columns receive no gradient and are an exact sample of the random init (flax lecun_normal kernel, zero bias)."""
import json, os, sys, gc
import numpy as np
from openpi.models import model as M

ROOT = "/coc/flash10/ksingara3/robot/checkpoints_l90c/jax"
ARMS = {  # name: (config, exp)
    "RAA-sL": ("pi05_g8v_raas_l_ae", "main"),
    "A0": ("pi05_g8v_raac_l_ae", "tl"),
    "A": ("pi05_g8v_raac_lf_ae", "tl"),
    "B": ("pi05_g8v_raac_lb_ae", "tl"),
    "A+0": ("pi05_g8v_raac_lp_ae", "tl"),
    "A+": ("pi05_g8v_raas_lp_ae", "main"),
}
OUT = "/coc/flash10/ksingara3/flawed-composition-vlas/results/flawed/raaf/stateproj_diag.json"
if os.path.exists(OUT):
    sys.exit("refusing to overwrite " + OUT)

def find(tree, path=()):
    if isinstance(tree, dict):
        for k, v in tree.items():
            if "state_proj" in str(k):
                return v
            r = find(v, path + (k,))
            if r is not None:
                return r
    return None

def unbox(x):
    while isinstance(x, dict) and "value" in x:
        x = x["value"]
    return np.asarray(x, dtype=np.float32)

res = {}
rng = np.random.default_rng(0)
X = rng.uniform(-1, 1, size=(500, 4)).astype(np.float32)   # synthetic normalised inputs, NOT real frames
for arm, (cfg, exp) in ARMS.items():
    res[arm] = {"config": cfg, "exp": exp}
    for step in (1000, 6000):
        p = f"{ROOT}/{cfg}/{exp}/{step}/params"
        params = M.restore_params(p, restore_type=np.ndarray)
        sp = find(params)
        W, b = unbox(sp["kernel"]), unbox(sp["bias"])
        del params; gc.collect()
        cn = np.linalg.norm(W, axis=1)            # per input dim (rows of (32,1024))
        live, pad = cn[:4], cn[4:]
        y = X @ W[:4] + b
        res[arm][str(step)] = {
            "W_shape": list(W.shape), "W_fro": float(np.linalg.norm(W)), "b_fro": float(np.linalg.norm(b)),
            "col_norm_live_dims0_3": live.tolist(), "col_norm_pad_mean": float(pad.mean()), "col_norm_pad_std": float(pad.std()),
            "col_norm_pad_min": float(pad.min()), "col_norm_pad_max": float(pad.max()),
            "live_over_pad_mean": (live / pad.mean()).tolist(),
            "W_fro_if_untouched_est": float(np.sqrt((pad.mean() ** 2) * 32)),
            "synthetic_out_norm_mean": float(np.linalg.norm(y, axis=1).mean()),
            "synthetic_out_norm_no_bias": float(np.linalg.norm(X @ W[:4], axis=1).mean()),
        }
        res[arm][f"_W{step}"] = W[:4].copy()
        print(arm, step, res[arm][str(step)], flush=True)
    d = res[arm]["_W6000"] - res[arm]["_W1000"]
    res[arm]["delta_1000_to_6000_live_row_norms"] = np.linalg.norm(d, axis=1).tolist()
    res[arm]["delta_rel_to_live_row_norm_6000"] = (np.linalg.norm(d, axis=1) / np.linalg.norm(res[arm]["_W6000"], axis=1)).tolist()
for a in res:
    for k in [k for k in res[a] if k.startswith("_W")]:
        del res[a][k]
res["_note"] = ("EXPLORATORY, NOT REGISTERED. Init is random (not reproducible here); pad columns (dims 4-31, zero input, no gradient) are the init reference. "
                "Synthetic outputs use uniform[-1,1] 4-d inputs (500 samples), not real frames. (c) vs other token norms skipped: needs a model forward.")
json.dump(res, open(OUT, "w"), indent=1)
