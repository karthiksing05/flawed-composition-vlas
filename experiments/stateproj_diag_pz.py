"""EXPLORATORY, DESCRIPTIVE, NOT REGISTERED (docs/notes/plan_oracle_pi0.md 3.4 step 7): how far did pi0's TRAINED state_proj move from its init (pi0_libero) in each pz arm?
Params only, no model build, CPU.  pi0's init IS saved (pi0_libero/params), so the change is exact: per live input dim (columns 0-3 of the (32, 1024) kernel: dx, dy, dz, role)
Frobenius change and the change relative to the init row norm, the whole-kernel and bias change, and the pad dims 4-31 (zero input, no gradient) as the control.
Output results/flawed/pz/stateproj_diag_pz.json (refuses to overwrite).  A flag per arm and step; ARMS may be restricted with --arms."""
import argparse
import gc
import json
import os
import sys

import numpy as np
from openpi.models import model as M

ROOT = "/coc/flash10/ksingara3/robot/checkpoints_l90c/jax"
INIT = "/coc/flash10/ksingara3/robot/openpi_cache/openpi-assets/checkpoints/pi0_libero/params"
ARMS = {"pzsL": ("pi0_g8v_raas_l_ae", "tl"), "pzA0": ("pi0_g8v_raac_l_ae", "tl")}
OUT = "/coc/flash10/ksingara3/flawed-composition-vlas/results/flawed/pz/stateproj_diag_pz.json"


def find(tree):
    if isinstance(tree, dict):
        for k, v in tree.items():
            if "state_proj" in str(k):
                return v
            r = find(v)
            if r is not None:
                return r
    return None


def unbox(x):
    while isinstance(x, dict) and "value" in x:
        x = x["value"]
    return np.asarray(x, dtype=np.float32)


def load(p):
    params = M.restore_params(p, restore_type=np.ndarray)
    sp = find(params)
    W, b = unbox(sp["kernel"]), unbox(sp["bias"])
    del params
    gc.collect()
    return W, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--steps", default="1000,6000")
    a = ap.parse_args()
    if os.path.exists(OUT):
        sys.exit("refusing to overwrite " + OUT)
    W0, b0 = load(INIT)
    res = {"init": {"W_fro": float(np.linalg.norm(W0)), "col_norm_live_dims0_3": np.linalg.norm(W0[:4], axis=1).tolist(), "col_norm_pad_mean": float(np.linalg.norm(W0[4:], axis=1).mean())}}
    for arm in a.arms.split(","):
        cfg, exp = ARMS[arm]
        res[arm] = {"config": cfg, "exp": exp}
        for step in [int(s) for s in a.steps.split(",")]:
            W, b = load(f"{ROOT}/{cfg}/{exp}/{step}/params")
            d = W - W0
            live = np.linalg.norm(d[:4], axis=1)
            res[arm][str(step)] = {
                "delta_fro_all": float(np.linalg.norm(d)), "delta_fro_rel_to_init_all": float(np.linalg.norm(d) / np.linalg.norm(W0)),
                "delta_row_norm_live_dims0_3": live.tolist(), "delta_rel_to_init_row_norm_live": (live / np.linalg.norm(W0[:4], axis=1)).tolist(),
                "delta_live_fro": float(np.linalg.norm(d[:4])), "delta_live_rel_fro": float(np.linalg.norm(d[:4]) / np.linalg.norm(W0[:4])),
                "delta_pad_fro_control": float(np.linalg.norm(d[4:])), "delta_pad_max_abs": float(np.abs(d[4:]).max()),
                "bias_delta_fro": float(np.linalg.norm(b - b0)), "bias_delta_rel": float(np.linalg.norm(b - b0) / max(np.linalg.norm(b0), 1e-9)),
            }
            print(arm, step, json.dumps(res[arm][str(step)]), flush=True)
    res["_note"] = ("EXPLORATORY, NOT REGISTERED. Init = pi0_libero's saved state_proj (exact). Live dims 0-3 = [dx, dy, dz, role] (they replaced eef xyz + axis-angle x of pi0_libero's proprio); "
                    "pad dims 4-31 receive zero input and no gradient (control).")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "x", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
