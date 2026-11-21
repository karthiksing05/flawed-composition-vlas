"""RAA-F Step 0, check S0-4 (docs/notes/raaf_plan.md 0.1, 5): the servo-null derivation.  openpi venv, CPU (numpy only).  sbatch only.

For every accepted relabelled copy d' (rows h0 .. e - 1 of d, actions + K delta, token rel + delta), compare per chunk start t the expert's target
under the SERVO-KEPT model (RAA-sL: chunk actions xyz minus clip(K rel_t, -M, M), the servo taken from the chunk's first frame) for the copy
against the original d at the same row.  Chunks are the loader's: row t + j, j < 10, clamped at the episode's last row (the copy ends at e - 1;
the original d continues).  Reported separately: chunks fully inside the copy (t + 9 <= e - 1 - h0) with both servos unclipped (the derivation's
condition: "all held and unclipped"), the same with a clipped servo, and the truncated tail chunks.  Also the servo-free target difference
(= K delta on every inside chunk), the size of the pressure that Route A trains on.

    python experiments/raaf_null.py --prep P --delta D --out O        (O must not exist)
"""
import argparse
import json
import os
import pickle

import numpy as np

K, M, H = 2.851209548740752, 0.9375, 10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", required=True)
    ap.add_argument("--delta", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    assert not os.path.exists(a.out), a.out
    P = pickle.load(open(a.prep, "rb"))
    D = json.load(open(a.delta, encoding="utf-8"))
    by = {(d["task"], d["demo"]): d for d in P["demos"]}
    acc = dict(inside_unclipped=[], inside_clipped=[], tail=[], free_inside=[])
    n_copies = 0
    for dd in D["demos"]:
        d = by[(dd["task"], dd["demo"])]
        h0, e = dd["h0"], dd["e"]
        L = e - h0
        for cp in dd["copies"]:
            n_copies += 1
            dl = np.asarray(cp["delta"])
            a_o = d["actions"]
            a_n = d["actions"][h0:e].copy()
            a_n[:, :3] += K * dl[None, :]
            n_orig = len(a_o)
            for t in range(L):
                idx_n = np.minimum(t + np.arange(H), L - 1)
                idx_o = np.minimum(h0 + t + np.arange(H), n_orig - 1)
                rel_o, rel_n = d["rel"][h0 + t], d["rel"][h0 + t] + dl
                so, sn = K * rel_o, K * rel_n
                clipped = bool(np.any(np.abs(so) > M) or np.any(np.abs(sn) > M))
                tgt_o = a_o[idx_o, :3] - np.clip(so, -M, M)[None, :]
                tgt_n = a_n[idx_n, :3] - np.clip(sn, -M, M)[None, :]
                diff = float(np.abs(tgt_n - tgt_o).max())
                inside = (t + H - 1) <= (L - 1)
                if inside:
                    (acc["inside_clipped"] if clipped else acc["inside_unclipped"]).append(diff)
                    acc["free_inside"].append(float(np.abs((a_n[idx_n, :3] - a_o[idx_o, :3])).max()))
                else:
                    acc["tail"].append(diff)

    def s(v):
        v = np.asarray(v, np.float64)
        return dict(n=int(len(v)), max=float(v.max()) if len(v) else None, median=float(np.median(v)) if len(v) else None,
                    frac_above_1e6=float((v > 1e-6).mean()) if len(v) else None)
    out = dict(copies=n_copies, inside_unclipped_servo=s(acc["inside_unclipped"]), inside_clipped_servo=s(acc["inside_clipped"]),
               tail_truncated=s(acc["tail"]), servo_free_target_change_inside=s(acc["free_inside"]),
               pass_s0_4=bool(len(acc["inside_unclipped"]) > 0 and max(acc["inside_unclipped"]) < 1e-6),
               rule="pass iff every inside, unclipped chunk has |servo-kept target(copy) - target(original)| < 1e-6")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print("[raaf_null] " + json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
