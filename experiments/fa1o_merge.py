"""FLAWED_COMPOSITION prototype A, stage 1o (section 5.27), DATA: the generated carries re-merged WITHOUT the no-op filter.
5.27's diagnostic: every generated merge (fa1g_plan -> gen_g, fa1k_plan -> gen_k, fa1l_cgen -> gen_lc) dropped no-op steps with
fa1_data.noop_mask "as for the demos".  The scripted planner's only no-op steps are its gripper DWELLS (phase 2: close in place for 12
steps; phase 6: open in place for 12 steps), so the merged data lost them: a generated grasp is one close row followed at once by the
lift, while the planner itself waited 12 steps for the fingers.  The demos' no-op steps are teleoperation idle frames; the
planner's dwell is part of what it executed.
This script rebuilds the same three merges, in the same order, demo ids and per-row fields (contact hold, grasp offset, lift, seg)
recomputed exactly as before, and changes ONE thing: every executed row is kept (no no-op filter).  --noop restores the filter and
must reproduce DATA/gen_lc.npz exactly (regression check).  No demo action is executed or replayed; the raw per-task files
(DATA/gen, DATA/gen_k, DATA/gen_lc) are the planner's own recorded runs.
-> DATA/gen_lo.npz (or, with --noop, DATA/gen_lo_noop.npz), results/flawed/a1o_merge.json

    python experiments/fa1o_merge.py [--noop]
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import fa1_common as FC                                                     # noqa: E402

OUT = os.path.join(ROOT, "results", "flawed")
KEYS = ("task", "demo", "step", "action", "ee_pos", "ee_rot", "grip", "pos", "yaw", "drawer", "prev_grip", "held", "off", "touch", "lift")
TASKS = ["K1", "K2", "K3", "K4", "K5", "K6"]
STAGES = (("gen", 100, False), ("gen_k", 150, False), ("gen_lc", 400, True))   # (raw dir, demo id base, has seg)


def main():
    import fa1b_common as FB
    from fa1_data import noop_mask
    ap = argparse.ArgumentParser()
    ap.add_argument("--noop", action="store_true", help="keep the old no-op filter (regression check against gen_lc.npz)")
    a = ap.parse_args()
    cols, seg = {x: [] for x in KEYS}, []
    rep = {}
    for src, base, has_seg in STAGES:
        r = dict(episodes=0, rows=0, rows_noop_filtered=0, dwell_rows_restored=0)
        for k in TASKS:
            G = dict(np.load(os.path.join(FC.DATA, src, f"{k}.npz")))
            tix = int(G["task_index"])
            for e in (np.unique(G["ep"]) if len(G["ep"]) else []):
                m = np.nonzero(G["ep"] == e)[0]
                mk = noop_mask(G["action"][m])
                keep = m[mk] if a.noop else m
                r["rows_noop_filtered"] += int(mk.sum())
                r["dwell_rows_restored"] += int((~mk).sum())
                hm, prev = FB.Held(), -1.0
                sg = int(G["seg"][m[0]]) if has_seg else 0
                for s, i in enumerate(keep):
                    lt, rt = bool(G["touch"][i][0]), bool(G["touch"][i][1])
                    h, off = hm.step(lt, rt, G["objp"][i], G["ee_pos"][i])
                    for x in ("action", "ee_pos", "ee_rot", "grip", "pos", "yaw", "drawer"):
                        cols[x].append(G[x][i])
                    cols["task"].append(tix); cols["demo"].append(base + int(e)); cols["step"].append(s); cols["prev_grip"].append(prev)
                    cols["held"].append(float(h)); cols["off"].append(off); cols["touch"].append((lt, rt))
                    cols["lift"].append(G["objp"][i][2] - hm.z0)
                    seg.append(sg)
                    prev = float(G["action"][i][6])
                r["episodes"] += 1
                r["rows"] += len(keep)
        rep[src] = r
        print(f"[fa1o] merge {src}: {json.dumps(r)}", flush=True)
    out = {x: np.asarray(v, np.float32 if x not in ("task", "demo", "step") else np.int32) for x, v in cols.items()}
    assert out["demo"].max() < 1000 and out["demo"].min() >= 100
    name = "gen_lo_noop.npz" if a.noop else "gen_lo.npz"
    np.savez_compressed(os.path.join(FC.DATA, name), gen=np.ones(len(out["task"]), np.int8), seg=np.asarray(seg, np.int8), **out)
    res = dict(file=name, noop_filter=a.noop, rows=int(len(out["task"])), stages=rep, generated=True)
    ref = dict(np.load(os.path.join(FC.DATA, "gen_lc.npz")))
    if a.noop:
        same = all(ref[x].shape == out[x].shape and np.array_equal(ref[x], out[x]) for x in KEYS)
        res["reproduces_gen_lc"] = bool(same)
        print(f"[fa1o] REGRESSION reproduces gen_lc.npz exactly: {same} ({len(ref['task'])} vs {len(out['task'])} rows)", flush=True)
    else:
        res["gen_lc_rows"] = int(len(ref["task"]))
    with open(os.path.join(OUT, "a1o_merge_noop.json" if a.noop else "a1o_merge.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
