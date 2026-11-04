"""Place diagnostic T1 (docs/PLACE_DIAG.md 1.3; m3_prereg pp_place): P_min's rollouts with the state read after each step.
fa1h_roll.episode, unchanged, the same seeds and states; env.step is wrapped to read the simulator, and nothing the policy sees changes.
Per episode and per radius of interest, the release state: the state just before the first open command at or after the hold first
reaches the radius.  LIBERO venv, CPU.  Dev states only.  -> results/flawed/pp_place_trace/P_s<seed>_<compound>_<layout>.json

    python experiments/pp_place_trace.py --seed 10 --task K1:new
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT]
import fa1_common as FC                                                     # noqa: E402
import fa1b_common as FB                                                    # noqa: E402
import fa1c_common as FK                                                    # noqa: E402
import fa1h_roll as RH                                                      # noqa: E402
from fa1g_read import radius                                                # noqa: E402
from pp_p_roll import load_any                                              # noqa: E402

OUT = os.path.join(ROOT, "results", "flawed")
PARTNERS = {"plate_1": ("akita_black_bowl_1", "cream_cheese_1", "wine_bottle_1"), "akita_black_bowl_1": ("cream_cheese_1", "wine_bottle_1")}


def tilt_deg(q_wxyz):
    w, x, y, z = [float(v) for v in q_wxyz]
    return float(math.degrees(math.acos(max(-1.0, min(1.0, 1.0 - 2.0 * (x * x + y * y))))))


class Rec:
    """Reads the state after each step, from the WAIT-th step on (the step of the episode's first track()), with its own Held."""

    def __init__(self, env, oe, te):
        self._e, self._oe, self._te, self.n, self.rows = env, oe, te, 0, []
        self._con, self._hm, self._dj = None, FB.Held(), None

    def __getattr__(self, k):
        return getattr(self._e, k)

    def step(self, a):
        r = self._e.step(a)
        self.n += 1
        if self.n < RH.WAIT:
            return r
        if self._con is None:
            self._con, self._dj = FB.Contact(self._e, [self._oe]), FC.drawer_joint(self._e)
        E = self._e.env
        h, _ = self._hm.step(*self._con.touching(self._oe), self._con.pos(self._oe), r[0]["robot0_eef_pos"])
        pos = FC.scene(self._e, r[0], self._dj)["pos"]
        bo = E.obj_body_id[self._oe]
        self.rows.append(dict(a6=float(a[6]), obj=pos[FC.ENT.index(self._oe)].copy(), tgt=pos[FC.ENT.index(self._te)].copy(),
                              table=float(pos[FK.TABLE, 2]), quat=np.array(E.sim.data.body_xquat[bo], np.float64), held=bool(h),
                              contact=bool(E.check_contact(E.get_object(self._oe), E.get_object(self._te)))))
        return r


def release(rows, r, zrest):
    """The release state for radius r: None if the hold never reaches r."""
    arr = next((i for i, x in enumerate(rows) if i >= 1 and x["held"] and np.linalg.norm((x["obj"] - x["tgt"])[:2]) <= r), None)
    if arr is None:
        return None
    j = next((i for i in range(arr, len(rows)) if rows[i]["a6"] < 0), None)
    if j is None:
        return dict(none=True, arrive=arr)
    x = rows[j - 1]
    f = x["obj"].copy()
    f[2] = f[2] - zrest + x["table"]
    rel = f - x["tgt"]
    return dict(none=False, arrive=arr, open=j, d=float(np.linalg.norm(rel[:2])), h=float(rel[2]), tilt=tilt_deg(x["quat"]),
                held_at_open=bool(x["held"]), tgt_moved=float(np.linalg.norm((x["tgt"] - rows[0]["tgt"])[:2])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--task", required=True)
    a = ap.parse_args()
    torch.set_num_threads(1)
    comp, lay = a.task.split(":")
    od = os.path.join(OUT, "pp_place_trace")
    os.makedirs(od, exist_ok=True)
    fn = os.path.join(od, f"P_s{a.seed}_{comp}_{lay}.json")
    if os.path.exists(fn):
        print("exists", fn, flush=True)
        return
    n2 = json.load(open(os.path.join(OUT, "n2_tasks.json"), encoding="utf-8"))
    cal = json.load(open(os.path.join(OUT, "a1_calib.json"), encoding="utf-8"))["pairs"]
    br = json.load(open(os.path.join(OUT, "n2_branches_main.json"), encoding="utf-8"))
    t = next(x for x in n2["tasks"] if x["compound"] == comp and x["layout"] == lay)
    radii = sorted({round(radius(o, t["target"], cal), 4) for o in PARTNERS[t["target"]]})
    m, _ = load_any(os.path.join(FC.DATA, "ckpt", f"pp_full_naf_s{a.seed}", "P.npz"))
    S = np.load(os.path.join(OUT, "n2_states", f"n2_goal_{t['tid']}.npy"))
    orig, eps = FC.fresh_env, []
    for b in br:
        rec = {}

        def mk(bddl, seed=7):
            rec["r"] = Rec(orig(bddl, seed), t["object"], t["target"])
            return rec["r"]

        FC.fresh_env = mk
        try:
            e = RH.episode(m, os.path.join(ROOT, t["bddl"]), S[int(b["trial"])], t["language"], b["seed_base"])
        finally:
            FC.fresh_env = orig
        rows = rec["r"].rows
        zr = e["onset_canon"]["zrest"] if e["onset_canon"] else None
        last = rows[-1]
        r = dict(trial=int(b["trial"]), success=e["success"], error=e["error"], steps=e["steps"], held_step=e["held_step"],
                 held_segments=e["held_segments"], held_end=e["held_end"], min_dxy_held=e["min_dxy_held"],
                 end=dict(d=float(np.linalg.norm((last["obj"] - last["tgt"])[:2])), z_rel=float(last["obj"][2] - last["tgt"][2]),
                          tilt=tilt_deg(last["quat"]), contact=last["contact"], tgt_moved=float(np.linalg.norm((last["tgt"] - rows[0]["tgt"])[:2]))),
                 release={f"{x:.4f}": release(rows, x, zr) for x in radii} if zr is not None else {f"{x:.4f}": None for x in radii})
        eps.append(r)
        print("[trace]", comp, lay, a.seed, b["trial"], e["success"], json.dumps(r["end"]), flush=True)
    with open(fn, "w", encoding="utf-8") as fh:
        json.dump(dict(compound=comp, layout=lay, seed=a.seed, tid=t["tid"], object=t["object"], target=t["target"], radii=radii,
                       host=os.uname().nodename, episodes=eps), fh, indent=1)


if __name__ == "__main__":
    main()
