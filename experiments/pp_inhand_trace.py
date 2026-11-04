"""In-hand test I1 and I2 (docs/PLACE_DIAG.md 4; m3_prereg pp_inhand): P_min's rollouts with the object's in-hand state read at arrival.
fa1h_roll.episode, unchanged, the same seeds and states.  --arm base changes nothing the policy sees.  --arm norot sets the commanded rotation a[3:6]
to 0 on every step where the oracle hold is on (a diagnostic, not a candidate fix).  LIBERO venv, CPU.  Dev states only.
-> results/flawed/pp_inhand_trace/<arm>_s<seed>_<compound>_<layout>.json

    python experiments/pp_inhand_trace.py --arm base --seed 10 --task K1:new
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


def mat_wxyz(q):
    w, x, y, z = [float(v) for v in q]
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def ang(R):
    return float(math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(R) - 1.0) / 2.0)))))


def zang(R0, R):
    return float(math.degrees(math.acos(max(-1.0, min(1.0, float(R0[:, 2] @ R[:, 2]))))))


class Rec:
    def __init__(self, env, oe, te, norot):
        self._e, self._oe, self._te, self._norot, self.n, self.rows = env, oe, te, norot, 0, []
        self._con, self._hm, self._dj = None, FB.Held(), None
        self.zeroed = 0

    def __getattr__(self, k):
        return getattr(self._e, k)

    def step(self, a):
        if self._norot and self._hm.on:
            a = list(a)
            self.zeroed += int(any(abs(x) > 0 for x in a[3:6]))
            a[3:6] = [0.0, 0.0, 0.0]
        r = self._e.step(a)
        self.n += 1
        if self.n < RH.WAIT:
            return r
        if self._con is None:
            self._con, self._dj = FB.Contact(self._e, [self._oe]), FC.drawer_joint(self._e)
        E = self._e.env
        h, _ = self._hm.step(*self._con.touching(self._oe), self._con.pos(self._oe), r[0]["robot0_eef_pos"])
        pos = FC.scene(self._e, r[0], self._dj)["pos"]
        q = np.asarray(r[0]["robot0_eef_quat"], np.float64)                  # robosuite: (x, y, z, w)
        self.rows.append(dict(obj=pos[FC.ENT.index(self._oe)].copy(), tgt=pos[FC.ENT.index(self._te)].copy(), table=float(pos[FK.TABLE, 2]),
                              Ro=mat_wxyz(E.sim.data.body_xquat[E.obj_body_id[self._oe]]), Re=mat_wxyz([q[3], q[0], q[1], q[2]]),
                              ee=np.asarray(r[0]["robot0_eef_pos"], np.float64).copy(), held=bool(h)))
        return r


def at_arrival(rows, r, zrest):
    arr = next((i for i, x in enumerate(rows) if i >= 1 and x["held"] and np.linalg.norm((x["obj"] - x["tgt"])[:2]) <= r), None)
    if arr is None:
        return None
    on = arr
    while on > 0 and rows[on - 1]["held"]:
        on -= 1
    x, o, s = rows[arr], rows[on], rows[0]
    f = x["obj"].copy()
    f[2] = f[2] - zrest + x["table"]
    rel = f - x["tgt"]
    h0, h1 = o["Re"].T @ o["Ro"], x["Re"].T @ x["Ro"]
    off0, off1 = o["Re"].T @ (o["obj"] - o["ee"]), x["Re"].T @ (x["obj"] - x["ee"])
    return dict(arrive=arr, onset=on, tilt=zang(s["Ro"], x["Ro"]), tilt_onset=zang(s["Ro"], o["Ro"]), hand_turn=ang(o["Re"].T @ x["Re"]), slip=ang(h0.T @ h1),
                offset_shift=float(np.linalg.norm(off1 - off0)), d=float(np.linalg.norm(rel[:2])), h=float(rel[2]),
                tilt_max_after=max(zang(s["Ro"], y["Ro"]) for y in rows[arr:]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=("base", "norot"))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--task", required=True)
    a = ap.parse_args()
    torch.set_num_threads(1)
    comp, lay = a.task.split(":")
    od = os.path.join(OUT, "pp_inhand_trace")
    os.makedirs(od, exist_ok=True)
    fn = os.path.join(od, f"{a.arm}_s{a.seed}_{comp}_{lay}.json")
    if os.path.exists(fn):
        print("exists", fn, flush=True)
        return
    n2 = json.load(open(os.path.join(OUT, "n2_tasks.json"), encoding="utf-8"))
    cal = json.load(open(os.path.join(OUT, "a1_calib.json"), encoding="utf-8"))["pairs"]
    br = json.load(open(os.path.join(OUT, "n2_branches_main.json"), encoding="utf-8"))
    t = next(x for x in n2["tasks"] if x["compound"] == comp and x["layout"] == lay)
    rad = radius(t["object"], t["target"], cal)
    m, _ = load_any(os.path.join(FC.DATA, "ckpt", f"pp_full_naf_s{a.seed}", "P.npz"))
    S = np.load(os.path.join(OUT, "n2_states", f"n2_goal_{t['tid']}.npy"))
    orig, eps = FC.fresh_env, []
    for b in br:
        rec = {}

        def mk(bddl, seed=7):
            rec["r"] = Rec(orig(bddl, seed), t["object"], t["target"], a.arm == "norot")
            return rec["r"]

        FC.fresh_env = mk
        try:
            e = RH.episode(m, os.path.join(ROOT, t["bddl"]), S[int(b["trial"])], t["language"], b["seed_base"])
        finally:
            FC.fresh_env = orig
        rows = rec["r"].rows
        last = rows[-1]
        zr = e["onset_canon"]["zrest"] if e["onset_canon"] else None
        f = last["obj"][2] - (zr if zr is not None else last["obj"][2]) + last["table"]
        r = dict(trial=int(b["trial"]), success=e["success"], error=e["error"], steps=e["steps"], held_step=e["held_step"], held_segments=e["held_segments"],
                 held_end=e["held_end"], min_dxy_held=e["min_dxy_held"], zeroed=rec["r"].zeroed,
                 end=dict(d=float(np.linalg.norm((last["obj"] - last["tgt"])[:2])), h=float(f - last["tgt"][2]), tilt=zang(rows[0]["Ro"], last["Ro"])),
                 arrival=at_arrival(rows, rad, zr) if zr is not None else None)
        eps.append(r)
        print("[inhand]", a.arm, comp, lay, a.seed, b["trial"], e["success"], flush=True)
    with open(fn, "w", encoding="utf-8") as fh:
        json.dump(dict(arm=a.arm, compound=comp, layout=lay, seed=a.seed, tid=t["tid"], object=t["object"], target=t["target"], radius=rad,
                       host=os.uname().nodename, episodes=eps), fh, indent=1)


if __name__ == "__main__":
    main()
