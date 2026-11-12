"""FX diagnostics: the reference for criterion (1) (docs/PLAN_ARCH_FAILURES.md section 1).  The scripted servo (pp_n4d.plan_episode
with servo=True, the atlas's planner, its controller and phase rules unchanged; a1g_keypoints K1 for bowl -> plate, K2 for bowl
-> stove) on each fx task's own start states, given the parsed referent (the task's `referent`, named to the planner) and, for
F6, the REMEMBERED target: the target's position is read once, at the first readout after the settle steps (t = 0), and that
value is used for the whole episode (`freeze`).  The servo reads oracle state and renders nothing, so F6's hide cannot touch it:
its row is the same for vis / hid / abs, which share states.  Noise rng = seed_base + 10, as the atlas's servo.
LIBERO venv, CPU (ControlEnv without cameras).

    python experiments/fx_ref.py --phase smoke|main [--tids 0+2] [--out DIR] [--pseed 11]
        -> DIR/servo_<tid>_<phase>[_p<pseed>].json   (fx_read averages the planner seeds per state)
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT]
import fx_tasks as FT                                                      # noqa: E402

KP = os.path.join(ROOT, "results", "flawed", "a1g_keypoints.json")


def spatial_ent(bddl, oe, te):
    """LIBERO-Spatial's scene: fa1_common's 9 slots re-filled with this scene's names (4 movable bodies, the referent and
    target first, then 5 main-table region sites; slot 8 is a table-surface site, as LIBERO-Goal's stove-front region is)."""
    import fa1_common as FC
    env = FC.make_env(bddl)
    m = env.env.sim.model
    sites = [m.site_id2name(i) for i in range(m.nsite)]
    tbl = [s for s in sites if s and s.startswith("main_table_") and s.endswith("_region")]
    mov = [oe, te] + [n for n in env.env.objects_dict if n not in (oe, te)][:2]
    env.close()
    assert len(tbl) >= 5 and len(mov) == 4, (tbl, mov)
    return mov + tbl[:5]


def run(a):
    import fa1_common as FC
    import pp_n4d as N
    meta = json.load(open(FT.META, encoding="utf-8"))
    kps = json.load(open(KP, encoding="utf-8"))["tasks"]
    br = json.load(open(os.path.join(FT.OUT, f"fx_branches_{a.phase}.json"), encoding="utf-8"))
    od = a.out or os.path.join(FT.OUT, "ref")
    os.makedirs(od, exist_ok=True)
    ent0, mov0, scene0 = list(FC.ENT), list(FC.MOVABLE), FC.scene
    mem = dict(ti=None, pos=None)

    def frozen(env, obs, dj):
        sc = scene0(env, obs, dj)
        if mem["ti"] is not None:
            if mem["pos"] is None:
                mem["pos"] = sc["pos"][mem["ti"]].copy()
            sc["pos"][mem["ti"]] = mem["pos"]
        return sc

    tids = [int(x) for x in a.tids.split("+")] if a.tids else [t["tid"] for t in meta["tasks"] if t["ref"]]
    for tid in tids:
        t = meta["tasks"][tid]
        if not t["ref"]:
            continue
        if a.kind == "pmin" and t["diag"] == "F7P":
            continue                                                       # P_min's features are LIBERO-Goal's 9 slots
        fn = os.path.join(od, (f"servo_{tid}_{a.phase}" + ("" if a.pseed == 10 else f"_p{a.pseed}") if a.kind == "servo"
                               else f"pmin_{tid}_{a.phase}_s{a.pseed}") + ".json")
        if os.path.exists(fn):
            print("[ref] exists", fn, flush=True)
            continue
        bddl = os.path.join(ROOT, t["bddl"])
        oe, te = t["referent"], t["target"]
        if a.kind == "pmin":
            import torch
            import fa1h_roll as RH
            from pp_p_roll import load_any
            torch.set_num_threads(1)
            model, _ = load_any(os.path.join(FC.DATA, "ckpt", f"pp_full_naf_s{a.pseed}", "P.npz"))
            roles0 = FC.roles
            FC.roles = lambda lang, _r=(oe, te): _r                       # the parsed referent and target, given
        if t["diag"] == "F7P":
            FC.ENT = spatial_ent(bddl, oe, te)
            FC.MOVABLE = FC.ENT[:4]
        else:
            FC.ENT, FC.MOVABLE = list(ent0), list(mov0)
        FC.scene = frozen if t["ref"]["freeze"] else scene0
        kp = kps[t["ref"]["kp"]]
        S = np.load(os.path.join(FT.ST, f"fx_{tid}.npy"))
        eps = []
        for b in br:
            mem["ti"], mem["pos"] = (FC.ENT.index(te) if t["ref"]["freeze"] else None), None
            try:
                if a.kind == "pmin":
                    e = RH.episode(model, bddl, S[int(b["trial"])], t["language"], b["seed_base"])
                    info = dict(success=e["success"], held_step=None, min_dxy_held=e.get("min_dxy_held"), steps=e.get("steps"))
                    if e.get("error"):
                        raise RuntimeError(e["error"])
                else:
                    _, info = N.plan_episode(bddl, S[int(b["trial"])], kp, oe, te, np.random.default_rng(int(b["seed_base"]) + a.pseed), servo=True)
                eps.append(dict(trial=int(b["trial"]), success=bool(info["success"]), held_step=info["held_step"],
                                min_dxy_held=info["min_dxy_held"], steps=info["steps"], frozen_target=None if mem["pos"] is None else mem["pos"].tolist(),
                                error=None))
            except Exception as ex:                                        # noqa: BLE001
                eps.append(dict(trial=int(b["trial"]), success=False, error=repr(ex)))
            print(f"[ref] {tid} {t['key']} state {b['trial']}: success={eps[-1]['success']} err={eps[-1]['error']}", flush=True)
        FC.scene, FC.ENT, FC.MOVABLE = scene0, list(ent0), list(mov0)
        if a.kind == "pmin":
            FC.roles = roles0
        res = dict(tid=tid, key=t["key"], referent=oe, target=te, ref=t["ref"], phase=a.phase, kind=a.kind, seed=a.pseed, eps=eps,
                   success_rate=float(np.mean([e["success"] for e in eps])))
        with open(fn, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=1)
        print(f"[ref] {tid} {t['key']} {a.kind} success {res['success_rate']:.2f}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=("smoke", "main"))
    ap.add_argument("--tids", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--pseed", type=int, default=10, help="servo: noise rng = seed_base + pseed (the atlas: 10); pmin: the checkpoint "
                    "seed pp_full_naf_s<pseed> (10, 11, 12)")
    ap.add_argument("--kind", default="servo", choices=("servo", "pmin"))
    run(ap.parse_args())
