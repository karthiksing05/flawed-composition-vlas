"""PLAN_PARTS step 1, the N4 instrument fixed (docs/PARTS_RESULTS.md section 18; m3_prereg pp_n4d2).

genfam  (LIBERO venv, CPU) --task N4|K1|K4: 50 generated carries from FAMILIAR-layout starts (N2's familiar sampler, seeds
        960000 + 1000 j + k, j = 0 / 1 / 2 for N4 / K1 / K4; no relocation), stage 1g's planner (pp_n4d.plan_episode, unchanged)
        with pp_n4d's keypoints (N4 composed by the relocation rule) -> DATA/gen/pp_n4d2_<task>.npz, results/flawed/pp_n4d2_gen_<task>.json
train   (GPU) --task T --seed s: pp_n4d.train's single-task fit (PartsFlow rank 32, P's recipe) on pp_n4d2_<task>.npz
        -> DATA/ckpt/pp_n4d2_<task>_s<seed>/P.npz
roll    (LIBERO venv, CPU) --ckpt C --name N --tids 22+6+0 [--oracle TASK]: fa1h_roll.episode unchanged (or the planner), dev states
        4-23, N2's seeds, with the instruction object's TILT recorded each step (angle of its body z-axis to world z, from the body
        quaternion) -> results/flawed/pp_n4d2_roll/<N>_s<seed>_main_<tid>.json; per episode max_tilt_held, final_tilt, tipped
read    (CPU) pp_n4d's read on pp_n4d2_roll (arms N4d2_<task>, P2r32, oracle) plus the gate and the tilt split
        -> results/flawed/pp_n4d2_read.json
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT]
import fa1_common as FC                                                     # noqa: E402
import pp_n4d as N                                                          # noqa: E402

OUT = os.path.join(ROOT, "results", "flawed")
RD = os.path.join(OUT, "pp_n4d2_roll")
# instrument v2 (PARTS_RESULTS 19): --v2 selects the servoing generator, seeds 970000 + 1000 j + k, files pp_n4d2v_*, arms N4d2v_*
V2 = "--v2" in sys.argv
V = "v" if V2 else ""
FAM_BASE = 970000 if V2 else 960000
J = {"N4": 0, "K1": 1, "K4": 2}
TIP = 45.0


def tilt_deg(q):
    """MuJoCo quaternion (w, x, y, z) -> angle (deg) between the body z-axis and world z."""
    w, x, y, z = q
    zz = 1.0 - 2.0 * (x * x + y * y)                                          # R[2, 2]
    return float(math.degrees(math.acos(max(-1.0, min(1.0, zz)))))


def genfam(task):
    import fa1b_common as FB
    from fa1_data import noop_mask
    n2 = json.load(open(os.path.join(OUT, "n2_tasks.json"), encoding="utf-8"))
    t = n2["tasks"][N.TID[task]]
    bddl = os.path.join(ROOT, t["bddl"])
    oe, te = N.TASKS[task]
    kp = N.keypoints(task)
    env0 = FC.make_env(bddl)
    cols = {x: [] for x in ("task", "demo", "step", "action", "ee_pos", "ee_rot", "grip", "pos", "yaw", "drawer", "prev_grip", "held", "off")}
    rep, kept, k, t0 = [], 0, 0, time.time()
    while kept < 50 and k < 400:
        seed = FAM_BASE + 1000 * J[task] + k
        k += 1
        np.random.seed(seed)
        env0.seed(seed)
        env0.reset()
        s0 = env0.sim.get_state().flatten()
        rng = np.random.default_rng(seed)
        ep, info = N.plan_episode(bddl, s0, kp, oe, te, rng, servo=V2)
        rep.append(dict(seed=seed, success=info["success"], phase=info["phase"]))
        if not info["success"]:
            continue
        A = np.asarray(ep["action"])
        keep = np.nonzero(noop_mask(A))[0]
        hm, prev = FB.Held(), -1.0
        for s, i in enumerate(keep):
            h, off = hm.step(bool(ep["touch"][i][0]), bool(ep["touch"][i][1]), ep["objp"][i], ep["ee_pos"][i])
            for x in ("action", "ee_pos", "ee_rot", "grip", "pos", "yaw", "drawer"):
                cols[x].append(ep[x][i])
            cols["task"].append(100); cols["demo"].append(kept); cols["step"].append(s); cols["prev_grip"].append(prev)
            cols["held"].append(float(h)); cols["off"].append(off)
            prev = float(A[i][6])
        kept += 1
    env0.close()
    out = {x: np.asarray(v, np.float32 if x not in ("task", "demo", "step") else np.int32) for x, v in cols.items()}
    np.savez_compressed(os.path.join(FC.DATA, "gen", f"pp_n4d2{V}_{task}.npz"), **out)
    with open(os.path.join(OUT, f"pp_n4d2{V}_gen_{task}.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(task=task, kept=kept, attempts=k, base=FAM_BASE + 1000 * J[task], rows=int(len(out["task"])), keypoints=kp,
                       episodes=rep, note="familiar-layout starts; stage 1g planner; no demo action executed"), fh, indent=1)
    print(f"[pp_n4d2 genfam] {task}: kept {kept} of {k} ({time.time() - t0:.0f}s)", flush=True)


def train(task, seed):
    """pp_n4d.train's single-task fit (same rows, recipe and class) on DATA/gen/pp_n4d2_<task>.npz."""
    import torch
    import fa1c_common as FK
    from fa1_train import chunks
    from fa1g_train import rot0_rows
    from pp_p_train import fit
    from lifelong_ood.policy import parts_flow as PF
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    G = dict(np.load(os.path.join(FC.DATA, "gen", f"pp_n4d2{V}_{task}.npz")))
    n = len(G["task"])
    oe, te = N.TASKS[task]
    oi, ti = np.full(n, FC.ENT.index(oe)), np.full(n, FC.ENT.index(te))
    zr, cy = FK.canon_rows(G, G["held"], oi, ti)
    r0 = rot0_rows(G, G["held"])
    X = chunks(G, 8)
    D = dict(np.load(os.path.join(FC.DATA, "demos.npz")))
    center = D["ee_pos"].mean(0).tolist()
    T = lambda v, dt=torch.float32: torch.tensor(np.asarray(v), dtype=dt, device=dev)   # noqa: E731
    b = dict(ee_pos=T(G["ee_pos"]), ee_rot=T(G["ee_rot"]), grip=T(G["grip"]), prev=T(G["prev_grip"]), pos=T(G["pos"]), yaw=T(G["yaw"]),
             drawer=T(G["drawer"]), oi=T(oi, torch.long), ti=T(ti, torch.long), held=T(G["held"]), off=T(G["off"]), zrest=T(zr), cyaw=T(cy),
             rot0=T(r0))
    size = np.array([FC.SIZE[e] for e in FC.ENT], np.float32)
    m = PF.PartsFlow(8, center=center, size=size, rank=32).to(dev)
    ema, curve, secs = fit(m, list(m.parameters()), b, T(np.clip(X, -1, 1)), 30000, 512, 3e-4, seed, 1e-4, log=f"n4d2 {task} s{seed}")
    d = os.path.join(FC.DATA, "ckpt", f"pp_n4d2{V}_{task}_s{seed}")
    os.makedirs(d, exist_ok=True)
    ema.cpu().save(os.path.join(d, "P.npz"), dict(tag=f"n4d2_{task}", seed=seed, tasks=[task], center=center, steps=30000, data="generated-familiar"))
    with open(os.path.join(OUT, f"pp_n4d2{V}_train_{task}_s{seed}.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(task=task, seed=seed, n_rows=n, n_episodes=int(len(np.unique(G["demo"]))), secs=secs, curve=curve), fh, indent=1)


def stats(task):
    """Descriptive data check (no rollout): field ranges of the generated rows against the human demos of the same task."""
    G = dict(np.load(os.path.join(FC.DATA, "gen", f"pp_n4d2{V}_{task}.npz")))
    D = dict(np.load(os.path.join(FC.DATA, "demos.npz")))
    Db = dict(np.load(os.path.join(FC.DATA, "demos_b.npz")))
    out = {}
    if task in FC.KNOWN:
        m = D["task"] == list(FC.KNOWN).index(task)
        H = {k: D[k][m] for k in ("action", "ee_pos", "grip", "prev_grip")}
        H["held"] = Db["held"][m]
        H["steps"] = np.bincount(D["demo"][m]).astype(float)
    for nm, X in (("gen", G), ("demo", H if task in FC.KNOWN else None)):
        if X is None:
            continue
        out[nm] = dict(rows=int(len(X["action"])), act_abs_mean=np.abs(X["action"]).mean(0).round(3).tolist(),
                       act_mean=X["action"].mean(0).round(3).tolist(), grip_mean=np.asarray(X["grip"]).mean(0).round(4).tolist(),
                       prev_mean=float(np.mean(X["prev_grip"])), held_share=float(np.mean(X["held"])),
                       ee_mean=np.asarray(X["ee_pos"]).mean(0).round(3).tolist())
        if nm == "gen":
            out[nm]["steps"] = float(np.mean(np.bincount(X["demo"])))
        else:
            out[nm]["steps"] = float(X["steps"][X["steps"] > 0].mean())
    print(f"[pp_n4d2 stats] {task}", json.dumps(out), flush=True)


class Rec:
    def __init__(self, env, oe, log):
        self._e, self._oe, self._log = env, oe, log

    def __getattr__(self, k):
        return getattr(self._e, k)

    def step(self, a):
        r = self._e.step(a)
        sim = self._e.env.sim
        self._log.append(tilt_deg(np.asarray(sim.data.body_xquat[self._e.env.obj_body_id[self._oe]])))
        return r

    def set_init_state(self, s):
        return self._e.set_init_state(s)

    def reset(self):
        return self._e.reset()


def roll(ckpt, name, tids, oracle_task=None):
    import torch
    import fa1h_roll as RH
    from pp_p_roll import load_any
    torch.set_num_threads(1)
    n2 = json.load(open(os.path.join(OUT, "n2_tasks.json"), encoding="utf-8"))
    br = {int(b["trial"]): b for b in json.load(open(os.path.join(OUT, "n2_branches_main.json"), encoding="utf-8"))}
    model, meta = (None, dict(seed=0)) if oracle_task else load_any(os.path.join(FC.DATA, "ckpt", ckpt))
    os.makedirs(RD, exist_ok=True)
    orig = FC.fresh_env
    for tid in [int(x) for x in tids.split("+")]:
        out = os.path.join(RD, f"{name}_s{meta['seed']}_main_{tid}.json")
        if os.path.exists(out):
            continue
        t = n2["tasks"][tid]
        S = np.load(os.path.join(OUT, "n2_states", f"n2_goal_{tid}.npy"))
        oe = t["object"]
        eps = []
        for k in N.DEV:
            log = []
            FC.fresh_env = lambda bddl, seed=7: Rec(orig(bddl, seed), oe, log)     # noqa: E731
            try:
                if oracle_task:
                    rng = np.random.default_rng(13000000 + 1000 * k)
                    _, e = N.plan_episode(os.path.join(ROOT, t["bddl"]), S[k], N.keypoints(oracle_task), *N.TASKS[oracle_task], rng)
                    e["held_step"] = e["held_step"]
                else:
                    e = RH.episode(model, os.path.join(ROOT, t["bddl"]), S[k], t["language"], br[k]["seed_base"])
            finally:
                FC.fresh_env = orig
            tl = np.array(log[10:] if not oracle_task else log[10:]) if log else np.zeros(1)
            hs = e["held_step"]
            e.update(trial=k, max_tilt_held=float(tl[hs:].max()) if hs is not None and len(tl) > hs else None, final_tilt=float(tl[-1]),
                     max_tilt=float(tl.max()), tipped=bool(tl[-1] > TIP))
            eps.append(e)
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(dict(ckpt=ckpt, name=name, ckpt_meta=meta, tasks={str(tid): dict(language=t["language"], episodes=eps)}), fh, indent=1)
        print(f"[pp_n4d2 roll] {name} task {tid} done", flush=True)


def gate():
    """The validation gate (section 18): single-task fit on familiar generated K1 / K4 within 0.15 of P on that task."""
    from fa1g_read import radius
    cal = json.load(open(os.path.join(OUT, "a1_calib.json"), encoding="utf-8"))["pairs"]
    res = {}
    for task in ("K1", "K4"):
        rad = radius(*N.TASKS[task], cal)
        a = N.arrays([os.path.join(RD, f"N4d2{V}_{task}_s{s}_main_{N.TID[task]}.json") for s in (0, 1, 2)], task, rad, np.zeros(2))[0]
        p = N.arrays([os.path.join(RD, f"P2r32_s{s}_main_{N.TID[task]}.json") for s in (10, 11, 12)], task, rad, np.zeros(2))[0]
        res[task] = dict(single_task=float(np.nanmean(a)), P=float(np.nanmean(p)), within_015=bool(np.nanmean(a) >= np.nanmean(p) - 0.15))
    res["valid"] = any(v["within_015"] for v in res.values() if isinstance(v, dict))
    return res


def read():
    """pp_n4d's read (arms, phases, release error, mapping), on pp_n4d2_roll, plus the gate and the tilt split of N4's failures."""
    N.read.__globals__["os"] = os
    orig_join = os.path.join
    R0 = orig_join(OUT, "pp_n4d_roll")

    def join(*p):
        s = orig_join(*p)
        if s.startswith(R0 + os.sep):
            base = os.path.basename(s)
            base = base.replace("N4d_", f"N4d2{V}_").replace("oracle_N4.json", "oracle_N4_s0_main_22.json") \
                       .replace("oracle_K4.json", "oracle_K4_s0_main_6.json").replace("oracle_K1.json", "oracle_K1_s0_main_0.json")
            return orig_join(RD, base)
        if s == orig_join(OUT, "pp_n4d_read.json"):
            return orig_join(OUT, "pp_n4d2_read.json")
        if s == orig_join(OUT, "pp_n4d_gen.json"):
            return orig_join(OUT, f"pp_n4d2{V}_gen_N4.json")
        return s
    os.path.join = join
    try:
        N.read()
    finally:
        os.path.join = orig_join
    out = json.load(open(os.path.join(OUT, "pp_n4d2_read.json"), encoding="utf-8"))
    out["gate"] = gate()
    split = {}
    for arm, files in (("P", [os.path.join(RD, f"P2r32_s{s}_main_22.json") for s in (10, 11, 12)]),
                       ("a", [os.path.join(RD, f"N4d2{V}_N4_s{s}_main_22.json") for s in (0, 1, 2)]),
                       ("o", [os.path.join(RD, "oracle_N4_s0_main_22.json")])):
        E = [e for f in files for e in N.ep_list(f)]
        fail = [e for e in E if not e["success"] and e["held_step"] is not None]
        tipped = [e for e in fail if e["tipped"]]
        split[arm] = dict(episodes=len(E), held_failures=len(fail), tipped=len(tipped), upright=len(fail) - len(tipped),
                          max_tilt_held_median=float(np.median([e["max_tilt_held"] for e in E if e["max_tilt_held"] is not None])) if E else None,
                          final_tilt_success_median=float(np.median([e["final_tilt"] for e in E if e["success"]])) if any(e["success"] for e in E) else None)
    out["tilt_split_N4"] = split
    out["instrument_valid"] = out["gate"]["valid"]
    json.dump(out, open(os.path.join(OUT, "pp_n4d2_read.json"), "w", encoding="utf-8"), indent=1)
    print("[pp_n4d2] gate", json.dumps(out["gate"]), flush=True)
    print("[pp_n4d2] tilt", json.dumps(split), flush=True)
    print("[pp_n4d2] verdict", out["verdict"], out["failing_phase"], "valid", out["instrument_valid"], flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("genfam", "train", "stats", "roll", "read", "gate"))
    ap.add_argument("--task", default="N4")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--name", default="")
    ap.add_argument("--tids", default="")
    ap.add_argument("--oracle", default="")
    ap.add_argument("--v2", action="store_true")                           # read at import (V2)
    a = ap.parse_args()
    if a.mode == "genfam":
        genfam(a.task)
    elif a.mode == "train":
        train(a.task, a.seed)
    elif a.mode == "stats":
        stats(a.task)
    elif a.mode == "roll":
        roll(a.ckpt, a.name, a.tids, a.oracle or None)
    elif a.mode == "gate":
        print("[pp_n4d2] gate", json.dumps(gate()), flush=True)
    else:
        read()
