"""L1 (docs/RIVAL_TESTS.md 2; m3_prereg pp_rival1): the library's floors, CPU, LIBERO venv, dev states and seeds only.
  k0 --cond nobottle|noplate --seed 10 --task K4:new    P_min's k = 1 creation with the created atom's slot embedding and low-rank delta zeroed
     -> results/flawed/pp_l1/k0_<cond>_s<seed>_<compound>_<layout>.json
  servo --seed 10 --task K1:new    the v2 servoing planner; a new pairing's keypoints follow pp_n4d.keypoints' rule for N4
     -> results/flawed/pp_l1/servo_s<seed>_<compound>_<layout>.json
  read    -> results/flawed/pp_l1_read.json
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT]
OUT = os.path.join(ROOT, "results", "flawed")
OD = os.path.join(OUT, "pp_l1")
SEEDS, B = (10, 11, 12), 2000
ATOM = {"nobottle": "wine_bottle_1", "noplate": "plate_1"}
S3 = {"nobottle": ("K4", "N4", "N5", "N6"), "noplate": ("K1", "N1", "N4")}
OBJ_K, TGT_K = {"cream_cheese_1": "K6", "wine_bottle_1": "K4"}, {"plate_1": "K1", "flat_stove_1_cook_region": "K2", "wooden_cabinet_1_top_side": "K3", "akita_black_bowl_1": "K6"}
n2 = json.load(open(os.path.join(OUT, "n2_tasks.json"), encoding="utf-8"))
ks = n2["states"]["main"]


def task(comp, lay):
    return next(t for t in n2["tasks"] if t["compound"] == comp and t["layout"] == lay)


def run_k0(a):
    import torch
    import fa1_common as FC
    import fa1h_roll as RH
    from pp_p_roll import load_any
    comp, lay = a.task.split(":")
    fn = os.path.join(OD, f"k0_{a.cond}_s{a.seed}_{comp}_{lay}.json")
    if os.path.exists(fn):
        return
    torch.set_num_threads(1)
    m, _ = load_any(os.path.join(FC.DATA, "ckpt", f"pp_{a.cond}_naf_s{a.seed}", f"P_{ATOM[a.cond]}_k1.npz"))
    e = FC.ENT.index(ATOM[a.cond])
    net, idx = (m.obj, int(m.opart[e])) if a.cond == "nobottle" else (m.tgt, int(m.tpart[e]))
    assert idx >= 0
    with torch.no_grad():
        net.emb[idx] = 0.0
        for ly in net.hidden:
            ly.B[idx] = 0.0
    t = task(comp, lay)
    S = np.load(os.path.join(OUT, "n2_states", f"n2_goal_{t['tid']}.npy"))
    eps = []
    for b in json.load(open(os.path.join(OUT, "n2_branches_main.json"), encoding="utf-8")):
        r = RH.episode(m, os.path.join(ROOT, t["bddl"]), S[int(b["trial"])], t["language"], b["seed_base"])
        eps.append(dict(trial=int(b["trial"]), success=r["success"], error=r["error"], held_step=r["held_step"]))
        print("[k0]", a.cond, a.seed, comp, lay, b["trial"], r["success"], flush=True)
    json.dump(dict(eps=eps), open(fn, "w", encoding="utf-8"), indent=1)


def kp_for(comp):
    kp = json.load(open(os.path.join(OUT, "a1g_keypoints.json"), encoding="utf-8"))["tasks"]
    if comp[0] == "K":
        return kp[comp]
    t = task(comp, "familiar")
    ko, kt = kp[OBJ_K[t["object"]]], kp[TGT_K[t["target"]]]
    out = dict(ko)
    out["release_off"] = (np.array(ko["grasp_off"]) + np.array(kt["release_off"]) - np.array(kt["grasp_off"])).tolist()
    out["object"], out["target"] = t["object"], t["target"]
    return out


def run_servo(a):
    import pp_n4d as N
    comp, lay = a.task.split(":")
    fn = os.path.join(OD, f"servo_s{a.seed}_{comp}_{lay}.json")
    if os.path.exists(fn):
        return
    t = task(comp, lay)
    kp = kp_for(comp)
    S = np.load(os.path.join(OUT, "n2_states", f"n2_goal_{t['tid']}.npy"))
    eps = []
    for b in json.load(open(os.path.join(OUT, "n2_branches_main.json"), encoding="utf-8")):
        rng = np.random.default_rng(int(b["seed_base"]) + a.seed)
        _, info = N.plan_episode(os.path.join(ROOT, t["bddl"]), S[int(b["trial"])], kp, t["object"], t["target"], rng, servo=True)
        eps.append(dict(trial=int(b["trial"]), success=info["success"], phase=info["phase"], held_step=info["held_step"]))
        print("[servo]", a.seed, comp, lay, b["trial"], info["success"], flush=True)
    json.dump(dict(eps=eps), open(fn, "w", encoding="utf-8"), indent=1)


def succ(path):
    R = json.load(open(path, encoding="utf-8"))
    E = R["eps"] if "eps" in R else [e for v in R["tasks"].values() for e in v["episodes"]]
    d = {int(e["trial"]): float(bool(e["success"])) for e in E}
    return np.array([d[k] for k in ks])


def stack(fmt, comps, lay):
    return np.stack([np.stack([succ(os.path.join(OUT, fmt.format(s=s, c=c, t=task(c, lay)["tid"], lay=lay))) for s in SEEDS]) for c in comps])


def boot(f):
    rng = np.random.default_rng(0)
    v = np.array([f(rng.choice(len(ks), len(ks))) for _ in range(B)])
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def read():
    res = dict(k0={}, servo={})
    for cond in S3:
        for lay in ("familiar", "new"):
            k0 = stack("pp_l1/k0_" + cond + "_s{s}_{c}_{lay}.json", S3[cond], lay)
            k1 = stack("pp_cons_roll/Pnaf_" + cond + "_k1_s{s}_main_{t}.json", S3[cond], lay)
            d = float(k1.mean() - k0.mean())
            ci = boot(lambda kk: float(k1[:, :, kk].mean() - k0[:, :, kk].mean()))
            v = "PART ADDS" if ci[0] > 0.0 and d > 0.10 else "BACKBONE SUFFICES"
            res["k0"][f"{cond}|{lay}"] = dict(s_k0=float(k0.mean()), s_k1=float(k1.mean()), delta=d, ci=ci, verdict=v,
                                              per_task={c: [float(k0[i].mean()), float(k1[i].mean())] for i, c in enumerate(S3[cond])})
            print("K0", cond, lay, json.dumps(res["k0"][f"{cond}|{lay}"]), flush=True)
    groups = dict(known=[f"K{i}" for i in range(1, 7)], new=[f"N{i}" for i in range(1, 7)])
    for g, comps in groups.items():
        for lay in ("familiar", "new"):
            sv = stack("pp_l1/servo_s{s}_{c}_{lay}.json", comps, lay)
            pm = stack("pp_cons_roll/P2naf_s{s}_main_{t}.json", comps, lay)
            d = float(pm.mean() - sv.mean())
            ci = boot(lambda kk: float(pm[:, :, kk].mean() - sv[:, :, kk].mean()))
            v = "P_MIN BEATS SERVO" if ci[0] > 0.0 and d > 0.10 else ("SERVO BEATS P_MIN" if ci[1] < 0.0 and d < -0.10 else "TIED")
            res["servo"][f"{g}|{lay}"] = dict(s_servo=float(sv.mean()), s_pmin=float(pm.mean()), delta=d, ci=ci, verdict=v,
                                              per_task={c: [float(sv[i].mean()), float(pm[i].mean())] for i, c in enumerate(comps)})
            print("SERVO", g, lay, json.dumps(res["servo"][f"{g}|{lay}"]), flush=True)
    json.dump(res, open(os.path.join(OUT, "pp_l1_read.json"), "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("k0", "servo", "read"))
    ap.add_argument("--cond")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--task")
    a = ap.parse_args()
    os.makedirs(OD, exist_ok=True)
    {"k0": run_k0, "servo": run_servo, "read": read}[a.mode](*([a] if a.mode != "read" else []))
