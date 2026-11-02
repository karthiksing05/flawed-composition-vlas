"""PLAN_PARTS confirmatory start states (docs/PARTS_RESULTS.md section 6, m3_prereg pp_l2r32): 50 FRESH start states per layout of the
N2 scene, never used by any earlier read.  N2's generator (experiments/fc_n2_tasks.py) unchanged in every rule: the same BDDLs
(results/flawed/n2_bddl), the same training box, footprints, settle check and acceptance rule; only the seed bases differ.
  familiar  LIBERO's sampler, seed 9100000 + k          (N2: 600000 + k, k = 0-23)
  new       the familiar state k with the four movable objects relocated, rng 9200000 + k   (N2: 700000 + k)
  noise     branch seed_base 23000000 + 1000 k           (N2: 13000000 + 1000 k)
k = 0-49.  No policy is run, no demo is read or executed.  LIBERO venv, EGL.
-> results/flawed/pp_conf_states/n2_goal_<tid>.npy [50, state_dim] (every task of a layout shares them, as in N2),
   results/flawed/pp_conf_branches.json, results/flawed/pp_conf_states.json

    python experiments/pp_conf_states.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fc_n2_tasks as G                                                     # noqa: E402

N, BASE_FAM, BASE_NEW, BASE_NOISE = 50, 9100000, 9200000, 23000000
OUTD = os.path.join(G.OUT, "pp_conf_states")


def main():
    from libero.libero import benchmark
    os.makedirs(OUTD, exist_ok=True)
    n2 = json.load(open(os.path.join(G.OUT, "n2_tasks.json"), encoding="utf-8"))
    paths = {t["compound"]: os.path.join(G.ROOT, t["bddl"]) for t in n2["tasks"]}
    mov = list(G.MOVR)
    suite = benchmark.get_benchmark_dict()["libero_goal"]()
    tidx = next(i for i in range(suite.n_tasks) if suite.get_task(i).name == "put_the_cream_cheese_in_the_bowl")
    stored = suite.get_task_init_states(tidx)
    env = G.make_env(paths["K1"])
    env.seed(0)
    tr = []
    for s in stored:
        env.reset()
        env.set_init_state(s)
        tr.append(G.positions(env, mov)[:, :2])
    tr = np.stack(tr)
    centre = tr.mean(0)
    assert np.allclose(centre, np.array(n2["training"]["centre"]), atol=1e-6), "training centre differs from N2's"
    offset = np.mean([centre[i] - np.array(G.REGION[n]) for i, n in enumerate(mov)], axis=0)
    fixn = list(G.FIXR)
    fixp = G.positions(env, fixn)[:, :2]
    lo, hi = centre.min(0) - G.BOX_PAD, centre.max(0) + G.BOX_PAD
    sf = (G.STOVE_FRONT[0] + offset[0], G.STOVE_FRONT[1] + offset[1], G.STOVE_FRONT[2] + offset[0], G.STOVE_FRONT[3] + offset[1])
    assert np.allclose(lo, n2["constraints"]["box"][0]) and np.allclose(hi, n2["constraints"]["box"][1]), "box differs from N2's"

    fam = []
    for k in range(N):
        np.random.seed(BASE_FAM + k)
        env.seed(BASE_FAM + k)
        env.reset()
        fam.append(env.sim.get_state().flatten())
    fam = np.stack(fam)
    for key in ("K4", "N4"):                                                  # the same sampler output under other BDDLs
        e2 = G.make_env(paths[key])
        for k in range(2):
            np.random.seed(BASE_FAM + k)
            e2.seed(BASE_FAM + k)
            e2.reset()
            assert np.allclose(e2.sim.get_state().flatten(), fam[k]), f"{key}: state {k} differs from K1's"
        e2.close()

    def ok(p):
        for i, nm in enumerate(mov):
            r = G.MOVR[nm]
            if np.any(p[i] < lo) or np.any(p[i] > hi):
                return False
            if np.linalg.norm(p[i] - centre[i]) < G.OWN_MIN:
                return False
            if any(np.linalg.norm(p[i] - centre[j]) < G.OTHER_MIN for j in range(len(mov)) if j != i):
                return False
            if any(np.linalg.norm(p[i] - p[j]) < r + G.MOVR[mov[j]] + G.GAP for j in range(i)):
                return False
            if any(np.linalg.norm(p[i] - fixp[f]) < r + G.FIXR[fn] + G.GAP for f, fn in enumerate(fixn)):
                return False
            cx, cy = np.clip(p[i][0], sf[0], sf[2]), np.clip(p[i][1], sf[1], sf[3])
            if np.hypot(p[i][0] - cx, p[i][1] - cy) < r + G.GAP:
                return False
        return True

    new, newpos, fampos, settle = [], [], [], []
    for k in range(N):
        rng = np.random.default_rng(BASE_NEW + k)
        env.reset()
        env.set_init_state(fam[k])
        p0 = G.positions(env, mov)
        fampos.append(p0[:, :2])
        for _ in range(10):
            env.step(G.DUMMY)
        dz_fam = G.positions(env, mov)[:, 2] - p0[:, 2]
        done, n_ok = False, 0
        for _ in range(200000):
            p = rng.uniform(lo, hi, size=(len(mov), 2))
            if not ok(p):
                continue
            n_ok += 1
            if n_ok > 400:
                break
            env.reset()
            env.set_init_state(fam[k])
            sim = env.env.sim
            for i, nm in enumerate(mov):
                a = G.qadr(env, nm)
                sim.data.qpos[a:a + 2] = p[i]
            sim.forward()
            s_new = sim.get_state().flatten()
            env.reset()
            env.set_init_state(s_new)
            q0 = G.positions(env, mov)
            for _ in range(10):
                env.step(G.DUMMY)
            q1 = G.positions(env, mov)
            d = float(max(np.linalg.norm(q1[:, :2] - q0[:, :2], axis=-1).max(), np.abs((q1[:, 2] - q0[:, 2]) - dz_fam).max()))
            if d <= G.SETTLE_MAX and np.allclose(q0[:, :2], p, atol=1e-4):
                new.append(s_new)
                newpos.append(p)
                settle.append(d)
                done = True
                break
        assert done, f"state {k}: no valid new layout ({n_ok} geometric candidates)"
    new = np.stack(new)
    env.close()
    old = np.load(os.path.join(G.OUT, "n2_states", "n2_goal_0.npy"))
    old_new = np.load(os.path.join(G.OUT, "n2_states", "n2_goal_1.npy"))
    dmin = float(min(np.abs(fam[:, None] - old[None]).max(-1).min(), np.abs(new[:, None] - old_new[None]).max(-1).min()))
    assert dmin > 1e-6, "a fresh state equals an N2 state"
    for t in n2["tasks"]:
        np.save(os.path.join(OUTD, f"{G.SUITE}_{t['tid']}.npy"), fam if t["layout"] == "familiar" else new)
    br = [dict(id=f"c{k}", trial=k, seed_base=BASE_NOISE + 1000 * k, at=0, variant="pp_conf") for k in range(N)]
    with open(os.path.join(G.OUT, "pp_conf_branches.json"), "w", encoding="utf-8") as fh:
        json.dump(br, fh, indent=1)
    meta = dict(n=N, base_familiar=BASE_FAM, base_new=BASE_NEW, base_noise=BASE_NOISE, states=list(range(N)),
                min_abs_diff_to_n2_states=dmin, settle_max_observed=max(settle),
                positions=dict(familiar=np.stack(fampos).tolist(), new=np.stack(newpos).tolist()))
    with open(os.path.join(G.OUT, "pp_conf_states.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1)
    print(f"[pp_conf] {N} familiar + {N} new states; min |diff| to N2's {dmin:.4f}; max settle {max(settle):.4f}", flush=True)


if __name__ == "__main__":
    main()
