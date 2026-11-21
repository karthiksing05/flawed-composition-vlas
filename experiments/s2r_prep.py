"""SR 89 step 2 (docs/notes/scene2_plan.md section 3): the REGISTERED scene-2 states and the resolver's axis constant.  LIBERO venv, EGL (run through sbatch).
    python experiments/s2r_prep.py states      -> results/flawed/sr89/main/{O/s2_living_<tid>.npy (tids 0-3, 64 states, seed 1,100,000 + k), branches_smoke.json (k 0-3), branches_main.json (k 4-63)}
                                                  (refuses to overwrite; every state is checked unequal to every earlier scene-2 state incl. the SR 89 probe / calibration states)
    python experiments/s2r_prep.py plate_axis  -> results/flawed/sr89/plate_axis.json: plate_1 ("the left plate" in the BDDL) and plate_2 base-frame positions on the SR 89 calibration + probe states
                                                  (geometry only, no rollout); the resolver (wa_s2r.py) names the left plate by the sign of the axis difference recorded here.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT]
OUT = os.path.join(ROOT, "results", "flawed")
PD = os.path.join(OUT, "sr89", "probe")
MD = os.path.join(OUT, "sr89", "main")
BASE, BR_BASE, N_STATES = 1100000, 48000000, 64
SMOKE, MAIN = list(range(0, 4)), list(range(4, 64))


def states():
    import s2
    import s2_tasks as T2
    import wa_s2 as W
    od = os.path.join(MD, "O")
    assert not os.path.exists(od), "exists: " + od
    M_ = s2.meta()
    env = T2.make_env(os.path.join(ROOT, M_["tasks"][0]["bddl"]))

    def draw(k):                                                            # s2_tasks.build's / wa_s2.states' rule with the registered base
        np.random.seed(BASE + k)
        env.seed(BASE + k)
        env.reset()
        return env.sim.get_state().flatten()
    S = np.stack([draw(k) for k in range(N_STATES)])
    env.close()
    old = W._old_states(s2) + [np.load(os.path.join(PD, "O", "%s_0.npy" % s2.SUITE)), np.load(os.path.join(PD, "calib_states.npy"))]
    for i in range(len(S)):
        for j in range(i):
            assert not np.allclose(S[i], S[j]), (i, j)
        for O_ in old:
            for j in range(len(O_)):
                assert not (O_[j].shape == S[i].shape and np.allclose(S[i], O_[j])), ("equals an earlier scene-2 state", i, j)
    os.makedirs(od)
    for t in M_["tasks"][:4]:
        np.save(os.path.join(od, "%s_%d.npy" % (s2.SUITE, t["tid"])), S)
    for ph, ks in (("smoke", SMOKE), ("main", MAIN)):
        with open(os.path.join(MD, "branches_%s.json" % ph), "w", encoding="utf-8") as fh:
            json.dump([dict(id="k%d" % k, trial=k, seed_base=BR_BASE + 1000 * k, at=0, variant="s2") for k in ks], fh, indent=1)
    print("[s2r] %d registered states (seed %d + k), none equal to an earlier scene-2 state (SR 50-72, s2rg, SR 89 probe + calibration); wrote %s" % (N_STATES, BASE, MD), flush=True)


def plate_axis():
    import s2
    import s2_tasks as T2
    p = os.path.join(OUT, "sr89", "plate_axis.json")
    assert not os.path.exists(p), "exists: " + p
    M_ = s2.meta()
    env = T2.make_env(os.path.join(ROOT, M_["tasks"][0]["bddl"]))
    env.seed(7)
    env.reset()
    rows = []
    for f in (os.path.join(PD, "calib_states.npy"), os.path.join(PD, "O", "%s_0.npy" % s2.SUITE)):
        X = np.load(f)
        for k in range(len(X)):
            env.set_init_state(X[k])
            for _ in range(s2.WAIT):
                env.step(T2.DUMMY)
            rows.append(T2.positions(env, ["plate_1", "plate_2"]))
    env.close()
    R = np.asarray(rows)                                                   # [n, 2, 3] base-frame positions
    d = R[:, 0, :] - R[:, 1, :]                                            # plate_1 ("left") minus plate_2 ("right")
    ax = int(np.argmax(np.abs(d).mean(0)))
    sign = float(np.sign(d[:, ax].mean()))
    assert (np.sign(d[:, ax]) == sign).all(), "the sides swap across states"
    res = dict(n_states=int(len(R)), axis=ax, left_is_greater=bool(sign > 0), mean_diff_m=d.mean(0).tolist(), min_abs_diff_axis_m=float(np.abs(d[:, ax]).min()),
               plate_1_mean=R[:, 0].mean(0).tolist(), plate_2_mean=R[:, 1].mean(0).tolist(),
               rule="left plate = the plate with the larger coordinate on `axis` iff left_is_greater, else the smaller; plate_1 is the BDDL's left plate")
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print("[s2r] plate axis", json.dumps(res), flush=True)


if __name__ == "__main__":
    {"states": states, "plate_axis": plate_axis}[sys.argv[1]]()
