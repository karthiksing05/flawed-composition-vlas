"""Extra FAMILIAR-layout start states for the N2 scene, beyond fc_n2_tasks.py's 24 (smoke 0-3, main 4-23): states k = 24..24+N-1,
drawn by the same rule (LIBERO's sampler on the template BDDL, np.random.seed and env.seed = 600000 + k).  For recording one
demonstration per new pairing (sc_pu_record.py), so the recording never touches a trial list's state.  Before writing, states 0-1
are regenerated and must equal the stored n2_goal_0.npy rows exactly (the rule reproduces).  LIBERO venv (py3.8), EGL.

    python experiments/fc_n2_extra.py [--n 12]
-> results/flawed/n2_states_record/states.npy [N, state_dim] (k = 24..) + meta.json.  Refuses to overwrite.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fc_n2_tasks as N2                                                    # noqa: E402

DST = os.path.join(N2.OUT, "n2_states_record")
K0 = N2.N_SMOKE + N2.N_MAIN                                                 # 24


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    a = ap.parse_args()
    fn = os.path.join(DST, "states.npy")
    assert not os.path.exists(fn), "exists: " + fn
    env = N2.make_env(os.path.join(N2.BD, "K1.bddl"))

    def draw(k):
        np.random.seed(N2.BASE_FAM + k)
        env.seed(N2.BASE_FAM + k)
        env.reset()
        return env.sim.get_state().flatten()

    ref = np.load(os.path.join(N2.ST, "n2_goal_0.npy"))
    for k in range(2):
        d = float(np.abs(draw(k) - ref[k]).max())
        print("[n2x] reproduce state %d: max |diff| %.3g" % (k, d), flush=True)
        assert d == 0.0, "the sampler rule does not reproduce stored state %d" % k
    S = np.stack([draw(k) for k in range(K0, K0 + a.n)])
    assert S.shape[1] == ref.shape[1]
    for i in range(len(S)):
        for j in range(i):
            assert not np.allclose(S[i], S[j]), "duplicate states %d %d" % (i, j)
    pos = []
    for s in S:
        env.reset()
        env.set_init_state(s)
        pos.append(N2.positions(env, list(N2.MOVR))[:, :2].tolist())
    env.close()
    os.makedirs(DST, exist_ok=True)
    np.save(fn + ".tmp.npy", S)
    os.replace(fn + ".tmp.npy", fn)
    with open(os.path.join(DST, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(k=list(range(K0, K0 + a.n)), seed_rule="np.random.seed = env.seed = %d + k" % N2.BASE_FAM, layout="familiar",
                       bddl="results/flawed/n2_bddl/K1.bddl (every task of the familiar layout shares the states)",
                       movable=list(N2.MOVR), positions_xy=pos, seed_base_rule="13000000 + 1000 k (sc_pu.record_states)"), fh, indent=1)
    print("[n2x] wrote %s %s" % (fn, S.shape), flush=True)


if __name__ == "__main__":
    main()
