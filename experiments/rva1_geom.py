"""RVA1 (docs/notes/plan_review_atlas.md 2.4): initial-state positions of every scene slot for the development states of N1-N6 / K1-K6 (CPU, LIBERO venv).
Every task of the familiar layout uses the SAME states, so one scene restore per state suffices: dev trials 4-23 (results/flawed/n2_states/n2_goal_0.npy rows t) and
trials 24-63 (n2_states_rx/n2_goal_0.npy rows t; 64 rows, the client indexes by trial).  Output results/flawed/rva1/geom_dev.json: {"<trial>": {entity: [x, y, z]}} (metres; entities of fa1_common.ENT:
movable bodies and target sites).  Fixtures do not move; the plate / bowl can be pushed (stated limit).  Refuses to overwrite.

    python experiments/rva1_geom.py        (LIBERO venv: VENV=libero through pz_cpu.sh)
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fa1_common as FC                                                      # noqa: E402

ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "results", "flawed", "rva1", "geom_dev.json")


def main():
    assert not os.path.exists(OUT), "never overwrite: " + OUT
    R = os.path.join(ROOT, "results", "flawed")
    main_s = np.load(os.path.join(R, "n2_states", "n2_goal_0.npy"))
    rx_s = np.load(os.path.join(R, "n2_states_rx", "n2_goal_0.npy"))
    assert main_s.shape[0] == 24 and rx_s.shape[0] == 64, (main_s.shape, rx_s.shape)
    print("[rva1_geom] rx rows 4-23 equal dev rows:", bool(np.allclose(main_s[4:24], rx_s[4:24])), flush=True)
    env = FC.make_env(os.path.join(R, "n2_bddl", "K1.bddl"))
    dj = FC.drawer_joint(env)
    res = {}
    for trial in range(4, 64):
        s = main_s[trial] if trial < 24 else rx_s[trial]
        env.reset()
        obs = env.set_init_state(s)
        sc = FC.scene(env, obs, dj)
        res[str(trial)] = {n: [float(v) for v in sc["pos"][i]] for i, n in enumerate(FC.ENT)}
    with open(OUT, "x", encoding="utf-8") as fh:
        json.dump(res, fh)
    print("[rva1_geom] wrote %d states -> %s" % (len(res), OUT), flush=True)
    t = res["4"]
    print("[rva1_geom] trial 4:", {k: [round(v, 3) for v in p] for k, p in t.items()}, flush=True)


if __name__ == "__main__":
    main()
