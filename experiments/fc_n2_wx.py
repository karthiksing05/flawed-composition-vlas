"""SR 87: a batch of NEW fresh familiar-layout start states for the N2 scene for the confirmatory W5 run (docs/notes/raasw5_plan.md):
states k = 580..639 (60), drawn by experiments/fc_n2_rx.py's rule unchanged (LIBERO's sampler on the familiar BDDL, np.random.seed and
env.seed = 600000 + k; k 0-23 fc_n2_tasks.py, 24-35 fc_n2_extra.py, 36-55 fc_n2_fresh.py, 56-115 fresh2, 116-155 fresh3, 156-195 fresh4,
236-275 s1x, 500-539 gx, 540-579 rx; fc_n2_rx.py's docstring leaves 580-639 free; k 196-235 and 276-499 are left to whoever claimed them).
SR 70-86 used k 4-23 and 540-579 for N1-N6 / K1-K6 (and the older sets in earlier SRs); none of 580-639.  No policy is run.
Before writing, the rule must reproduce stored states exactly (k = 0, 23, 540, 579); every new state must differ (not np.allclose) from every
other new state and from every stored N2 familiar state (all sets above).
LIBERO venv (py3.8), EGL.    python experiments/fc_n2_wx.py
-> results/flawed/n2_states_wx/ n2_goal_<tid>.npy [60, state_dim] for every FAMILIAR-layout tid (row t = state k = 580 + t; the harness trial index t = 0..59),
   states.npy, meta.json;  results/flawed/n2_branches_wx.json  (trial t, seed_base 13000000 + 1000 k, N2's convention).
Refuses to overwrite.  Stored files are only read.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fc_n2_tasks as N2                                                    # noqa: E402

DST = os.path.join(N2.OUT, "n2_states_wx")
SETS = ("n2_states_record", "n2_states_fresh", "n2_states_fresh2", "n2_states_fresh3", "n2_states_fresh4", "n2_states_s1x", "n2_states_gx", "n2_states_rx")
CONF = os.path.join(N2.OUT, "pp_conf_states", "n2_goal_0.npy")
K0, N = 580, 60
BR = os.path.join(N2.OUT, "n2_branches_wx.json")


def main():
    fn = os.path.join(DST, "states.npy")
    for p in (fn, BR):
        assert not os.path.exists(p), "exists: " + p
    tasks = json.load(open(os.path.join(N2.OUT, "n2_tasks.json"), encoding="utf-8"))["tasks"]
    fam = [t for t in tasks if t["layout"] == "familiar"]
    assert len(fam) == 16 and all(t["tid"] % 2 == 0 for t in fam), [t["tid"] for t in fam]
    ref = np.load(os.path.join(N2.ST, "n2_goal_0.npy"))
    old = [ref] + [np.load(os.path.join(N2.OUT, d, "states.npy")) for d in SETS]
    rx = old[-1]
    assert rx.shape == (40, ref.shape[1]), rx.shape
    conf = np.load(CONF) if os.path.exists(CONF) else None
    if conf is not None and conf.shape[1] == ref.shape[1]:
        old.append(conf)
    old = np.concatenate(old)
    env = N2.make_env(os.path.join(N2.BD, "K1.bddl"))

    def draw(k):
        np.random.seed(N2.BASE_FAM + k)
        env.seed(N2.BASE_FAM + k)
        env.reset()
        return env.sim.get_state().flatten()

    for k, want in ((0, ref[0]), (23, ref[23]), (540, rx[0]), (579, rx[39])):
        d = float(np.abs(draw(k) - want).max())
        print("[n2wx] reproduce state %d: max |diff| %.3g" % (k, d), flush=True)
        assert d == 0.0, "the sampler rule does not reproduce stored state %d" % k
    S = np.stack([draw(k) for k in range(K0, K0 + N)])
    assert S.shape == (N, ref.shape[1]), S.shape
    for i in range(N):
        for j in range(i):
            assert not np.allclose(S[i], S[j]), "duplicate new states %d %d" % (K0 + i, K0 + j)
        assert not any(np.allclose(S[i], o) for o in old), "new state %d equals a stored state" % (K0 + i)
    print("[n2wx] %d new states distinct from each other and from %d stored states" % (N, old.shape[0]), flush=True)
    os.makedirs(DST, exist_ok=True)
    for t in fam:
        out = os.path.join(DST, "n2_goal_%d.npy" % t["tid"])
        assert not os.path.exists(out), "exists: " + out
        np.save(out, S)
    br = [dict(id="k%d" % (K0 + t), trial=t, seed_base=13000000 + 1000 * (K0 + t), at=0, variant="n2") for t in range(N)]
    with open(BR, "w", encoding="utf-8") as fh:
        json.dump(br, fh, indent=1)
    with open(os.path.join(DST, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(k=list(range(K0, K0 + N)), seed_rule="np.random.seed = env.seed = %d + k" % N2.BASE_FAM, layout="familiar", trial_to_k="k = trial + %d" % K0,
                       distinct_from=dict(stored_rows=int(old.shape[0]), sets=list(SETS)), branches=os.path.relpath(BR, N2.ROOT)), fh, indent=1)
    np.save(fn + ".tmp.npy", S)
    os.replace(fn + ".tmp.npy", fn)
    print("[n2wx] wrote %s %s; branches %s" % (fn, S.shape, BR), flush=True)


if __name__ == "__main__":
    main()
