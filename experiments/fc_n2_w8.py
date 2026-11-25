"""NPHS COPY of fc_n2_w5.py (docs/notes/plan_necessity_power.md 3.2 step 7; plan_review_atlas risk 1): EIGHTH fresh state set k = 940..999 (N/K only), n2_states_w7 added to the refusal list (tag nphp).  fc_n2_w5.py docstring: FIFTH fresh familiar-layout N2 state batch k = 760..819 (60) (raaf_plan 7.1; docs/notes/plan_review_atlas.md 3.1), same rule (np.random.seed and env.seed = 600000 + k), refuses states equal to any stored set incl. SR 91 wz (700-759).  Original docstring follows:
states k = 580..639 (60), drawn by experiments/fc_n2_rx.py's rule unchanged (LIBERO's sampler on the familiar BDDL, np.random.seed and
env.seed = 600000 + k; k 0-23 fc_n2_tasks.py, 24-35 fc_n2_extra.py, 36-55 fc_n2_fresh.py, 56-115 fresh2, 116-155 fresh3, 156-195 fresh4,
236-275 s1x, 500-539 gx, 540-579 rx; fc_n2_rx.py's docstring leaves 580-639 free; k 196-235 and 276-499 are left to whoever claimed them).
SR 70-86 used k 4-23 and 540-579 for N1-N6 / K1-K6 (and the older sets in earlier SRs); none of 580-639.  No policy is run.
Before writing, the rule must reproduce stored states exactly (k = 0, 23, 540, 579); every new state must differ (not np.allclose) from every
other new state and from every stored N2 familiar state (all sets above).
LIBERO venv (py3.8), EGL.    python experiments/fc_n2_w8.py
-> results/flawed/n2_states_w8/ n2_goal_<tid>.npy [60, state_dim] for every FAMILIAR-layout tid (row t = state k = 940 + t; the harness trial index t = 0..59),
   states.npy, meta.json;  results/flawed/n2_branches_w8.json  (trial t, seed_base 13000000 + 1000 k, N2's convention).
Refuses to overwrite.  Stored files are only read.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fc_n2_tasks as N2                                                    # noqa: E402

DST = os.path.join(N2.OUT, "n2_states_w8")
SETS = ("n2_states_record", "n2_states_fresh", "n2_states_fresh2", "n2_states_fresh3", "n2_states_fresh4", "n2_states_s1x", "n2_states_gx", "n2_states_rx", "n2_states_wx", "n2_states_wy", "n2_states_wz", "n2_states_w5", "n2_states_w6", "n2_states_w7")
CONF = os.path.join(N2.OUT, "pp_conf_states", "n2_goal_0.npy")
K0, N = 940, 60
BR = os.path.join(N2.OUT, "n2_branches_w8.json")


def main():
    fn = os.path.join(DST, "states.npy")
    for p in (fn, BR):
        assert not os.path.exists(p), "exists: " + p
    tasks = json.load(open(os.path.join(N2.OUT, "n2_tasks.json"), encoding="utf-8"))["tasks"]
    fam = [t for t in tasks if t["layout"] == "familiar"]
    assert len(fam) == 16 and all(t["tid"] % 2 == 0 for t in fam), [t["tid"] for t in fam]
    ref = np.load(os.path.join(N2.ST, "n2_goal_0.npy"))
    old = [ref] + [np.load(os.path.join(N2.OUT, d, "states.npy")) for d in SETS]
    rx = old[8]; wxs = old[9]; wys = old[10]; wzs = old[11]; w5s = old[12]; w6s = old[13]; w7s = old[14]
    assert rx.shape == (40, ref.shape[1]) and wxs.shape == (60, ref.shape[1]) and wys.shape == (60, ref.shape[1]) and wzs.shape == (60, ref.shape[1]) and w5s.shape == (60, ref.shape[1]) and w6s.shape == (60, ref.shape[1]) and w7s.shape == (60, ref.shape[1]), (rx.shape, wxs.shape, wys.shape, wzs.shape)
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

    for k, want in ((0, ref[0]), (23, ref[23]), (540, rx[0]), (579, rx[39]), (580, wxs[0]), (639, wxs[59]), (640, wys[0]), (699, wys[59]), (700, wzs[0]), (759, wzs[59]), (760, w5s[0]), (819, w5s[59]), (820, w6s[0]), (879, w6s[59]), (880, w7s[0]), (939, w7s[59])):
        d = float(np.abs(draw(k) - want).max())
        print("[n2w8] reproduce state %d: max |diff| %.3g" % (k, d), flush=True)
        assert d == 0.0, "the sampler rule does not reproduce stored state %d" % k
    S = np.stack([draw(k) for k in range(K0, K0 + N)])
    assert S.shape == (N, ref.shape[1]), S.shape
    for i in range(N):
        for j in range(i):
            assert not np.allclose(S[i], S[j]), "duplicate new states %d %d" % (K0 + i, K0 + j)
        assert not any(np.allclose(S[i], o) for o in old), "new state %d equals a stored state" % (K0 + i)
    print("[n2w8] %d new states distinct from each other and from %d stored states" % (N, old.shape[0]), flush=True)
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
    print("[n2w8] wrote %s %s; branches %s" % (fn, S.shape, BR), flush=True)


if __name__ == "__main__":
    main()
