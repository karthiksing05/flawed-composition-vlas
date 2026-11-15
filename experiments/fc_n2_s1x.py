"""A batch of NEW fresh familiar-layout start states for the N2 scene (scene 1), for S1-CROSS (m3_prereg s1_cross;
docs/SCAFFOLD_RESULTS.md 65): states k = 236..275, drawn by experiments/fc_n2_fresh.py's rule unchanged (LIBERO's sampler on the
familiar BDDL, np.random.seed and env.seed = 600000 + k; the rule fc_n2_tasks.py used for k = 0-23, fc_n2_extra.py for 24-35,
fc_n2_fresh.py for 36-55, fc_n2_fresh2.py for 56-115, fc_n2_fresh3.py for 116-155 and fc_n2_fresh4.py for 156-195).  k 196-235 are
skipped on purpose, so that a concurrent batch that takes the next block cannot coincide with this one.  This file is fc_n2_fresh4.py
with only K0, the output names and the duplicate / reproduction checks widened to the fourth batch.  No policy is run.
Before writing, the rule must reproduce stored states exactly: k = 0, 1, 23 (n2_states/n2_goal_0.npy), k = 24, 35
(n2_states_record/states.npy), k = 36, 55 (n2_states_fresh), k = 56, 115 (n2_states_fresh2), k = 116, 155 (n2_states_fresh3) and
k = 156, 195 (n2_states_fresh4).  Every new state must differ (not np.allclose) from every other new state, from every stored N2
familiar state (k = 0-195) and from PLAN_PARTS's confirmatory familiar states (results/flawed/pp_conf_states/n2_goal_0.npy), when that
file exists.
LIBERO venv (py3.8), EGL.

    python experiments/fc_n2_s1x.py
-> results/flawed/n2_states_s1x/
     states.npy [40, state_dim] (rows k = 236..275) + meta.json
     n2_goal_<tid>.npy [64, state_dim] for every FAMILIAR-layout tid: rows 0-23 = the stored n2_states/n2_goal_<tid>.npy unchanged,
         rows 24-63 = states.npy.  So the harness's trial index t >= 24 addresses state k = t + 212.
   results/flawed/n2_branches_s1x.json        trials 24..63, seed_base 13000000 + 1000 k (k = trial + 212), N2's convention
   results/flawed/n2_branches_s1x_smoke.json  its first two entries (unused; the smoke runs on stored states 0-1)
Refuses to overwrite.  Stored files are only read.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fc_n2_tasks as N2                                                    # noqa: E402

DST = os.path.join(N2.OUT, "n2_states_s1x")
REC = os.path.join(N2.OUT, "n2_states_record", "states.npy")               # fc_n2_extra.py: k = 24..35
FR1 = os.path.join(N2.OUT, "n2_states_fresh", "states.npy")                # fc_n2_fresh.py: k = 36..55
FR2 = os.path.join(N2.OUT, "n2_states_fresh2", "states.npy")               # fc_n2_fresh2.py: k = 56..115
FR3 = os.path.join(N2.OUT, "n2_states_fresh3", "states.npy")               # fc_n2_fresh3.py: k = 116..155
FR4 = os.path.join(N2.OUT, "n2_states_fresh4", "states.npy")               # fc_n2_fresh4.py: k = 156..195
CONF = os.path.join(N2.OUT, "pp_conf_states", "n2_goal_0.npy")             # pp_conf_states.py familiar, seed 9100000 + k
N_STORED = N2.N_SMOKE + N2.N_MAIN                                           # 24 rows per stored task file
K0, N = 236, 40                                                             # k = 236..275
SHIFT = K0 - N_STORED                                                       # harness trial t >= 24  ->  k = t + 212
BR = os.path.join(N2.OUT, "n2_branches_s1x.json")
BR_SMOKE = os.path.join(N2.OUT, "n2_branches_s1x_smoke.json")


def main():
    fn = os.path.join(DST, "states.npy")
    for p in (fn, BR, BR_SMOKE):
        assert not os.path.exists(p), "exists: " + p
    tasks = json.load(open(os.path.join(N2.OUT, "n2_tasks.json"), encoding="utf-8"))["tasks"]
    fam = [t for t in tasks if t["layout"] == "familiar"]
    assert len(fam) == 16 and all(t["tid"] % 2 == 0 for t in fam), [t["tid"] for t in fam]
    stored = {t["tid"]: np.load(os.path.join(N2.ST, "n2_goal_%d.npy" % t["tid"])) for t in fam}
    ref = stored[0]
    assert ref.shape[0] == N_STORED, ref.shape
    for tid, s in stored.items():
        assert s.shape == ref.shape and np.array_equal(s, ref), "familiar tid %d does not share tid 0's states" % tid
    rec, fr1, fr2, fr3, fr4 = (np.load(p) for p in (REC, FR1, FR2, FR3, FR4))
    assert rec.shape == (12, ref.shape[1]) and fr1.shape == (20, ref.shape[1]) and fr2.shape == (60, ref.shape[1]) \
        and fr3.shape == (40, ref.shape[1]) and fr4.shape == (40, ref.shape[1]), \
        (rec.shape, fr1.shape, fr2.shape, fr3.shape, fr4.shape)
    conf = np.load(CONF) if os.path.exists(CONF) else None
    print("[n2s1x] pp_conf familiar states: %s" % (None if conf is None else str(conf.shape)), flush=True)

    env = N2.make_env(os.path.join(N2.BD, "K1.bddl"))

    def draw(k):
        np.random.seed(N2.BASE_FAM + k)
        env.seed(N2.BASE_FAM + k)
        env.reset()
        return env.sim.get_state().flatten()

    for k, want in ((0, ref[0]), (1, ref[1]), (23, ref[23]), (24, rec[0]), (35, rec[11]), (36, fr1[0]), (55, fr1[19]),
                    (56, fr2[0]), (115, fr2[59]), (116, fr3[0]), (155, fr3[39]), (156, fr4[0]), (195, fr4[39])):
        d = float(np.abs(draw(k) - want).max())
        print("[n2s1x] reproduce state %d: max |diff| %.3g" % (k, d), flush=True)
        assert d == 0.0, "the sampler rule does not reproduce stored state %d" % k
    S = np.stack([draw(k) for k in range(K0, K0 + N)])
    assert S.shape == (N, ref.shape[1]), S.shape
    old = [ref, rec, fr1, fr2, fr3, fr4] + ([conf] if conf is not None and conf.shape[1] == ref.shape[1] else [])
    old = np.concatenate(old)
    for i in range(N):
        for j in range(i):
            assert not np.allclose(S[i], S[j]), "duplicate new states %d %d" % (K0 + i, K0 + j)
        assert not any(np.allclose(S[i], o) for o in old), "new state %d equals a stored state" % (K0 + i)
    print("[n2s1x] %d new states distinct from each other and from %d stored states" % (N, old.shape[0]), flush=True)
    pos = []
    for s in S:
        env.reset()
        env.set_init_state(s)
        pos.append(N2.positions(env, list(N2.MOVR))[:, :2].tolist())
    env.close()

    os.makedirs(DST, exist_ok=True)
    for tid, s in stored.items():
        out = os.path.join(DST, "n2_goal_%d.npy" % tid)
        assert not os.path.exists(out), "exists: " + out
        H = np.concatenate([s, S])
        assert H.shape[0] == N_STORED + N and np.array_equal(H[:N_STORED], s) and np.array_equal(H[N_STORED:], S)
        np.save(out, H)
    br = [dict(id="k%d" % (t + SHIFT), trial=t, seed_base=13000000 + 1000 * (t + SHIFT), at=0, variant="n2")
          for t in range(N_STORED, N_STORED + N)]
    for p, b in ((BR, br), (BR_SMOKE, br[:2])):
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(b, fh, indent=1)
    with open(os.path.join(DST, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(k=list(range(K0, K0 + N)), seed_rule="np.random.seed = env.seed = %d + k" % N2.BASE_FAM, layout="familiar",
                       bddl="results/flawed/n2_bddl/K1.bddl (every task of the familiar layout shares the states)",
                       reproduced=dict(n2_goal_0=[0, 1, 23], n2_states_record=[24, 35], n2_states_fresh=[36, 55],
                                       n2_states_fresh2=[56, 115], n2_states_fresh3=[116, 155], n2_states_fresh4=[156, 195]),
                       skipped="k 196-235 (left for a concurrent batch)",
                       distinct_from=dict(stored_rows=int(old.shape[0]), pp_conf=conf is not None),
                       harness=dict(files=["n2_goal_%d.npy" % t for t in sorted(stored)], rows="0-23 = stored n2_states (k = 0-23), "
                                    "24-63 = states.npy (k = 236-275)", trial_to_k="k = trial + %d for trial >= 24" % SHIFT),
                       branches=dict(main=os.path.relpath(BR, N2.ROOT), smoke=os.path.relpath(BR_SMOKE, N2.ROOT),
                                     seed_base_rule="13000000 + 1000 k"),
                       movable=list(N2.MOVR), positions_xy=pos), fh, indent=1)
    np.save(fn + ".tmp.npy", S)
    os.replace(fn + ".tmp.npy", fn)
    print("[n2s1x] wrote %s %s; %d harness files; branches %s" % (fn, S.shape, len(stored), BR), flush=True)


if __name__ == "__main__":
    main()
