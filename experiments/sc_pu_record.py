"""Record one demonstration per new pairing from the extra start states (fc_n2_extra.py, k = 24..), with sc_pu.py's own recorder
(oracle arm: U on simulator poses), unchanged.  sc_pu.record_states takes every index of the state array that neither trial list
uses, so appending the extra states after the 24 stored ones makes exactly k = 24.. available and keeps main states 4-23 out.
LIBERO venv (py3.8), EGL, one GPU.

    python experiments/sc_pu_record.py --index i --record DIR [--nrec 1] [--maxtry 12]
index: sc_pu.combos() (seed x compound x layout); only familiar-layout combos are allowed (the extra states are familiar).
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sc_pu as PU                                                          # noqa: E402

EXTRA = os.path.join(PU.OUT, "n2_states_record", "states.npy")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, required=True)
    ap.add_argument("--record", required=True)
    ap.add_argument("--nrec", type=int, default=1)
    ap.add_argument("--maxtry", type=int, default=12)
    a = ap.parse_args()
    a.arm = "oracle"
    assert not PU.SMOKE, "unset SC_SMOKE: this records from the extra states only"
    s, tid = PU.combos()[a.index]
    t = PU.tasks()["tasks"][tid]
    assert t["tid"] == tid and t["layout"] == "familiar", (tid, t["layout"])
    S = np.load(os.path.join(PU.OUT, "n2_states", "n2_goal_%d.npy" % tid))
    X = np.load(EXTRA)
    assert len(S) == 24 and X.shape[1] == S.shape[1], (S.shape, X.shape)
    todo, used = PU.record_states(np.concatenate([S, X]))
    assert [j for j, _ in todo] == list(range(24, 24 + len(X))), (todo, used)
    print("[pu_rec] tid %d %s seed %d: recording from states %d-%d" % (tid, t["compound"], s, todo[0][0], todo[-1][0]), flush=True)
    PU.record(a, s, t, np.concatenate([S, X]))


if __name__ == "__main__":
    main()
