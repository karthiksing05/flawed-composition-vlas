"""FLAWED_COMPOSITION prototype A, stage 1c read (section 5.7; m3_prereg tag pa1c).  CPU.  The engine is experiments/fa1x_read.py
(stage 1b's estimators, C1-C4 and verdict order): A = Ac (seeds 0-2, results/flawed/a1c_roll); ablations Ac_nobear, Ac_noft (seed 0);
H = Hb and R = Rb are stage 1b's runs (a1b_roll, the same protocol, states and noise); Ab (1b, seeds 0-2) and stage 1's A beside it.
-> results/flawed/a1c_read.json

    python experiments/fa1c_read.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fa1x_read as RX                                                       # noqa: E402

if __name__ == "__main__":
    RX.run(RX.Read("Ac", ("Ac_nobear", "Ac_noft"), [("Ab", (0, 1, 2)), ("A_stage1", (0,))]), "a1c_read.json")
