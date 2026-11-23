"""PZDEV dev read (experiments/prereg/pzdev_pred.txt): the RAA-F dev-read pipeline (raaf_dev_read.py, imported unchanged) on the two pi0 arms.
Phase A (K1-K6, N1-N6; dev trials 4-23 from dev_<route>/raaf<route>_main_<tid>.json, new trials 24-63 from raaf<route>_rx_<tid>.json): paired bootstrap B = 2000, rng 0
(raa_read.diff: tasks and states resampled).  Phase B (F7 pool, F6_b, F6_a, K1t, K2t; dev_<route>/wa/full_<tid>.json): wa_read.bootm.
Registered contrasts: (d1) pzA0 - pzsL on N; (d2) pzA0 - A0 (pi0.5) on N; (d3) pzsL - raasl (pi0.5) on N; K non-inferiority at -.05 for pzA0 vs pzsL.  Also K and Phase B contrasts for the same pairs.
Writes once.

    python experiments/pzdev_read.py --out results/flawed/pz/pzdev_read.json
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import raaf_dev_read as RD                                                   # noqa: E402
import raa_read as RR                                                        # noqa: E402

ROUTES = ["A0", "pzsL", "pzA0"]                                              # A0 = pi0.5 servo-off comparator (results/flawed/raaf/dev_A0); raasl added by the pipeline
PAIRS = [("pzA0", "pzsL"), ("pzA0", "A0"), ("pzsL", "raasl"), ("pzA0", "raasl")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = os.path.abspath(a.out)
    assert not os.path.exists(out), "never overwrite: %s" % out
    arms = ["raasl"] + ROUTES
    res = dict(B=RR.B, routes=ROUTES, pairs=PAIRS, phaseA=RD.phaseA(arms, PAIRS), phaseB=RD.phaseB(arms, PAIRS))
    print("[pzdev-read]", json.dumps(res), flush=True)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "x", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print("[pzdev-read] -> %s" % out, flush=True)


if __name__ == "__main__":
    main()
