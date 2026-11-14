"""F6O2's reference on the new states: experiments/fx_ref.py, imported and run unchanged (scripted servo with the plate pose
frozen at t = 0, or P_min given the referent; fx tid 2 = F6_disp_vis, ref K1 freeze), with fx_tasks' META / OUT / ST pointed
at results/flawed/f6o2/refsrc (written by f6o2_tasks.py: fx's task table, fx tid 2's states = f6o2's rows, f6o2's branches as
fx_branches_<phase>.json).  The covers do not collide and the reference renders nothing, so it is the reference for every f6o2
condition on these states, as fx's was for f6o.  LIBERO venv.

    python experiments/f6o2_ref.py --phase smoke|main [--pseed 10] [--kind servo|pmin]   -> results/flawed/f6o2/ref/
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]
import fx_tasks as FT                                                      # noqa: E402
import f6o2_tasks as O2                                                    # noqa: E402

FT.META = os.path.join(O2.REFSRC, "fx_tasks.json")
FT.OUT = O2.REFSRC
FT.ST = O2.REFSRC
import fx_ref as XF                                                        # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=("smoke", "main"))
    ap.add_argument("--pseed", type=int, default=10)
    ap.add_argument("--kind", default="servo", choices=("servo", "pmin"))
    a = ap.parse_args()
    a.tids, a.out = "2", os.path.join(O2.OUT, "ref")
    assert XF.FT is FT and FT.ST == O2.REFSRC
    XF.run(a)
