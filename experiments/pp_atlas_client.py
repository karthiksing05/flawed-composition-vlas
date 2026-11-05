"""The break atlas's VLA client (docs/BREAK_ATLAS.md; m3_prereg pp_atlas): N2's client, unchanged, with the simulator read after each step
into <out>_track.npz.  Prompts, images, state, seeds and episode protocol are N2's.  LIBERO venv (py3.8).

    python experiments/pp_atlas_client.py --port P --suite n2_goal --tasks 16 --trials 1 --fresh results/flawed/n2_states \
        --branches results/flawed/n2_branches_main.json --videos 0 --bind track --out X.json
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fc_n2_client as FN                                                   # noqa: E402
import pp_atlas as PA                                                       # noqa: E402

LC = FN.LC
TID = int(sys.argv[sys.argv.index("--tasks") + 1])
OUTF = sys.argv[sys.argv.index("--out") + 1]
TASK = FN.N2["tasks"][TID]
assert TASK["tid"] == TID
ORIG, RECS = LC.OffScreenRenderEnv, []


class ClientRec(PA.Rec):
    def close(self):
        if self.n > 0:
            RECS.append(self.data())
            PA.save(OUTF.replace(".json", "_track.npz"), RECS)
        self._e.close()


LC.OffScreenRenderEnv = lambda *a, **k: ClientRec(ORIG(*a, **k), TASK["object"], TASK["target"])

if __name__ == "__main__":
    assert "--fresh" in sys.argv and "--branches" in sys.argv
    LC.main()
