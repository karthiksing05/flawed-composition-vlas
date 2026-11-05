"""The hold gate's client (docs/HOLD_GATE.md; m3_prereg pp_gate): the atlas's client with the gripper forced shut while held and away from the target,
and in arm GA the known target prompt while held.  LIBERO venv (py3.8).  GATE_ARM: G | GA.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fc_n2_client as FN                                                   # noqa: E402
import pp_atlas as PA                                                       # noqa: E402

LC = FN.LC
ARM = os.environ["GATE_ARM"]
assert ARM in ("G", "GA")
R_GATE = 0.05
TGT_PARTNER = {"plate_1": "K1", "flat_stove_1_cook_region": "K2", "wooden_cabinet_1_top_side": "K3", "akita_black_bowl_1": "K6"}
TID = int(sys.argv[sys.argv.index("--tasks") + 1])
OUTF = sys.argv[sys.argv.index("--out") + 1]
TASK = FN.N2["tasks"][TID]
assert TASK["tid"] == TID
TEXT = {t["compound"]: t for t in FN.N2["tasks"] if t["layout"] == "familiar"}
POST = TASK["language"]
if ARM == "GA":
    assert TASK["compound"][0] == "N"
    k = TEXT[TGT_PARTNER[TASK["target"]]]
    assert k["target"] == TASK["target"]
    POST = k["language"]
ORIG, RECS, CUR = LC.OffScreenRenderEnv, [], {}


class GateRec(PA.Rec):
    def __init__(self, *a):
        super().__init__(*a)
        self.fired = 0
        CUR["env"] = self

    def held_now(self):
        return bool(self.rows["held"] and self.rows["held"][-1])

    def step(self, a):
        R = self.rows
        if self.held_now() and float(np.linalg.norm((R["obj"][-1] - R["tgt"][-1])[:2])) > R_GATE:
            a = list(a)
            if a[6] <= 0:
                self.fired += 1
            a[6] = 1.0
        return super().step(a)

    def close(self):
        if self.n > 0:
            d = self.data()
            d["fired"] = np.array(self.fired)
            RECS.append(d)
            PA.save(OUTF.replace(".json", "_track.npz"), RECS)
        self._e.close()


class SwitchClient(LC._wcp.WebsocketClientPolicy):
    def infer(self, el):
        el = dict(el)
        el["prompt"] = POST if CUR["env"].held_now() else TASK["language"]
        return super().infer(el)


LC.OffScreenRenderEnv = lambda *a, **k: GateRec(ORIG(*a, **k), TASK["object"], TASK["target"])
LC._wcp.WebsocketClientPolicy = SwitchClient

if __name__ == "__main__":
    assert "--fresh" in sys.argv and "--branches" in sys.argv
    LC.main()
