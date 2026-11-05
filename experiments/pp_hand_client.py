"""The hand-over's client (docs/HAND_OVER.md; m3_prereg pp_hand): the atlas's client until the first hold, P_min's actions after it.
LIBERO venv (py3.8).  P_min steps its own tracking exactly as fa1h_roll.episode does, from the episode's start.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fc_n2_client as FN                                                   # noqa: E402
import pp_atlas as PA                                                       # noqa: E402
import torch                                                                # noqa: E402
import fa1_common as FC                                                     # noqa: E402
import fa1b_common as FB                                                    # noqa: E402
import fa1c_common as FK                                                    # noqa: E402
from lifelong_ood.policy import factored_flow as FF                         # noqa: E402
from pp_p_roll import load_any                                              # noqa: E402

LC = FN.LC
REPLAN = 5
P_SEED = 10
TID = int(sys.argv[sys.argv.index("--tasks") + 1])
OUTF = sys.argv[sys.argv.index("--out") + 1]
TASK = FN.N2["tasks"][TID]
assert TASK["tid"] == TID
torch.set_num_threads(1)
M, _ = load_any(os.path.join(FC.DATA, "ckpt", f"pp_full_naf_s{P_SEED}", "P.npz"))
OE, TE = TASK["object"], TASK["target"]
OI, TI = FC.ENT.index(OE), FC.ENT.index(TE)
ORIG, RECS, CUR = LC.OffScreenRenderEnv, [], {}


class HandRec(PA.Rec):
    def __init__(self, *a):
        super().__init__(*a)
        self.on, self.handed, self.calls, self.plan = False, -1, 0, []
        self.hm, self.cn, self.obs, self.prev = None, None, None, -1.0
        self.h_on, self.off_, self.zr, self.cy = False, np.zeros(3), 0.0, 0.0
        self.rot0, self.d0t, self.ee0 = None, None, None
        CUR["env"] = self

    def _tgt(self):
        p = FC.scene(self._e, self.obs, self._dj)["pos"]
        return p[TI], float(p[FK.TABLE, 2])

    def _track(self):
        """fa1h_roll.episode's track(), onset_rot and onset_d, with the policy gate off (pgate None)."""
        was = self.h_on if self.hm is not None else False
        if self.hm is None:
            self.hm, self.cn = FB.Held(), FK.Canon()
            self.ee0 = FC.scene(self._e, self.obs, self._dj)["ee_pos"].copy()
        h, o = self.hm.step(*self._con.touching(OE), self._con.pos(OE), self.obs["robot0_eef_pos"])
        z, c = self.cn.step(h, self._con.pos(OE), self._tgt)
        self.h_on, self.off_, self.zr, self.cy = h, o, z, c
        self.rot0 = (FC.scene(self._e, self.obs, self._dj)["ee_rot"].copy() if h and not was else (self.rot0 if h else None))
        self.d0t = (float(np.linalg.norm(np.asarray(self._con.pos(OE))[:2] - np.asarray(self._tgt()[0])[:2])) if h and not was else (self.d0t if h else None))

    def _part_action(self):
        if not self.plan:
            b = FF.batch_from_scene(FC.scene(self._e, self.obs, self._dj), OI, TI, self.prev, held=float(self.h_on), off=self.off_, zrest=self.zr, cyaw=self.cy, rot0=self.rot0)
            if float(getattr(M, "hscale", 0.0)) > 0:
                b["ee0"] = torch.tensor(np.asarray(self.ee0), dtype=torch.float32)[None]
            if bool(getattr(M, "tscale", False)) and self.d0t is not None:
                b["d0t"] = torch.tensor([self.d0t], dtype=torch.float32)
            g = torch.Generator().manual_seed(int(CUR["seed_base"]) + self.calls)
            ch = M.sample(b, generator=g)[0].numpy()
            self.plan = [np.clip(x, -1.0, 1.0) for x in ch[:REPLAN]]
            self.calls += 1
        return self.plan.pop(0).tolist()

    def step(self, a):
        if self.on:
            a = self._part_action()
        r = super().step(a)
        if self.n < PA.WAIT:
            return r
        self.obs = r[0]
        self.prev = float(a[6]) if self.hm is not None else -1.0
        self._track()
        if not self.on and self.h_on:
            self.on, self.handed = True, len(self.rows["held"]) - 1
        return r

    def close(self):
        if self.n > 0:
            d = self.data()
            d["handed"] = np.array(self.handed)
            RECS.append(d)
            PA.save(OUTF.replace(".json", "_track.npz"), RECS)
        self._e.close()


class HandClient(LC._wcp.WebsocketClientPolicy):
    def infer(self, el):
        CUR["seed_base"] = el["meta/seed_base"]
        if CUR["env"].on:                                                    # the env ignores this plan after the hand-over
            return {"actions": np.zeros((REPLAN, 7))}
        return super().infer(el)


LC.OffScreenRenderEnv = lambda *a, **k: HandRec(ORIG(*a, **k), OE, TE)
LC._wcp.WebsocketClientPolicy = HandClient

if __name__ == "__main__":
    assert "--fresh" in sys.argv and "--branches" in sys.argv
    LC.main()
