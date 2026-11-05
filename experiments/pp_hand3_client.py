"""The closed-loop locator client (docs/LOCATOR.md 10; m3_prereg pp_hand3).  LIBERO venv (py3.8).  LOC_SEED: the locator's seed.
pp_hand2_client's H1 arm, with P seeing the object's horizontal position from one look: the raw output of CP's network, then hand + offset during a hold.
"""
import os
import sys

os.environ["HAND_ARM"] = "H1"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np                                                          # noqa: E402
import pp_hand2_client as H2                                                # noqa: E402
import pp_loc as L                                                          # noqa: E402
import torch                                                                # noqa: E402

FC, FB, FK, LC = H2.FC, H2.FB, H2.FK, H2.LC
SEED = int(os.environ["LOC_SEED"])
HEAD = {"cream_cheese_1": 0, "wine_bottle_1": 1, "akita_black_bowl_1": 2}[H2.OE]
Fz = np.load(L.FRAMES)
K, R = Fz["K"], Fz["R"]
Z = float(np.nanmean(Fz["lab_C"][(Fz["demo"] < L.N_TRAIN_DEMO) & Fz["valid"][:, HEAD], HEAD, 2]))
NET = L.net()
NET.load_state_dict(torch.load(os.path.join(FC.DATA, "ckpt", f"pp_loc2_CP_s{SEED}.pt"), map_location="cpu"))
NET.eval()
CPU = torch.device("cpu")


class LocRec(H2.HandRec):
    def __init__(self, *a):
        super().__init__(*a)
        self.est, self.eoff, self.est_rows = None, None, []

    def _look(self):
        img = np.ascontiguousarray(self.obs["agentview_image"][::-1])[None]
        rc = L.locate(NET, img, CPU)[:, HEAD]
        self.est = L.lift(K, R, rc, np.array([Z]))[0, :2].copy()

    def _scene(self):
        sc = dict(FC.scene(self._e, self.obs, self._dj))
        p = np.array(sc["pos"], copy=True)
        p[H2.OI, :2] = self.est
        sc["pos"] = p
        return sc

    def _track(self):
        """pp_hand2_client.HandRec._track's H1 path, with the estimate in place of the object's horizontal position."""
        first = self.hm is None
        was = self.h_on and not first
        sc = FC.scene(self._e, self.obs, self._dj)
        if first:
            self.hm, self.cn = FB.Held(), FK.Canon()
            self.ee0 = sc["ee_pos"].copy()
            self.det, self.cnp = H2.ProprioHeld(H2.PG["tw"], H2.PG["lift"]), FK.Canon()
            self._look()
        po = np.asarray(self._con.pos(H2.OE), np.float64)
        h, o = self.hm.step(*self._con.touching(H2.OE), po, self.obs["robot0_eef_pos"])
        self.cn.step(h, po, self._tgt)                                      # the oracle's own tracker, stepped as in H1; P does not see it
        hp = self.det.step(sc["grip"], self.prev, sc["ee_pos"][2])
        if hp and not was:
            self.eoff = self.est - sc["ee_pos"][:2]
        if hp:
            self.est = sc["ee_pos"][:2] + self.eoff
        pe = np.array([self.est[0], self.est[1], po[2]])
        zp, cp = self.cnp.step(hp, pe, self._tgt)
        self.h_on, self.zr, self.cy, self.off_ = hp, zp, cp, o
        self.rot0 = sc["ee_rot"].copy() if hp and not was else (self.rot0 if hp else None)
        self.d0t = float(np.linalg.norm(self.est - np.asarray(self._tgt()[0])[:2])) if hp and not was else (self.d0t if hp else None)
        self.est_rows.append(self.est.copy())

    def data(self):
        d = super().data()
        d["est"] = np.array(self.est_rows, np.float32).reshape(-1, 2)
        return d


LC.OffScreenRenderEnv = lambda *a, **k: LocRec(H2.ORIG(*a, **k), H2.OE, H2.TE)

if __name__ == "__main__":
    assert "--fresh" in sys.argv and "--branches" in sys.argv
    LC.main()
