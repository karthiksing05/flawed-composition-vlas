"""Arm LR (docs/LOCATOR.md 11; m3_prereg pp_hand4).  LIBERO venv (py3.8).  LOC_SEED: the locator's seed.
pp_hand3_client's LS, with the estimate updating every LOOK steps while no hold is on and the gripper command is open.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np                                                          # noqa: E402
import pp_hand3_client as H3                                                # noqa: E402

H2, FC, FB, FK, LC, L = H3.H2, H3.FC, H3.FB, H3.FK, H3.LC, H3.L
LOOK, KEEP = 5, 3


class LookRec(H3.LocRec):
    def __init__(self, *a):
        super().__init__(*a)
        self.buf, self.nt = [], 0

    def _track(self):
        first = self.hm is None
        was = self.h_on and not first
        sc = FC.scene(self._e, self.obs, self._dj)
        if first:
            self.hm, self.cn = FB.Held(), FK.Canon()
            self.ee0 = sc["ee_pos"].copy()
            self.det, self.cnp = H2.ProprioHeld(H2.PG["tw"], H2.PG["lift"]), FK.Canon()
        po = np.asarray(self._con.pos(H2.OE), np.float64)
        h, o = self.hm.step(*self._con.touching(H2.OE), po, self.obs["robot0_eef_pos"])
        hp = self.det.step(sc["grip"], self.prev, sc["ee_pos"][2])
        if first or (not hp and not self.prev > 0 and self.nt % LOOK == 0):
            self._look()
            self.buf = (self.buf + [self.est.copy()])[-KEEP:]
            self.est = np.median(self.buf, 0)
        self.nt += 1
        self.cn.step(h, po, self._tgt)
        if hp and not was:
            self.eoff, self.buf = self.est - sc["ee_pos"][:2], []
        if hp:
            self.est = sc["ee_pos"][:2] + self.eoff
        pe = np.array([self.est[0], self.est[1], po[2]])
        zp, cp = self.cnp.step(hp, pe, self._tgt)
        self.h_on, self.zr, self.cy, self.off_ = hp, zp, cp, o
        self.rot0 = sc["ee_rot"].copy() if hp and not was else (self.rot0 if hp else None)
        self.d0t = float(np.linalg.norm(self.est - np.asarray(self._tgt()[0])[:2])) if hp and not was else (self.d0t if hp else None)
        self.est_rows.append(self.est.copy())


LC.OffScreenRenderEnv = lambda *a, **k: LookRec(H2.ORIG(*a, **k), H2.OE, H2.TE)

if __name__ == "__main__":
    assert "--fresh" in sys.argv and "--branches" in sys.argv
    LC.main()
