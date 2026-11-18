"""RAA-c (SR 77) server-side unit check, CPU, openpi venv, no model: RAAServer.final with arm raac on stub inputs.  Object role -> state [obj - eef, 1]; held target
role with an offset -> [target - eef - offset, 2] and the prompt is the role phrase with NO suffix; an unrouted instruction or a refused lexicon -> zeros(4); the
8-dim client state never reaches the policy.  (The 4-episode smoke on the 20-step checkpoint never held an object, so this covers the held branch.)"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]
import raa as RA
import sc_rv as RV


class Pol:
    def infer(self, o):
        self.o = o
        return {}


srv = object.__new__(RA.RAAServer)
srv.RV, srv.arm, srv.diag, srv.T16, srv.n, srv.noch = RV, "raac", None, RV.t16(), 0, 0
text = "put the bowl on the plate"
P = srv.T16.given(text)
assert P is not None
ee = np.array([0.1, 0.2, 0.3])
st8 = np.concatenate([ee, np.zeros(5)])
obj, tgt, off = np.array([0.15, 0.1, 0.1]), np.array([0.4, 0.25, 0.12]), np.array([0.01, -0.04, -0.03])
raa = dict(ok=True, obj=obj.tolist(), tgt=tgt.tolist(), off_ok=True, off=off.tolist(), toff=off.tolist())
pol = Pol()
# object role
srv.cur = dict(text=text, raa=raa, t=0, ep=0)
srv.final({"prompt": P[0], "observation/state": st8.copy()}, pol)
assert pol.o["prompt"] == P[0] and "rel:" not in pol.o["prompt"]
assert np.allclose(pol.o["observation/state"], np.r_[obj - ee, 1.0], atol=1e-6) and pol.o["observation/state"].dtype == np.float32, pol.o["observation/state"]
print("[raac-final] object role:", np.round(pol.o["observation/state"], 4).tolist(), repr(pol.o["prompt"]))
# held target role, offset fixed at onset
srv.final({"prompt": P[1], "observation/state": st8.copy()}, pol)
assert pol.o["prompt"] == P[1] and "rel:" not in pol.o["prompt"]
assert np.allclose(pol.o["observation/state"], np.r_[tgt - ee - off, 2.0], atol=1e-6), pol.o["observation/state"]
print("[raac-final] held target role:", np.round(pol.o["observation/state"], 4).tolist(), repr(pol.o["prompt"]))
# held, no offset yet: eef-relative, counted, still 4-dim
srv.cur = dict(text=text, raa=dict(raa, off_ok=False), t=1, ep=0)
srv.final({"prompt": P[1], "observation/state": st8.copy()}, pol)
assert np.allclose(pol.o["observation/state"], np.r_[tgt - ee, 2.0], atol=1e-6) and getattr(srv, "nooff", 0) == 1
# lexicon refused
srv.cur = dict(text=text, raa=dict(ok=False, why="x"), t=2, ep=0)
srv.final({"prompt": P[0], "observation/state": st8.copy()}, pol)
assert np.all(pol.o["observation/state"] == 0) and pol.o["observation/state"].shape == (4,) and srv.noch == 1
# unrouted instruction (K7-like / anything T16 does not route)
unr = [k for k, v in srv.T16._GIVEN.items() if v is None][0]
srv.cur = dict(text=unr, raa={}, t=3, ep=0)
assert srv.T16.given(srv.cur["text"]) is None
srv.final({"prompt": srv.cur["text"], "observation/state": st8.copy()}, pol)
assert np.all(pol.o["observation/state"] == 0) and pol.o["observation/state"].shape == (4,)
print("[raac-final] refused lexicon and unrouted: zeros(4); ok")
