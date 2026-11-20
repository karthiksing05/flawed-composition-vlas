"""SR 82 diagnostic server (not registered): experiments/wa_s.py serve, unchanged, with a per-query log of the model OUTPUT next to the servo it contains.
Per query: episode, t, the 4-dim channel state sent (dx dy dz [m], role id), the first 5 action rows (xyz) returned, and the servo that was added
(raas_model's rule recomputed from the same state: clip(K * rel, +-max_step) * (role id > 1.5), raw action units).  expert = action - servo.
   SR82_LOG=file.jsonl python experiments/sr82_diag_serve.py --ckpt D --port P --seg-ckpt F --config C   (openpi venv; same arguments as wa_s.py serve)"""
import argparse, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]
import wa_s
import sc_rv as RV
RV._openpi()
from lifelong_ood.policy import serve as S
from lifelong_ood.policy.raas_model import servo_fields
G = servo_fields()
LOG = os.environ["SR82_LOG"]
_orig = S.GuidedPolicy.infer
def infer(self, obs, **kw):
    r = _orig(self, obs, **kw)
    st = np.asarray(obs["observation/state"], np.float64).ravel()[:4]
    act = np.asarray(r["actions"], np.float64)[:5, :3]
    m = 1.0 if st[3] > 1.5 else 0.0
    sv = np.clip(G["servo_k"] * st[:3], -G["servo_max"], G["servo_max"]) * m
    if not obs.get("meta/ff", False):
        with open(LOG, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(dict(ep=int(obs.get("meta/episode", -1)), t=int(obs.get("meta/t", -1)), state=st.round(4).tolist(), role=float(st[3]),
                                     act=act.mean(0).round(3).tolist(), servo=sv.round(3).tolist(), expert=(act.mean(0) - sv).round(3).tolist(), prompt=str(obs.get("prompt")))) + "\n")
    return r
S.GuidedPolicy.infer = infer
ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True); ap.add_argument("--port", type=int, required=True); ap.add_argument("--seed", type=int, default=13)
ap.add_argument("--config", default=""); ap.add_argument("--seg-ckpt", dest="seg_ckpt", default=""); ap.add_argument("--diag", default="")
ap.add_argument("--chan-diag", dest="chan_diag", default="")
wa_s.serve_cmd(ap.parse_args())
