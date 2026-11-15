"""S2FIX3 detector sidecar (docs/SCAFFOLD_RESULTS.md 63; m3_prereg tag s2_fix3): the repo's off-the-shelf open-vocabulary detector,
OWLv2 (lifelong_ood/policy/where.OwlDetector's checkpoint google/owlv2-base-patch16-ensemble, the detector of role_hooks.py /
serve_guided role mode / b1_where_probe W3), served to the LIBERO client (py3.8, transformers 4.21 has no OWLv2) over a local
multiprocessing.connection socket.  Runs in the openpi venv (py3.11).  No simulator access: it sees only the RGB image it is sent.

Request  dict(img=<bytes, uint8 HxWx3 RGB>, shape=[H, W, 3], queries=[str, ...], topk=int)
Reply    dict(boxes=[[x0, y0, x1, y1] px in the sent image], scores=[[sigmoid score per query]], n=int)  -- the topk boxes by their max
         query score; only builtin types cross the socket (the two venvs' numpy versions differ).

    python experiments/s2fix3_det.py --port P [--device cuda]
"""
import argparse
import os
import sys
import time
from multiprocessing.connection import Listener

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT]
AUTH = b"s2fix3-owlv2"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    import numpy as np
    import torch
    from lifelong_ood.policy.where import OwlDetector
    det = OwlDetector(device=a.device)
    name = det.model.config._name_or_path
    lis = Listener(("127.0.0.1", a.port), authkey=AUTH)
    print("[s2fix3_det] %s on %s: detector listening on %d" % (name, a.device, a.port), flush=True)
    n_req = 0
    while True:
        conn = lis.accept()
        try:
            while True:
                try:
                    req = conn.recv()
                except EOFError:
                    break
                if req == "ping":
                    conn.send(dict(ok=True, model=name))
                    continue
                t0 = time.time()
                H, W, C = [int(v) for v in req["shape"]]
                img = np.frombuffer(req["img"], np.uint8).reshape(H, W, C)
                qs = list(req["queries"])
                inp = det.proc(text=[qs], images=img, return_tensors="pt").to(det.device)
                with torch.no_grad():
                    out = det.model(**inp)
                sc = torch.sigmoid(out.logits[0]).float().cpu().numpy()          # [n_boxes, Q]
                bx = out.pred_boxes[0].float().cpu().numpy()                     # [n_boxes, 4] cxcywh, normalised to the padded square
                S = float(max(H, W))
                k = int(min(int(req.get("topk", 60)), len(sc)))
                idx = np.argsort(-sc.max(1))[:k]
                boxes = [[float((bx[i, 0] - bx[i, 2] / 2) * S), float((bx[i, 1] - bx[i, 3] / 2) * S),
                          float((bx[i, 0] + bx[i, 2] / 2) * S), float((bx[i, 1] + bx[i, 3] / 2) * S)] for i in idx]
                conn.send(dict(boxes=boxes, scores=[[float(v) for v in sc[i]] for i in idx], n=k, secs=time.time() - t0))
                n_req += 1
                if n_req % 200 == 1:
                    print("[s2fix3_det] request %d: %d queries, %.3fs" % (n_req, len(qs), time.time() - t0), flush=True)
        finally:
            conn.close()


if __name__ == "__main__":
    main()
