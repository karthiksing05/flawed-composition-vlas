"""SR 78 diagnosis: does the SERVED path add the servo?  (sbatch only; one GPU; openpi venv.)

experiments/raas_check.py check 4 tests PI0PytorchS.sample_actions.  Serving (experiments/raa.py -> lifelong_ood/policy/serve.py GuidedPolicy.infer) never calls
model.sample_actions: it calls PolicyBackbone.sample (backbone.py, its own Euler loop on v_full) and PolicyBackbone.unnormalize, neither of which knows the servo.
This script (a) greps both files for the servo, (b) runs the REAL served path (observation -> prefix -> sample -> unnormalize) on real held-target demo frames with the
RAA-s checkpoint, and compares the demo action chunk with: served as-is, served + servo (what training assumed), and RAA-c's served chunk; (c) the model's own
sample_actions (with servo) vs the backbone path on the same noise = the size of the missing term.
    python experiments/raas_diag.py --raas CKPT --raac CKPT
"""
import argparse, io, json, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = ["/coc/flash10/ksingara3/robot/openpi/src", ROOT]
LR = "/coc/flash10/ksingara3/robot/data/hf/lerobot/lol/libero_g8v_raac"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raas", required=True)
    ap.add_argument("--raac", required=True)
    ap.add_argument("--eps", default="0,40,80,120,200")
    a = ap.parse_args()
    for f in ("lifelong_ood/policy/backbone.py", "lifelong_ood/policy/serve.py"):
        txt = open(os.path.join(ROOT, f), encoding="utf-8").read()
        print("[diag] %s mentions servo/raas: %s" % (f, ("servo" in txt) or ("raas" in txt)), flush=True)
    import pyarrow.parquet as pq
    from PIL import Image
    from lifelong_ood.policy import raas_model as RS
    from lifelong_ood.policy.backbone import PolicyBackbone
    import torch
    ns = json.load(open(RS.NORM, encoding="utf-8"))["norm_stats"]
    g = json.load(open(RS.GATE, encoding="utf-8"))
    tasks = {}
    for l in open(os.path.join(LR, "meta", "tasks.jsonl"), encoding="utf-8"):
        d = json.loads(l); tasks[d["task_index"]] = d["task"]
    # frames
    frames = []
    for ep in [int(x) for x in a.eps.split(",")]:
        t = pq.read_table(os.path.join(LR, "data", "chunk-000", "episode_%06d.parquet" % ep)).to_pydict()
        st = np.asarray(t["state"], np.float32); ac = np.asarray(t["actions"], np.float32)
        idx = np.where(st[:, 3] == 2.0)[0]
        for i in idx[:: max(1, len(idx) // 3)][:3]:
            if i + 10 > len(ac): continue
            dec = lambda c: np.asarray(Image.open(io.BytesIO(t[c][i]["bytes"])).convert("RGB"))
            frames.append(dict(ep=ep, i=int(i), obs={"observation/image": dec("image"), "observation/wrist_image": dec("wrist_image"),
                                                     "observation/state": st[i], "prompt": tasks[t["task_index"][i]]}, demo=ac[i:i + 10, :7]))
    print("[diag] %d held-target frames from eps %s" % (len(frames), a.eps), flush=True)
    res = {}
    for tag, ck, name in (("raas", a.raas, "pi05_g8v_raas_ae"), ("raac", a.raac, "pi05_g8v_raac_ae")):
        bk = PolicyBackbone(name, checkpoint_dir=ck, device="cuda")
        rows = []
        for f in frames:
            o = bk.observation(f["obs"]); pre = bk.prefix(o)
            nz = np.random.default_rng(f["i"]).standard_normal((1, bk.dim_full)).astype(np.float32)
            served = bk.unnormalize(bk.sample(1, pre, seed=0, num_steps=10, noise=nz), state=o.state[0].float().cpu().numpy())[0]   # exactly GuidedPolicy.infer
            r = dict(served=served)
            if tag == "raas":
                sn = RS.servo_norm_torch(bk.model.config, o.state, 32)[0, 0, :3].cpu().numpy()          # normalised-space servo
                rn = bk.sample(1, pre, seed=0, num_steps=10, noise=nz).reshape(bk.H, 7).copy()
                rn[:, :3] += sn
                r["served_plus_servo"] = bk.unnormalize(rn.reshape(1, -1), state=o.state[0].float().cpu().numpy())[0]
                r["servo_raw"] = (sn * (np.asarray(ns["actions"]["q99"][:3]) - np.asarray(ns["actions"]["q01"][:3]) + 1e-6) / 2)
            rows.append(r)
        res[tag] = rows
        del bk; torch.cuda.empty_cache()
    def err(x, f):
        x = np.asarray(x).reshape(10, 7); return float(np.abs(x[:, :3] - f["demo"][:, :3]).mean()), float(np.linalg.norm((x[:, :3] - f["demo"][:, :3]).mean(0)))
    keys = [("raas served as-is (what serving did)", lambda i: res["raas"][i]["served"]), ("raas served + servo (what training assumed)", lambda i: res["raas"][i]["served_plus_servo"]),
            ("raac served", lambda i: res["raac"][i]["served"])]
    for n, fn in keys:
        e = np.array([err(fn(i), f) for i, f in enumerate(frames)])
        print("[diag] %-45s xyz mean|a - demo| %.3f   |mean chunk bias| %.3f" % (n, e[:, 0].mean(), e[:, 1].mean()), flush=True)
    sv = np.array([np.abs(r["servo_raw"]).mean() for r in res["raas"]])
    dm = np.array([np.abs(f["demo"][:, :3]).mean() for f in frames])
    print("[diag] mean |servo| (raw action units) %.3f  vs mean |demo xyz| %.3f; servo range per frame %s" % (sv.mean(), dm.mean(), np.round(sv, 2).tolist()), flush=True)
    for i, f in enumerate(frames[:6]):
        print("[diag] ep %d t %d  demo[0,:3] %s  served %s  served+servo %s  raac %s  servo %s" % (f["ep"], f["i"], np.round(f["demo"][0, :3], 2).tolist(),
              np.round(res["raas"][i]["served"].reshape(10, 7)[0, :3], 2).tolist(), np.round(res["raas"][i]["served_plus_servo"].reshape(10, 7)[0, :3], 2).tolist(),
              np.round(res["raac"][i]["served"].reshape(10, 7)[0, :3], 2).tolist(), np.round(res["raas"][i]["servo_raw"], 2).tolist()), flush=True)


if __name__ == "__main__":
    main()
