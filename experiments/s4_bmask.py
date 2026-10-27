"""PLAN_SOLVE S4 (docs/PLAN_S4.md): the deployable binders' masks on every S0 TRAINING frame (the robust adapter's 'actual binder'
mask source).  For each frame of results/bind/s0/data/<tid>.npz: the policy's agentview tokens (embed_prefix, the cell's prompt) ->
s2_serve.Binder(kind).masks for the cell's object / goal phrase (results/bind/s4/phrases.json), kind in {s1b, s1c}.
openpi venv, GPU.  -> results/bind/s4/bmask/<tid>.npz: {kind}_mo, {kind}_mg [n, 256], {kind}_conf [n, 2, 6]."""
import json, os, sys, time
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "experiments"), "/coc/flash10/ksingara3/robot/openpi/src"]


def main():
    import torch
    from lifelong_ood.policy.backbone import PolicyBackbone
    import s2_serve as S2
    bk = PolicyBackbone(os.environ.get("BACKBONE_CONFIG") or "pi05_l90c", checkpoint_dir=os.environ.get("BACKBONE_CKPT") or None,
                        device="cuda")
    m = bk.model
    binders = {k: S2.Binder(bk, k) for k in ("s1b", "s1c")}
    ph = json.load(open(os.path.join(ROOT, "results/bind/s4/phrases.json")))
    od = os.path.join(ROOT, "results/bind/s4/bmask")
    os.makedirs(od, exist_ok=True)
    t0 = time.time()
    for tid, p in sorted(ph.items()):
        if p["obj_phrase"] is None:          # parse does not align (2 cells): no binder masks
            continue
        out_f = os.path.join(od, tid.replace(":", "_") + ".npz")
        if os.path.exists(out_f):
            continue
        z = np.load(os.path.join(ROOT, p["file"]), allow_pickle=True)
        n = len(z["frames"])
        res = {f"{k}_{r}": np.zeros((n, 256), np.float32) for k in binders for r in ("mo", "mg")}
        res.update({f"{k}_conf": np.zeros((n, 2, 6), np.float32) for k in binders})
        for g0 in range(0, n, 16):
            ods = [{"observation/image": z["img"][i], "observation/wrist_image": z["wimg"][i],
                    "observation/state": np.asarray(z["state8"][i], np.float32), "prompt": str(z["prompt"])} for i in range(g0, min(n, g0 + 16))]
            with torch.no_grad():
                images, img_masks, lt, lm, _ = m._preprocess_observation(bk.observations(ods), train=False)
                embs, _, _ = m.embed_prefix(images, img_masks, lt, lm)
            for j in range(embs.shape[0]):
                i = g0 + j
                for k, b in binders.items():
                    mo, io = b.masks(z["img"][i], embs[j, :256], "object", p["obj_phrase"])
                    mg, ig = b.masks(z["img"][i], embs[j, :256], "goal", p["goal_phrase"])
                    res[f"{k}_mo"][i], res[f"{k}_mg"][i], res[f"{k}_conf"][i] = mo, mg, [io["conf"], ig["conf"]]
        np.savez(out_f + ".tmp.npz", **res)
        os.replace(out_f + ".tmp.npz", out_f)
        print(f"[s4 bmask] {tid}: {n} frames ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
