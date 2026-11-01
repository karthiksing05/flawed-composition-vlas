"""FLAWED_COMPOSITION prototype A, stage 1m DIAGNOSTIC (section 5.23; descriptive): what the occupancy input of arm Ao is used for.
On approach rows (hold off) of K1-K6 (demos + gen_lc), each non-object entity is classed NEAR (footprint clearance to the object
<= 0.10 m) or FAR (> 0.10 m).  The object term's sampled action chunk (fixed noise) is compared before and after moving
  far   every FAR entity by 5 cm in a random xy direction (NEAR ones and the object kept)
  near  every NEAR entity likewise (rows with at least one near entity)
A grasp that reads obstacles should react to `near` and barely to `far`; a large `far` response means the term keys on geometry
that cannot constrain the grasp (in generated data every layout is unique, so the far geometry identifies the episode).
Also the share of rows with a near entity, by source (demo / generated).
CPU -> results/flawed/a1m_sens.json

    FA1M_MODELS="l_s12:Ao+lk_s12:Ao" python experiments/fa1m_sens.py
"""
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
import fa1_common as FC                                                     # noqa: E402
import fa1c_common as FK                                                    # noqa: E402
from fa1g_train import rot0_rows                                            # noqa: E402
from lifelong_ood.policy import factored_flow as FF                         # noqa: E402

N, NEAR = 3000, 0.10
MODELS = [tuple(x.split(":")) for x in os.environ.get("FA1M_MODELS", "l_s12:Ao").split("+")]
FIELDS = ("task", "demo", "step", "action", "ee_pos", "ee_rot", "grip", "pos", "yaw", "drawer", "prev_grip")


def main():
    torch.set_num_threads(4)
    D0 = dict(np.load(os.path.join(FC.DATA, "demos.npz")))
    Db0 = dict(np.load(os.path.join(FC.DATA, "demos_b.npz")))
    G = dict(np.load(os.path.join(FC.DATA, "gen_lc.npz")))
    D = {x: np.concatenate([D0[x], G[x].astype(D0[x].dtype)]) for x in FIELDS}
    Db = {x: np.concatenate([Db0[x], G[x].astype(np.float32)]) for x in ("held", "off")}
    src = np.r_[np.zeros(len(D0["action"]), int), np.ones(len(G["action"]), int)]
    n2 = json.load(open(os.path.join(ROOT, "results", "flawed", "n2_tasks.json"), encoding="utf-8"))
    lang = {t["compound"]: t["language"] for t in n2["tasks"]}
    rl = [FC.roles(lang[k]) for k in FC.KNOWN]
    oi = np.array([FC.ENT.index(rl[t][0]) for t in D["task"]])
    ti = np.array([FC.ENT.index(rl[t][1]) for t in D["task"]])
    zr, cy = FK.canon_rows(D, Db["held"], oi, ti)
    R0 = rot0_rows(D, Db["held"])
    size = np.array([FC.SIZE[e] for e in FC.ENT])
    rng = np.random.default_rng(0)
    pool = np.nonzero((D["task"] <= 5) & (Db["held"] < 0.5))[0]
    j = rng.choice(pool, N, replace=False)
    ii = np.arange(N)
    po = D["pos"][j, oi[j]]
    clr = np.linalg.norm(D["pos"][j][:, :, :2] - po[:, None, :2], axis=-1) - size[None, :, 0]
    clr[ii, oi[j]] = np.inf
    near = clr <= NEAR
    near[:, FK.TABLE] = False
    far = (clr > NEAR) & np.isfinite(clr)
    far[:, FK.TABLE] = False
    ang = rng.uniform(0, 2 * np.pi, (N, FF.N_ENT))
    sh = np.stack([np.cos(ang), np.sin(ang), np.zeros_like(ang)], -1) * 0.05
    # stage 1m: entities beyond Ap's radius OCC_R both before and after the shift (Ap must not react to them at all)
    clr2 = np.linalg.norm(D["pos"][j][:, :, :2] + sh[..., :2] - po[:, None, :2], axis=-1) - size[None, :, 0]
    strict = (clr > FF.OCC_R) & (clr2 > FF.OCC_R) & np.isfinite(clr)
    strict[:, FK.TABLE] = False
    T = lambda v, dt=torch.float32: torch.tensor(np.asarray(v), dtype=dt)   # noqa: E731

    def batch(pos):
        return dict(ee_pos=T(D["ee_pos"][j]), ee_rot=T(D["ee_rot"][j]), grip=T(D["grip"][j]), prev=T(D["prev_grip"][j]), pos=T(pos),
                    yaw=T(D["yaw"][j]), drawer=T(D["drawer"][j]), oi=T(oi[j], torch.long), ti=T(ti[j], torch.long), held=T(Db["held"][j]),
                    off=T(Db["off"][j]), zrest=T(zr[j]), cyaw=T(cy[j]), rot0=T(R0[j]))

    out = dict(rows=N, near_radius=NEAR, share_rows_with_near={s: float(near[src[j] == i].any(1).mean()) for i, s in enumerate(("demo", "gen"))},
               models={})
    x0 = torch.randn(N, 8, 7, generator=torch.Generator().manual_seed(1))
    for ck, arm in MODELS:
        m, _ = FF.FactoredFlow.load(os.path.join(FC.DATA, "ckpt", ck, f"{arm}.npz"))

        def act(pos):
            x = x0.clone()
            with torch.no_grad():
                for k in range(10):
                    t = torch.full((N,), k / 10)
                    x = x + m.velocity(batch(pos), x, t) / 10
            return x

        base = act(D["pos"][j])
        d_far = (act(D["pos"][j] + np.where(far[..., None], sh, 0.0)) - base).abs().amax((1, 2)).numpy()
        d_near = (act(D["pos"][j] + np.where(near[..., None], sh, 0.0)) - base).abs().amax((1, 2)).numpy()
        d_strict = (act(D["pos"][j] + np.where(strict[..., None], sh, 0.0)) - base).abs().amax((1, 2)).numpy()
        hn = near.any(1)
        r = dict(far_mean=float(d_far.mean()), far_p90=float(np.percentile(d_far, 90)),
                 beyond_radius_mean=float(d_strict.mean()), beyond_radius_max=float(d_strict.max()),
                 far_mean_demo=float(d_far[src[j] == 0].mean()), far_mean_gen=float(d_far[src[j] == 1].mean()),
                 near_mean=float(d_near[hn].mean()) if hn.any() else None, near_p90=float(np.percentile(d_near[hn], 90)) if hn.any() else None,
                 rows_with_near=int(hn.sum()), mean_abs_action=float(base.abs().mean()))
        out["models"][f"{ck}/{arm}"] = r
        print(f"[fa1m] sens {ck}/{arm}: {json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})}", flush=True)
    print(f"[fa1m] near share {out['share_rows_with_near']}", flush=True)
    with open(os.path.join(ROOT, "results", "flawed", "a1m_sens.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
