"""MT (SR 81, m3_prereg tag mt; docs/notes/mt_plan.md): MOVED-TARGET states.  DATA AND RENDER CHECK ONLY: no policy runs here.
LIBERO venv (py3.8), EGL, one GPU.  fx_tasks.py and the n2 files are imported or read, never edited.

Claim under test: "a VLA learns an object's trained position rather than the object: if the target is moved to new coordinates, the arm
still goes to the trained coordinates."

Scene: N2's LIBERO-Goal scene 1 (results/flawed/n2_bddl/K1.bddl, K2.bddl: same scene, the instruction and goal differ).
Entities that are moved: the PLATE (K1's target; the stove, K2's target, is a fixture and cannot move) and the BOWL (both tasks' object).
Demo range: LIBERO's sampler puts each object in a 2 cm x 2 cm region; n2_tasks.json training.centre / spread (50 stored sampler
states: spread .0273 m, the largest distance from the centre) is the radius of the demo range of an entity's xy.  The build also reads
the 50 official init states of LIBERO-Goal's "put the bowl on the plate" / "put the bowl on the stove" (the demos' start states) and
records their range.  Trained position ("mode") of an entity = training.centre (the mean of the stored sampler states).

Cells (cid; tid = n2 task, K1 = 0, K2 = 2):  bins by the moved entity's distance d from its trained centre
    0 K1_in    tid 0  nothing moved (sampler state; d <= demo spread by construction)    control for tid 0
    1 T_near   tid 0  PLATE moved, d in [0.04, 0.07] m  (1.5-2.6 x spread: just outside the demo range)
    2 T_far    tid 0  PLATE moved, d in [0.14, 0.30] m  (5-11 x spread; the ring of F6's displaced plate)
    3 O1_near  tid 0  BOWL moved,  d in [0.04, 0.07]
    4 O1_far   tid 0  BOWL moved,  d in [0.14, 0.30]
    5 K2_in    tid 2  nothing moved                                                      control for tid 2
    6 O2_near  tid 2  BOWL moved, near;   7 O2_far  tid 2  BOWL moved, far
Each cell: 4 smoke states then 40 main states (candidate j = 0, 1, ...; the first 4 accepted are smoke).
Seed rule (none of fx / f6o / f6o2 / rg_comp / steer / s2rg / cov_gen: 810000-990000, 14000000 ...): candidate j of cell cid draws its
start state with LIBERO's sampler on the cell's BDDL, np.random.seed = env.seed = 1050000 + 100 cid + j; a moved entity's new xy comes
from numpy default_rng(1060000 + 100 cid + j): uniform angle, uniform radius in the bin, first draw that passes the checks;
state id k = 100 cid + (accepted index a), branch seed_base = 17000000 + 1000 k.
Checks per candidate: xy inside n2's box; at least 0.75 x (r + r') from every other movable and from each fixture footprint and from the
stove-front region (r: fx_tasks.MOVR / FIXR; the trained plate sits at 0.141 from the bowl, 0.75 x (0.10 + 0.06) = 0.12); the state
settles (10 dummy steps) with nothing else moving more than 2 cm and the moved entity within 1e-4 of its target; the render check: every
movable has agentview pixels >= .5 x its median in the control cell, the moved entity >= .4 x.

    python experiments/mt_tasks.py
-> results/flawed/mt/{states/n2_goal_0.npy [220, D], n2_goal_2.npy [132, D], mt_tasks.json, mt_br_<cell>_<smoke|main>.json, check_<cell>.png}
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT]
import fx_tasks as FT                                                       # noqa: E402

OUT = os.path.join(ROOT, "results", "flawed", "mt")
ST = os.path.join(OUT, "states")
META = os.path.join(OUT, "mt_tasks.json")
N2F = os.path.join(ROOT, "results", "flawed", "n2_tasks.json")
SEED_FAM, SEED_DISP, BR_BASE = 1050000, 1060000, 17000000
N_SMOKE, N_MAIN = 4, 40
NEAR, FAR = (0.04, 0.07), (0.14, 0.30)
B1, PL = "akita_black_bowl_1", "plate_1"
# cid, name, tid, moved entity (None = control), bin range, task key
CELLS = [
    (0, "K1_in", 0, None, None, "K1"), (1, "T_near", 0, PL, NEAR, "K1"), (2, "T_far", 0, PL, FAR, "K1"),
    (3, "O1_near", 0, B1, NEAR, "K1"), (4, "O1_far", 0, B1, FAR, "K1"),
    (5, "K2_in", 2, None, None, "K2"), (6, "O2_near", 2, B1, NEAR, "K2"), (7, "O2_far", 2, B1, FAR, "K2"),
]
PER = N_SMOKE + N_MAIN


def row0(cid):
    return (cid if cid < 5 else cid - 5) * PER                              # row of a cell's first state in its tid's file


def clear2(p, ent, P, mov4, lo, hi, fixp, sf):
    r = FT.MOVR[ent]
    if np.any(p < lo) or np.any(p > hi):
        return False
    for n in mov4:
        if n != ent and np.linalg.norm(p - P[n]) < 0.75 * (r + FT.MOVR[n]):
            return False
    for f, fn in enumerate(FT.FIXR):
        if np.linalg.norm(p - fixp[f]) < 0.75 * (r + FT.FIXR[fn]):
            return False
    cx, cy = np.clip(p[0], sf[0], sf[2]), np.clip(p[1], sf[1], sf[3])
    return float(np.hypot(p[0] - cx, p[1] - cy)) >= 0.75 * r


def official_range(mov4, centre):
    """The 50 official init states of LIBERO-Goal's K1 / K2 tasks (the demos' start states): range of the bowl's and plate's xy."""
    out = {}
    try:
        from libero.libero import benchmark
        bd = benchmark.get_benchmark_dict()["libero_goal"]()
        for lang in ("put the bowl on the plate", "put the bowl on the stove"):
            ti = [i for i, t in enumerate(bd.get_task_names()) if t.replace("_", " ") == lang]
            if not ti:
                out[lang] = "task not found"
                continue
            t = bd.get_task(ti[0])
            S = bd.get_task_init_states(ti[0])
            env = FT.make_env(os.path.join(FT.LIB, t.problem_folder, t.bddl_file))
            env.seed(0)
            rows = {n: [] for n in (B1, PL)}
            for s in S:
                FT.load(env, np.asarray(s))
                P = FT.pos(env, [B1, PL])
                for n in rows:
                    rows[n].append(P[n][:2].tolist())
            env.close()
            d = {}
            for n, r in rows.items():
                r = np.array(r)
                c = np.array(centre[mov4.index(n)])
                d[n] = dict(n=int(len(r)), mean=r.mean(0).tolist(), xmin=r[:, 0].min().item(), xmax=r[:, 0].max().item(),
                            ymin=r[:, 1].min().item(), ymax=r[:, 1].max().item(), max_dist_from_n2_centre=float(np.linalg.norm(r - c, axis=1).max()))
            out[lang] = d
    except Exception as ex:                                                 # noqa: BLE001
        out["error"] = repr(ex)[:300]
    return out


def main():
    import imageio
    for p in (META, ST):
        assert not os.path.exists(p), "exists: " + p
    os.makedirs(ST)
    n2 = json.load(open(N2F, encoding="utf-8"))
    mov4 = n2["movable"]
    centre = np.array(n2["training"]["centre"])
    spread = float(n2["training"]["spread"])
    c = n2["constraints"]
    lo, hi = np.array(c["box"][0]) - FT.BOX_EXTRA, np.array(c["box"][1]) + FT.BOX_EXTRA
    fixp, sf = np.array(c["fixture_xy"]), c["stove_front_world"]
    off = official_range(mov4, centre)
    print("[mt] official init-state range:", json.dumps(off)[:600], flush=True)
    envs = {k: FT.make_env(os.path.join(ROOT, "results", "flawed", "n2_bddl", k + ".bddl")) for k in ("K1", "K2")}
    for e in envs.values():
        e.seed(0)
    pxmed, accepted, rejected = {}, {}, {}
    files = {0: [None] * (5 * PER), 2: [None] * (3 * PER)}
    meta_states = {}
    for cid, name, tid, ent, rng_bin, key in CELLS:
        env = envs[key]
        acc, rej = [], []
        j = 0
        while len(acc) < PER:
            assert j < 100, (name, "too many rejects", rej[:5])
            seed = SEED_FAM + 100 * cid + j
            base = FT.sampled(env, seed)
            cand = dict(cell=name, j=j, seed=seed)
            j += 1
            if ent is None:
                s, p = base, None
            else:
                _, q0, q1 = FT.settle(env, base, mov4, steps=10)
                dz = np.array([q1[n][2] - q0[n][2] for n in mov4])
                P0 = FT.pos(env, mov4)
                FT.load(env, base)
                P0 = {n: FT.pos(env, mov4)[n][:2] for n in mov4}
                rng = np.random.default_rng(SEED_DISP + 100 * cid + (j - 1))
                ci = centre[mov4.index(ent)]
                s, p, tries = None, None, 0
                for _ in range(20000):
                    r = rng.uniform(*rng_bin)
                    th = rng.uniform(0, 2 * np.pi)
                    q = ci + r * np.array([np.cos(th), np.sin(th)])
                    if not clear2(q, ent, P0, mov4, lo, hi, fixp, sf):
                        continue
                    tries += 1
                    if tries > 60:
                        break
                    sn = FT.edit(env, base, xyz={ent: q})
                    _, a0, a1 = FT.settle(env, sn, mov4, steps=10)
                    others = [n for n in mov4 if n != ent]
                    d = max(FT.xy_moved(a0, a1, mov4),
                            float(np.abs(np.array([a1[n][2] - a0[n][2] for n in mov4]) - dz).max()))
                    if d <= FT.SETTLE_MAX and np.allclose(a0[ent][:2], q, atol=1e-4):
                        s, p = sn, q
                        break
                if s is None:
                    rej.append([cand["j"], "no spot"])
                    continue
            # render check
            img = FT.view(env, s)
            px = {n: FT.pixels(env, FT.subtree_geoms(env, n), "agentview") for n in mov4}
            if name == "K1_in" or name == "K2_in":
                pass
            elif key in pxmed:
                bad = [n for n in mov4 if px[n] < (0.4 if n == ent else 0.5) * pxmed[key][n]]
                if bad:
                    rej.append([cand["j"], "render: low pixels " + str({n: px[n] for n in bad})])
                    continue
            FT.load(env, s)
            P = FT.pos(env, mov4)
            dist = {n: float(np.linalg.norm(P[n][:2] - centre[mov4.index(n)])) for n in (B1, PL)}
            if ent is None:
                assert max(dist.values()) <= spread + 0.01, (name, dist)    # sampler state: inside the demo range (table offset slack)
            cand.update(state=s, img=img, px=px, dist=dist)
            acc.append(cand)
        if ent is None:
            pxmed[key] = {n: float(np.median([a["px"][n] for a in acc])) for n in mov4}
        accepted[name], rejected[name] = acc, rej
        print("[mt] cell %s: %d accepted, %d rejected; dist plate/bowl %s" % (
            name, len(acc), len(rej), [(round(a["dist"][PL], 3), round(a["dist"][B1], 3)) for a in acc[:3]]), flush=True)
        for a, cand in enumerate(acc):
            files[tid][row0(cid) + a] = cand["state"]
        cell_meta = []
        for a, cand in enumerate(acc):
            cell_meta.append(dict(a=a, k=100 * cid + a, j=cand["j"], seed=cand["seed"], row=row0(cid) + a, smoke=a < N_SMOKE,
                                  dist_plate=cand["dist"][PL], dist_bowl=cand["dist"][B1], px=cand["px"]))
        meta_states[name] = cell_meta
        imageio.imwrite(os.path.join(OUT, "check_%s.png" % name), np.concatenate([np.concatenate(a_["img"], axis=0) for a_ in acc[:4]], axis=1))
        for part, lo_a, hi_a in (("smoke", 0, N_SMOKE), ("main", N_SMOKE, PER)):
            br = [dict(id="mt%d" % (100 * cid + a), trial=row0(cid) + a, seed_base=BR_BASE + 1000 * (100 * cid + a), at=0, variant="n2")
                  for a in range(lo_a, hi_a)]
            with open(os.path.join(OUT, "mt_br_%s_%s.json" % (name, part)), "w", encoding="utf-8") as fh:
                json.dump(br, fh, indent=1)
    for tid, rows in files.items():
        assert all(r is not None for r in rows)
        np.save(os.path.join(ST, "n2_goal_%d.npy" % tid), np.stack(rows))
    for e in envs.values():
        e.close()
    meta = dict(cells=[dict(cid=c_[0], name=c_[1], tid=c_[2], moved=c_[3], bin=c_[4], key=c_[5], row0=row0(c_[0])) for c_ in CELLS],
                trained=dict(plate=centre[mov4.index(PL)].tolist(), bowl=centre[mov4.index(B1)].tolist(), spread=spread),
                official_init=off, seeds=dict(sampler=SEED_FAM, disp=SEED_DISP, branch=BR_BASE), px_median=pxmed,
                n_smoke=N_SMOKE, n_main=N_MAIN, states=meta_states, rejected={k: v for k, v in rejected.items()})
    with open(META, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1)
    print("[mt] done", flush=True)


if __name__ == "__main__":
    main()
