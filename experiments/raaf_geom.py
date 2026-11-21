"""RAA-F Step 0 (docs/notes/raaf_plan.md 2.3, 2.4, 5; checks S0-1, S0-2, S0-3): the per-demo feasible set F_d of goal shifts, the physics
validation, the delta draws with the clip check, and the dose of Route A.  LIBERO venv (py3.8), CPU (MuJoCo physics only, no rendering).  sbatch only.

    python experiments/raaf_geom.py --prep P/prep.pkl --out-dir D        (D must not hold geom.json / raaf_delta.json yet)

Reads the pickle of experiments/raaf_prep.py.  For each K1-K6 demo with a hold and a release: the carried object's position p_d at the demo's last
frame (success fired there) and the target's pose are read from the demo's own recorded state in the task env.  F_d = { delta in R^2 on a 1 mm
grid, |dx|, |dy| <= 15 cm : the success predicate holds for p_d + (delta, 0) with the predicate's half-sizes shrunk by the margin }, evaluated with
LIBERO's own formulas: movable targets (plate, bowl): |obj_xy - tgt_xy| < 0.03 - margin (ObjectState.check_ontop); site targets: SiteObject.under
(size_xy - margin, the z test unchanged), the vectorised copy of which is asserted equal to the real `under`.  Then delta ~ Uniform(F_d minus
{|delta| < 1 cm}) per copy (M = 2), seed np.random.default_rng(92_000_000 + 1000 tid + 10 demo + copy), redrawn up to 20 times when a relabelled
action leaves [-.9375, .9375] on xyz.  Physics (S-A2): 10 demos per class, the object teleported to p_d + delta at the last frame, 50 dummy steps,
the task's success predicate must still hold (the same check at delta = 0 is the control); more than 10% failures raises that class's margin to 2 cm.
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT]
RV_ROOT = "/coc/flash10/ksingara3/robot/data/fa1/rv"
NORM = "/coc/flash10/ksingara3/robot/checkpoints_l90c/assets/pi05_g8v_raac/lol/libero_g8v_raac/norm_stats.json"
CLASS = {"K1": "plate", "K6": "bowl", "K2": "stove", "K3": "cabinet", "K4": "cabinet", "K5": "rack"}
SPAN, MIN_NORM, M_COPY, N_DRAW = 150, 10, 2, 20            # mm, mm
K, MAXA = 2.851209548740752, 0.9375
DISK = 0.03


def grid():
    g = np.arange(-SPAN, SPAN + 1)
    dx, dy = np.meshgrid(g, g, indexing="ij")
    return dx.ravel(), dy.ravel()                            # mm


def feasible(sc, margin):
    """sc: dict(kind, p [3], t [3] | site pos / mat / size) -> boolean mask over the grid (True = predicate holds with the margin)."""
    dx, dy = grid()
    p = sc["p"][None, :] + np.stack([dx / 1000.0, dy / 1000.0, np.zeros_like(dx, dtype=float)], 1)
    if sc["kind"] == "disk":
        return np.linalg.norm(p[:, :2] - sc["t"][None, :2], axis=1) < DISK - margin
    v = (sc["mat"] @ (p - sc["site"][None, :]).T).T
    s = sc["size"]
    return (s[2] - 0.005 < v[:, 2]) & (v[:, 2] < s[2] + 0.10) & np.all(np.abs(v[:, :2]) < (s[:2] - margin)[None, :], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--nphys", type=int, default=10)
    a = ap.parse_args()
    geom_p, delta_p = os.path.join(a.out_dir, "geom.json"), os.path.join(a.out_dir, "raaf_delta.json")
    assert not os.path.exists(geom_p) and not os.path.exists(delta_p), a.out_dir
    os.makedirs(a.out_dir, exist_ok=True)
    import fa1_common as FC
    import sc_rv as RV
    P = pickle.load(open(a.prep, "rb"))
    demos = P["demos"]
    nst = json.load(open(NORM, encoding="utf-8"))["norm_stats"]
    aq01, aq99 = np.asarray(nst["actions"]["q01"][:3]), np.asarray(nst["actions"]["q99"][:3])
    sq01, sq99 = np.asarray(nst["state"]["q01"][:3]), np.asarray(nst["state"]["q99"][:3])

    # ---- per task: scene read at each demo's last frame
    envs, inf = {}, {}
    for c in sorted({d["task"] for d in demos}):
        env = FC.make_env(RV._bddl(c))
        env.seed(0)
        env.reset()
        envs[c] = env
    n_succ, n_tot = 0, 0
    for d in demos:
        c, env = d["task"], envs[d["task"]]
        raw = np.load(RV.raw_path(RV_ROOT, c))
        st = raw["states_%d" % d["demo"]]
        # the demo's settled last frame: the raw final state if the task's success predicate holds there, else the last kept row's
        d["ok_src"] = None
        for src, row in (("raw_last", st[-1]), ("kept_last", st[int(d["kept"][-1])])):
            env.env.timestep, env.env.done = 0, False
            env.set_init_state(row)
            if bool(env.check_success()):
                d["ok_src"], d["st_last"] = src, np.asarray(row).copy()
                break
        if d["ok_src"] is None:
            d["st_last"] = np.asarray(st[-1]).copy()
            env.env.timestep, env.env.done = 0, False
            env.set_init_state(d["st_last"])
        inner, sim = env.env, env.env.sim
        p = np.asarray(sim.data.body_xpos[inner.obj_body_id[d["g"]]], np.float64).copy()
        sc = dict(p=p)
        if d["rslot"] in FC.MOVABLE:
            sc.update(kind="disk", t=np.asarray(sim.data.body_xpos[inner.obj_body_id[d["rslot"]]], np.float64).copy())
        else:
            so = inner.object_sites_dict[d["rslot"]]
            sc.update(kind="site", site=np.asarray(sim.data.get_site_xpos(d["rslot"]), np.float64).copy(),
                      mat=np.asarray(sim.data.get_site_xmat(d["rslot"]), np.float64).reshape(3, 3).copy(),
                      size=np.asarray(so.size, np.float64).copy())
            # the vectorised copy must equal LIBERO's own SiteObject.under (on the real point and on 200 random shifted points)
            rng = np.random.default_rng(0)
            for _ in range(200):
                q = p + np.r_[rng.uniform(-0.15, 0.15, 2), 0.0]
                real = bool(so.under(sc["site"], sc["mat"], q))
                v = sc["mat"] @ (q - sc["site"])
                mine = bool((sc["size"][2] - 0.005 < v[2] < sc["size"][2] + 0.10) and np.all(np.abs(v[:2]) < sc["size"][:2]))
                assert real == mine, (c, d["demo"], q)
        n_tot += 1
        n_succ += int(d["ok_src"] is not None)
        d["sc"] = sc
    print("[raaf_geom] scenes read for %d demos; success predicate true at the last frame in %d" % (n_tot, n_succ), flush=True)
    for d in demos:
        d["last_ok"] = None

    # ---- class margins, F_d, physics validation (S-A2)
    margin = {k: 0.010 for k in set(CLASS.values())}
    dx, dy = grid()
    far = np.hypot(dx, dy) >= MIN_NORM

    def fsets(m):
        for d in demos:
            f = feasible(d["sc"], m[CLASS[d["task"]]]) & far & (d["ok_src"] is not None)
            d["F"] = np.nonzero(f)[0]
            d["F0"] = bool(feasible(d["sc"], 0.0)[(SPAN) * (2 * SPAN + 1) + SPAN])        # delta = 0 under the unshrunk predicate (sanity)

    def physics(cls, m):
        """teleport test on the first nphys eligible demos of the class (cabinet: K3 and K4 alternately)."""
        pool = [d for d in demos if CLASS[d["task"]] == cls and len(d["F"]) > 0]
        if cls == "cabinet":
            k3 = [d for d in pool if d["task"] == "K3"]
            k4 = [d for d in pool if d["task"] == "K4"]
            pool = [x for pair in zip(k3, k4) for x in pair]
        pool = pool[:a.nphys]
        res = []
        for d in pool:
            rng = np.random.default_rng(92_900_000 + 1000 * d["tid"] + d["demo"])
            idx = int(d["F"][rng.integers(len(d["F"]))])
            row = dict(task=d["task"], demo=d["demo"], dxy_mm=[int(dx[idx]), int(dy[idx])])
            for tag, shift in (("shift", np.array([dx[idx] / 1000.0, dy[idx] / 1000.0, 0.0])), ("control", np.zeros(3))):
                env = envs[d["task"]]
                raw = np.load(RV.raw_path(RV_ROOT, d["task"]))
                env.env.timestep, env.env.done = 0, False
                env.set_init_state(d["st_last"])
                inner, sim = env.env, env.env.sim
                obj = inner.get_object(d["g"])
                j = obj.joints[0]
                q = np.asarray(sim.data.get_joint_qpos(j), np.float64).copy()
                q[:3] += shift
                sim.data.set_joint_qpos(j, q)
                sim.data.set_joint_qvel(j, np.zeros(6))
                sim.forward()
                for _ in range(50):
                    env.step(FC.DUMMY)
                row[tag] = bool(env.check_success())
            res.append(row)
        return res

    phys, margin_log = {}, []
    fsets(margin)
    for cls in sorted(set(CLASS.values())):
        r = physics(cls, margin)
        nf = sum(1 for x in r if not x["shift"])
        margin_log.append(dict(cls=cls, margin_cm=100 * margin[cls], n=len(r), fail=nf, control_fail=sum(1 for x in r if not x["control"])))
        if r and nf > 0.10 * len(r):
            margin[cls] = 0.020
            fsets(margin)
            r2 = physics(cls, margin)
            margin_log.append(dict(cls=cls, margin_cm=100 * margin[cls], n=len(r2), fail=sum(1 for x in r2 if not x["shift"]),
                                   control_fail=sum(1 for x in r2 if not x["control"]), raised=True))
            r = r2
        phys[cls] = r
        print("[raaf_geom] physics %s: %s" % (cls, json.dumps(margin_log[-1])), flush=True)

    # ---- draws with the clip check (S-A1)
    copies, dropped, redraws = [], [], 0
    for d in demos:
        d["copies"] = []
        if len(d["F"]) == 0:
            continue
        seg = d["actions"][d["h0"]:d["e"], :3]
        if len(seg) == 0:
            continue
        for m in range(M_COPY):
            rng = np.random.default_rng(92_000_000 + 1000 * d["tid"] + 10 * d["demo"] + m)
            ok = False
            for tr in range(N_DRAW):
                idx = int(d["F"][rng.integers(len(d["F"]))])
                dl = np.array([dx[idx] / 1000.0, dy[idx] / 1000.0, 0.0])
                if np.all(np.abs(seg + K * dl[None, :]) <= MAXA + 1e-12):
                    ok = True
                    break
                redraws += 1
            if ok:
                d["copies"].append(dict(copy=m, delta=dl.tolist(), tries=tr + 1))
            else:
                dropped.append(dict(task=d["task"], demo=d["demo"], copy=m))
    n_try = sum(1 for d in demos if len(d["F"]) > 0 for _ in range(M_COPY))

    # ---- S0-1 eligibility and S0-3 dose
    per = {}
    for c in sorted({d["task"] for d in demos}):
        v = [d for d in demos if d["task"] == c]
        held = sum(int(d["H"].sum()) for d in v)
        elig_rows = sum(d["e"] - d["h0"] for d in v if d["copies"])
        per[c] = dict(demos=len(v), demos_with_F=sum(1 for d in v if len(d["F"]) > 0), demos_with_copy=sum(1 for d in v if d["copies"]),
                      held_rows=held, rows_with_copy=elig_rows, frac_held_rows_eligible=elig_rows / max(held, 1),
                      F_size_median_cells=float(np.median([len(d["F"]) for d in v])),
                      delta_norm_cm_median=float(np.median([100 * np.hypot(*cp["delta"][:2]) for d in v for cp in d["copies"]] or [np.nan])),
                      delta_norm_cm_p90=float(np.percentile([100 * np.hypot(*cp["delta"][:2]) for d in v for cp in d["copies"]] or [np.nan], 90)))
    held_all = sum(int(d["H"].sum()) for d in demos)
    elig_all = sum(d["e"] - d["h0"] for d in demos if d["copies"])
    # variance share of the K delta term among the held-phase xyz labels (originals: term 0; each accepted copy adds its own rows)
    lab_o, lab_c, term_c, tok_out, tok_n, act_out_any, act_out_dim, act_n, base_out_any, base_out_dim, base_n = [], [], [], 0, 0, 0, np.zeros(3), 0, 0, np.zeros(3), 0
    chg = []
    for d in demos:
        Ho = d["H"]
        lab_o.append(d["actions"][Ho, :3])
        seg = slice(d["h0"], d["e"])
        for cp in d["copies"]:
            dl = np.asarray(cp["delta"])
            ap_ = d["actions"][seg, :3] + K * dl[None, :]
            lab_c.append(ap_)
            term_c.append(np.tile(K * dl[None, :], (len(ap_), 1)))
            chg.append(K * float(np.hypot(dl[0], dl[1])))
            out = (ap_ < aq01[None, :]) | (ap_ > aq99[None, :])
            act_out_any += int(out.any(1).sum())
            act_out_dim += out.sum(0)
            act_n += len(ap_)
            o0 = d["actions"][seg, :3]
            out0 = (o0 < aq01[None, :]) | (o0 > aq99[None, :])
            base_out_any += int(out0.any(1).sum())
            base_out_dim += out0.sum(0)
            base_n += len(o0)
            tk = d["rel"][seg] + dl[None, :]
            tout = (tk < sq01[None, :]) | (tk > sq99[None, :])
            tok_out += int(tout.any(1).sum())
            tok_n += len(tk)
    Lo = np.concatenate(lab_o) if lab_o else np.zeros((0, 3))
    Lc = np.concatenate(lab_c) if lab_c else np.zeros((0, 3))
    Tc = np.concatenate(term_c) if term_c else np.zeros((0, 3))
    pool_lab = np.concatenate([Lo, Lc])
    pool_term = np.concatenate([np.zeros_like(Lo), Tc])
    share_pool = float(pool_term.var(0).sum() / pool_lab.var(0).sum())
    share_copies = float(Tc.var(0).sum() / Lc.var(0).sum()) if len(Lc) else float("nan")
    n_cp = sum(len(d["copies"]) for d in demos)
    out = dict(
        n_demos=len(demos), held_rows_all=held_all, rows_with_copy=elig_all, frac_held_rows_eligible=elig_all / max(held_all, 1),
        per_task=per, margin_cm_by_class={k: 100 * v for k, v in margin.items()}, margin_log=margin_log, physics=phys,
        scenes_success_at_last_frame=[n_succ, n_tot],
        copies=n_cp, copy_attempts=n_try, copies_dropped_by_clip=len(dropped), frac_dropped_by_clip=len(dropped) / max(n_try, 1), redraws=redraws, dropped=dropped[:50],
        dose=dict(var_share_pooled_held_labels=share_pool, var_share_copies_only=share_copies,
                  median_label_change_action_units=float(np.median(chg)) if chg else float("nan"),
                  p10_label_change=float(np.percentile(chg, 10)) if chg else float("nan"), p90_label_change=float(np.percentile(chg, 90)) if chg else float("nan"),
                  relabelled_rows=int(len(Lc)), original_held_rows=int(len(Lo))),
        quantile_fractions=dict(relabelled_rows=act_n, actions_outside_q01_q99_any_dim=act_out_any / max(act_n, 1),
                                actions_outside_per_dim=(act_out_dim / max(act_n, 1)).tolist(),
                                originals_same_rows_outside_any_dim=base_out_any / max(base_n, 1),
                                originals_same_rows_outside_per_dim=(base_out_dim / max(base_n, 1)).tolist(),
                                state_token_outside_q01_q99_any_dim=tok_out / max(tok_n, 1)),
        kill=dict(k1_frac_below_30pct=bool(elig_all / max(held_all, 1) < 0.30), k1_var_share_below_0p5pct=bool(share_pool < 0.005),
                  k3_dropped_above_5pct=bool(len(dropped) / max(n_try, 1) > 0.05),
                  stop_quantile_above_2pct=bool(act_out_any / max(act_n, 1) > 0.02)),
        k=K, max_action=MAXA, seeds="default_rng(92_000_000 + 1000 tid + 10 demo + copy), tid = K index - 1",
        grid=dict(span_mm=SPAN, min_norm_mm=MIN_NORM, copies_per_demo=M_COPY, max_draws=N_DRAW))
    with open(geom_p, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    delta = [dict(task=d["task"], demo=d["demo"], h0=d["h0"], e=d["e"], r=d["r"], copies=d["copies"], F_cells=int(len(d["F"])),
                  cls=CLASS[d["task"]], margin_cm=100 * margin[CLASS[d["task"]]]) for d in demos]
    with open(delta_p, "w", encoding="utf-8") as fh:
        json.dump(dict(k=K, max_action=MAXA, margin_cm_by_class={k: 100 * v for k, v in margin.items()}, demos=delta), fh)
    print("[raaf_geom] " + json.dumps({k: out[k] for k in ("n_demos", "held_rows_all", "rows_with_copy", "frac_held_rows_eligible", "copies", "copies_dropped_by_clip",
                                                       "frac_dropped_by_clip", "dose", "quantile_fractions", "kill")}, indent=1), flush=True)
    for c, v in per.items():
        print("[raaf_geom] %s %s" % (c, json.dumps(v)), flush=True)


if __name__ == "__main__":
    main()
