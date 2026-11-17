"""RAA place-step diagnosis from existing records only (docs/SCAFFOLD_RESULTS.md 73; exploratory, no new rollouts).

Inputs: results/flawed/raa/raa_{main,rx}_<tid>.json (episode records), rgp_* (same), chan_raa_*.jsonl (per-query channel: role, rel_cm
= estimate fed to the model, true_cm = simulator truth, err_cm, src).  Not logged: end-effector pose, gripper state, release step, the
policy's actions.  The end-effector displacement is recovered from the target-role channel (the target is static, so d(rel) = -d(eef)).

    python experiments/raa_diag.py            -> results/flawed/raa_diag.json (refuses to overwrite)
"""
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "flawed")
RD = os.path.join(OUT, "raa")
DEV = set(range(4, 24))


def load_eps(stem, tid):
    by = {}
    for f in (os.path.join(RD, "%s_main_%d.json" % (stem, tid)), os.path.join(RD, "%s_rx_%d.json" % (stem, tid))):
        for e in json.load(open(f, encoding="utf-8"))["tasks"][str(tid)]["episodes"]:
            by[int(e["trial"])] = e
    return by


def load_chan():
    """(text, ep) -> list of queries of the LAST run (t restarts at 0)."""
    runs = {}
    for f in sorted(glob.glob(os.path.join(RD, "chan_raa_*.jsonl"))):
        cur = {}
        for ln in open(f, encoding="utf-8"):
            r = json.loads(ln)
            k = (r["text"], int(r["ep"]))
            if r["t"] == 0:
                cur[k] = []
            cur.setdefault(k, []).append(r)
        for k, v in cur.items():
            runs[k] = v
    return runs


def pct(a, q):
    return float(np.percentile(a, q)) if len(a) else None


def main():
    outp = os.path.join(OUT, "raa_diag.json")
    if os.path.exists(outp):
        sys.exit("refusing to overwrite " + outp)
    meta = json.load(open(os.path.join(OUT, "n2_tasks.json"), encoding="utf-8"))
    tasks = [t for t in meta["tasks"] if t["layout"] == "familiar" and t["compound"][1] in "123456" ]
    chan = load_chan()
    res = dict(note="exploratory; existing rollout records only", per_task={}, notlogged=["eef pose", "gripper state / release step",
                                                                                         "actions", "simulator target position in base frame"])
    # tolerance proxy: successes' minimum 3-D / xy distance to the channel target during the hold
    succ_min = []
    rows = {}
    for t in tasks:
        tid, cp, txt = t["tid"], t["compound"], t["language"]
        ra, rg = load_eps("raa", tid), load_eps("rgp", tid)
        for k, e in ra.items():
            q = chan.get((txt, k))
            row = dict(task=cp, trial=k, success=bool(e["success"]), steps=e["steps"], bound_step=e.get("bound_step"), have_chan=q is not None)
            if q:
                t_ = np.array([x["t"] for x in q])
                role = np.array([x["role"] == "target" for x in q])
                rel = np.array([x["rel_cm"] for x in q], float)
                tru = np.array([x["true_cm"] for x in q], float)
                err = np.array([x["err_cm"] for x in q], float)
                src = [x["src"] for x in q]
                row.update(nq=len(q), last_t=int(t_[-1]), n_target_q=int(role.sum()), n_flip_back=int(np.sum(np.diff(role.astype(int)) < 0)))
                if role.any():
                    i0 = int(np.argmax(role))
                    row["t_switch"] = int(t_[i0])
                    row["switch_minus_bound"] = None if e.get("bound_step") is None else int(t_[i0] - e["bound_step"])
                    tt = tru[role]
                    d3 = np.linalg.norm(tt, axis=1)
                    dxy = np.linalg.norm(tt[:, :2], axis=1)
                    row.update(d3_switch=float(d3[0]), dxy_switch=float(dxy[0]), d3_min=float(d3.min()), dxy_min=float(dxy.min()),
                               d3_final=float(d3[-1]), dxy_final=float(dxy[-1]), z_final=float(tt[-1, 2]), final_rel=[float(v) for v in tt[-1]],
                               switch_rel=[float(v) for v in tt[0]],
                               # eef displacement over the hold = rel_switch - rel_final (target static); fraction of the needed move done
                               frac_xy=float(1 - dxy[-1] / max(dxy[0], 1e-6)),
                               move_last10q=float(np.linalg.norm(tt[-1] - tt[max(0, len(tt) - 11)])) if len(tt) > 1 else 0.0,
                               e_target_max=float(err[role].max()), e_target_med=float(np.median(err[role])),
                               src_target={s: src_i for s, src_i in zip(*np.unique(np.array(src)[role], return_counts=True))})
                    row["src_target"] = {str(k2): int(v) for k2, v in row["src_target"].items()}
                    # target-role segments; a target -> object flip is the release signature (ProprioHeld off)
                    seg = []
                    s = None
                    for j, v in enumerate(role):
                        if v and s is None:
                            s = j
                        if (not v) and s is not None:
                            seg.append((s, j - 1))
                            s = None
                    if s is not None:
                        seg.append((s, len(role) - 1))
                    row["n_target_segments"] = len(seg)
                    a, b = seg[0]
                    row["released"] = bool(b + 1 < len(role))
                    row["held_to_end"] = bool(seg[-1][1] == len(role) - 1)
                    row["t_hold_first"] = [int(t_[a]), int(t_[b])]
                    rl = tru[b]
                    row["rel_at_release"] = [float(v) for v in rl]
                    row["d_release_xy"] = float(np.linalg.norm(rl[:2]))
                    row["d_release_3d"] = float(np.linalg.norm(rl))
                    if b + 1 < len(role):
                        row["t_release"] = int(t_[b + 1])
                        # object position relative to the target at the release: obj_rel(next) - tgt_rel(last) (eef moves little in 5 steps)
                        po = tru[b + 1] - rl
                        row["placed_off"] = [float(v) for v in po]
                        row["placed_off_xy"] = float(np.linalg.norm(po[:2]))
                        row["steps_after_release"] = int(e["steps"] - t_[b + 1])
                else:
                    row["t_switch"] = None
                obj = ~role
                if obj.any():
                    row["e_obj_max"] = float(err[obj].max())
                    row["obj_q_after_bound"] = int(np.sum(obj & (t_ > (e.get("bound_step") or 10 ** 9) + 5)))
                    row["target_q_before_bound"] = int(np.sum(role & (t_ < (e.get("bound_step") or -1) - 5)))
            row["rgp_success"] = bool(rg[k]["success"]) if k in rg else None
            row["rgp_steps"] = rg[k]["steps"] if k in rg else None
            rows.setdefault(cp, []).append(row)
            if row["success"] and "dxy_min" in row:
                succ_min.append((row["dxy_min"], row["d3_min"]))
    sm = np.array(succ_min)
    TOL_XY = float(np.percentile(sm[:, 0], 90))
    res["tolerance_proxy"] = dict(rule="90th pct of successes' min xy distance (cm) to the channel target during the hold", xy_cm=TOL_XY,
                                  succ_dxy_min_p50=pct(sm[:, 0], 50), succ_dxy_min_p90=pct(sm[:, 0], 90), succ_d3_min_p90=pct(sm[:, 1], 90), n=len(sm))
    cats_all = {}
    for cp, R in rows.items():
        fails = [r for r in R if not r["success"]]
        succ = [r for r in R if r["success"]]
        cat = {"never_switched_to_target": 0, "never_approached_target(min_xy>3*tol)": 0, "partial_approach(tol<min_xy<=3*tol)": 0,
               "reached_target_xy(min_xy<=tol) but no success": 0, "no_chan_record": 0}
        sub = dict(reached_stalled_300=0, reached_ended_early=0, approach_timeout_300=0)
        for r in fails:
            if not r["have_chan"]:
                cat["no_chan_record"] += 1
            elif r["t_switch"] is None:
                cat["never_switched_to_target"] += 1
            elif r["dxy_min"] > 3 * TOL_XY:
                cat["never_approached_target(min_xy>3*tol)"] += 1
                if r["steps"] >= 300:
                    sub["approach_timeout_300"] += 1
            elif r["dxy_min"] > TOL_XY:
                cat["partial_approach(tol<min_xy<=3*tol)"] += 1
            else:
                cat["reached_target_xy(min_xy<=tol) but no success"] += 1
                sub["reached_stalled_300" if r["steps"] >= 300 else "reached_ended_early"] += 1
        fr = [r for r in fails if r.get("t_switch") is not None]
        fin = np.array([r["final_rel"] for r in fr]) if fr else np.zeros((0, 3))
        sfin = np.array([r["final_rel"] for r in succ if r.get("t_switch") is not None]) if succ else np.zeros((0, 3))
        tgt_err_all = [r["e_target_max"] for r in R if "e_target_max" in r]
        res["per_task"][cp] = dict(
            n=len(R), n_success=len(succ), n_fail=len(fails), categories=cat, sub=sub,
            fail_timeout300=sum(r["steps"] >= 300 for r in fails),
            switch_minus_bound_med=pct([r["switch_minus_bound"] for r in R if r.get("switch_minus_bound") is not None], 50),
            switch_minus_bound_range=[min([r["switch_minus_bound"] for r in R if r.get("switch_minus_bound") is not None] or [None]),
                                      max([r["switch_minus_bound"] for r in R if r.get("switch_minus_bound") is not None] or [None])],
            early_switch_eps=sum(r.get("target_q_before_bound", 0) > 0 for r in R), late_switch_eps=sum(r.get("obj_q_after_bound", 0) > 0 for r in R),
            flip_back_eps=sum(r.get("n_flip_back", 0) > 0 for r in R),
            fail_dxy_min=dict(p10=pct([r["dxy_min"] for r in fr], 10), p50=pct([r["dxy_min"] for r in fr], 50), p90=pct([r["dxy_min"] for r in fr], 90)),
            succ_dxy_min=dict(p50=pct([r["dxy_min"] for r in succ if "dxy_min" in r], 50), p90=pct([r["dxy_min"] for r in succ if "dxy_min" in r], 90)),
            fail_dxy_final_p50=pct([r["dxy_final"] for r in fr], 50), fail_frac_xy_p50=pct([r["frac_xy"] for r in fr], 50),
            fail_final_rel_mean=[float(v) for v in fin.mean(0)] if len(fin) else None, fail_final_rel_sd=[float(v) for v in fin.std(0)] if len(fin) else None,
            succ_final_rel_mean=[float(v) for v in sfin.mean(0)] if len(sfin) else None, succ_final_rel_sd=[float(v) for v in sfin.std(0)] if len(sfin) else None,
            fail_stalled_move_last10q_lt1cm=sum(r["move_last10q"] < 1.0 for r in fr),
            channel_target_err_max_p50=pct(tgt_err_all, 50), channel_target_err_max_p90=pct(tgt_err_all, 90), channel_target_err_max_max=max(tgt_err_all) if tgt_err_all else None,
            channel_target_err_gt3cm_eps=sum(v > 3 for v in tgt_err_all), channel_target_err_gt3cm_in_fail=sum(r.get("e_target_max", 0) > 3 for r in fails),
            channel_obj_err_gt5cm_in_fail=sum(r.get("e_obj_max", 0) > 5 for r in fails),
            src_target_fail={}, rgp_succ=sum(1 for r in R if r["rgp_success"]), raa_fail_rgp_succ=sum(1 for r in fails if r["rgp_success"]),
            raa_steps_succ_med=pct([r["steps"] for r in succ], 50), rgp_steps_succ_med=pct([r["rgp_steps"] for r in R if r["rgp_success"]], 50),
        )
        for r in fr:
            for s, v in r.get("src_target", {}).items():
                res["per_task"][cp]["src_target_fail"][s] = res["per_task"][cp]["src_target_fail"].get(s, 0) + v
        for k2, v in cat.items():
            cats_all[k2] = cats_all.get(k2, 0) + v
    # release-based categories (tolerance from successes' last target-role distance)
    allr = [r for R in rows.values() for r in R]
    sfin = np.array([[r["dxy_final"], r["d3_final"]] for r in allr if r["success"] and "dxy_final" in r])
    TX, T3 = float(np.percentile(sfin[:, 0], 90)), float(np.percentile(sfin[:, 1], 90))
    res["tolerance_release"] = dict(rule="90th pct over successes of the last target-role distance to target (cm)", xy=TX, d3=T3,
                                    succ_dxy_final_p50=pct(sfin[:, 0], 50), n=len(sfin))
    rc = {}
    for cp, R in rows.items():
        c = dict(n_fail=0, held_to_end_near=0, held_to_end_far=0, released_near=0, released_far=0, no_target_role=0, retried=0,
                 chan_err_gt3_any=0, chan_err_gt3_and_far=0, placed_off_xy_fail=[], placed_off_xy_succ=[], d_release_xy_fail=[], z_fail=[])
        for r in R:
            if r["success"]:
                if "placed_off_xy" in r:
                    c["placed_off_xy_succ"].append(r["placed_off_xy"])
                continue
            c["n_fail"] += 1
            if r.get("t_switch") is None:
                c["no_target_role"] += 1
                continue
            far = r["dxy_final"] > TX if r["held_to_end"] else r["d_release_xy"] > TX
            if r["held_to_end"]:
                c["held_to_end_far" if far else "held_to_end_near"] += 1
            else:
                c["released_far" if far else "released_near"] += 1
                c["placed_off_xy_fail"].append(r["placed_off_xy"])
                c["d_release_xy_fail"].append(r["d_release_xy"])
            c["retried"] += int(r["n_target_segments"] > 1)
            if r["e_target_max"] > 3:
                c["chan_err_gt3_any"] += 1
                c["chan_err_gt3_and_far"] += int(far)
        for k2 in ("placed_off_xy_fail", "placed_off_xy_succ", "d_release_xy_fail"):
            v = c.pop(k2)
            c[k2 + "_p50_p90_n"] = [pct(v, 50), pct(v, 90), len(v)]
        c.pop("z_fail")
        rc[cp] = c
    res["release_categories"] = rc
    res["categories_total"] = cats_all
    res["n_fail_total"] = sum(v["n_fail"] for v in res["per_task"].values())
    res["episodes"] = rows
    with open(outp, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print("[raa-diag]", json.dumps({k: v for k, v in res.items() if k != "episodes"}), flush=True)


if __name__ == "__main__":
    main()
