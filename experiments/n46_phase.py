"""N46: why do N4 (bottle -> plate) and N6 (bottle -> bowl) resist every front-end arm (SR 46-49)?  Descriptive per-episode
breakdown of EXISTING rollouts only (no simulation, no policy).  docs/SCAFFOLD_RESULTS.md 51.  LIBERO venv or any numpy venv, CPU.

    python experiments/n46_phase.py --out results/flawed/n46_phase.json        (refuses to overwrite)

Sources (all read-only; each used only if both <stem>.json and <stem>_track.npz exist):
    main  (states 4-23)   results/flawed/pp_atlas/{pi05,groot,pi0,xvla,smolvla}_<tid>     arm I
                          results/flawed/pp_atlas/servo_s10_<tid>, pmin_s1{0,1,2}_<tid>    references (oracle state)
                          results/flawed/tv_frozen/{pi05,groot}_{tv,tvp,w}_<tid>
    fresh (trials 24-43)  results/flawed/pp_atlas_fresh/{pi05,groot,pi0,xvla,smolvla}_<tid>  arm I
                          results/flawed/tv_fresh/{pi05_tv,pi05_tvp,groot_tv}_<tid>; results/flawed/tv_attr/pi05_{p,wp}_<tid>
    fresh2 (trials 24-83) results/flawed/tv_g/groot_{I,tv,w}_<tid>
Tasks (familiar): N4 22, N6 26 (the question); N1 16, N5 24 (same target / same object controls); K1 0, K4 6, K6 10 (partners).

Per episode: pp_atlas.classify's stage and subtype (unchanged), mapped to a phase:
    GRASP other_object -> grasp_wrong_object; GRASP none -> grasp_miss; HANDOFF / TRANSPORT -> transport;
    RELEASE away -> release_away; RELEASE never_let_go -> never_let_go; PHYSICS -> physics (seated within 3 cm, check failed).
plus geometry from the oracle track (instrument only): lift (max object z rise), object at the last held step against the target, the
cabinet top (K4's destination) and the rack (K5's); progress toward the target and toward the cabinet over the hold; the entity nearest
the object at the last held step and at the end (horizontal); final tilt; oracle hold segments; open commands while held; and, for the
front-end arms, the detector's first onset against the oracle hold and the per-episode paint IoU / active leak (from <stem>_tv.json).

PREDICTIONS (written before this script ran; my record on this project is poor):
  P1 GR00T N4 / N6: grasp succeeds (grasp phases <= .10); failures are post-grasp; the bottle is lifted high (median lift > .15 m, a
     K4 cabinet-top carry) and released away from the target; under I and tv alike (the views do not touch this phase).
  P2 pi0.5 N4 / N6: failures mostly post-grasp too; N6 failures end near the bowl (within .10 m) more often than GR00T's, with tipping
     (physics or release_away with final tilt > 45 deg) the largest single class; N4 failures are release_away short of the plate.
  P3 Front end: on N4 / N6 the detector fires in >= .95 of episodes and paint IoU >= .95, no worse than on N1 / N5, so the front end
     is not where N4 / N6 fail.
  P4 References: servo and P_min succeed on N4 / N6 main states at >= .75 (task physically feasible from these states).
  P5 Cabinet pull: among GR00T N4 / N6 failures, the object at the last held step is nearer the cabinet top than the target in
     >= .30 of episodes; for pi0.5 in < .30.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT]
import pp_atlas as PA                                                       # noqa: E402

OUT = PA.OUT
ENT = ["akita_black_bowl_1", "cream_cheese_1", "wine_bottle_1", "plate_1", "flat_stove_1_cook_region", "wooden_cabinet_1_top_side",
       "wine_rack_1_top_region", "wooden_cabinet_1_top_region", "main_table_stove_front_region"]   # fa1_common.ENT (track 'final' rows)
SHORT = dict(akita_black_bowl_1="bowl", cream_cheese_1="cheese", wine_bottle_1="bottle", plate_1="plate", flat_stove_1_cook_region="stove",
             wooden_cabinet_1_top_side="cab_top", wine_rack_1_top_region="rack", wooden_cabinet_1_top_region="drawer",
             main_table_stove_front_region="stove_front")
CAB, RACK = ENT.index("wooden_cabinet_1_top_side"), ENT.index("wine_rack_1_top_region")
COMPS = ("N4", "N6", "N1", "N5", "K1", "K4", "K6")
PHASE = {("GRASP", "other_object"): "grasp_wrong_object", ("GRASP", "none"): "grasp_miss", ("HANDOFF", ""): "transport",
         ("TRANSPORT", ""): "transport", ("RELEASE", "away"): "release_away", ("RELEASE", "never_let_go"): "never_let_go",
         ("PHYSICS", ""): "physics", ("success", ""): "success"}
PHASES = ("success", "grasp_wrong_object", "grasp_miss", "transport", "release_away", "never_let_go", "physics")
FL = OUT


def sources():
    S = []
    for m in ("pi05", "groot", "pi0", "xvla", "smolvla"):
        S.append(("main", m, "I", os.path.join(FL, "pp_atlas"), [m + "_{tid}"]))
    S.append(("main", "servo", "ref", os.path.join(FL, "pp_atlas"), ["servo_s10_{tid}"]))
    S.append(("main", "pmin", "ref", os.path.join(FL, "pp_atlas"), ["pmin_s10_{tid}", "pmin_s11_{tid}", "pmin_s12_{tid}"]))
    for m in ("pi05", "groot"):
        for arm in ("tv", "tvp", "w"):
            S.append(("main", m, arm, os.path.join(FL, "tv_frozen"), ["%s_%s_{tid}" % (m, arm)]))
    for m in ("pi05", "groot", "pi0", "xvla", "smolvla"):
        S.append(("fresh", m, "I", os.path.join(FL, "pp_atlas_fresh"), [m + "_{tid}"]))
    for m, arm in (("pi05", "tv"), ("pi05", "tvp"), ("groot", "tv")):
        S.append(("fresh", m, arm, os.path.join(FL, "tv_fresh"), ["%s_%s_{tid}" % (m, arm)]))
    for arm in ("p", "wp"):
        S.append(("fresh", "pi05", arm, os.path.join(FL, "tv_attr"), ["pi05_%s_{tid}" % arm]))
    for arm in ("I", "tv", "w"):
        S.append(("fresh2", "groot", arm, os.path.join(FL, "tv_g"), ["groot_%s_{tid}" % arm]))
    return S


def trials(n2, setname):
    return dict(main=list(n2["states"]["main"]), fresh=list(range(24, 44)), fresh2=list(range(24, 84)))[setname]


def h(a, b):
    return float(np.linalg.norm((np.asarray(a) - np.asarray(b))[:2]))


def nearest(p, final, skip):
    best = None
    for i, n in enumerate(ENT):
        if n == skip:
            continue
        d = h(p, final[i])
        if best is None or d < best[1]:
            best = (SHORT[n], d)
    return best


def episode(e, r, t, side):
    obj, tgt = t["object"], t["target"]
    held = np.asarray(r["held"], bool)
    O, T, EE, F = r["obj"], r["tgt"], r["ee"], r["final"]
    d = np.linalg.norm((O - T)[:, :2], axis=1)
    st, sub, last = PA.classify(held, d, bool(e["success"]), e.get("bound"), obj)
    out = dict(trial=int(e["trial"]), success=bool(e["success"]), stage=st, sub=sub, phase=PHASE[(st, sub)], bound=e.get("bound"),
               steps=int(len(held)), lift=float(O[:, 2].max() - O[0, 2]) if len(O) else None,
               hold_segments=int(np.sum(np.diff(np.r_[0, held.astype(int)]) == 1)), held_at_end=bool(held[-1]) if len(held) else None,
               final_tilt=PA.tilt_deg(r["quat"][-1]) if len(held) else None)
    oi = ENT.index(obj)
    fo = F[oi]
    out["final_d_tgt"] = float(d[-1]) if len(d) else None
    out["final_dz_tgt"] = float(O[-1, 2] - T[-1, 2]) if len(O) else None
    nf = nearest(fo, F, obj)
    out["final_nearest"], out["final_nearest_d"] = nf[0], nf[1]
    idx = np.flatnonzero(held)
    if len(idx):
        on, la = int(idx[0]), int(idx[-1])
        cab, rack = F[CAB], F[RACK]
        dt0, dtl = float(d[on]), float(d[la])
        dc0, dcl = h(O[on], cab), h(O[la], cab)
        dch = [h(O[i], cab) for i in idx]
        nl = nearest(O[la], F, obj)
        a6h = np.asarray(r["a6"])[idx]
        out.update(onset=on, last_held=la, d_tgt_onset=dt0, d_tgt_last=dtl, dmin_tgt_held=float(d[idx].min()),
                   d_cab_onset=dc0, d_cab_last=dcl, dmin_cab_held=float(min(dch)), d_rack_last=h(O[la], rack),
                   prog_tgt=(1.0 - dtl / dt0) if dt0 > 1e-6 else None, prog_cab=(1.0 - dcl / dc0) if dc0 > 1e-6 else None,
                   lift_held=float(O[idx, 2].max() - O[0, 2]), z_last=float(O[la, 2] - O[0, 2]),
                   last_nearest=nl[0], last_nearest_d=nl[1], cab_nearer_than_tgt_at_last=bool(dcl < dtl),
                   open_cmds_held=int(np.sum(a6h < 0)), ee_obj_gap_last=float(np.linalg.norm(EE[la] - O[la])),
                   tilt_last_held=PA.tilt_deg(r["quat"][la]))
    if side is not None:
        f0, of = side.get("first_onset"), side.get("oracle_first_held")
        out.update(det_fired=f0 is not None, det_onsets=len(side.get("onsets") or []), det_onset_minus_oracle=(f0 - of) if (f0 is not None and of is not None) else None,
                   det_oracle_held_at_onset=side.get("oracle_held_at_onset"))
        P = side.get("paint")
        if P:
            for c, k in (("agentview", "agent"), ("robot0_eye_in_hand", "wrist")):
                out["paint_iou_" + k] = (P[c]["inter"] / float(P[c]["union"])) if P[c]["union"] else None
                out["leak_" + k] = (P[c]["act_painted"] / float(P[c]["act_px"])) if P[c]["act_px"] else None
    return out


def load(stem, t, ks):
    R = json.load(open(stem + ".json", encoding="utf-8"))
    E = R["eps"] if "eps" in R else [e for v in R["tasks"].values() for e in v["episodes"]]
    Tr = PA.load_track(stem + "_track.npz")
    sf = stem + "_tv.json"
    SD = json.load(open(sf, encoding="utf-8"))["eps"] if os.path.exists(sf) else None
    assert len(E) == len(ks) == len(Tr), (stem, len(E), len(ks), len(Tr))
    if SD is not None:
        assert len(SD) == len(E), (sf, len(SD), len(E))
    rows = []
    for j, (k, e, r) in enumerate(zip(ks, E, Tr)):
        assert int(e["trial"]) == k and e.get("error") is None, (stem, k, e.get("error"))
        s = SD[j] if SD is not None else None
        if s is not None and s.get("trial") is not None:
            assert int(s["trial"]) == k, (sf, s["trial"], k)
        rows.append(episode(e, r, t, s))
    return rows


def med(v):
    v = [x for x in v if x is not None]
    return float(np.median(v)) if v else None


def mean(v):
    v = [x for x in v if x is not None]
    return float(np.mean(v)) if v else None


def summarize(rows):
    n = len(rows)
    fail = [r for r in rows if not r["success"]]
    post = [r for r in fail if r["phase"] not in ("grasp_miss", "grasp_wrong_object")]
    s = dict(n=n, success=mean([float(r["success"]) for r in rows]), phases={p: sum(r["phase"] == p for r in rows) / float(n) for p in PHASES})
    s["n_fail"], s["n_post_grasp_fail"] = len(fail), len(post)
    s["lift_med_all"] = med([r.get("lift_held") for r in rows])
    s["lift_med_success"] = med([r.get("lift_held") for r in rows if r["success"]])
    s["lift_med_post_fail"] = med([r.get("lift_held") for r in post])
    s["z_last_med_post_fail"] = med([r.get("z_last") for r in post])
    s["d_tgt_last_med_post_fail"] = med([r.get("d_tgt_last") for r in post])
    s["dmin_tgt_held_med_post_fail"] = med([r.get("dmin_tgt_held") for r in post])
    s["prog_tgt_med_post_fail"] = med([r.get("prog_tgt") for r in post])
    s["prog_cab_med_post_fail"] = med([r.get("prog_cab") for r in post])
    s["cab_nearer_post_fail"] = mean([float(r["cab_nearer_than_tgt_at_last"]) for r in post if "cab_nearer_than_tgt_at_last" in r])
    s["final_within10_post_fail"] = mean([float(r["final_d_tgt"] is not None and r["final_d_tgt"] < 0.10) for r in post])
    s["final_tilt_gt45_post_fail"] = mean([float(r["final_tilt"] is not None and r["final_tilt"] > 45) for r in post])
    s["final_tilt_gt45_success"] = mean([float(r["final_tilt"] is not None and r["final_tilt"] > 45) for r in rows if r["success"]])
    s["held_at_end_post_fail"] = mean([float(bool(r["held_at_end"])) for r in post])
    s["multi_hold_post_fail"] = mean([float(r["hold_segments"] > 1) for r in post])
    cnt = {}
    for r in post:
        cnt[r.get("last_nearest")] = cnt.get(r.get("last_nearest"), 0) + 1
    s["last_held_nearest_post_fail"] = cnt
    cnt = {}
    for r in post:
        cnt[r["final_nearest"]] = cnt.get(r["final_nearest"], 0) + 1
    s["final_nearest_post_fail"] = cnt
    if rows and "det_fired" in rows[0]:
        s["det_fired"] = mean([float(r["det_fired"]) for r in rows])
        s["det_multi"] = mean([float(r["det_onsets"] > 1) for r in rows])
        s["det_onset_minus_oracle_med"] = med([r.get("det_onset_minus_oracle") for r in rows])
        s["det_oracle_held_at_onset"] = mean([None if r.get("det_oracle_held_at_onset") is None else float(r["det_oracle_held_at_onset"]) for r in rows])
        s["det_fired_fail"] = mean([float(r["det_fired"]) for r in fail])
        s["det_multi_fail"] = mean([float(r["det_onsets"] > 1) for r in fail])
        s["paint_iou_agent_med"] = med([r.get("paint_iou_agent") for r in rows])
        s["paint_iou_wrist_med"] = med([r.get("paint_iou_wrist") for r in rows])
        s["paint_iou_wrist_min"] = min([r["paint_iou_wrist"] for r in rows if r.get("paint_iou_wrist") is not None], default=None)
        s["leak_wrist_max"] = max([r["leak_wrist"] for r in rows if r.get("leak_wrist") is not None], default=None)
    return s


def fmt(x):
    if x is None:
        return "-"
    if isinstance(x, float):
        return "%.2f" % x
    return str(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = os.path.abspath(a.out)
    assert not os.path.exists(out), "never overwrite a result file: %s" % out
    n2 = PA.tasks()
    T = {t["compound"]: t for t in n2["tasks"] if t["layout"] == "familiar"}
    for c in COMPS:
        print("task", c, T[c]["tid"], repr(T[c]["language"]), T[c]["object"], "->", T[c]["target"], flush=True)
    res = dict(sources=[], per_episode={}, summary={})
    for setname, m, arm, d, stems in sources():
        ks = trials(n2, setname)
        key = "%s|%s|%s" % (setname, m, arm)
        for c in COMPS:
            files = [os.path.join(d, st.format(tid=T[c]["tid"])) for st in stems]
            if not all(os.path.exists(f + ".json") and os.path.exists(f + "_track.npz") for f in files):
                continue
            rows = []
            for f in files:
                rows += load(f, T[c], ks)
            res["per_episode"]["%s|%s" % (key, c)] = rows
            res["summary"]["%s|%s" % (key, c)] = summarize(rows)
        res["sources"].append(key)
    print("\n== per (set, model, arm, task): n succ | phases succ/gWrong/gMiss/transport/relAway/neverLetGo/physics | post-grasp fails: "
          "lift z_last d_tgt_last prog_tgt prog_cab cabNearer final<10cm tilt>45 heldEnd multiHold | last-held nearest | final nearest", flush=True)
    for k, s in res["summary"].items():
        ph = "/".join("%.2f" % s["phases"][p] for p in PHASES)
        print("%-28s n%3d %.2f | %s | nf %2d lift %s z %s dT %s pT %s pC %s cab %s near %s tilt %s held %s multi %s | %s | %s" % (
            k, s["n"], s["success"], ph, s["n_post_grasp_fail"], fmt(s["lift_med_post_fail"]), fmt(s["z_last_med_post_fail"]),
            fmt(s["d_tgt_last_med_post_fail"]), fmt(s["prog_tgt_med_post_fail"]), fmt(s["prog_cab_med_post_fail"]), fmt(s["cab_nearer_post_fail"]),
            fmt(s["final_within10_post_fail"]), fmt(s["final_tilt_gt45_post_fail"]), fmt(s["held_at_end_post_fail"]), fmt(s["multi_hold_post_fail"]),
            json.dumps(s["last_held_nearest_post_fail"]), json.dumps(s["final_nearest_post_fail"])), flush=True)
    print("\n== success-episode lift (median) and final tilt > 45 among successes", flush=True)
    for k, s in res["summary"].items():
        print("%-28s lift_succ %s tilt45_succ %s lift_all %s" % (k, fmt(s["lift_med_success"]), fmt(s["final_tilt_gt45_success"]), fmt(s["lift_med_all"])), flush=True)
    print("\n== front end (detector, paint) per arm and task", flush=True)
    for k, s in res["summary"].items():
        if "det_fired" in s:
            print("%-28s fired %s (fail %s) multi %s (fail %s) onset-oracle %s oracleHeldAtOnset %s paintIoU agent %s wrist %s (min %s) leakMax %s" % (
                k, fmt(s["det_fired"]), fmt(s["det_fired_fail"]), fmt(s["det_multi"]), fmt(s["det_multi_fail"]), fmt(s["det_onset_minus_oracle_med"]),
                fmt(s["det_oracle_held_at_onset"]), fmt(s["paint_iou_agent_med"]), fmt(s["paint_iou_wrist_med"]), fmt(s["paint_iou_wrist_min"]),
                fmt(s["leak_wrist_max"])), flush=True)
    print("\n== per-episode, N4 / N6, failures only (trial phase lift z_last dT_last prog_tgt prog_cab lastNearest finalNearest tilt det)", flush=True)
    for k, rows in res["per_episode"].items():
        if not (k.endswith("|N4") or k.endswith("|N6")):
            continue
        fr = [r for r in rows if not r["success"]]
        line = " ".join("%d:%s:%s/%s/%s/%s/%s:%s>%s:%s%s" % (
            r["trial"], r["phase"][:6], fmt(r.get("lift_held")), fmt(r.get("z_last")), fmt(r.get("d_tgt_last")), fmt(r.get("prog_tgt")),
            fmt(r.get("prog_cab")), r.get("last_nearest"), r["final_nearest"], "%.0f" % r["final_tilt"] if r["final_tilt"] is not None else "-",
            (":d%d" % r["det_onsets"]) if "det_onsets" in r else "") for r in fr)
        print("%-28s %s" % (k, line), flush=True)
    tmp = out + ".tmp%d" % os.getpid()
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    os.replace(tmp, out)
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
