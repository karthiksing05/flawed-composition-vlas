"""Arm HP's read (experiments/sc_hp.py): HP paired with the intact atlas runs and with arm H (docs/HAND_OVER.md 3, 6.1), per cell
(new pairings N1-N6 / known K1, K2, K3, K4, K6 x familiar / new layout), bootstrapped jointly over tasks and states (2,000 draws,
rng 0 per estimate), as pp_hand_read.py does.  U alone from learned masks (sc_seg pu_learned, seed 10; SCAFFOLD_RESULTS 40) is
paired on success only (no tracks).  CPU, LIBERO venv.  No registered verdict: HAND_OVER 3's mapping is printed for HP, labelled
"unregistered", for the user to fix before reading.

    python experiments/sc_hp_read.py [--models pi05,groot] [--out results/flawed/sc_hp_read.json]
Sources (from pp_hand_read.py): intact = results/flawed/pp_atlas/<model>_<tid>; H = results/flawed/pp_hand/<model>_H_<tid>;
HP = results/flawed/sc_hp/<model>_s10_<tid>; each <stem>.json + <stem>_track.npz.  A model with any HP file missing (or not
hp_complete) is skipped and its missing tids printed.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pp_atlas as PA                                                       # noqa: E402

n2 = PA.tasks()
ks = n2["states"]["main"]
tid = {(t["compound"], t["layout"]): t for t in n2["tasks"]}
HD = os.path.join(PA.OUT, "pp_hand")
HPD = os.path.join(PA.OUT, "sc_hp")
UD = os.path.join(PA.OUT, "sc_seg", "pu_learned")
ARMS = ("intact", "H", "HP")
CELLS = dict(new_pairings=PA.NEW, known=PA.KNOWN)


SELFTEST = False                                                            # --selftest: HP's stems point at H's files


def stem(model, arm, t):
    if arm == "H" or (arm == "HP" and SELFTEST):
        return os.path.join(HD, "%s_H_%d" % (model, t["tid"]))
    if arm == "HP":
        return os.path.join(HPD, "%s_s10_%d" % (model, t["tid"]))
    return os.path.join(PA.OD, "%s_%d" % (model, t["tid"]))


def episodes(R):
    return R["eps"] if "eps" in R else [e for v in R["tasks"].values() for e in v["episodes"]]


def load(st, t):
    """-> (stage indicators per state, handed [states], episode records), as pp_hand_read.load plus the episodes."""
    R = json.load(open(st + ".json", encoding="utf-8"))
    E = episodes(R)
    z = np.load(st + "_track.npz")
    Tr = PA.load_track(st + "_track.npz")
    assert len(E) == len(ks) == len(Tr) and all(e.get("error") is None for e in E), (st, len(E), len(Tr))
    A = {s: np.zeros(len(ks)) for s in PA.STAGES}
    handed = np.full(len(ks), -1.0)
    for j, (e, r) in enumerate(zip(E, Tr)):
        assert int(e["trial"]) == ks[j], (st, j, e["trial"])
        d = np.linalg.norm((r["obj"] - r["tgt"])[:, :2], axis=1)
        A[PA.classify(r["held"], d, bool(e["success"]), e.get("bound"), t["object"])[0]][j] = 1.0
        if "e%d_handed" % j in z:
            handed[j] = float(int(z["e%d_handed" % j]))
    return A, handed, E


def load_u(t):
    fn = os.path.join(UD, "learned_s10_%d.json" % t["tid"])
    E = episodes(json.load(open(fn, encoding="utf-8")))
    assert [int(e["trial"]) for e in E] == list(ks), fn
    return np.array([float(e["success"]) for e in E])


def boot(f, nt):
    rng = np.random.default_rng(0)
    v = [f(rng.choice(nt, nt), rng.choice(len(ks), len(ks))) for _ in range(PA.B)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def missing(model):
    out = []
    for c in PA.NEW + PA.KNOWN:
        for lay in ("familiar", "new"):
            st = stem(model, "HP", tid[(c, lay)])
            ok = os.path.exists(st + ".json") and os.path.exists(st + "_track.npz")
            if ok and not SELFTEST:
                ok = bool(json.load(open(st + ".json", encoding="utf-8")).get("hp_complete"))
            if not ok:
                out.append("%s %s (tid %d)" % (c, lay, tid[(c, lay)]["tid"]))
    return out


def q(v, p):
    v = [x for x in v if x is not None]
    return round(float(np.percentile(v, p)), 3) if v else None


def hp_log(E_by_task):
    """Pooled per-episode HP log: trigger, hand-over row, oracle hold at the hand-over, pose errors (cm)."""
    eps = [e for E in E_by_task for e in E]
    h = [e["hp"] for e in eps]
    fired = [x["trigger_fired"] for x in h]
    succ = [bool(e["success"]) for e in eps]
    hand = [x["handed"] for x in h if x["handed"] >= 0]
    orc = [x["oracle_held_at_handover"] for x in h if x["handed"] >= 0]
    at = [x["at_handover"] for x in h if x.get("at_handover")]

    def pe(k, f="med"):
        return [x["pose_err_cm"][k][f] if x["pose_err_cm"].get(k) else None for x in h]
    return dict(
        episodes=len(eps), trigger_fired=float(np.mean(fired)),
        success_given_fired=float(np.mean([s for s, f in zip(succ, fired) if f])) if any(fired) else None,
        success_given_not_fired=float(np.mean([s for s, f in zip(succ, fired) if not f])) if not all(fired) else None,
        handover_row=dict(med=q(hand, 50), p10=q(hand, 10), p90=q(hand, 90)),
        oracle_held_at_handover=float(np.mean(orc)) if orc else None,
        roles_match=float(np.mean([x["roles_match"] for x in h])),
        err_episode_median_cm={k: dict(med=q(pe(k), 50), p90=q(pe(k), 90)) for k in ("obj_pre_vla", "obj_pre_u", "obj_held", "tgt")},
        err_episode_p90_cm={k: dict(med=q(pe(k, "p90"), 50), p90=q(pe(k, "p90"), 90)) for k in ("obj_pre_vla", "obj_pre_u", "obj_held", "tgt")},
        at_handover={k: dict(med=q([a[k] for a in at], 50), p90=q([a[k] for a in at], 90)) for k in ("obj_cm", "tgt_cm", "d0t_err_cm", "zrest_err_cm", "cyaw_err_deg")},
        seg_ms_per_pred=q([x["seg"]["ms_per_pred"] for x in h], 50))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="pi05,groot")
    ap.add_argument("--out", default=os.path.join(PA.OUT, "sc_hp_read.json"))
    ap.add_argument("--selftest", action="store_true", help="HP := H's files (pairing / bootstrap check: HP - H must be 0, "
                    "HP - intact must equal H - intact, familiar N: pp_hand_read's numbers); no hp log")
    a = ap.parse_args()
    global SELFTEST
    SELFTEST = a.selftest
    res = {}
    for model in [m for m in a.models.replace("+", ",").split(",") if m]:
        miss = missing(model)
        if miss:
            print(model, "SKIPPED: HP missing or incomplete for", miss, flush=True)
            res[model] = dict(skipped=miss)
            continue
        res[model] = {}
        for lay in ("familiar", "new"):
            D = {ar: {c: load(stem(model, ar, tid[(c, lay)]), tid[(c, lay)]) for c in PA.NEW + PA.KNOWN} for ar in ARMS}
            U = {c: load_u(tid[(c, lay)]) for c in PA.NEW + PA.KNOWN}
            si0 = np.arange(len(ks))
            rl = {}
            for cell, names in CELLS.items():
                nt = len(names)
                ti0 = np.arange(nt)

                def S(ar, ti, si, names=names):
                    if ar == "U":
                        return float(np.mean([U[names[i]][si].mean() for i in ti]))
                    return float(np.mean([D[ar][names[i]][0]["success"][si].mean() for i in ti]))

                def delta(x, y, nt=nt, ti0=ti0, S=S):
                    e = S(x, ti0, si0) - S(y, ti0, si0)
                    lo, hi = boot(lambda t, s: S(x, t, s) - S(y, t, s), nt)
                    return dict(delta=e, ci=[lo, hi])
                r = dict(s={ar: S(ar, ti0, si0) for ar in ARMS + ("U",)})
                r["HP_minus_intact"] = delta("HP", "intact")
                r["HP_minus_H"] = delta("HP", "H")
                r["H_minus_intact"] = delta("H", "intact")
                r["HP_minus_U"] = delta("HP", "U")
                r["per_task"] = {ar: {c: {s: float(D[ar][c][0][s].mean()) for s in PA.STAGES} for c in names} for ar in ARMS}
                r["per_task_success_U"] = {c: float(U[c].mean()) for c in names}
                for ar in ("H", "HP"):
                    h = np.stack([D[ar][c][1] for c in names])
                    sc = np.stack([D[ar][c][0]["success"] for c in names])
                    r["handed_share_" + ar] = float((h >= 0).mean())
                    r["success_given_handed_" + ar] = float(sc[h >= 0].mean()) if (h >= 0).any() else None
                hh, hp = np.stack([D["H"][c][1] for c in names]), np.stack([D["HP"][c][1] for c in names])
                both = (hh >= 0) & (hp >= 0)
                r["handover_row_HP_minus_H"] = dict(n=int(both.sum()), med=q((hp - hh)[both].tolist(), 50), p10=q((hp - hh)[both].tolist(), 10),
                                                    p90=q((hp - hh)[both].tolist(), 90))
                r["hp_log"] = None if SELFTEST else hp_log([D["HP"][c][2] for c in names])
                rl[cell] = r
            # the breaks under each arm, as pp_hand_read (new pairings against their known partners, same arm)
            ti0 = np.arange(len(PA.NEW))

            def excess(ar, nm, ti, si):
                out = []
                for i in ti:
                    c = PA.NEW[i]
                    ko, kt = PA.PAIR[c]
                    out.append(sum(D[ar][c][0][n][si] - 0.5 * (D[ar][ko][0][n][si] + D[ar][kt][0][n][si]) for n in nm).mean())
                return float(np.mean(out))

            def est(ar, nm):
                e = excess(ar, nm, ti0, si0)
                ci = boot(lambda t, s: excess(ar, nm, t, s), len(PA.NEW))
                return dict(E=e, ci=ci, breaks=bool(ci[0] > 0.0 and e > 0.10))
            rl["E"] = {ar: {g: est(ar, nm) for g, nm in list(PA.GROUP.items()) + [(s, (s,)) for s in PA.STAGES[1:]]} for ar in ARMS}
            n = rl["new_pairings"]["HP_minus_intact"]
            raises = bool(n["ci"][0] > 0.0 and n["delta"] > 0.10)
            w, x = rl["E"]["HP"]["within"]["breaks"], rl["E"]["HP"]["cross"]["breaks"]
            rl["mapping_unregistered"] = ("NO FIX" if not raises else
                                          {(False, False): "FIXED", (True, False): "RAISED, WITHIN STILL BREAKS",
                                           (False, True): "RAISED, CROSS STILL BREAKS", (True, True): "RAISED, BOTH STILL BREAK"}[(w, x)])
            k = rl["known"]["s"]
            rl["known_cost_unregistered"] = bool(k["intact"] - k["HP"] > 0.05)
            res[model][lay] = rl
            for cell in CELLS:
                r = rl[cell]
                print(model, lay, cell, json.dumps({kk: r[kk] for kk in ("s", "HP_minus_intact", "HP_minus_H", "H_minus_intact", "HP_minus_U",
                                                                         "handed_share_HP", "success_given_handed_HP", "handover_row_HP_minus_H")}), flush=True)
                print("  hp_log", json.dumps(r["hp_log"]), flush=True)
                print("  per_task_HP", json.dumps({c: v["success"] for c, v in r["per_task"]["HP"].items()}), flush=True)
            print(model, lay, "mapping (unregistered):", rl["mapping_unregistered"], "known cost:", rl["known_cost_unregistered"], flush=True)
            print("  E", json.dumps({ar: {g: rl["E"][ar][g] for g in ("within", "cross")} for ar in ARMS}), flush=True)
    tmp = a.out + ".tmp%d" % os.getpid()
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    os.replace(tmp, a.out)
    print("->", a.out, flush=True)


if __name__ == "__main__":
    main()
