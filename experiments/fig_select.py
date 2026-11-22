"""Applies the fixed selection rule of docs/notes/fig_plan.md section 2 (general rules) to the display rollouts of
results/flawed/fig_frames/** and writes paper/figs/frames/selection.json.  Openpi venv (numpy only), via sbatch.

RULE (fixed before any frame was looked at; DEVELOPMENT states only, n2_states + n2_branches_main.json, rg_comp main states, mt states):
  per entry, over the entry's first 10 trials (branches-file order), a trial QUALIFIES iff the failing arm fails with the failure's own
  signature AND the comparator arm succeeds on the same trial of the same fresh run.  The shown episode = the lowest-index qualifying trial.
  A16's extra episodes = the next three qualifying trials.  If none of the first 10 qualifies, 10 more trials (skip 10) are run into
  <dir>_more and the same rule is applied to those; the record says so.  Signatures (functions below) are fixed here:
    F1   failed, held, and the held object's last-held xy within 3 cm of the bowl (the trained destination)
    F1m  failed, held, lift while held (max obj z - first obj z) > .22 m (n46_lift's high carry)
    F2   failed and the first-held object ('bound') is another object
    F7   failed after a grasp (the referent is not logged: a bowl was grasped and the episode failed)
    F6   failed, held, last-held xy farther than 3 cm from the plate (released away)
    NL   failed (comparator: none; every system fails)
    MV   release closer to the trained plate position than to the actual plate by > 3 cm (SR 81's rule), failed;  comparator I succeeds
    A14  RAA-sL fails and RAA-sL-W5 succeeds on the trial
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fig_frames as FR                                                                   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
FF = os.path.join(REPO, "results", "flawed", "fig_frames")
OUTJ = os.path.join(REPO, "paper", "figs", "frames", "selection.json")
SEAT = 0.03


def load(stem, task=None):
    """-> {trial: dict(success, bound, tk)} for <stem>.json + _track.npz + _disp.npz (fig_frames.load_episodes), or None."""
    if not all(os.path.exists(stem + s) for s in (".json", "_track.npz", "_disp.npz")):
        return None
    try:
        L = FR.load_episodes(stem, task)
    except Exception as ex:
        print("WARN", stem, ex, flush=True)
        return None
    return {e["trial"]: dict(success=e["success"], bound=e["bound"], steps=e["steps"], tk=e["tk"], idx=i) for i, e in enumerate(L)}


def xy(a, b):
    return float(np.linalg.norm(np.asarray(a)[:2] - np.asarray(b)[:2]))


def last_held(tk):
    h = np.flatnonzero(tk["held"])
    return int(h[-1]) if len(h) else None


def sig_f1(e, tk, task):
    r = last_held(tk)
    return (not e["success"]) and r is not None and xy(tk["obj"][r], tk["final"][0]) <= SEAT


def sig_f1m(e, tk, task):
    h = np.flatnonzero(tk["held"])
    return (not e["success"]) and len(h) > 0 and float(tk["obj"][h, 2].max() - tk["obj"][0, 2]) > 0.22


def sig_f2(e, tk, task):
    return (not e["success"]) and e["bound"] not in (None, task["object"])


def sig_f7(e, tk, task):
    return (not e["success"]) and last_held(tk) is not None


def sig_f6(e, tk, task):
    r = last_held(tk)
    return (not e["success"]) and r is not None and xy(tk["obj"][r], tk["tgt"][r]) > SEAT


def sig_nl(e, tk, task):
    return not e["success"]


def sig_mv(e, tk, task):
    r = last_held(tk)
    if e["success"] or r is None:
        return False
    cen = task["centre"]
    return xy(tk["obj"][r], tk["tgt"][r]) - xy(tk["obj"][r], cen) > SEAT


SIG = dict(f1=sig_f1, f1m=sig_f1m, f2=sig_f2, f7=sig_f7, f6=sig_f6, nl=sig_nl, mv=sig_mv, any=lambda e, tk, t: not e["success"])


def E(key, sig, fail, comp, extra=(), reg="", obj="akita_black_bowl_1", tgt=None, model=None, cell=None):
    return dict(key=key, sig=sig, fail=fail, comp=list(comp), extra=list(extra), reg=reg, model=model)


# stems are <dir>/<file stem> under results/flawed/fig_frames; a trailing "_more" dir holds the skip-10 rerun
ENTRIES = [
    E("F1_N1", "f1", "F1/pi05_I_16", ["F1/pi05_rgp_16"], ["F1/pi05_sb_16", "F1/groot_I_16"], "atlas E +.450 [.28, .61] (Table 1); ReGuide .929, tvpb .938 vs .454"),
    E("F1_N2", "f1", "F1/pi05_I_18", ["F1/pi05_rgp_18"], ["F1/pi05_sb_18", "F1/groot_I_18"], "atlas E +.450 [.28, .61] (Table 1); ReGuide .929"),
    E("F1m_N4", "f1m", "F1m/pi05_g_22", ["F1m/pi05_sb_22"], [], "high-carry share .225, N4+N6 .625 (arm g); arm sb .062 and .812 (SR 53)"),
    E("F1m_N6", "f1m", "F1m/pi05_g_26", ["F1m/pi05_sb_26"], [], "high-carry share .225, N4+N6 .625 (arm g); arm sb .062 and .812 (SR 53)"),
    E("F2_pi0_N1", "f2", "F2_pi0/pi0_I_16", ["F2_pi0/pi0_rgp_16"], [], "grasp excess +.650 [.39, .88] (pi0); .679 end at the grasp; ReGuide .608 (SR 65)", model="pi0"),
    E("F2_pi0_N4", "f2", "F2_pi0/pi0_I_22", ["F2_pi0/pi0_rgp_22"], [], "grasp excess +.650 [.39, .88] (pi0); ReGuide .608 (SR 65)", model="pi0"),
    E("F2_smolvla_N1", "f2", "F2_smolvla/smolvla_I_16", ["F2_smolvla/smolvla_rgp_16"], [], "grasp excess +.833 [.67, .96] (SmolVLA); .829 end at the grasp; ReGuide .604 (SR 65)", model="smolvla"),
    E("F2_smolvla_N4", "f2", "F2_smolvla/smolvla_I_22", ["F2_smolvla/smolvla_rgp_22"], [], "grasp excess +.833 [.67, .96] (SmolVLA); ReGuide .604 (SR 65)", model="smolvla"),
    E("F7_relL", "f7", "F7/pi05_I_0", ["F7/pi05_rc_0"], ["F7/groot_I_0"], "pi0.5 .40 (SP), resolver 1.00 (SR 45.2)"),
    E("F7_relR", "f7", "F7/pi05_I_1", ["F7/pi05_rc_1"], ["F7/groot_I_1"], "pi0.5 .40 (SP), resolver 1.00 (SR 45.2)"),
    E("F7_neg", "f7", "F7/pi05_I_3", ["F7/pi05_rc_3"], ["F7/groot_I_3"], "pi0.5 .20 (NG), resolver .775 (SR 45.2)"),
    E("F6_b", "f6", "F6/pi05_I_8", ["F6/pi05_rc_8"], ["F6/pi05_I_9"], "cost of covering (d minus b) .38 [.28, .48]; RC with pose memory .800 vs .517 intact (SR 70)"),
    E("NL_N1new", "nl", "NL/pi05_I_17", [], [], "all four systems fail on the same new-layout states (new-layout limit, DRAFT 4.6)"),
    E("NL_N6new", "nl", "NL/pi05_I_27", [], [], "all four systems fail on the same new-layout states (new-layout limit, DRAFT 4.6)"),
    E("MV_Tfar", "mv", "MV/r2_T_far", ["MV/I_T_far"], ["MV/raa_T_far", "MV/rgp_T_far"], "I .975, 0 of 40 trained releases; R2 releases at the trained position on 31 of 40 (SR 81)"),
    E("S2_N1", "any", "S2/pi05_I_2", ["S2/pi05_rgp_2"], ["S2_pi0/pi0_I_2"], "0 of 720 new-pairing episodes succeed (pi0.5); ReGuide with perceived poses .708 on N1+N2 (SR 68)"),
    E("S2_N2", "any", "S2/pi05_I_3", ["S2/pi05_rgp_3"], ["S2_pi0/pi0_I_3"], "0 of 720 new-pairing episodes succeed (pi0.5); ReGuide with perceived poses N2 1.00 (SR 68); pi0 .993 of 145 placements go to the start-side plate"),
    E("A14_N1", "any", "A14_raasl/raac_raac_16", ["A14_w5/raac_raac_16"], [], "RAA-sL against RAA-sL-W5 (SR 87/91)"),
    E("A14_N4", "any", "A14_raasl/raac_raac_22", ["A14_w5/raac_raac_22"], [], "RAA-sL against RAA-sL-W5 (SR 87/91)"),
]


def task_info(key, tid=None):
    n2 = json.load(open(os.path.join(REPO, "results", "flawed", "n2_tasks.json"), encoding="utf-8"))
    return n2


def main():
    n2 = json.load(open(os.path.join(REPO, "results", "flawed", "n2_tasks.json"), encoding="utf-8"))
    rgt = json.load(open(os.path.join(REPO, "results", "flawed", "rg_comp", "rg_comp_tasks.json"), encoding="utf-8"))
    cen = np.array(n2["training"]["centre"][3], np.float64)                 # plate (MOVABLE order bowl, cheese, bottle, plate)
    sel = dict(rule=__doc__.split("RULE")[1].split('"""')[0].strip(), dev_states_only=True, entries={})

    def task_of(stem):
        last = stem.split("_")[-1]
        d = stem.split("/")[0]
        if last.isdigit() and d.startswith(("F7", "F6")):
            return rgt["tasks"][int(last)]
        if d.startswith("S2"):
            return dict(object="", target="")
        if last.isdigit():
            return n2["tasks"][int(last)]
        return dict(object="akita_black_bowl_1", target="plate_1")

    def stem_trials(stem):
        d, f = stem.split("/")
        out = {}
        for sub in (d, d + "_more"):
            r = load(os.path.join(FF, sub, f), task_of(stem))
            if r:
                for t, v in r.items():
                    out[t] = dict(v, more=sub.endswith("_more"))
        return out

    for en in ENTRIES:
        f0 = stem_trials(en["fail"])
        if not f0:
            sel["entries"][en["key"]] = dict(status="missing", fail=en["fail"])
            continue
        # the task of the failing stem: tid is the last _ field (cells for MV)
        last = en["fail"].split("_")[-1]
        if en["sig"] in ("f7", "f6"):
            task = rgt["tasks"][int(last)]
        elif en["sig"] == "mv":
            task = dict(object="akita_black_bowl_1", target="plate_1", centre=cen)
        else:
            task = n2["tasks"][int(last)] if last.isdigit() else dict(object="")
        task = dict(task, centre=cen)
        comps = [stem_trials(c) for c in en["comp"]]
        extras = {x: stem_trials(x) for x in en["extra"]}
        order = sorted(f0, key=lambda t: (f0[t]["more"], list(f0).index(t)))     # file order, first 10 then the more-rerun
        rows = []
        for t in order:
            v = f0[t]
            ok_f = bool(SIG[en["sig"]](v, v["tk"], task))
            ok_c = all((t in c and c[t]["success"]) for c in comps) if en["comp"] else True
            rows.append(dict(trial=t, more=v["more"], fail_success=v["success"], signature=ok_f, comp_success=[(c[t]["success"] if t in c else None) for c in comps],
                             qualifies=ok_f and ok_c,
                             extra_success={k: (x[t]["success"] if t in x else None) for k, x in extras.items()}))
        first = [r for r in rows if not r["more"]]
        q1 = [r for r in first if r["qualifies"]]
        qa = [r for r in rows if r["qualifies"]]
        d = dict(status="ok" if qa else "none_qualifies", fail=en["fail"], comp=en["comp"], extra=en["extra"], signature=en["sig"], registered=en["reg"],
                 n_first10=len(first), n_qualifying_first10=len(q1), n_more=len(rows) - len(first), n_qualifying_more=len(qa) - len(q1),
                 used_more=(not q1 and bool(qa)), chosen=(qa[0]["trial"] if qa else None), further=[r["trial"] for r in qa[1:4]],
                 chosen_outcome=(None if not qa else dict(fail_arm_success=qa[0]["fail_success"], comp_success=qa[0]["comp_success"], extra_success=qa[0]["extra_success"])),
                 trials=rows)
        sel["entries"][en["key"]] = d
        print(en["key"], d["status"], "chosen", d["chosen"], "first10 qualifying %d/%d" % (len(q1), len(first)), "more", d["n_qualifying_more"], flush=True)
    os.makedirs(os.path.dirname(OUTJ), exist_ok=True)
    with open(OUTJ, "w", encoding="utf-8") as f:
        json.dump(sel, f, indent=1)
    print("wrote", OUTJ)


if __name__ == "__main__":
    main()
