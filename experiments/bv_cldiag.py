"""PLAN_BINDVLA 12.4: closed-loop slot diagnostic on dev rollouts recorded with `bv_eval.py --set dev --slotlog` (descriptive).

Per episode (step 0 of the instruction only):
  first grasp g / release r  = b1_render.events on the executed gripper command (the training-referent rule);
  contacted object           = the movable object displaced most (> 3 cm) between g and r (or g + 60) while closed;
  named object / target      = the step-0 object / goal phrase matched by word overlap to the scene's object / fixture names
                               (obj_of_interest first);
  slot masks                 = recomputed with the checkpoint from the EXACT policy inputs recorded at the query before g (object
                               slot), before r (goal slot) and at the first query (both): the prefix is deterministic, so these
                               are the masks the policy acted on;
  a slot "is on" an entity   = the entity's oracle coverage in the 3 x 3 token window around the mask argmax is >= 0.05.
Each failed episode gets one cause:
  start_miss  the object slot is off the named object at the first query (a binding miss, as offline);
  drift       on it at the first query, off it at the grasp query (closed-loop covariate shift)            (b);
  motor       on it at the grasp query, but the gripper contacts another object or nothing                 (a);
  place       the contacted object is the named one, the episode still fails (transport / place)           (c);
  no_grasp    no grasp event.
First-contact object per task: named / training-habit (named by a training task of the same scene) / other.

    python experiments/bv_cldiag.py --tag bindvla_c1main@NNNN --ckpt results/bindvla/c1main/ckpt/step_NNNN.pt
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from collections import Counter, defaultdict

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "experiments")]
EVAL = os.path.join(ROOT, "results", "bindvla", "eval")
STOP = {"the", "of", "a", "1", "2", "3", "4", "region", "init"}


def words(s):
    return {w for w in re.split(r"[_\s]+", str(s).lower()) if w and w not in STOP and not w.isdigit()}


def match(phrase, names, prefer=()):
    """best name for a phrase by word overlap (fraction of the phrase's words); obj_of_interest wins ties."""
    pw = words(phrase)
    if not pw:
        return None
    best, bs = None, 0.0
    for n in list(prefer) + [x for x in names if x not in prefer]:
        s = len(pw & words(n)) / len(pw)
        if s > bs:
            best, bs = n, s
    return best if bs > 0 else None


def on(seg, names, tok, ent):
    if ent is None or ent not in names:
        return None
    r, c = divmod(int(tok), 16)
    win = [rr * 16 + cc for rr in range(max(r - 1, 0), min(r + 2, 16)) for cc in range(max(c - 1, 0), min(c + 2, 16))]
    return bool(float(seg[names.index(ent)][win].astype(np.float32).sum()) >= 0.05)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--ckpt", required=True)
    a = ap.parse_args()
    import b1_render as BR
    import bv_c
    import bv_train as T
    from lifelong_ood import l90c
    from lifelong_ood.policy.backbone_bindvla import BindVLABackbone
    tname = a.tag.replace("@", "_")
    jsons = sorted(glob.glob(os.path.join(EVAL, f"devlog_{tname}_*.json")))
    assert jsons, f"no devlog_{tname}_*.json"
    sp = json.load(open(os.path.join(ROOT, "results", "l90c", "split.json"), encoding="utf-8"))
    bin_of = {t: k for k, v in sp["dev"].items() for t in v}
    from lifelong_ood.policy import factors as F
    tasks = {t.tid: t for t in F.load(("libero_90",))}
    store = T.Store()
    emap = bv_c.entity_map(store)
    bk = BindVLABackbone(a.ckpt, device="cuda")
    rows = []
    for js in jsons:
        res = json.load(open(js, encoding="utf-8"))
        suite = res.get("suite") or os.path.basename(js).split(f"devlog_{tname}_")[1].rsplit("_", 1)[0]
        rec_dir = js[:-5] + "_rec"
        for f in sorted(glob.glob(os.path.join(rec_dir, "*.npz"))):
            z = np.load(f, allow_pickle=True)
            if "qimg" not in z:
                continue
            tid = f"{suite}:{int(z['task'])}"
            lang = res["tasks"][str(int(z["task"]))]["language"]
            l90c.check_task(tid)
            scene = tasks[tid].scene
            o_ph, g_ph = bv_c.role_phrases(lang)[0]
            seg_names = [str(x) for x in z["seg_names"]]
            ooi = [str(x) for x in z["ooi"]]
            named = match(o_ph, seg_names, ooi)
            target = match(g_ph, seg_names, ooi[1:] + ooi[:1]) if g_ph else None
            act = z["action"]
            ev = BR.events(act[:, 6])
            g, r = (ev[0] if ev else (None, None))
            contacted = None
            if g is not None and "obj_pos" in z:
                P = z["obj_pos"]
                end = min(r if r is not None else g + 60, len(P) - 1)
                disp = np.linalg.norm(P[end] - P[g], axis=-1)
                if disp.max() > 0.03:
                    contacted = str(z["obj_names"][int(np.argmax(disp))])
            qt = np.asarray(z["qt"])
            qi = {"start": 0}
            if g is not None:
                qi["grasp"] = int(np.searchsorted(qt, g, side="right") - 1)
            if r is not None:
                qi["release"] = int(np.searchsorted(qt, r, side="right") - 1)
            obs = [dict({"observation/image": z["qimg"][i], "observation/wrist_image": z["qwrist"][i], "observation/state": z["qstate"][i],
                         "prompt": lang}) for i in qi.values()]
            with torch.no_grad():
                _, _, dg = bk.bv.prefix_cache(bk.observations(obs).batch)
            M = dg["mask"][:, 0].float().cpu().numpy()                       # [n, 2, 256] step 0
            tok = {k: (int(M[j, 0].argmax()), int(M[j, 1].argmax())) for j, k in enumerate(qi)}
            seg = {k: z["qseg"][i] for k, i in qi.items()}
            row = dict(tid=tid, bin=bin_of.get(tid), trial=int(z["trial"]), success=bool(z["success"]), named=named, target=target,
                       contacted=contacted, grasp=g, release=r,
                       obj_on_named_start=on(seg["start"], seg_names, tok["start"][0], named),
                       goal_on_target_start=on(seg["start"], seg_names, tok["start"][1], target))
            if "grasp" in qi:
                row.update(obj_on_named_grasp=on(seg["grasp"], seg_names, tok["grasp"][0], named),
                           obj_on_contacted_grasp=on(seg["grasp"], seg_names, tok["grasp"][0], contacted))
            if "release" in qi:
                row.update(goal_on_target_release=on(seg["release"], seg_names, tok["release"][1], target))
            habit_ph = {ph for ph in emap.get(scene, {}) if ph != o_ph}
            if contacted is None:
                row["first_contact"] = "none"
            elif contacted == named:
                row["first_contact"] = "named"
            elif any(match(ph, [contacted]) for ph in habit_ph):
                row["first_contact"] = "habit"
            else:
                row["first_contact"] = "other"
            if row["success"]:
                row["cause"] = "success"
            elif g is None:
                row["cause"] = "no_grasp"
            elif not row["obj_on_named_start"]:
                row["cause"] = "start_miss"
            elif not row.get("obj_on_named_grasp"):
                row["cause"] = "drift"
            elif contacted != named:
                row["cause"] = "motor"
            else:
                row["cause"] = "place"
            rows.append(row)
    out = dict(tag=a.tag, ckpt=a.ckpt, episodes=len(rows))

    def rate(rs, k):
        v = [r[k] for r in rs if r.get(k) is not None]
        return dict(n=len(v), rate=float(np.mean(v))) if v else None
    for nm, sel in [("all", lambda r: True)] + [(f"bin_{b}", (lambda b: lambda r: r["bin"] == b)(b)) for b in ("0.1", "0.3", "0.5", "part")]:
        rs = [r for r in rows if sel(r)]
        if not rs:
            continue
        out[nm] = dict(episodes=len(rs), success=float(np.mean([r["success"] for r in rs])),
                       causes=dict(Counter(r["cause"] for r in rs)), first_contact=dict(Counter(r["first_contact"] for r in rs)),
                       **{k: rate(rs, k) for k in ("obj_on_named_start", "obj_on_named_grasp", "obj_on_contacted_grasp",
                                                   "goal_on_target_start", "goal_on_target_release")})
    per = defaultdict(list)
    for r in rows:
        per[r["tid"]].append(r)
    out["per_task"] = {t: dict(bin=rs[0]["bin"], named=rs[0]["named"], target=rs[0]["target"], success=float(np.mean([r["success"] for r in rs])),
                               first_contact=dict(Counter(r["first_contact"] for r in rs)),
                               contacted=dict(Counter(str(r["contacted"]) for r in rs)), causes=dict(Counter(r["cause"] for r in rs)))
                       for t, rs in sorted(per.items())}
    out["unmatched_named"] = sorted({r["tid"] for r in rows if r["named"] is None})
    p = os.path.join(ROOT, "results", "bindvla", f"cldiag_{tname}.json")
    json.dump(dict(summary=out, rows=rows), open(p, "w", encoding="utf-8"), indent=1, default=str)
    print(json.dumps({k: v for k, v in out.items() if k != "per_task"}, indent=1, default=str))


if __name__ == "__main__":
    main()
