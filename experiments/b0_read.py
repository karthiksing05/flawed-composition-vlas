"""PLAN_BINDER B0 read (docs/PLAN_BINDER.md; pre-registration experiments/prereg/pb0.py, ledger m3_prereg tag pb0): pretrained
vision-language backbones (experiments/b0_vlm.py; results/bind/b0/<arm>__<config>.npz) as binders on S1's 240 dev units, scored
EXACTLY as S1 / S1c / S3b (s1_read.hit: the peak token lies in the target and is covered by it at least as much as by every
competitor; s1_read.iou), with the same two-level task-cluster bootstrap (d4_read.rate).  CPU (chunks env).

BARS (S1c's, point estimates; per (arm, config)):  (i) held-out depth-1 hit >= 0.70 (:60, :38, :16's goal; 15 units, 3 tasks);
(ii) hit on ALL depth-2 units with a SEEN body type >= 0.70 (:1, :29, :32, :40; 20 units, 4 tasks); (iii) depth-0 hit >= GroundingDINO
alone (S1's B1a on the same units, 0.786).  An (arm, config) PASSES iff all three hold.
VERDICT: B0 PASS iff at least one arm's PRIMARY config passes.  Primary = phrase-only prompt; PaliGemma-family arms: detect.
Arms: pi05_base, qwen, molmo (primary; the PaliGemma mix / pt-224 arms were not run: gated, no local weights); pi05_l90c is supplementary (reported, not in the verdict).  Every other config
(segment, the instruction-context variant) is reported against the same bars as secondary (multiplicity: 3 primary, 5 secondary).
Also reported: every config by depth x {all, seen, held-out, held-out body-seen, held-out body-unseen} x role, the unseen-caddy units
(depth-2 held-out with an unseen body type), parse rates, and S1b / S1c / RoboPoint / GroundingDINO / B1b on the same cells.

    python experiments/b0_read.py [--record]  -> results/bind/b0.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
S1 = os.path.join(ROOT, "results", "bind", "s1")
OD = os.path.join(ROOT, "results", "bind", "b0")
PRIMARY = {"pi05_base": "detect_phrase", "qwen": "bbox_phrase", "molmo": "point_phrase"}
NOT_RUN = {"pg_mix": "PaliGemma mix (not the policy's base), google/paligemma-3b-mix-224: not run: gated, no local weights",
           "pg_pt": "PaliGemma pt-224 (the policy's base), google/paligemma-3b-pt-224: not run: gated, no local weights"}
SUPPLEMENTARY = ("pi05_l90c",)
SPLITS = ("all", "seen", "heldout", "heldout_body_seen", "heldout_body_unseen")


def main():
    import d4_read as DR
    import s1_read as SR
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", action="store_true")
    a = ap.parse_args()
    lab = json.load(open(os.path.join(S1, "labels.json"), encoding="utf-8"))
    s1 = {f"{u['role']}|{u['frame']}": u for u in json.load(open(os.path.join(ROOT, "results", "bind", "s1.json")))["units"]}
    ref = {}
    for nm in ("s1b", "s1c", "s3b"):     # S1b / S1c / RoboPoint hits on the same units (same loop order over labels["dev"]; asserted)
        js = json.load(open(os.path.join(ROOT, "results", "bind", f"{nm}.json")))
        ref[nm] = [(u["tid"], u["role"], u["phrase"], u["s1b_hit"]) for u in js["units"]]
        assert len(ref[nm]) == len(lab["dev"])
    files = sorted(glob.glob(os.path.join(OD, "*__*.npz")))
    preds = {}
    for f in files:
        arm, cfg = os.path.basename(f)[:-4].split("__")
        pz = np.load(f, allow_pickle=True)
        preds[(arm, cfg)] = {str(k): (int(p), m, json.loads(str(mt))) for k, p, m, mt in zip(pz["keys"], pz["peak"], pz["mask"], pz["meta"])}
        assert len(preds[(arm, cfg)]) == len(lab["dev"]), f
    for arm, cfg in PRIMARY.items():
        assert (arm, cfg) in preds, f"primary config missing: {arm} {cfg}"
    cache = {}

    def frame(f):
        if f not in cache:
            z = np.load(os.path.join(ROOT, f), allow_pickle=True)
            cache[f] = (z, *SR.masks(z), [str(b) for b in z["body_names"]])
        return cache[f]
    seen_types = set()
    for e in lab["train"]:
        _, names, _, bodies = frame(e["frame"])
        for t in e["targets"]:
            seen_types.add(SR.itype(t) if t in bodies else SR.competitors(t, names, bodies)[1])
    ck = [f"{arm}__{cfg}" for arm, cfg in sorted(preds)]
    units = []
    for ix, e in enumerate(lab["dev"]):
        z, names, cov, bodies = frame(e["frame"])
        tg = e["targets"][0]
        t = cov[names.index(tg)]
        comp, btype = SR.competitors(tg, names, bodies)
        if tg in bodies:
            btype = SR.itype(tg)
        cc = [cov[names.index(c)] for c in comp]
        key = f"{e['role']}|{e['frame']}"
        o = s1[key]
        for nm in ref:
            assert ref[nm][ix][:3] == (e["tid"], e["role"], e["phrase"]), (nm, ix)
        u = dict(tid=e["tid"], role=e["role"], phrase=e["phrase"], target=tg, depth=e["depth"], seen=e["seen"], body_type=btype,
                 body_seen=btype in seen_types, b1b=o["b1b_hit"], b1a=o["b1a_hit"], anchor=o["anchor_hit"],
                 s1b=ref["s1b"][ix][3], s1c=ref["s1c"][ix][3], robopoint=ref["s3b"][ix][3])
        for (arm, cfg), pr in preds.items():
            pk, msk, meta = pr[key]
            c = f"{arm}__{cfg}"
            u[c] = SR.hit(pk, t, cc)
            u[c + "__iou"] = SR.iou(np.asarray(msk, bool), t)
            u[c + "__parsed"] = float(meta.get("parsed", "none") != "none")
            u[c + "__answer"] = meta.get("answer", "")[:160]
        units.append(u)

    def per(us, k):
        d = {}
        for u in us:
            if u.get(k) is not None:
                d.setdefault(u["tid"], []).append(float(u[k]))
        return DR.rate(d)

    def sel(depth, split, role=None):
        return [u for u in units if u["depth"] == depth and (role is None or u["role"] == role) and (
            split == "all" or (split == "seen" and u["seen"]) or (split == "heldout" and not u["seen"])
            or (split == "heldout_body_seen" and not u["seen"] and u["body_seen"])
            or (split == "heldout_body_unseen" and not u["seen"] and not u["body_seen"]))]
    methods = ck + ["s1b", "s1c", "robopoint", "b1a", "b1b"]
    cells = {}
    for depth in (0, 1, 2):
        for split in SPLITS:
            for role in (None, "object", "goal"):
                us = sel(depth, split, role)
                if not us:
                    continue
                k = f"d{depth}_{split}" + (f"_{role}" if role else "")
                cells[k] = dict(n=len(us), tasks=sorted({u["tid"] for u in us}), hit={m: per(us, m) for m in methods},
                                iou={c: per(us, c + "__iou")["mean"] for c in ck}, parsed={c: per(us, c + "__parsed")["mean"] for c in ck})
    d2s = [u for u in units if u["depth"] == 2 and u["body_seen"]]
    cells["d2_body_seen_all"] = dict(n=len(d2s), tasks=sorted({u["tid"] for u in d2s}), hit={m: per(d2s, m) for m in methods},
                                     iou={c: per(d2s, c + "__iou")["mean"] for c in ck}, parsed={c: per(d2s, c + "__parsed")["mean"] for c in ck})
    cad = [u for u in units if "caddy" in (u["body_type"] or "")]
    cells["caddy_units"] = dict(n=len(cad), tasks=sorted({u["tid"] for u in cad}), hit={m: per(cad, m) for m in methods},
                                iou={c: per(cad, c + "__iou")["mean"] for c in ck}, parsed={c: per(cad, c + "__parsed")["mean"] for c in ck})
    gd0 = cells["d0_all"]["hit"]["b1a"]["mean"]
    bars = {}
    for c in ck:
        arm, cfg = c.split("__")
        i_ = cells["d1_heldout"]["hit"][c]
        ii = cells["d2_body_seen_all"]["hit"][c]
        iii = cells["d0_all"]["hit"][c]
        b = dict(arm=arm, config=cfg, role=("primary" if PRIMARY.get(arm) == cfg else "supplementary" if arm in SUPPLEMENTARY
                                            else "secondary"),
                 i_depth1_heldout=i_, i_pass=bool(i_["mean"] is not None and i_["mean"] >= 0.70),
                 ii_depth2_body_seen=ii, ii_pass=bool(ii["mean"] is not None and ii["mean"] >= 0.70),
                 iii_depth0=iii, iii_groundingdino=gd0, iii_pass=bool(iii["mean"] is not None and iii["mean"] >= gd0),
                 caddy=cells["caddy_units"]["hit"][c], parse_rate=per(units, c + "__parsed")["mean"])
        b["pass"] = b["i_pass"] and b["ii_pass"] and b["iii_pass"]
        bars[c] = b
    verdict = "PASS" if any(bars[f"{arm}__{cfg}"]["pass"] for arm, cfg in PRIMARY.items()) else "FAIL"
    pt = {}
    for u in units:
        p = pt.setdefault(f"{u['tid']}|{u['role']}", dict(phrase=u["phrase"], target=u["target"], depth=u["depth"], seen=u["seen"],
                                                         body_seen=u["body_seen"], hits={m: [] for m in methods}))
        for m in methods:
            p["hits"][m].append(u[m])
    pt = {k: dict(v, hits={m: float(np.mean(x)) for m, x in v["hits"].items()}) for k, v in pt.items()}
    json.dump(dict(verdict=verdict, bars=bars, cells=cells, per_task=pt, units=units, seen_body_types=sorted(seen_types),
                   primary=PRIMARY, supplementary=list(SUPPLEMENTARY), not_run=NOT_RUN),
              open(os.path.join(ROOT, "results", "bind", "b0.json"), "w", encoding="utf-8"), indent=1)
    for k in ("d0_all", "d0_seen", "d0_heldout", "d1_all", "d1_seen", "d1_heldout", "d2_all", "d2_seen", "d2_heldout",
              "d2_body_seen_all", "d2_heldout_body_unseen", "caddy_units"):
        v = cells.get(k)
        if not v:
            continue
        print(f"[b0 read] {k}: n {v['n']} ({len(v['tasks'])} tasks)", flush=True)
        for m in methods:
            h = v["hit"][m]
            print(f"[b0 read]     {m:28s} {h['mean']} [{h['lo']}, {h['hi']}]" + (f" | parsed {v['parsed'][m]}" if m in v["parsed"] else ""), flush=True)
    for k, v in sorted(pt.items()):
        print(f"[b0 read] {k}: {v['phrase']!r} -> {v['target']} d{v['depth']} seen {v['seen']} body_seen {v['body_seen']}: "
              + " ".join(f"{m.replace('__', '/')} {x:.2f}" for m, x in v["hits"].items()), flush=True)
    for c, b in bars.items():
        print(f"[b0 read] BAR {c} ({b['role']}): i {b['i_depth1_heldout']['mean']} {b['i_pass']} | ii {b['ii_depth2_body_seen']['mean']} "
              f"{b['ii_pass']} | iii {b['iii_depth0']['mean']} vs {gd0} {b['iii_pass']} | PASS {b['pass']} | caddy {b['caddy']['mean']} | "
              f"parse {b['parse_rate']}", flush=True)
    print(f"[b0 read] VERDICT {verdict}", flush=True)
    if a.record:
        from lifelong_ood import ledger
        print(ledger.record("b0_vlm", bench="libero_90_c", tag="read", seed=0, command="python experiments/b0_read.py --record",
                            metrics=dict(verdict=verdict, bars=bars, cells=cells, not_run=NOT_RUN),
                            bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0),
                            config=dict(note="B0 pretrained VLM backbones as binders (S1 hit metric)"), script="experiments/b0_read.py"))


if __name__ == "__main__":
    main()
