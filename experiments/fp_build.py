"""Failure-predictor development table (docs/PLAN_GOAL3.md pillar B).  CPU, numpy only; run through sbatch.
One row per (model, task, state) from EXISTING frozen-model atlas rollouts (no new rollouts):
  scene 1: results/flawed/pp_atlas (dev states 4-23, 5 models, familiar layout) + pp_atlas_fresh (states 36-55)
  scene 2: results/flawed/s2_roll (standard layout, pi05/pi0/groot, arm I) + s2cue5_roll (pi05 cue layouts A2 PA2 PD2 Sx PSx Sm)
Features use only pre-rollout information (task spec, training index, initial scene geometry).  Labels (success, class) use the stored episode
json + track npz; classes mirror rva1_read.classify (G grasp, W wrong target, O elsewhere, H held to the end, M centimetre miss).
Output: results/flawed/fp_dev/table.csv (never overwritten)."""
import csv, json, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "results", "flawed")
OUT = os.path.join(R, "fp_dev", "table.csv")
MODELS = ["pi05", "pi0", "groot", "xvla", "smolvla"]
KN = ["K1", "K2", "K3", "K4", "K6", "N1", "N2", "N3", "N4", "N5", "N6"]
BOWL, CHEESE, BOTTLE, PLATE = "akita_black_bowl_1", "cream_cheese_1", "wine_bottle_1", "plate_1"
STOVE, CAB, RACK = "flat_stove_1_cook_region", "wooden_cabinet_1_top_side", "wine_rack_1_top_region"
RECEPT = [PLATE, STOVE, CAB, RACK, BOWL]
TRAINED = {CHEESE: [BOWL], BOTTLE: [CAB, RACK], BOWL: [PLATE, STOVE, CAB]}          # rva1_read.PARTNERS
OBJ_TARGETS = {CHEESE: 1, BOTTLE: 2, BOWL: 3}                                      # docs/THEORY_COMPOSITION.md partner counts
TGT_HOLDERS = {PLATE: 1, STOVE: 1, CAB: 2, BOWL: 1, RACK: 1}
PARTNER_K = {"N1": ["K6", "K1"], "N2": ["K6", "K2"], "N3": ["K6", "K3"], "N4": ["K4", "K1"], "N5": ["K4", "K2"], "N6": ["K4", "K6"]}  # BREAK_ATLAS 3
D_ARR, D_PART = 0.10, 0.08
S2_MOV = ["porcelain_mug_1", "white_yellow_mug_1", "red_coffee_mug_1", "plate_1", "plate_2"]
S2_TRAINED = {"porcelain_mug_1": "plate_1", "white_yellow_mug_1": "plate_2"}
S2_PARTNER_K = {"N1": ["K1", "K2"], "N2": ["K2", "K1"]}


def jl(p):
    return json.load(open(p, encoding="utf-8"))


def xy(p):
    return np.asarray(p, float)[:2]


def episodes_tracks(dirname, stem):
    """episodes of one (model, tid) file with their track dicts (episode i <-> e{i}); None if missing."""
    jp = os.path.join(dirname, stem + ".json")
    tp = os.path.join(dirname, stem + "_track.npz")
    if not (os.path.exists(jp) and os.path.exists(tp)):
        return None
    j = jl(jp)
    eps = list(j["tasks"].values())[0]["episodes"]
    z = np.load(tp)
    n = int(z["n"])
    tr = [{k: z["e%d_%s" % (i, k)] for k in ("held", "obj", "tgt", "final") if "e%d_%s" % (i, k) in z.files} for i in range(n)]
    return eps, tr


def classify(ep, tr, obj_name, partner_xy):
    """S / G / H / W / O / M from the stored episode + track (mirrors rva1_read.classify; track instead of channel log)."""
    if ep.get("error"):
        return "E", None
    if ep["success"]:
        return "S", None
    held = np.asarray(tr["held"], bool)
    last = min(int(ep["steps"]), len(held) - 1)
    if ep.get("bound") != obj_name or not held[: last + 1].any():
        return "G", None
    if held[last]:
        return "H", None
    t = int(np.nonzero(held[: last + 1])[0][-1])
    o, g = xy(tr["obj"][t]), xy(tr["tgt"][t])
    d = float(np.linalg.norm(o - g))
    if d >= D_ARR:
        near = any(np.linalg.norm(o - xy(p)) < D_PART for p in partner_xy)
        return ("W" if near else "O"), d
    return "M", d


def vote_score(votes, instr, ncues):
    """cue-vote rule: 0.5 + 0.5 * (best non-instructed votes - instructed votes) / ncues  (> .5 = predicts a wrong receptacle)."""
    other = max([v for k, v in votes.items() if k != instr] + [0])
    return 0.5 + 0.5 * (other - votes.get(instr, 0)) / max(ncues, 1), ("W" if other > votes.get(instr, 0) else "S")


def main():
    assert not os.path.exists(OUT), "never overwrite: " + OUT
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    log = []
    meta = jl(os.path.join(R, "n2_tasks.json"))["tasks"]
    key_tid = {t["key"]: t for t in meta}
    geom_dev = jl(os.path.join(R, "rva1", "geom_dev.json"))
    ent = ["akita_black_bowl_1", "cream_cheese_1", "wine_bottle_1", "plate_1", STOVE, CAB, RACK]
    rack_med = np.median([g[RACK] for g in geom_dev.values()], axis=0)
    # ---- scene-1 initial entity positions per trial, assembled from step-0 tracks of the K tasks (no outcome used)
    asm = {}
    for setdir, trials in (("pp_atlas", range(4, 24)), ("pp_atlas_fresh", range(36, 56))):
        for m in MODELS:
            for ck, (nm_o, nm_t) in {"K1": (BOWL, PLATE), "K2": (None, STOVE), "K3": (None, CAB), "K4": (BOTTLE, None), "K6": (CHEESE, BOWL)}.items():
                t = key_tid.get(ck + "_fam")
                if t is None:
                    continue
                et = episodes_tracks(os.path.join(R, setdir), "%s_%d" % (m, t["tid"]))
                if et is None:
                    continue
                for ep, tr in zip(*et):
                    d = asm.setdefault((setdir, ep["trial"]), {})
                    for nm, arr in ((nm_o, "obj"), (nm_t, "tgt")):
                        if nm:
                            d.setdefault(nm, []).append(np.asarray(tr[arr][0], float))
    pos1 = {}
    spread, devdiff = 0.0, 0.0
    for (sd, tri), d in asm.items():
        pp = {n: np.mean(v, 0) for n, v in d.items()}
        spread = max(spread, max(float(np.abs(np.asarray(v) - pp[n]).max()) for n, v in d.items()))
        pp.setdefault(RACK, np.asarray(rack_med))
        pos1[(sd, tri)] = pp
        if sd == "pp_atlas" and str(tri) in geom_dev:
            for n, v in pp.items():
                if n != RACK:
                    devdiff = max(devdiff, float(np.abs(np.asarray(geom_dev[str(tri)][n])[:2] - v[:2]).max()))
    log.append("scene-1 position assembly: %d states, max cross-model spread %.4f m, max |assembled - geom_dev| xy %.4f m (dev states)" % (len(pos1), spread, devdiff))
    rows = []
    # ---------------- scene 1
    for sd, sname in (("pp_atlas", "dev"), ("pp_atlas_fresh", "fresh")):
        for m in MODELS:
            for ck in KN:
                t = key_tid.get(ck + "_fam")
                et = episodes_tracks(os.path.join(R, sd), "%s_%d" % (m, t["tid"])) if t else None
                if et is None:
                    log.append("missing %s %s %s" % (sd, m, ck)); continue
                obj, tgt = t["object"], t["target"]
                for ep, tr in zip(*et):
                    P = pos1.get((sd, ep["trial"]))
                    if P is None:
                        continue
                    o = xy(P[obj])
                    cand = [r for r in RECEPT if r != obj]
                    dist = {r: float(np.linalg.norm(o - xy(P[r]))) for r in cand}
                    di = dist[tgt]
                    dother = min(v for r, v in dist.items() if r != tgt)
                    rank = sorted(dist.values()).index(di)
                    nearest = min(dist, key=dist.get)
                    tr_list = [r for r in TRAINED[obj]]
                    tc = sum(1 for r in tr_list if r != tgt)
                    votes = {nearest: 1}
                    for r in tr_list:
                        votes[r] = votes.get(r, 0) + 1
                    v0s, v0c = vote_score(votes, tgt, 2)
                    cls, d = classify(ep, tr, obj, [xy(P[r]) for r in TRAINED[obj] if r != tgt])
                    rows.append(dict(model=m, scene=1, set=sname, layout="familiar", task=ck, trial=ep["trial"], cluster="1_%s_%d" % (sname, ep["trial"]),
                                     success=int(ep["success"]), cls=cls, f_novel=int(ck.startswith("N")), f_obj_targets=OBJ_TARGETS[obj], f_tgt_holders=TGT_HOLDERS[tgt],
                                     f_tc_conflict=int(tc > 0), f_nearer_conf=int(dother < di), f_near_margin=float(np.clip(di - dother, -0.5, 0.5)), f_instr_rank=rank,
                                     f_sal_conf="", f_lang=0, f_occl=0, v0_score=v0s, v0_cls=v0c, bound=ep.get("bound"), d_release="" if d is None else d))
    # ---------------- scene 2
    s2t = jl(os.path.join(R, "s2", "s2_tasks.json"))
    s2pos = s2t["positions"]
    s2k = {t["key"]: t for t in s2t["tasks"]}
    st5 = jl(os.path.join(R, "s2cue5", "s2cue5_states.json"))
    rowsk = {r["k"]: r for r in st5["rows"]}
    log.append("s2cue5 row keys: %s" % sorted(rowsk[4].keys()))
    def s2_features(layout_xy, seg, obj, tgt, mug_tr):
        ox = xy(layout_xy[obj])
        dL, dR = (float(np.linalg.norm(ox - xy(layout_xy[p]))) for p in ("plate_1", "plate_2"))
        nearer = None if abs(dL - dR) < 0.02 else ("plate_1" if dL < dR else "plate_2")
        sal = None
        if seg is not None:
            a, b = seg["plate_1"]["px"], seg["plate_2"]["px"]
            sal = "plate_1" if a >= 1.5 * b else ("plate_2" if b >= 1.5 * a else None)
        votes, nc = {}, 1
        votes[mug_tr] = 1
        for c in (nearer, sal):
            if c:
                votes[c] = votes.get(c, 0) + 1
                nc += 1
        v0s, v0c = vote_score(votes, tgt, nc)
        other = "plate_2" if tgt == "plate_1" else "plate_1"
        return dict(f_tc_conflict=int(mug_tr != tgt), f_nearer_conf=int(nearer == other) if nearer else 0,
                    f_near_margin=float(np.clip((dL - dR) * (1 if tgt == "plate_1" else -1), -0.5, 0.5)),
                    f_instr_rank=0 if (nearer in (None, tgt)) else 1, f_sal_conf="" if seg is None else int(sal == other), v0_score=v0s, v0_cls=v0c), other
    s2sets = [("std", "pp_s2", [("pi05", "pi05_I"), ("pi0", "pi0_I"), ("groot", "groot_I")], os.path.join(R, "s2_roll"))]
    s2sets += [(m5, "cue5", [("pi05", "pi05_I")], os.path.join(R, "s2cue5_roll", m5)) for m5 in ("A2", "PA2", "PD2", "Sx", "PSx", "Sm")]
    for sname, kind, mlist, d in s2sets:
        for mname, stem in mlist:
            for ck in ("K1", "K2", "N1", "N2"):
                tk = s2k[ck]
                et = episodes_tracks(d, "%s_%d" % (stem, tk["tid"]))
                if et is None:
                    log.append("missing s2 %s %s %s" % (sname, mname, ck)); continue
                obj, tgt = tk["object"], tk["target"]
                for ep, tr in zip(*et):
                    k = ep["trial"]
                    if kind == "cue5":
                        rk = rowsk.get(k)
                        if rk is None or sname not in rk:
                            continue
                        lay = rk[sname]["xy"]; seg = rk[sname].get("seg")
                    else:
                        lay = {n: s2pos[k][i] for i, n in enumerate(S2_MOV)}
                        seg = rowsk[k]["orig"].get("seg") if k in rowsk else None
                    f, other = s2_features(lay, seg, obj, tgt, S2_TRAINED[obj])
                    # class: G wrong mug / H held / W ends near the other plate / M near instructed plate / O
                    held = np.asarray(tr["held"], bool); last = min(int(ep["steps"]), len(held) - 1)
                    if ep["success"]:
                        cls = "S"
                    elif ep.get("bound") != obj or not held[: last + 1].any():
                        cls = "G"
                    elif held[last]:
                        cls = "H"
                    else:
                        fin = np.asarray(tr["final"], float)
                        fo = xy(fin[S2_MOV.index(obj)])
                        dP = float(np.linalg.norm(fo - xy(fin[S2_MOV.index(other)])))
                        dT = float(np.linalg.norm(fo - xy(fin[S2_MOV.index(tgt)])))
                        cls = "W" if dP < D_ARR else ("M" if dT < D_ARR else "O")
                    rows.append(dict(model=mname, scene=2, set=sname, layout=sname, task=ck, trial=k, cluster="2_%d" % k, success=int(ep["success"]), cls=cls,
                                     f_novel=int(ck.startswith("N")), f_obj_targets=1, f_tgt_holders=1, f_lang=0, f_occl=0, bound=ep.get("bound"), d_release="", **f))
    # ---------------- base rate: success on the known partners (leave the state out); the row's own episode is never used
    ks = {}
    for r in rows:
        if r["task"].startswith("K"):
            ks.setdefault((r["model"], r["scene"], r["set"], r["task"]), {})[r["trial"]] = r["success"]
    for r in rows:
        pk = (S2_PARTNER_K if r["scene"] == 2 else PARTNER_K).get(r["task"], [r["task"]])
        vals = []
        for p in pk:
            dd = ks.get((r["model"], r["scene"], r["set"], p), {})
            vals += [v for t, v in dd.items() if t != r["trial"]]
        r["f_base_rate"] = float(np.mean(vals)) if vals else ""
    cols = ["model", "scene", "set", "layout", "task", "trial", "cluster", "success", "cls", "f_novel", "f_obj_targets", "f_tgt_holders", "f_tc_conflict",
            "f_nearer_conf", "f_near_margin", "f_instr_rank", "f_sal_conf", "f_lang", "f_occl", "f_base_rate", "v0_score", "v0_cls", "bound", "d_release"]
    with open(OUT, "x", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})
    from collections import Counter
    log.append("rows %d" % len(rows))
    for key, n in sorted(Counter((r["scene"], r["set"], r["model"]) for r in rows).items()):
        log.append("  n %s = %d" % (key, n))
    log.append("classes: %s" % dict(Counter(r["cls"] for r in rows)))
    with open(os.path.join(R, "fp_dev", "build_log.txt"), "x", encoding="utf-8") as fh:
        fh.write("\n".join(log) + "\n")
    print("\n".join(log))


if __name__ == "__main__":
    main()
