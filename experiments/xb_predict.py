"""XB frozen-predictor predictions (experiments/prereg/xb_pred.txt).  CPU, numpy only.  Run AFTER the partner rollouts and BEFORE any held-out rollout.
Writes results/bench/xb/pred_<bench>.json (refuses to overwrite) and prints the sha256 of every frozen file and of the output.

  python experiments/xb_predict.py <bench>      bench in mesa | bimesa | robocasa | robotwin

Predictors (all frozen before the pre-registration): v0 cue vote (fp_build.vote_score), v1 = results/flawed/fp_dev/v1_frozen.json (sha256 6bd3f414...), b2 = base rate only.
v2 (f00ec7c1...) needs the language-attention probe, which exists for no xb model: not applied (logged; both frozen file hashes are verified).
Features per episode, from the policy-free reset state: f_novel (held-out cell); f_obj_targets / f_tgt_holders (training partner counts, log in the model); f_tc_conflict (the object has
a trained partner receptacle other than the instructed one); f_nearer_conf / f_near_margin / f_instr_rank (object-to-receptacle xy distances at reset over the receptacles in the scene);
f_sal_* not available (v1 impute value); f_lang = f_occl = 0 (v1 coefficient 0); f_base_rate = the model's success on this benchmark's partner cells (pooled episodes).
A feature a benchmark cannot supply (RoboTwin: no competing receptacle) takes v1's stored impute value (= training mean).  Cell score = mean over the cell's episodes.
"""
import glob, hashlib, importlib.util, json, os, sys
import numpy as np

ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
XB = ROOT + "/results/bench/xb"
FD = ROOT + "/results/flawed/fp_dev"
AUD = ROOT + "/results/bench"
DOCS = "/coc/flash10/ksingara3/robot/bench/mesa/mesa-env/docs/mesa_bench/"
MODELS = {"mesa": ["pi0", "pi05", "pi0fast", "groot"], "bimesa": ["pi0", "pi05", "pi0fast", "groot"], "robocasa": ["pi0", "pi05", "groot15", "xvla"], "robotwin": ["xvla", "pi05", "smolvla"]}
RKEYS = ["tray", "plate", "bowl", "basket", "pan", "cutting_board", "pot", "baking", "drainer", "box", "cabinet", "microwave", "sink", "stove", "oven", "drawer", "fridge", "blender", "toaster", "dishwasher"]
FEATS = None


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def load_frozen():
    out = {}
    for tag in ("v1", "v2"):
        p = f"{FD}/{tag}_frozen.json"; want = open(f"{FD}/{tag}_frozen.sha256").read().split()[0]; got = sha(p)
        assert got == want, f"{tag} frozen file hash mismatch {got} != {want}"
        out[tag] = (json.load(open(p)), got)
        print(f"[sha256] {p} {got}  (matches {tag}_frozen.sha256)", flush=True)
    assert out["v1"][1].startswith("6bd3f414") and out["v2"][1].startswith("f00ec7c1")
    return out


def v1_prob(F, feats):
    names = F["features"]
    x = np.array([feats.get(n, np.nan) for n in names], float)
    for n in ("f_obj_targets", "f_tgt_holders"):
        i = names.index(n); x[i] = np.log(x[i]) if x[i] > 0 else np.nan
    x[names.index("f_sal_missing")] = 1.0
    x = np.where(np.isnan(x), np.array(F["impute"], float), x)
    z = F["intercept"] + sum(F["coef"][n] * (x[i] - F["mean"][i]) / F["std"][i] for i, n in enumerate(names))
    return float(1 / (1 + np.exp(-z)))


def vote_score(votes, instr, ncues):                         # copied from fp_build.py
    other = max([v for k, v in votes.items() if k != instr] + [0])
    return 0.5 + 0.5 * (other - votes.get(instr, 0)) / max(ncues, 1)


def kc(s):
    return s.lower().replace(" ", "_").replace("-", "_")


def find(cats, key):
    toks = kc(key).split("_")
    return [n for n, c in cats.items() if all(t in kc(c) or t in kc(n) for t in toks)]


def geometry(o_xy, tgt_xys, other_xys):
    """-> nearer_conf, near_margin, instr_rank (fp_build definitions)."""
    dt = min(np.linalg.norm(o_xy - t) for t in tgt_xys)
    do = [float(np.linalg.norm(o_xy - p)) for p in other_xys]
    if not do:
        return dict(f_nearer_conf=0.0, f_near_margin=0.0, f_instr_rank=0.0), None
    allr = sorted([dt] + do)
    return dict(f_nearer_conf=float(min(do) < dt), f_near_margin=float(np.clip(dt - min(do), -0.5, 0.5)), f_instr_rank=float(allr.index(dt))), int(np.argmin(do))


class Ctx:
    pass


def trained_maps_mesa(bench):
    spec = importlib.util.spec_from_file_location("bam", ROOT + "/experiments/bench_audit_mesa.py"); bam = importlib.util.module_from_spec(spec); spec.loader.exec_module(bam)
    md = bam.md_tables(DOCS + ("task_suites.md" if bench == "mesa" else "task_suites_bimanual.md"))
    sec = "MESA-70 (`mesa-70`)" if bench == "mesa" else "BiMESA-ID (`bimesa-id`)"
    o2r, r2o = {}, {}
    for tid, ins in md[sec]:
        p = bam.parse(ins)
        if not p:
            continue
        objs, fx, region, var = p; rf = fx + (":" + region if region else "")
        for o in objs:
            o2r.setdefault(o, set()).add(rf); r2o.setdefault(rf, set()).add(o)
    return o2r, r2o


def main():
    bench = sys.argv[1]
    dry = os.environ.get("XB_PRED_DRY") == "1"
    out = f"{XB}/dry/pred_{bench}_dry.json" if dry else f"{XB}/pred_{bench}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    assert not os.path.exists(out), "never overwrite"
    frozen = load_frozen(); V1 = frozen["v1"][0]
    cells = json.load(open(f"{XB}/cells_{bench}.json"))
    res = dict(bench=bench, frozen_sha256={"v1": frozen["v1"][1], "v2": frozen["v2"][1]}, v2_applied=False, v2_note="needs the language-attention probe; none exists for any xb model",
               prereg_sha256=sha(ROOT + "/experiments/prereg/xb_pred.txt"), models={}, notes=[])
    # ---------------- cell definitions: cid -> dict(ctype, objs, rec, trained, n_obj, n_rec) and episode providers
    cd = {}
    if bench in ("mesa", "bimesa"):
        o2r, r2o = trained_maps_mesa(bench)
        for ct, info in (("heldout", cells["heldout_info"]), ("partner", cells["partner_info"])):
            for t, d in info.items():
                cd[t] = dict(ctype=ct, objs=d["objects"], rec=d["receptacle"], trained=sorted({r for o in d["objects"] for r in o2r.get(o, set())}),
                             n_obj=min(len(o2r.get(o, ())) for o in d["objects"]), n_rec=len(r2o.get(d["receptacle"], ())))
        eps_of = lambda m, cid, c: ([json.load(open(f)) for f in sorted(glob.glob(f"{XB}/roll/{bench}/{m}/{cid}/ep*.json"))] if c["ctype"] == "partner"
                                    else [json.load(open(f)) for f in sorted(glob.glob(f"{XB}/reset/{bench}/{cid}/ep*.json"))])
    elif bench == "robocasa":
        aud = json.load(open(f"{AUD}/robocasa/audit/pepijn223__robocasa_pretrain_human300_v4.json"))
        pc = {}
        for k, n in aud["pair_object_family_counts"].items():
            o, fam = k.split(" || ")
            if n > 0:
                pc.setdefault(o, {})[fam] = n
        fam_objs = {}
        for o, d in pc.items():
            for fam in d:
                fam_objs.setdefault(fam, set()).add(o)
        final = json.load(open(f"{XB}/cells_robocasa_final.json"))["families"]
        for fam, d in final.items():
            for ct in ("heldout", "partner"):
                for o in d[ct]["cells"]:
                    cid = f"{fam.replace('->', '_to_')}__{o.replace(' ', '_')}"
                    cd[cid] = dict(ctype=ct, objs=[o], rec=fam.split("->")[1], fam=fam, trained=sorted({f.split("->")[1] for f in pc.get(o, {})}), n_obj=len(pc.get(o, {})), n_rec=len(fam_objs.get(fam, ())))
        eps_of = lambda m, cid, c: ([json.load(open(f)) for f in sorted(glob.glob(f"{XB}/roll/robocasa/{m}/{cid}/ep*.json"))] if c["ctype"] == "partner"
                                    else [json.load(open(f)) for f in sorted(glob.glob(f"{XB}/reset/robocasa/{cid}/ep*.json"))])
    else:
        final = json.load(open(f"{XB}/cells_robotwin_final.json"))["tasks"]
        for task, d in final.items():
            aud = json.load(open(f"{AUD}/robotwin/audit/{task}.json"))["instance"]
            pcnt = {tuple(k.split(" || ")): n for k, n in aud["pair_counts"].items()}
            a2b, b2a = {}, {}
            for (a_, b_), n in pcnt.items():
                if n > 0:
                    a2b.setdefault(a_, set()).add(b_); b2a.setdefault(b_, set()).add(a_)
            for ct in ("heldout", "partner"):
                for A, Bb in d[ct]:
                    cid = f"{task}/{A.replace('/', '-')}__{Bb.replace('/', '-')}"
                    cd[cid] = dict(ctype=ct, objs=[A], rec=Bb, trained=sorted(a2b.get(A, ())), n_obj=len(a2b.get(A, ())), n_rec=len(b2a.get(Bb, ())), A=A, B=Bb, task=task)
        eps_of = lambda m, cid, c: ([json.load(open(f)) for f in sorted(glob.glob(f"{XB}/roll/robotwin/{m}/{cid}/ep*.json"))] if c["ctype"] == "partner"
                                    else [dict(reset_pos=s["reset_pos"]) for s in json.load(open(f"{XB}/rt_seeds/{c['task']}/{c['A'].replace('/', '-')}__{c['B'].replace('/', '-')}.json"))["seeds"][:10]])
    def feats_episode(c, e):
        if bench in ("mesa", "bimesa"):
            cats, pos = e["objects_categories"], {n: np.array(p) for n, p in e["reset_pos"].items()}
            ob = [n for n in find(cats, c["objs"][0]) if n in pos]
            tg = [n for n in (find(cats, c["rec"].split(":")[0]) or find(cats, c["rec"].split(":")[0].split()[-1])) if n in pos]
            oth = sorted({n for k in RKEYS for n in find(cats, k) if n in pos and n not in ob and n not in tg})
            if not ob or not tg:
                return None
            g, near = geometry(pos[ob[0]][:2], [pos[t][:2] for t in tg], [pos[n][:2] for n in oth])
            nearest = kc(cats[oth[near]]).replace("robocasa_", "").split("_")[0] if near is not None and g["f_nearer_conf"] else c["rec"].split(":")[0]
            return g, nearest
        if bench == "robocasa":
            pos = {n: np.array(p) for n, p in e["reset_pos"].items()}; fx = e.get("fixtures", {})
            if "obj" not in pos:
                return None
            dkey = {"microwave": "microwave", "sink": "sink", "cabinet": "cab"}[c["rec"]]
            fxx = {n: np.array(v["pos"][:2]) for n, v in fx.items() if any(k in (n + " " + v.get("cls", "")).lower() for k in RKEYS + ["cab"])}
            tg = [n for n in fxx if dkey in n.lower() or dkey in fx[n].get("cls", "").lower()]
            if not tg:
                return None
            o = pos["obj"][:2]; tgn = min(tg, key=lambda n: np.linalg.norm(o - fxx[n]))
            g, near = geometry(o, [fxx[tgn]], [fxx[n] for n in fxx if n != tgn])
            nm = [n for n in fxx if n != tgn]
            nearest = ([k for k in RKEYS if k in nm[near].lower()] or [nm[near].lower()])[0] if near is not None and g["f_nearer_conf"] else c["rec"]
            return g, nearest
        return dict(), None                                               # robotwin: no competing receptacle in the scene
    for m in MODELS[bench]:
        part = [(cid, c) for cid, c in cd.items() if c["ctype"] == "partner"]
        succ = [bool(e["success"]) for cid, c in part for e in eps_of(m, cid, c)]
        if not succ:
            continue
        base = float(np.mean(succ)); cm = dict(base_rate=base, n_partner_eps=len(succ), cells={})
        for cid, c in cd.items():
            eps = eps_of(m, cid, c); p1, p0, matched = [], [], 0
            for e in eps:
                ef = feats_episode(c, e)
                geo, nearest = (ef if ef is not None else ({}, None))
                matched += int(ef is not None)
                tgt = c["rec"].split(":")[0] if bench != "robotwin" else c["rec"]
                trained = c["trained"] if bench == "robotwin" else [r.split(":")[0] for r in c["trained"]]
                tc = float(any(r != tgt for r in trained))
                f = dict(f_novel=float(c["ctype"] == "heldout"), f_obj_targets=max(1, c["n_obj"]), f_tgt_holders=max(1, c["n_rec"]), f_tc_conflict=tc, f_lang=0.0, f_occl=0.0, f_base_rate=base, **geo)
                votes = {}
                if nearest:
                    votes[nearest] = 1
                for r in trained:
                    votes[r] = votes.get(r, 0) + 1
                p1.append(v1_prob(V1, f)); p0.append(vote_score(votes, tgt, 2))
            cm["cells"][cid] = dict(ctype=c["ctype"], n_episodes=len(eps), n_feature_matched=matched, v1=float(np.mean(p1)) if p1 else None, v0=float(np.mean(p0)) if p0 else None, b2=1 - base,
                                    object_partner_count=c["n_obj"], receptacle_partner_count=c["n_rec"])
        res["models"][m] = cm
        print(f"[{bench}/{m}] base rate {base:.3f} over {len(succ)} partner episodes; cells {len(cm['cells'])}; feature-matched episodes "
              f"{sum(v['n_feature_matched'] for v in cm['cells'].values())}/{sum(v['n_episodes'] for v in cm['cells'].values())}", flush=True)
    with open(out, "x") as fh:
        json.dump(res, fh, indent=1)
    print(f"[sha256] {out} {sha(out)}", flush=True)


if __name__ == "__main__":
    main()
