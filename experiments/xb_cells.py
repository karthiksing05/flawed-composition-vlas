"""XB cell selection (experiments/prereg/xb_pred.txt).  CPU only; reads results/bench/*/audit; writes results/bench/xb/cells_<bench>.json (refuses to overwrite).
Benchmarks: mesa, bimesa, robocasa, robotwin.   Draw rules follow the pre-registration exactly:
  mesa / bimesa : held-out = composite tasks whose every (object, receptacle) slot is held out against the training set (fine receptacle key);
                  partners = every training-suite task sharing a held-out task's object or receptacle.  50 episodes per task, fixed init states 0-49.
  robocasa      : families counter->{microwave, sink, cabinet}; per family np.random.default_rng(0) draws 10 held-out cells from zero-count (object category,
                  family) cells; np.random.default_rng(1) draws 10 partner cells from cells with count >= 3 in the same family.  The full permutation is stored:
                  a placeability failure (3 resets) replaces the cell by the next draw (done by xb_roll_robocasa.py --check, which writes cells_robocasa_final.json).
  robotwin      : place_a2b_left / place_a2b_right; per task rng(0) draws 20 held-out instance pairs, rng(1) draws 20 partner pairs (count >= 2); permutations stored
                  for the expert-seed/forced-pair replacement rule.
Draw = rng.permutation(n_candidates) over candidates sorted lexicographically; the first k are the cells, later ones are replacements.
"""
import importlib.util, json, os, re, sys
import numpy as np

ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
B = "/coc/flash10/ksingara3/robot/bench"
AUD = ROOT + "/results/bench"
OUT = AUD + "/xb"
DOCS = B + "/mesa/mesa-env/docs/mesa_bench/"

spec = importlib.util.spec_from_file_location("bench_audit_mesa", ROOT + "/experiments/bench_audit_mesa.py")
bam = importlib.util.module_from_spec(spec); spec.loader.exec_module(bam)   # parse(), md_tables()


def wjson(name, obj):
    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/{name}", "x", encoding="utf-8") as fh:     # never overwrite
        json.dump(obj, fh, indent=1)


def key_of(p):
    objs, fx, region, var = p
    return objs, fx + (":" + region if region else "")


def mesa_like(name, comp_cells_file, train_section, train_md, comp_section, comp_md, ep_files, eval_ho, eval_pa, train_ds):
    cells = json.load(open(f"{AUD}/mesa/audit/{comp_cells_file}"))
    by_task = {}
    for r in cells:
        by_task.setdefault(r["task"], []).append(r)
    held = [t for t, rs in by_task.items() if all(r.get("parsed") and r["held_out_fine"] for r in rs)]
    comp_instr = dict(bam.md_tables(DOCS + comp_md)[comp_section])
    train_instr = dict(bam.md_tables(DOCS + train_md)[train_section])
    pc = json.load(open(f"{AUD}/mesa/audit/partner_counts__{train_ds}.json"))
    ho_info = {}
    for t in held:
        rs = by_task[t]
        objs = [r["object"] for r in rs]; rec = rs[0]["fixture"] + (":" + rs[0]["region"] if rs[0]["region"] else "")
        n_obj = min(pc["object_to_n_receptacles"].get(o, 0) for o in objs)
        n_rec = pc["receptacle_to_n_objects"].get(rec, 0)
        ho_info[t] = dict(instruction=comp_instr[t], objects=objs, receptacle=rec, variant=rs[0]["variant"], object_partner_count=n_obj, receptacle_partner_count=n_rec,
                          partners_objects={r["object"]: r["partners_same_object_other_receptacle"] for r in rs}, partners_receptacle=rs[0]["partners_same_receptacle_other_object"])
    # training tasks parsed
    tr = {}
    for tid, ins in train_instr.items():
        p = bam.parse(ins)
        if p:
            tr[tid] = key_of(p) + (p[3], ins)
    partners = {}
    for tid, (objs, rec, var, ins) in tr.items():
        for t, h in ho_info.items():
            share_obj = sorted(set(objs) & set(h["objects"])); share_rec = rec == h["receptacle"]
            if share_obj or share_rec:
                d = partners.setdefault(tid, dict(instruction=ins, objects=objs, receptacle=rec, variant=var, for_heldout=[]))
                d["for_heldout"].append(dict(task=t, shares_object=share_obj, shares_receptacle=share_rec))
    for tid, d in partners.items():
        d["object_partner_count"] = min(pc["object_to_n_receptacles"].get(o, 0) for o in d["objects"])
        d["receptacle_partner_count"] = pc["receptacle_to_n_objects"].get(d["receptacle"], 0)
    unparsed = sorted(set(train_instr) - set(tr))
    out = dict(bench=name, rule="held-out = composite tasks with every slot held_out_fine against the training set; partners = training tasks sharing a held-out task's object or fine receptacle",
               n_episodes_per_task=50, init_states="0-49 (the suite's fixed)", eval_set_heldout=eval_ho, eval_set_partner=eval_pa, train_dataset=train_ds,
               n_heldout=len(held), heldout_tasks=sorted(held), heldout_info=ho_info, n_partner=len(partners), partner_tasks=sorted(partners), partner_info=partners,
               training_tasks_unparsed=unparsed, excluded_composite=sorted(set(by_task) - set(held)))
    wjson(f"cells_{name}.json", out)
    print(f"{name}: held-out {len(held)} partners {len(partners)} (training tasks parsed {len(tr)}/{len(train_instr)}; unparsed {unparsed})")
    return out


def robocasa():
    fam_task = {"counter->microwave": "PickPlaceCounterToMicrowave", "counter->sink": "PickPlaceCounterToSink", "counter->cabinet": "PickPlaceCounterToCabinet"}
    dfile = f"{AUD}/robocasa/audit/pepijn223__robocasa_pretrain_human300_v4.json"
    d = json.load(open(dfile))
    reg = json.load(open(f"{AUD}/robocasa/audit/object_registry.json"))["cats"]
    ok = lambda o: (o.replace(" ", "_") in reg) and any(reg[o.replace(" ", "_")]["graspable"].values())
    pair = {}
    for k, n in d["pair_object_family_counts"].items():
        o, fam = k.split(" || ")
        pair[(o, fam)] = n
    objn = d["object_episodes"]
    out = {"bench": "robocasa", "datafile": dfile, "rule": "see module docstring", "episodes_per_cell": 20, "seeds": "0-19", "families": {}}
    zero_counts = {}
    for fam in fam_task:
        zc = sorted(c["object"] for c in d["zero_cells"] if c["family"] == fam and ok(c["object"]) and objn.get(c["object"], 0) >= 5)
        zero_counts[fam] = len(zc)
    # audit names: the three families all present; keep all three (the rule picks the three with most zero cells among exactly these three)
    for fam, task in fam_task.items():
        zc = sorted({c["object"] for c in d["zero_cells"] if c["family"] == fam and ok(c["object"]) and objn.get(c["object"], 0) >= 5})
        pc = sorted(o for (o, f), n in pair.items() if f == fam and n >= 3 and ok(o))
        h_perm = np.random.default_rng(0).permutation(len(zc)); p_perm = np.random.default_rng(1).permutation(len(pc))
        out["families"][fam] = dict(task=task, n_zero_candidates=len(zc), n_partner_candidates=len(pc),
                                    heldout_draw=[zc[i] for i in h_perm], partner_draw=[pc[i] for i in p_perm],
                                    heldout_first10=[zc[i] for i in h_perm[:10]], partner_first10=[pc[i] for i in p_perm[:10]],
                                    partner_counts={o: pair[(o, fam)] for o in pc},
                                    heldout_object_partners={o: sorted([(f, n) for (oo, f), n in pair.items() if oo == o and n > 0], key=lambda t: -t[1]) for o in zc},
                                    object_episodes={o: objn.get(o, 0) for o in set(zc) | set(pc)},
                                    family_n_objects=len({o for (o, f), n in pair.items() if f == fam and n > 0}))
        print(f"robocasa {fam}: zero candidates {len(zc)}, partner candidates {len(pc)}; held-out first10 {out['families'][fam]['heldout_first10']}")
    wjson("cells_robocasa.json", out)


def robotwin():
    out = {"bench": "robotwin", "episodes_per_cell": 10, "seeds_from": 100000, "tasks": {}}
    for task in ("place_a2b_left", "place_a2b_right"):
        d = json.load(open(f"{AUD}/robotwin/audit/{task}.json"))["instance"]
        pc = {tuple(k.split(" || ")): n for k, n in d["pair_counts"].items()}
        cat = lambda s: s.split("/")[0]
        zero = sorted((c["A"], c["B"]) for c in d["zero_cells"] if cat(c["A"]) != cat(c["B"]) and d["A_episodes"][c["A"]] >= 5 and d["B_episodes"][c["B"]] >= 5)
        part = sorted(p for p, n in pc.items() if n >= 2 and cat(p[0]) != cat(p[1]))
        hp = np.random.default_rng(0).permutation(len(zero)); pp = np.random.default_rng(1).permutation(len(part))
        out["tasks"][task] = dict(n_zero_candidates=len(zero), n_partner_candidates=len(part),
                                  heldout_draw=[list(zero[i]) for i in hp], partner_draw=[list(part[i]) for i in pp],
                                  heldout_first20=[list(zero[i]) for i in hp[:20]], partner_first20=[list(part[i]) for i in pp[:20]],
                                  pair_count={f"{a} || {b}": pc.get((a, b), 0) for a, b in [tuple(x) for x in [zero[i] for i in hp[:20]] + [part[i] for i in pp[:20]]]},
                                  A_partners={a: d["A_to_n_B_partners"][a] for a in d["A_to_n_B_partners"]}, B_partners=d["B_to_n_A_partners"],
                                  A_episodes=d["A_episodes"], B_episodes=d["B_episodes"])
        print(f"robotwin {task}: zero candidates {len(zero)}, partner candidates {len(part)}")
    wjson("cells_robotwin.json", out)


def _main():
    which = sys.argv[1:] or ["mesa", "bimesa", "robocasa", "robotwin"]
    if "mesa" in which:
        mesa_like("mesa", "cells__mesa-composite__vs__mesa-70-lerobot.json", "MESA-70 (`mesa-70`)", "task_suites.md", "MESA-Composite (`mesa-composite`)", "task_suites.md",
                  None, "mesa-composite", "mesa-70", "mesa-70-lerobot")
    if "bimesa" in which:
        mesa_like("bimesa", "cells__bimesa-composite__vs__bimesa-57-lerobot.json", "BiMESA-ID (`bimesa-id`)", "task_suites_bimanual.md", "BiMESA-Composite (`bimesa-composite`)",
                  "task_suites_bimanual.md", None, "bimesa-composite", "bimesa-id", "bimesa-57-lerobot")
    if "robocasa" in which:
        robocasa()
    if "robotwin" in which:
        robotwin()
    if "robotwin_final" in which:
        robotwin_final()


def robotwin_final():
    """After the policy-free seed prep: the first 20 draws per (task, cell type) whose prep file shows 10 expert-solvable seeds (replacement = next draw; logged)."""
    cells = json.load(open(f"{OUT}/cells_robotwin.json"))["tasks"]
    out = {"bench": "robotwin", "rule": "first 20 draws with a feasible prep file (10 expert-solved seeds, forced pair verified)", "tasks": {}, "log": {}}
    for task, d in cells.items():
        out["tasks"][task] = {}
        for ctype, draw in (("heldout", d["heldout_draw"]), ("partner", d["partner_draw"])):
            keep, log = [], []
            for A, B in draw:
                fp = f"{OUT}/rt_seeds/{task}/{A.replace('/', '-')}__{B.replace('/', '-')}.json"
                if not os.path.exists(fp):
                    log.append(dict(A=A, B=B, status="no prep file")); continue
                r = json.load(open(fp))
                log.append(dict(A=A, B=B, status="ok" if r["feasible"] else "infeasible", n_seeds=r["n"], n_skipped=len(r["skipped"])))
                if r["feasible"] and len(keep) < 20:
                    keep.append([A, B])
            out["tasks"][task][ctype] = keep
            out["log"][f"{task}/{ctype}"] = [l for l in log if l["status"] != "ok" or [l["A"], l["B"]] in keep]
            print(task, ctype, "kept", len(keep), "of", len(log), "examined; replaced:", [l for l in log if l["status"] == "infeasible"][:5])
    with open(f"{OUT}/cells_robotwin_final.json", "x") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    _main()
