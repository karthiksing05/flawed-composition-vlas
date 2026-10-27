"""PLAN_BINDER B1.3 (docs/PLAN_BINDER.md section 13): freeze the generator's MCD split and layout plan (b1_3_gen.py; seeded, no
rendering).  Chunks env, CPU.  Also lists every LIBERO task that is NOT a LIBERO-90-C training cell (all five suites; read from
the BDDL files and results/bind/s1/tasks.json's training list; results/l90c/split_sealed.json is never opened), with its scene,
family, object / fixture types and role phrases, so b1_3_units.py can keep their scene + phrase combinations out of the
generated training pool.

    python experiments/b1_3_plan.py  -> results/bind/b1_3/plan.json
"""
from __future__ import annotations

import collections
import glob
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
B13 = os.path.join(ROOT, "results", "bind", "b1_3")
BDDL = "/coc/flash10/ksingara3/robot/openpi/third_party/libero/libero/libero/bddl_files"
VERB = re.compile(r"^(open|close|turn on|turn off|pick up|push|put|place|stack)\s+")
FAM_OF = {"KITCHEN": "kitchen", "LIVING_ROOM": "living", "STUDY": "study"}


def heldout_libero():
    import s1_labels as SL
    from lifelong_ood.policy import primitives as P
    tasks = json.load(open(os.path.join(ROOT, "results", "bind", "s1", "tasks.json"), encoding="utf-8"))
    train = {(x["scene"], " ".join(x["lang"].lower().split())) for x in tasks.values() if x["set"] == "train"}
    out, n_train_files = [], 0
    for f in sorted(glob.glob(os.path.join(BDDL, "*", "*.bddl"))):
        txt = open(f, encoding="utf-8").read()
        lang = " ".join(re.search(r"\(:language ([^)]*)\)", txt).group(1).lower().split())
        base = os.path.basename(f)
        m = re.match(r"^([A-Z_]+_SCENE\d+)_", base)
        scene = m.group(1) if m else os.path.basename(os.path.dirname(f)) + ":" + base
        if (scene, lang) in train:
            n_train_files += 1
            continue
        prob = re.search(r"\(problem (\w+)\)", txt).group(1)
        fam = next((v for k, v in FAM_OF.items() if scene.startswith(k + "_SCENE")), None)
        if fam is None and prob == "LIBERO_Tabletop_Manipulation":
            fam = "tabletop"
        types = sorted(set(re.findall(r" - ([a-z_0-9]+)\s*$", txt, flags=re.M)))
        ph = {"object": [], "goal": []}
        for c in P.steps_of(lang):
            o, g = SL.split_step(c)
            ph["object"].append(SL.norm(VERB.sub("", o.strip())))
            if g:
                ph["goal"].append(SL.norm(g))
        out.append(dict(file=os.path.relpath(f, BDDL), scene=scene, family=fam, problem=prob, lang=lang, types=types, phrases=ph))
    return out, n_train_files


def main():
    import b1_3_gen as G
    sp = G.plan_split()
    lays = G.plan_layouts(sp)
    for k, L in enumerate(lays):
        L["seed"] = 9_400_000 + k
    ho, ntr = heldout_libero()
    cells = collections.Counter((t["depth"], t["bin"]) for t in sp["test"])
    roles = collections.Counter((t["depth"], t["bin"], t["tuple"][0]) for t in sp["test"])
    os.makedirs(B13, exist_ok=True)
    out = dict(split=sp, layouts=lays, heldout_libero=ho, n_training_task_files=ntr,
               cells={f"d{d}_{b}": n for (d, b), n in sorted(cells.items())},
               roles={f"d{d}_{b}_{r}": n for (d, b, r), n in sorted(roles.items())})
    json.dump(out, open(os.path.join(B13, "plan.json"), "w", encoding="utf-8"), indent=1)
    print(f"[b1.3 plan] restart {sp['restart']}: D_A(plan, depth-matched) {sp['d_atom_plan']:.4f}; {len(sp['held_pairs'])} held-out pairs; "
          f"{len(sp['test'])} test tuples; {len(sp['train_tuples'])} training tuples of {sp['n_feasible']}", flush=True)
    print(f"[b1.3 plan] cells {out['cells']}", flush=True)
    print(f"[b1.3 plan] roles {out['roles']}", flush=True)
    print(f"[b1.3 plan] layouts: {collections.Counter(L['set'] for L in lays)}; held-out LIBERO task files {len(ho)} "
          f"(training-task files skipped {ntr})", flush=True)
    for t in sp["test"]:
        print(f"[b1.3 plan]   test d{t['depth']} bin {t['bin']} u {t['u']:.2f} {t['tuple']}", flush=True)
    print(f"[b1.3 plan] held pairs: {sp['held_pairs']}", flush=True)


if __name__ == "__main__":
    main()
