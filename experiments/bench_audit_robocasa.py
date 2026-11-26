"""RoboCasa365 pick-and-place co-occurrence audit: (object category, source fixture -> destination fixture) in the data the
released checkpoints were trained on (CPU only; only LeRobot meta/ files are downloaded, no videos, no simulator rollouts).

The per-episode language instruction of the PickPlace atomic tasks is "Pick the <obj> from the <src> and place it in/on the <dst>."
(robocasa/environments/kitchen/atomic/kitchen_pick_place.py), so (object, src, dst) is recovered from the instruction text and
the per-episode task assignment in meta/episodes*.  No ep_meta.json is needed (those live in the heavy extras/ folders).

Datasets (HF):
  pepijn223/robocasa_pretrain_human300_v4        card-cited training set of the community X-VLA (deksprime/xvla-base-robocasa-human300-ri);
                                                 same 300-task pretrain-human data as the official multitask pi0 / pi0.5 / GR00T N1.5
  ember-lab-berkeley/robocasa365-pretrain-atomic / -composite   LeRobot v3 mirror of the official pretrain human data (cross-check)
  lerobot/robocasa_target_human_unified          trained on by lerobot/smolvla_robocasa (target split: 50 tasks, 10 target kitchens)
Output: results/bench/robocasa/audit/ (refuses to overwrite).
"""
import collections, glob, json, os, re, subprocess

REPO = "/coc/flash10/ksingara3/flawed-composition-vlas"
B = "/coc/flash10/ksingara3/robot/bench"
OUT = REPO + "/results/bench/robocasa/audit"
META = B + "/robocasa/meta"
VENV_PY = B + "/robocasa/venv_sim/bin/python"
DATASETS = ["pepijn223/robocasa_pretrain_human300_v4", "ember-lab-berkeley/robocasa365-pretrain-atomic",
            "ember-lab-berkeley/robocasa365-pretrain-composite", "lerobot/robocasa_target_human_unified"]
MIN_PARTNER = 5

PAT = re.compile(r"^pick (?:up )?the (.+?) from (?:the |a )?(.+?) and place it (?:in|on|into|onto|next to|inside)(?: the| a)? (.+?)\.?$", re.I)
FIXTURES = ["fridge drawer", "fridge shelf", "toaster oven", "stand mixer", "microwave", "blender", "counter", "cabinet", "sink",
            "stove", "oven", "drawer", "toaster", "plate", "pan", "pot", "bowl", "basket", "tray", "rack", "shelf", "cutting board"]


def fx(s):
    s = s.lower()
    for f in FIXTURES:
        if f in s:
            return f
    return s.split()[-1] if s.split() else s


def load(d):
    import pandas as pd
    root = f"{META}/{d.replace('/', '__')}/meta"
    df = pd.read_parquet(root + "/tasks.parquet")
    if "task" in df.columns:
        tasks = {int(r.task_index): r.task for r in df.itertuples()}
    else:
        tasks = {int(v): k for k, v in zip(df.index, df["task_index"])}
    cnt = collections.Counter(); n_ep = 0
    for f in sorted(glob.glob(root + "/episodes/**/*.parquet", recursive=True)):
        e = pd.read_parquet(f); n_ep += len(e)
        if "tasks" in e.columns:
            for ts in e["tasks"]:
                for t in (list(ts) if not isinstance(ts, str) else [ts]):
                    cnt[t] += 1
        elif "task_index" in e.columns:
            for i in e["task_index"]:
                cnt[tasks[int(i)]] += 1
    return tasks, cnt, n_ep


def registry():
    code = r"""
import json
from robocasa.models.objects.kitchen_objects import OBJ_CATEGORIES, OBJ_GROUPS
out = {}
for c, regs in OBJ_CATEGORIES.items():
    out[c] = {"registries": sorted(regs.keys()), "graspable": {r: bool(getattr(m, "graspable", False)) for r, m in regs.items()},
              "is_group_key": c in OBJ_GROUPS}
print("REG_JSON" + json.dumps({"cats": out, "groups": {g: len(v) for g, v in OBJ_GROUPS.items()}}))
"""
    p = subprocess.run([VENV_PY, "-c", code], capture_output=True, text=True, cwd=B + "/tmp", env={**os.environ, "MUJOCO_GL": "egl"})
    for line in p.stdout.splitlines():
        if line.startswith("REG_JSON"):
            return json.loads(line[8:])
    print("registry dump failed:", p.stderr[-800:]); return None


def main():
    from huggingface_hub import snapshot_download
    os.makedirs(OUT, exist_ok=False)  # never overwrite
    for d in DATASETS:
        snapshot_download(d, repo_type="dataset", local_dir=f"{META}/{d.replace('/', '__')}", allow_patterns=["meta/*", "meta/**/*"])
    reg = registry()
    if reg:
        json.dump(reg, open(OUT + "/object_registry.json", "w"), indent=1)
    summary = {}
    for d in DATASETS:
        tasks, cnt, n_ep = load(d)
        parsed, unparsed = collections.Counter(), collections.Counter()
        templ = collections.Counter()
        for t, n in cnt.items():
            m = PAT.match(t.strip())
            if not m:
                unparsed[t] += n; continue
            o, s, dd = m.group(1).strip().lower(), fx(m.group(2)), fx(m.group(3))
            parsed[(o, s, dd)] += n
            templ[(s, dd)] += n
        tot = sum(parsed.values())
        obj = collections.Counter(); dst = collections.Counter(); fam = collections.Counter(); od = collections.Counter(); osd = collections.Counter()
        for (o, s, dd), n in parsed.items():
            obj[o] += n; dst[dd] += n; fam[(s, dd)] += n; od[(o, dd)] += n; osd[(o, s + "->" + dd)] += n
        print(f"\n=== {d}: task strings {len(tasks)}, episodes {n_ep}, pick-place-parsed episodes {tot} in {len(fam)} (src,dst) families, "
              f"{len(obj)} object strings; unparsed episodes {sum(unparsed.values())}", flush=True)
        print("  families:", dict(fam.most_common(40)))
        print("  top objects:", dict(obj.most_common(25)))
        print("  unparsed examples:", list(unparsed.items())[:8])
        # candidate held-out cells: (object, dst) with zero count, object >= MIN_PARTNER episodes elsewhere, dst >= MIN_PARTNER with other objects
        cells = []
        fam_list = sorted(fam)
        for o in sorted(obj):
            if obj[o] < MIN_PARTNER:
                continue
            for (s, dd) in fam_list:
                key = (o, s + "->" + dd)
                if osd[key] == 0 and fam[(s, dd)] >= MIN_PARTNER:
                    cells.append({"object": o, "family": s + "->" + dd, "object_episodes_total": obj[o],
                                  "family_episodes": fam[(s, dd)],
                                  "object_partner_families": sorted([(k[1], c) for k, c in osd.items() if k[0] == o], key=lambda t: -t[1])[:6],
                                  "family_partner_objects": sorted([(k[0], c) for k, c in osd.items() if k[1] == s + "->" + dd], key=lambda t: -t[1])[:6]})
        # registry realizability: category string must be a graspable category in some registry
        real = None
        if reg:
            cats = reg["cats"]
            def ok(o):
                c = o.replace(" ", "_")
                return c in cats and any(cats[c]["graspable"].values())
            real = [c for c in cells if ok(c["object"])]
        # per-family realizable counts
        per_fam = collections.Counter(c["family"] for c in (real if real is not None else cells))
        json.dump({"dataset": d, "n_episodes": n_ep, "pickplace_episodes": tot, "families": {f"{s}->{dd}": n for (s, dd), n in fam.items()},
                   "object_episodes": dict(obj), "dst_episodes": dict(dst), "pair_object_family_counts": {f"{o} || {k}": n for (o, k), n in osd.items()},
                   "unparsed_top": unparsed.most_common(40), "n_zero_cells_object_ge5": len(cells), "n_zero_cells_realizable_registry": None if real is None else len(real),
                   "zero_cells_per_family_realizable": dict(per_fam), "zero_cells": cells[:5000]},
                  open(f"{OUT}/{d.replace('/', '__')}.json", "w"), indent=1)
        summary[d] = {"n_episodes": n_ep, "pickplace_episodes": tot, "n_families": len(fam), "n_objects": len(obj),
                      "n_zero_cells": len(cells), "n_zero_cells_realizable": None if real is None else len(real),
                      "per_family": dict(per_fam)}
        print(f"  zero-count (object, family) cells with object>={MIN_PARTNER} eps and family>={MIN_PARTNER}: {len(cells)}; "
              f"graspable-category realizable: {None if real is None else len(real)}; per family {dict(per_fam)}", flush=True)
    json.dump(summary, open(OUT + "/summary.json", "w"), indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
