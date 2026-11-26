"""RoboTwin 2.0 (A, B) co-occurrence audit over the released demos (CPU only, no simulator, no rendering).

The released per-task zips (HF dataset TianxingChen/RoboTwin2.0, dataset/<task>/aloha-agilex_{clean_50,randomized_500}.zip)
contain scene_info.json with, per episode, the objects the task sampled ("info": {"{A}": "057_toycar/base2", "{B}": ...}).
We read only that member of each zip through HTTP range requests (zip central directory + one small file); no demos are downloaded.

Training-data map (see docs/notes/bench_audit.md; unverified parts flagged there):
  X-VLA cotrain ckpt       RoboTwin-cotrain-aloha_agilex-ee-0   -> assumed clean_50 + randomized_500 per task (550 x 50 tasks)
  pi0.5 cotrain ckpt       RoboTwin-lerobot_v30-aloha_agilex-joint-0 (HF lerobot_dataset/RoboTwin_lerobot_v30.zip)
  SmolVLA lerobot/smolvla_robotwin  -> lerobot/robotwin_unified (27.5k episodes = 550 x 50)
Cells are counted at two levels: category (A cat, B cat) and instance (A cat/baseN, B cat/baseN).
Output: results/bench/robotwin/audit/ (refuses to overwrite).
"""
import collections, json, os, re, sys, zipfile

REPO = "/coc/flash10/ksingara3/flawed-composition-vlas"
OUT = REPO + "/results/bench/robotwin/audit"
RT = "/coc/flash10/ksingara3/robot/bench/robotwin/RoboTwin"
TASKS = ["place_a2b_left", "place_a2b_right", "place_object_basket", "place_object_scale", "place_object_stand",
         "place_cans_plasticbox", "place_container_plate", "place_bread_basket", "place_bread_skillet", "place_phone_stand",
         "place_mouse_pad", "place_fan"]
SETS = ["aloha-agilex_clean_50", "aloha-agilex_randomized_500"]
MIN_PARTNER = 5


def read_scene_info(fs, task, st):
    p = f"datasets/TianxingChen/RoboTwin2.0/dataset/{task}/{st}.zip"
    with fs.open(p, "rb") as f:
        zf = zipfile.ZipFile(f)
        name = [i for i in zf.namelist() if i.endswith("scene_info.json")][0]
        return json.loads(zf.read(name)), len(zf.namelist())


def cat_of(v):
    return v.split("/")[0] if isinstance(v, str) else str(v)


def main():
    from huggingface_hub import HfFileSystem
    os.makedirs(OUT, exist_ok=False)  # never overwrite
    fs = HfFileSystem()
    summary = {}
    for task in TASKS:
        eps = []  # (set, episode key, info dict)
        for st in SETS:
            try:
                si, n = read_scene_info(fs, task, st)
            except Exception as e:  # noqa
                print(f"[{task}] {st}: cannot read ({type(e).__name__}: {e})", flush=True); continue
            for k, v in si.items():
                eps.append((st, k, v.get("info", {})))
        if not eps:
            continue
        keys = collections.Counter(tuple(sorted(i.keys())) for _, _, i in eps)
        print(f"\n=== {task}: {len(eps)} episodes; info key sets {dict(keys)}", flush=True)
        # A/B style: both {A} and {B} present
        ab = [(s, k, i) for s, k, i in eps if "{A}" in i and "{B}" in i]
        res = {"n_episodes": len(eps), "n_with_A_and_B": len(ab), "info_keysets": {"|".join(k): v for k, v in keys.items()}}
        if ab:
            for level, f in (("category", cat_of), ("instance", lambda v: v)):
                pair = collections.Counter((f(i["{A}"]), f(i["{B}"])) for _, _, i in ab)
                A = collections.Counter(f(i["{A}"]) for _, _, i in ab)
                Bc = collections.Counter(f(i["{B}"]) for _, _, i in ab)
                allA, allB = sorted(A), sorted(Bc)
                zero = [(a, b) for a in allA for b in allB if pair[(a, b)] == 0 and A[a] >= MIN_PARTNER and Bc[b] >= MIN_PARTNER]
                # partner counts
                nb = {a: len({b for (x, b) in pair if x == a}) for a in allA}
                na = {b: len({a for (a, y) in pair if y == b}) for b in allB}
                cells = []
                for (a, b) in zero:
                    cells.append({"A": a, "B": b, "A_episodes": A[a], "B_episodes": Bc[b],
                                  "A_partners_in_B_slot": nb[a], "B_partners_in_A_slot": na[b],
                                  "known_partner_pairs_A": sorted([(y, c) for (x, y), c in pair.items() if x == a], key=lambda t: -t[1])[:8],
                                  "known_partner_pairs_B": sorted([(x, c) for (x, y), c in pair.items() if y == b], key=lambda t: -t[1])[:8]})
                res[level] = {"n_A": len(allA), "n_B": len(allB), "n_pairs_seen": len(pair), "n_pairs_possible": len(allA) * len(allB),
                              "n_zero_cells_A_B_each_ge5": len(zero), "A_episodes": dict(A), "B_episodes": dict(Bc),
                              "A_to_n_B_partners": nb, "B_to_n_A_partners": na, "pair_counts": {f"{a} || {b}": c for (a, b), c in sorted(pair.items())},
                              "zero_cells": cells[:3000]}
                print(f"  [{level}] A={len(allA)} B={len(allB)} pairs seen {len(pair)}/{len(allA) * len(allB)}; zero-count cells with A,B each >= {MIN_PARTNER}: {len(zero)}", flush=True)
        else:
            # generic: report distribution of each key (for tasks whose second slot is fixed or unnamed)
            for k in sorted({kk for _, _, i in eps for kk in i}):
                c = collections.Counter(cat_of(i.get(k)) for _, _, i in eps)
                res.setdefault("slot_distributions", {})[k] = dict(c.most_common(30))
        summary[task] = res
        json.dump(res, open(f"{OUT}/{task}.json", "w"), indent=1)
    json.dump({t: {k: (v if not isinstance(v, dict) or "zero_cells" not in v else {kk: vv for kk, vv in v.items() if kk not in ("zero_cells", "pair_counts")}) for k, v in r.items()} for t, r in summary.items()},
              open(f"{OUT}/summary.json", "w"), indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
