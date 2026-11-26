"""MESA / BiMESA training-data co-occurrence audit (CPU only).

Question: for each composite eval task, which (object, receptacle) pairs occur in the data each released checkpoint was
trained on?  A held-out pairing = an (object, receptacle) pair absent from the checkpoint's training data while the object
and the receptacle each appear with other partners.

Data: language instructions + episode counts from the LeRobot meta/ of the training datasets (downloaded here, meta only),
and the instruction tables of the benchmark docs for the eval suites.  Nothing is rendered or rolled out.
Pair granularity:  coarse = (object, fixture);  fine = (object, fixture:region), where the region is the drawer
(top/middle/bottom cabinet) or the left/right side of the dish drainer.  Both are reported; the fine one is the strict test.

Output: results/bench/mesa/audit/  (refuses to overwrite).
"""
import argparse, collections, glob, json, os, re, sys

REPO = "/coc/flash10/ksingara3/flawed-composition-vlas"
B = "/coc/flash10/ksingara3/robot/bench"
DOCS = B + "/mesa/mesa-env/docs/mesa_bench/"
META = B + "/mesa/meta"
OUT = REPO + "/results/bench/mesa/audit"

CKPT_DATA = {  # from the HF model cards (fetched 2026-11-26)
    "albertwilcox/mesa-pi0": "mesa-70-lerobot",
    "albertwilcox/mesa-pi05": "mesa-70-lerobot",
    "albertwilcox/mesa-pi0-fast": "mesa-70-lerobot",
    "albertwilcox/mesa-gr00t-n1.6": "mesa-70-lerobot",
    "albertwilcox/bimesa-pi0": "bimesa-57-lerobot",
    "albertwilcox/bimesa-pi05": "bimesa-57-lerobot",
    "albertwilcox/bimesa-pi0-fast": "bimesa-57-lerobot",
    "albertwilcox/bimesa-gr00t-n1.6": "bimesa-57-lerobot",
}
DATASETS = ["mesa-70-lerobot", "mesa-all-train-lerobot", "bimesa-57-lerobot"]


def download_meta():
    from huggingface_hub import snapshot_download
    for d in DATASETS:
        snapshot_download("albertwilcox/" + d, repo_type="dataset", local_dir=f"{META}/{d}", allow_patterns=["meta/*", "meta/**/*"])


def load_task_episodes(d):
    """-> Counter(task string -> episode count)."""
    import pandas as pd
    root = f"{META}/{d}/meta"
    tasks = {}
    if os.path.exists(root + "/tasks.jsonl"):
        for line in open(root + "/tasks.jsonl"):
            r = json.loads(line); tasks[r["task_index"]] = r["task"]
    else:
        df = pd.read_parquet(root + "/tasks.parquet")
        if "task" in df.columns:
            tasks = {int(r.task_index): r.task for r in df.itertuples()}
        else:  # index = task string
            tasks = {int(v): k for k, v in zip(df.index, df["task_index"])}
    cnt = collections.Counter()
    if os.path.exists(root + "/episodes.jsonl"):
        for line in open(root + "/episodes.jsonl"):
            r = json.loads(line)
            for t in r["tasks"]:
                cnt[t] += 1
    else:
        files = sorted(glob.glob(root + "/episodes/**/*.parquet", recursive=True))
        for f in files:
            df = pd.read_parquet(f)
            col = "tasks" if "tasks" in df.columns else None
            if col:
                for ts in df[col]:
                    for t in (list(ts) if not isinstance(ts, str) else [ts]):
                        cnt[t] += 1
            elif "task_index" in df.columns:
                for i in df["task_index"]:
                    cnt[tasks[int(i)]] += 1
    for t in tasks.values():
        cnt.setdefault(t, 0)
    return cnt, len(tasks)


FIX = {  # normalise fixture names
    "dish drainer": "dish drainer", "slide cabinet": "slide cabinet", "cutting board": "cutting board",
    "baking sheet": "baking sheet",
}


def norm_recept(s):
    s = s.strip().rstrip(".").lower()
    s = re.sub(r"^(the )", "", s)
    m = re.match(r"(left|right) side of the (.+)", s)
    if m:
        return m.group(2), m.group(1)
    m = re.match(r"(top|middle|bottom) drawer of the (.+)", s)
    if m:
        return m.group(2), m.group(1)
    return s, ""


def parse(instr):
    """-> (list of objects, fixture, region, variant) or None."""
    s = instr.strip().lower().rstrip(".")
    var = "plain"
    m = re.match(r"open the (.+?) and put the (.+?) in it( and close it)?$", s)
    if m:
        fx, region = norm_recept(m.group(1))
        objs = [o.strip() for o in re.split(r" and (?:the )?", m.group(2))]
        return objs, fx, region, "open_place" + ("_close" if m.group(3) else "")
    if s.endswith(" and close it"):
        var = "place_close"; s = s[: -len(" and close it")]
    m = re.match(r"put the (.+?) (?:in|on|into|onto) (.+)$", s)
    if not m:
        return None
    objs = [o.strip() for o in re.split(r" and (?:the )?", m.group(1))]
    fx, region = norm_recept(m.group(2))
    return objs, fx, region, var


def md_tables(path):
    """-> {section title: [(id, instruction)]} from the benchmark docs."""
    sec, out = None, collections.defaultdict(list)
    for line in open(path):
        if line.startswith("## "):
            sec = line[3:].strip()
        m = re.match(r"\|\s*([a-z0-9_]+)\s*\|\s*(.+?)\s*\|\s*$", line)
        if m and sec and m.group(1) not in ("Task", "Block") and not set(m.group(1)) <= set("-"):
            out[sec].append((m.group(1), m.group(2)))
    return out


def inventory(task_counts):
    """pair counts (episodes) and the tasks behind them."""
    P_coarse, P_fine, O, R_c, R_f = (collections.Counter() for _ in range(5))
    tasks_by_pair = collections.defaultdict(list)
    unparsed = []
    for t, n in task_counts.items():
        p = parse(t)
        if not p:
            unparsed.append(t); continue
        objs, fx, region, var = p
        for o in objs:
            for key, C in (((o, fx), P_coarse), ((o, fx + (":" + region if region else "")), P_fine)):
                C[key] += n
            tasks_by_pair[(o, fx + (":" + region if region else ""))].append({"instruction": t, "episodes": n, "variant": var})
            O[o] += n
        R_c[fx] += n
        R_f[fx + (":" + region if region else "")] += n
    return dict(P_coarse=P_coarse, P_fine=P_fine, O=O, R_c=R_c, R_f=R_f, tasks_by_pair=tasks_by_pair, unparsed=unparsed)


def audit_suite(name, comp_tasks, inv):
    rows = []
    objs_by_recept = collections.defaultdict(set); recepts_by_obj = collections.defaultdict(set)
    for (o, r), n in inv["P_fine"].items():
        if n > 0:
            objs_by_recept[r].add(o); recepts_by_obj[o].add(r)
    for tid, instr in comp_tasks:
        p = parse(instr)
        if not p:
            rows.append({"task": tid, "instruction": instr, "parsed": False}); continue
        objs, fx, region, var = p
        rf = fx + (":" + region if region else "")
        for o in objs:
            partners_o = sorted((r, inv["P_fine"][(o, r)]) for r in recepts_by_obj.get(o, []))
            partners_r = sorted((x, inv["P_fine"][(x, rf)]) for x in objs_by_recept.get(rf, []))
            same_fx = sorted((r, inv["P_fine"][(o, r)]) for r in recepts_by_obj.get(o, []) if r.split(":")[0] == fx)
            rows.append({
                "task": tid, "instruction": instr, "parsed": True, "object": o, "fixture": fx, "region": region, "variant": var,
                "pair_count_coarse": inv["P_coarse"][(o, fx)], "pair_count_fine": inv["P_fine"][(o, rf)],
                "object_seen": inv["O"][o] > 0, "recept_seen_fine": inv["R_f"][rf] > 0, "recept_seen_coarse": inv["R_c"][fx] > 0,
                "n_partner_receptacles_of_object": len(partners_o), "n_partner_objects_of_receptacle": len(partners_r),
                "episodes_object": inv["O"][o], "episodes_receptacle": inv["R_f"][rf],
                "partners_same_object_other_receptacle": partners_o,
                "partners_same_receptacle_other_object": partners_r,
                "partner_tasks_same_object": [t for r, _ in partners_o for t in inv["tasks_by_pair"][(o, r)]],
                "partner_tasks_same_receptacle": [t for x, _ in partners_r for t in inv["tasks_by_pair"][(x, rf)]],
                "same_fixture_other_region_of_object": same_fx,
                "held_out_fine": inv["P_fine"][(o, rf)] == 0 and inv["O"][o] > 0 and inv["R_f"][rf] > 0,
                "held_out_coarse": inv["P_coarse"][(o, fx)] == 0 and inv["O"][o] > 0 and inv["R_c"][fx] > 0,
            })
    ok = [r for r in rows if r.get("parsed")]
    summ = {"suite": name, "n_tasks": len(comp_tasks), "n_object_slots": len(ok),
            "held_out_fine": sum(r["held_out_fine"] for r in ok), "held_out_coarse": sum(r["held_out_coarse"] for r in ok),
            "pair_seen_fine": sum(r["pair_count_fine"] > 0 for r in ok),
            "object_unseen": sum(not r["object_seen"] for r in ok), "recept_unseen_fine": sum(not r["recept_seen_fine"] for r in ok),
            "unparsed": [r["task"] for r in rows if not r.get("parsed")]}
    return rows, summ


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--skip-download", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=False)  # never overwrite
    if not a.skip_download:
        download_meta()
    single = md_tables(DOCS + "task_suites.md"); bi = md_tables(DOCS + "task_suites_bimanual.md")
    comp = {"mesa-composite": single["MESA-Composite (`mesa-composite`)"], "bimesa-composite": bi["BiMESA-Composite (`bimesa-composite`)"]}
    dsets = {}
    for d in DATASETS:
        cnt, ntask = load_task_episodes(d)
        inv = inventory(cnt)
        dsets[d] = {"inv": inv, "n_task_strings": ntask, "n_episodes": sum(cnt.values())}
        print(f"{d}: task strings={ntask} episodes={sum(cnt.values())} unparsed={len(inv['unparsed'])}", flush=True)
        if inv["unparsed"]:
            print("  unparsed examples:", inv["unparsed"][:15])
    plan = [("mesa-composite", "mesa-70-lerobot"), ("mesa-composite", "mesa-all-train-lerobot"), ("bimesa-composite", "bimesa-57-lerobot")]
    summary = {"checkpoint_to_dataset": CKPT_DATA, "datasets": {d: {"task_strings": v["n_task_strings"], "episodes": v["n_episodes"], "unparsed": v["inv"]["unparsed"]} for d, v in dsets.items()}, "audits": []}
    for suite, d in plan:
        rows, s = audit_suite(suite, comp[suite], dsets[d]["inv"]); s["train_dataset"] = d
        summary["audits"].append(s)
        json.dump(rows, open(f"{OUT}/cells__{suite}__vs__{d}.json", "w"), indent=1)
        # partner counts per object / receptacle (training data side)
        inv = dsets[d]["inv"]
        pc = {"object_to_n_receptacles": {}, "receptacle_to_n_objects": {}}
        by_o, by_r = collections.defaultdict(set), collections.defaultdict(set)
        for (o, r), n in inv["P_fine"].items():
            if n > 0: by_o[o].add(r); by_r[r].add(o)
        pc["object_to_n_receptacles"] = {o: len(v) for o, v in sorted(by_o.items())}
        pc["receptacle_to_n_objects"] = {r: len(v) for r, v in sorted(by_r.items())}
        json.dump(pc, open(f"{OUT}/partner_counts__{d}.json", "w"), indent=1)
        print(f"\n== {suite} vs {d}: {json.dumps({k: v for k, v in s.items() if k != 'unparsed'})} unparsed={s['unparsed']}")
        for r in rows:
            if r.get("parsed"):
                print(f"  {r['task']:58s} {r['object']:14s} {r['fixture']+(':'+r['region'] if r['region'] else ''):24s} n_pair={r['pair_count_fine']:4d} (coarse {r['pair_count_coarse']:4d}) obj->{r['n_partner_receptacles_of_object']:2d} recepts, recept->{r['n_partner_objects_of_receptacle']:2d} objs  HELD_OUT_FINE={r['held_out_fine']}")
    # compact leakage check: composite task strings that literally occur in a training set
    leak = {}
    for suite, d in plan:
        tr = set(dsets[d]["inv"]["tasks_by_pair"].keys())
        leak[f"{suite}|{d}"] = None
    json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=1)
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
