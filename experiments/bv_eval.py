"""PLAN_BINDVLA section 6: closed-loop reads of BindVLA and the public SmolVLA reference, through the repo's rollout harness
(roll_pack.sbatch -> roll.sh -> serve_guided.py MODE=ours in the SmolVLA venv -> libero_client.py -> roll_record.py).

    python experiments/bv_eval.py --set parity --model bindvla --ckpt results/bindvla/main/ckpt/step_030000.pt --launch
    python experiments/bv_eval.py --set parity --model smolvla_libero --launch      # the reference on its own 40 tasks
    python experiments/bv_eval.py --set dev --model bindvla --ckpt ... --launch     # needs the m3_prereg row tag pbv
    python experiments/bv_eval.py --read [--record]

Sets: parity = the model's TRAINING tasks (BindVLA: split.json train, 76; smolvla_libero: its 40 fine-tuning tasks); dev = the
split's dev tasks (15 CD-bin + 14 part; every one passes l90c.check_task, so no sealed task can be launched).  10 episodes per
task (init states 0-9), MODE=ours (10 Euler steps), replan 5 (the client's default).  Ledger rows: bv_parity / bv_dev, tag =
<model>@<step>.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
R = os.path.join(ROOT, "results", "bindvla", "eval")
TRIALS = 10
CHUNK = 5
PER_GPU = 3
EXCLUDE = "sky1,hal,tars,t1000,vicki,johnny5,c3po,olivaw,megabot,ripl-s1"
SUITES40 = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
SMOL_PY = "/coc/flash10/ksingara3/robot/smolvla/.venv/bin/python"


def task_groups(which, model):
    from lifelong_ood import l90c
    s = l90c.load_split()
    if which == "parity" and model == "smolvla_libero":
        return {su: list(range(10)) for su in SUITES40}
    if which == "parity":
        tids = s["train"]
    else:
        tids = [t for b in ("0.1", "0.3", "0.5", "part") for t in s["dev"][b]]
        for t in tids:
            l90c.check_task(t)                                   # refuses a sealed task
    g = {}
    for t in tids:
        su, i = t.split(":")
        g.setdefault(su, []).append(int(i))
    return {k: sorted(v) for k, v in g.items()}


def tag_of(a):
    if a.model == "smolvla_libero":
        return "smolvla_libero"
    import re
    m = re.search(r"results/bindvla/([^/]+)/ckpt/step_(\d+)\.pt", os.path.abspath(a.ckpt))
    assert m, a.ckpt
    return f"bindvla_{m.group(1)}@{int(m.group(2))}"


def launch(a):
    if a.set == "dev":
        from lifelong_ood import ledger
        need = "pbv2" if any(x in a.ckpt for x in ("main2", "vanilla")) else "pbv"
        assert any(r.get("tag") == need for r in ledger.load("m3_prereg")), f"dev read needs the pre-registration row (m3_prereg tag {need})"
    tag = tag_of(a)
    os.makedirs(os.path.join(R, "packs"), exist_ok=True)
    specs = []
    for suite, ids in task_groups(a.set, a.model).items():
        for k in range(0, len(ids), CHUNK):
            part = ids[k:k + CHUNK]
            name = f"{a.set}{'log' if a.slotlog else ''}_{tag.replace('@', '_')}_{suite}_{part[0]}-{part[-1]}"
            out = os.path.join(R, name + ".json")
            if os.path.exists(out):
                continue
            sp = dict(MODE="ours", SUITE=suite, TASKS="+".join(map(str, part)), TRIALS=TRIALS, FIRST=0, EXPERIMENT=f"bv_{a.set}{'log' if a.slotlog else ''}",
                      TAG=tag, OUT=out, BACKBONE_CONFIG="smolvla", SERVER_PY=SMOL_PY,
                      **({"BACKBONE_CKPT": os.path.abspath(a.ckpt)} if a.model != "smolvla_libero" else {}),
                      **(dict(RECORD=1, LIGHT=1, OBJECTS=1, ALL_OBJECTS=1, REC_QIMG=1) if a.slotlog else {}))
            specs.append(sp)
    jobs = []
    for k in range(0, len(specs), PER_GPU):
        path = os.path.join(R, "packs", f"{a.set}{'log' if a.slotlog else ''}_{tag.replace('@', '_')}_{k // PER_GPU:03d}.tsv")
        with open(path, "w", encoding="utf-8") as fh:
            for sp in specs[k:k + PER_GPU]:
                fh.write("\t".join(f"{kk}={v}" for kk, v in sp.items()) + "\n")
        args = [f"--partition={a.partition}", f"--account={a.partition}", "--qos=short", "--time=04:00:00", "--gres=gpu:a40:1",
                "--requeue", f"--exclude={EXCLUDE}"] + ([f"--dependency=afterok:{a.after}"] if a.after else []) + \
               [f"--export=ALL,SPECS={path}", "experiments/roll_pack.sbatch"]
        if a.dry:
            print("[dry] sbatch", " ".join(args), [sp["SUITE"] + ":" + sp["TASKS"] for sp in specs[k:k + PER_GPU]])
            continue
        j = subprocess.run(["sbatch", "--parsable"] + args, capture_output=True, text=True, cwd=ROOT).stdout.strip()
        jobs.append(j)
        print(f"pack {os.path.basename(path)} {[sp['SUITE'] + ':' + sp['TASKS'] for sp in specs[k:k + PER_GPU]]} -> job {j}", flush=True)
    print("jobs:", ":".join(jobs))


def per_task(exp, tag):
    from lifelong_ood import ledger
    out = {}
    for r in ledger.load(exp, tag=tag):
        for k, v in r["metrics"]["tasks"].items():
            out[f"{r['bench']}:{k}"] = dict(sr=v["success_rate"], n=v["n"], language=v["language"])
    return out


def boot_mean(v, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    v = np.asarray(v, float)
    b = [rng.choice(v, len(v)).mean() for _ in range(n)]
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def read(a):
    from lifelong_ood import l90c, ledger
    s = l90c.load_split()
    res = {}
    tags = sorted(t for t in {r["tag"] for r in ledger.load("bv_parity")} | {r["tag"] for r in ledger.load("bv_dev")} if not str(t).startswith("read_"))
    for tag in tags:
        par = per_task("bv_parity", tag)
        dev = per_task("bv_dev", tag)
        row = {}
        if par:
            v = [x["sr"] for x in par.values()]
            row["parity"] = dict(tasks=len(par), episodes=int(sum(x["n"] for x in par.values())), mean=float(np.mean(v)), ci95=boot_mean(v),
                                 by_suite={su: float(np.mean([x["sr"] for k, x in par.items() if k.startswith(su + ":")]))
                                           for su in sorted({k.split(":")[0] for k in par})})
        if dev:
            row["dev"] = {}
            for b in ("0.1", "0.3", "0.5", "part"):
                v = [dev[t]["sr"] for t in s["dev"][b] if t in dev]
                if v:
                    row["dev"][b] = dict(tasks=len(v), mean=float(np.mean(v)), ci95=boot_mean(v), per_task={t: dev[t]["sr"] for t in s["dev"][b] if t in dev})
        res[tag] = row
    ref = res.get("smolvla_libero", {}).get("parity")
    for tag, row in res.items():
        if tag != "smolvla_libero" and ref and "parity" in row:
            row["parity"]["bar_0.8x_ref"] = 0.8 * ref["mean"]
            row["parity"]["reached"] = bool(row["parity"]["mean"] >= 0.8 * ref["mean"])
    os.makedirs(R, exist_ok=True)
    json.dump(res, open(os.path.join(R, "bv_eval.json"), "w", encoding="utf-8"), indent=1)
    print(json.dumps(res, indent=1))
    if a.record:
        for tag, row in res.items():
            for k in row:
                print(ledger.record(f"bv_{k}", bench="libero_90_c", tag=f"read_{tag}", seed=7, metrics=row[k],
                                    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config=dict(trials=TRIALS),
                                    script="experiments/bv_eval.py", command=" ".join(sys.argv)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=("parity", "dev"), default="parity")
    ap.add_argument("--slotlog", action="store_true", help="PLAN_BINDVLA 12.4: record each query's policy inputs + oracle coverage (ledger bv_devlog)")
    ap.add_argument("--model", choices=("bindvla", "smolvla_libero"), default="bindvla")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--launch", action="store_true")
    ap.add_argument("--read", action="store_true")
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--after", default="")
    ap.add_argument("--partition", default="overcap")
    a = ap.parse_args()
    if a.launch:
        launch(a)
    if a.read:
        read(a)
