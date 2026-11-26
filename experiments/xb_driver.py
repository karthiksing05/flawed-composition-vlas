"""XB stage driver (runs as a short CPU sbatch job via docs/notes/xb_jobs/xb_driver.sh; each run re-submits itself with --begin=now+25min until xb_read has written read_main.json).
`sbatch --dependency` is unusable on this cluster (afterok rejected, afterany not honoured), so the stage order is enforced here:
  1. any (bench, model, cell-type, chunk) whose rollout episodes are incomplete and which has no queued/running xb job is (re)submitted (episodes are resumable);
  2. when every active model has all partner episodes of a benchmark, xb_predict.py writes pred_<bench>.json (once) and only then are that benchmark's held-out jobs submitted;
  3. when every expected episode of every benchmark exists, xb_classes.py and xb_read.py run (once).  Incomplete -> no read (k2).
Hard guards: GPU-h cap 300 (no submissions at >= 285), at most 60 xb jobs queued, at most MAX_ITER driver iterations.
active models: results/bench/xb/active_models.json  ({bench: [model, ...]}), maintained by hand when a model fails its checks (reason logged in docs/notes/xb_status.md).
"""
import glob, json, os, subprocess, sys, time

ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
XB = ROOT + "/results/bench/xb"
D = ROOT + "/docs/notes/xb_jobs"
PY = "/coc/flash10/ksingara3/robot/openpi/.venv/bin/python"
MAX_ITER = 60
CHUNKS = {"mesa": dict(partner=4, heldout=2), "bimesa": dict(partner=4, heldout=1), "robocasa": dict(partner=3, heldout=2), "robotwin": dict(partner=2, heldout=2)}
SCRIPT = {"mesa": "xb_roll_mesa.sh", "bimesa": "xb_roll_mesa.sh", "robocasa": "xb_roll_robocasa.sh", "robotwin": "xb_roll_rt.sh"}


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()


def log(msg):
    line = time.strftime("%FT%T") + " " + msg
    print(line, flush=True)
    open(XB + "/driver.log", "a").write(line + "\n")


def gpu_hours():
    t = 0.0
    for l in sh("sacct -u $USER -S 2026-11-26 -X -n -P -o JobName,ElapsedRaw,AllocTRES").splitlines():
        p = l.split("|")
        if len(p) == 3 and p[0].startswith("xb_") and "gres/gpu" in p[2]:
            n = 1
            for tok in p[2].split(","):
                if tok.startswith("gres/gpu") and "=" in tok:
                    n = int(tok.split("=")[1])
            t += float(p[1]) * n / 3600
    return t


def queued():
    return {l.split()[1]: l.split()[0] for l in sh("squeue -u $USER -h -o '%i %j' | grep ' xb_'").splitlines() if l.strip()}


def cells(bench, ctype):
    """ordered list of (cell id, n episodes, task-for-rt) exactly as the harness orders them."""
    if bench in ("mesa", "bimesa"):
        c = json.load(open(f"{XB}/cells_{bench}.json"))
        return [(t, 50, None) for t in (c["heldout_tasks"] if ctype == "heldout" else c["partner_tasks"])]
    if bench == "robocasa":
        f = json.load(open(f"{XB}/cells_robocasa_final.json"))["families"]
        return [(f"{fam.replace('->', '_to_')}__{o.replace(' ', '_')}", 20, None) for fam, d in f.items() for o in d[ctype]["cells"]]
    f = json.load(open(f"{XB}/cells_robotwin_final.json"))["tasks"]
    return [(f"{t}/{A.replace('/', '-')}__{B.replace('/', '-')}", 10, t) for t, d in f.items() for A, B in d[ctype]]


def chunk_cells(bench, ctype, ci, cn):
    cl = cells(bench, ctype)
    if bench == "robotwin":
        out = []
        for t in ("place_a2b_left", "place_a2b_right"):
            tl = [c for c in cl if c[2] == t]
            out += [c for k, c in enumerate(tl) if k % cn == ci]
        return out
    return [c for k, c in enumerate(cl) if k % cn == ci]


def n_done(bench, model, cid):
    return len(glob.glob(f"{XB}/roll/{bench}/{model}/{cid}/ep*.json"))


def incomplete_chunks(bench, model, ctype):
    out = []
    for ci in range(CHUNKS[bench][ctype]):
        miss = sum(max(0, n - n_done(bench, model, cid)) for cid, n, _ in chunk_cells(bench, ctype, ci, CHUNKS[bench][ctype]))
        if miss:
            out.append((ci, miss))
    return out


def submit(bench, model, ctype, ci):
    cn = CHUNKS[bench][ctype]
    ph = "par" if ctype == "partner" else "ho"
    nm = f"xb_{bench}_{model}_{ph}_{ci}"
    if bench in ("mesa", "bimesa"):
        env = f"BENCH={bench},MODEL={model},CELLTYPE={ctype},CHUNK={ci}/{cn},NSHARD=4"
        names = [(nm, env)]
    elif bench == "robocasa":
        env = f"MODE=roll,MODEL={model},CELLTYPE={ctype},CHUNK={ci}/{cn},NSHARD=3"
        names = [(f"xb_rc_{model}_{ph}_{ci}", env)]
    else:
        names = [(f"xb_rt_{model}_{ph}_{t[len('place_a2b_'):]}_{ci}", f"MODE=roll,TASK={t},MODEL={model},CELLTYPE={ctype},CHUNK={ci}/{cn}") for t in ("place_a2b_left", "place_a2b_right")]
    q = queued()
    for name, env in names:
        if name in q:
            continue
        if bench == "robotwin":
            t = "place_a2b_" + name.rsplit("_", 2)[1]
            if sum(max(0, n - n_done(bench, model, cid)) for cid, n, tt in chunk_cells(bench, ctype, ci, cn) if tt == t) == 0:
                continue
        jid = sh(f"cd {ROOT}; sbatch --parsable -J {name} --export=ALL,{env} {D}/{SCRIPT[bench]}")
        log(f"submitted {name} -> {jid}")
        open(f"{D}/xb_jobids.tsv", "a").write(f"{jid}\t{name}\t{ctype}\tdriver\n")


def main():
    it_path = f"{XB}/driver_iter.txt"
    it = int(open(it_path).read()) + 1 if os.path.exists(it_path) else 1
    open(it_path, "w").write(str(it))
    if os.path.exists(f"{XB}/read_main.json"):
        log("read_main.json exists: driver done"); return
    gh = gpu_hours(); log(f"iteration {it}: xb GPU-h {gh:.1f}")
    active = json.load(open(f"{XB}/active_models.json"))
    for b, cs in json.load(open(f"{XB}/candidate_models.json")).items():          # a candidate becomes active once its smoke job produced an episode (server + client work)
        for m in cs:
            if m not in active[b] and any(json.load(open(f)).get("error") is None for f in glob.glob(f"{XB}/roll/{b}/{m}/**/ep*.json", recursive=True)[:5]):
                active[b].append(m); log(f"activated {b}/{m} (smoke episode exists)")
    q = queued()
    all_complete = True
    for bench, models in active.items():
        if not os.path.exists(f"{XB}/cells_{bench}.json") or (bench == "robocasa" and not os.path.exists(f"{XB}/cells_robocasa_final.json")) or \
                (bench == "robotwin" and not os.path.exists(f"{XB}/cells_robotwin_final.json")):
            all_complete = False; continue
        part_ok = True
        for m in models:
            for ci, miss in incomplete_chunks(bench, m, "partner"):
                part_ok = False
                if gh < 285 and len(q) < 60:
                    submit(bench, m, "partner", ci)
        pred = f"{XB}/pred_{bench}.json"
        if part_ok and not os.path.exists(pred):
            if bench in ("mesa", "bimesa") and not all(len(glob.glob(f"{XB}/reset/{bench}/*/ep*.json")) >= 50 * len(json.load(open(f"{XB}/cells_{bench}.json"))["heldout_tasks"]) for _ in [0]):
                log(f"{bench}: reset-state pass incomplete; waiting"); all_complete = False; continue
            if bench == "robocasa" and sum(1 for _ in glob.glob(f"{XB}/reset/robocasa/*/ep*.json")) < sum(n for _, n, _ in cells(bench, "heldout")):
                log("robocasa: reset-state pass incomplete; waiting"); all_complete = False; continue
            log(f"{bench}: all partner episodes present -> xb_predict")
            out = sh(f"cd {ROOT}; {PY} experiments/xb_predict.py {bench} 2>&1 | tail -12")
            log(out.replace("\n", " | "))
        if not os.path.exists(pred):
            all_complete = False; continue
        for m in models:
            for ci, miss in incomplete_chunks(bench, m, "heldout"):
                all_complete = False
                if gh < 285 and len(q) < 60:
                    submit(bench, m, "heldout", ci)
        if not part_ok:
            all_complete = False
    if all_complete and not os.path.exists(f"{XB}/classes.json"):
        log(sh(f"cd {ROOT}; {PY} experiments/xb_classes.py 2>&1 | tail -20").replace("\n", " | "))
    if all_complete:
        log(sh(f"cd {ROOT}; {PY} experiments/xb_read.py 2>&1 | tail -40").replace("\n", " | "))
    if not os.path.exists(f"{XB}/read_main.json") and it < MAX_ITER and gh < 298:
        jid = sh(f"cd {ROOT}; sbatch --parsable -t 03:00:00 --begin=now+25minutes -J xb_driver --export=ALL,CMD=experiments/xb_driver.py {D}/xb_cpu.sh")
        log(f"re-submitted driver -> {jid}")
    else:
        log("driver stops (done, cap, or max iterations)")


if __name__ == "__main__":
    main()
