"""XR port data step: per-frame object / receptacle poses for every MESA-70 demonstration (docs/notes/xr_port_plan.md).

MESA-70-lerobot stores only robot proprio + images + actions (meta/info.json, no object state).  The source `albertwilcox/mesa-70-hdf5`
stores the flattened MuJoCo state of every frame (`data/demo_k/states`) and the per-demo env args (`data.attrs["env_args"]`, with the parsed
BDDL problem) and the scene xml (`data/demo_k.attrs["model_file"]`), so the poses are replayed exactly as mesa/utils/replay.py does.
The hdf5 files are 4.6-11 GB each (473 GB in all, images included), so the `extract` stage reads ONLY states / attrs by HTTP range requests.

Stages (two venvs, because the sim env lacks huggingface_hub):
  extract   openpi venv (huggingface_hub + h5py):  hdf5 -> <out>/src/<task>.pkl.gz    (states, model xml, env args, demo lengths)
  replay    mesa venv (robosuite + mesa):          src pkl -> <out>/mesa70/<task>.npz + <task>.json   (CPU only, no rendering)
  map       any venv (numpy only):                 hdf5 demo -> lerobot episode_index (LCS on lengths; lerobot truncates at first success)
  vocab     any venv (numpy only):                 <out>/mesa70/*.json -> <out>/vocab.json, and the lerobot-episode index <out>/index.json
  table     any venv (numpy only):                 per-frame arrays in LeRobot global-index order -> <out>/xr_table.npz (training lookup)

Per frame t (aligned with LeRobot frame_index t = hdf5 states[t], the state the observation was rendered from):
  eef[3], obj[3], tgt[3]            world-frame positions: end effector, instructed object, current target (receptacle body or region site)
  rel_o2t = tgt - obj               object-to-receptacle offset (the quantity the task text calls the offset)
  rel_e2o = obj - eef, rel_e2t = tgt - eef
  chan[4] = [rel(3), role]          LIBERO's role channel (experiments/sc_rv.py chan_state): rel = active entity minus end effector, role 1 = object
                                    (until the hold), 2 = target (while held), 0 = none (open / close subtasks, after completion; rel = 0)
  role, subtask (index into demonstration_states, n_sub = all done), grasp (sim check_grasp now), (jaw width and eef are in the LeRobot parquet already)
Per episode (json): lerobot episode_index, hdf5 key, instruction, object / receptacle names and categories, receptacle fine key, subtask list.

Usage:
  extract: python xr_mesa_poses.py extract --tasks a,b|all [--max-eps N] [--out DIR]
  replay : python xr_mesa_poses.py replay  --tasks a,b|all [--max-eps N] [--out DIR]
  vocab  : python xr_mesa_poses.py vocab   [--out DIR]
Resumable (an existing output is skipped).  Nothing is written to the repo.
"""
import argparse, glob, gzip, json, os, pickle, re, sys, time
import numpy as np

OUT = "/coc/flash10/ksingara3/robot/bench/mesa/xr_poses"
PSTRIDE = int(os.environ.get("XR_PSTRIDE", "3"))
META = "/coc/flash10/ksingara3/robot/bench/mesa/meta/mesa-70-lerobot/meta"
HF_REPO = "albertwilcox/mesa-70-hdf5"
HF_HOME = "/coc/flash10/ksingara3/robot/bench/hf"


def all_tasks():
    from_cache = sorted(glob.glob(OUT + "/src/*.pkl.gz"))
    lst = OUT + "/hdf5_files.json"
    if os.path.exists(lst):
        return json.load(open(lst))
    return [os.path.basename(p)[:-len(".pkl.gz")] for p in from_cache]


def lerobot_lengths():
    L = {}
    for line in open(META + "/episodes.jsonl", encoding="utf-8"):
        d = json.loads(line)
        L[d["episode_index"]] = (d["length"], d["tasks"][0])
    return L


# ------------------------------------------------------------------ extract (openpi venv)
def _retry(fn, what=""):
    """HF rate-limits anonymous range requests (429): back off and retry."""
    for k in range(12):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            if k == 11:
                raise
            w = min(30 * (k + 1), 300)
            print(f"retry {what} after {type(e).__name__} in {w}s", flush=True)
            time.sleep(w)


def extract(a):
    os.environ["HF_HOME"] = HF_HOME
    import h5py
    from huggingface_hub import HfApi, HfFileSystem
    os.makedirs(OUT + "/src", exist_ok=True)
    if not os.path.exists(OUT + "/hdf5_files.json"):
        files = sorted(s.rfilename[:-5] for s in HfApi().dataset_info(HF_REPO).siblings if s.rfilename.endswith(".hdf5"))
        json.dump(files, open(OUT + "/hdf5_files.json", "w"))
    tasks = json.load(open(OUT + "/hdf5_files.json")) if a.tasks == "all" else a.tasks.split(",")
    si, sn = [int(x) for x in a.shard.split("/")]
    tasks = tasks[si::sn]
    fs = HfFileSystem()
    for task in tasks:
        dst = f"{OUT}/src/{task}.pkl.gz" if a.max_eps is None else f"{OUT}/src/{task}.n{a.max_eps}.pkl.gz"
        if os.path.exists(dst):
            continue
        t0 = time.time()
        with fs.open(f"datasets/{HF_REPO}/{task}.hdf5", "rb", block_size=4 * 2 ** 20) as fh, h5py.File(fh, "r") as f:
            demos = sorted(_retry(lambda: list(f["data"].keys()), "keys"), key=lambda s: int(s.split("_")[1]))
            if a.max_eps is not None:
                demos = demos[:a.max_eps]
            rec = dict(task=task, env_args=_retry(lambda: f["data"].attrs["env_args"], "env_args"), demos=[])
            if isinstance(rec["env_args"], bytes):
                rec["env_args"] = rec["env_args"].decode()
            def one(k):
                g = f["data"][k]
                mf = g.attrs["model_file"]
                return dict(key=k, states=g["states"][()], n_actions=int(g["actions"].shape[0]), model_file=mf.decode() if isinstance(mf, bytes) else mf)
            for k in demos:
                rec["demos"].append(_retry(lambda: one(k), k))
        tmp = dst + ".tmp"
        with gzip.open(tmp, "wb", compresslevel=3) as o:
            pickle.dump(rec, o)
        os.replace(tmp, dst)
        print(f"extract {task}: {len(rec['demos'])} demos, {os.path.getsize(dst) / 1e6:.1f} MB, {time.time() - t0:.0f}s", flush=True)


# ------------------------------------------------------------------ replay (mesa venv)
def cat_of(env, name):
    if name in env.objects_dict:
        return getattr(env.objects_dict[name], "category_name", name)
    if name in env.fixtures_dict:
        return getattr(env.fixtures_dict[name], "category_name", name)
    if name in getattr(env, "object_sites_dict", {}):
        par = env.object_sites_dict[name].parent_name
        return cat_of(env, par) if par != name else name
    return name


def pos_of(env, name):
    return np.asarray(env.object_states_dict[name].get_geom_state()["pos"], np.float64).copy()


def norm_cat(c):
    """robocasa_mushroom_0 -> mushroom, cabinet834 -> cabinet, bowl_drainer -> bowl_drainer (instance ids and the robocasa_ prefix removed)."""
    c = re.sub(r"^robocasa_", "", str(c))
    return re.sub(r"[_]?\d+", "", c).strip("_")


def fine_key(env, name):
    """cabinet middle region -> 'cabinet:middle'; dish drainer left region -> 'drainer:left'; otherwise the category."""
    c = norm_cat(cat_of(env, name))
    m = re.search(r"_(top|middle|bottom|left|right)_region$", name)
    return f"{c}:{m.group(1)}" if m else c


def replay(a):
    os.environ.setdefault("MUJOCO_GL", "egl")
    import robosuite
    import mesa  # noqa: F401
    from mesa.sim.envs import BDDLBaseDomain  # noqa: F401  registers the envs with robosuite
    os.makedirs(OUT + "/mesa70", exist_ok=True)
    if a.tasks == "all":
        tasks = json.load(open(OUT + "/hdf5_files.json"))
    else:
        tasks = a.tasks.split(",")
    si, sn = [int(x) for x in a.shard.split("/")]
    tasks = tasks[si::sn]
    lo, hi = (0, 10 ** 9) if a.demos is None else [int(x) for x in a.demos.split(":")]
    sfx = "" if a.demos is None else f".d{lo}-{hi}"
    for task in tasks:
        dst = f"{OUT}/mesa70/{task}{sfx}.npz"
        if os.path.exists(dst):
            continue
        t0 = time.time()
        if not os.path.exists(f"{OUT}/src/{task}{a.src_sfx}.pkl.gz"):
            print(f"skip {task}: no src", flush=True)
            continue
        rec = pickle.load(gzip.open(f"{OUT}/src/{task}{a.src_sfx}.pkl.gz", "rb"))
        rec["demos"] = rec["demos"][lo:hi]
        env_meta = json.loads(rec["env_args"])
        multi = "env_kwargs" not in env_meta and "demo_0" in env_meta
        env, env_key = None, None
        cols = {k: [] for k in ("ep", "frame", "eef", "obj", "tgt", "chan", "role", "subtask", "grasp", "demo")}
        meta = []
        for di, d in enumerate(rec["demos"], start=lo):
            em = env_meta[d["key"]] if multi else env_meta
            kw = json.loads(json.dumps(em["env_kwargs"]))
            kw.update(camera_names=[], use_camera_obs=False, has_offscreen_renderer=False, has_renderer=False, return_2d_bboxes=False, return_3d_bboxes=False)
            kk = json.dumps(kw, sort_keys=True)
            if kk != env_key:
                if env is not None:
                    env.close()
                env = robosuite.make(em["env_name"], **kw)
                env_key = kk
            states = d["states"]
            env.reset_to(dict(states=states[0], model=d["model_file"]))
            pp = env.parsed_problem
            sub = [list(x) for x in pp["demonstration_states"]]
            gs = [i for i, x in enumerate(sub) if x[0] == "grasp"]
            place = [i for i, x in enumerate(sub) if x[0] in ("on", "in")]
            obj_name = sub[gs[0]][1] if gs else None
            rec_name = sub[place[-1]][2] if place else None
            ep = -1                                              # resolved by the `map` stage (lerobot dropped some demos and truncates at success)
            meta.append(dict(episode_index=ep, hdf5_key=d["key"], length=int(len(states)), first_success=-1, demo=di,
                             instruction=" ".join(pp["language_instruction"]), object=obj_name, object_cat=norm_cat(cat_of(env, obj_name)) if obj_name else None,
                             receptacle=rec_name, receptacle_cat=norm_cat(cat_of(env, rec_name)) if rec_name else None, receptacle_fine=fine_key(env, rec_name) if rec_name else None,
                             subtasks=sub, n_sub=len(sub)))
            cur = 0
            rb = env.robots[0]
            bid = env.sim.model.body_name2id(rb.robot_model.root_body)
            meta[-1]["base_pos"] = env.sim.data.body_xpos[bid].tolist(); meta[-1]["base_quat"] = env.sim.data.body_xquat[bid].tolist()
            eef_sid = rb.eef_site_id[rb.arms[0]]
            if di == lo:
                o0 = env._get_observations(force_update=True)
                assert np.allclose(o0["robot0_eef_pos"], env.sim.data.site_xpos[eef_sid], atol=1e-5), "eef site mismatch"
            first_succ = -1
            grasp = False
            for t in range(len(states)):
                env.sim.set_state_from_flattened(states[t])
                env.sim.forward()
                eef = np.array(env.sim.data.site_xpos[eef_sid], np.float64)
                if t % PSTRIDE == 0:          # predicates are the expensive part: evaluated every PSTRIDE frames, held in between (subtask change latency <= PSTRIDE-1 frames)
                    # latch forward through satisfied subtasks (close needs the preceding open: the latch only moves in order)
                    while cur < len(sub) and env._eval_predicate(sub[cur]):
                        cur += 1
                    grasp = bool(obj_name and env.object_states_dict[obj_name].check_grasp())
                kind = sub[cur][0] if cur < len(sub) else "done"
                o = pos_of(env, obj_name) if obj_name else np.zeros(3)
                if kind in ("on", "in"):
                    tg = pos_of(env, sub[cur][2])
                elif kind in ("open", "close"):
                    tg = pos_of(env, sub[cur][1])
                elif kind == "grasp":
                    tg = pos_of(env, rec_name) if rec_name else np.zeros(3)
                else:
                    tg = pos_of(env, rec_name) if rec_name else np.zeros(3)
                if kind == "grasp":
                    role, rel = 1, o - eef
                elif kind in ("on", "in"):
                    role, rel = 2, tg - eef
                else:
                    role, rel = 0, np.zeros(3)
                cols["ep"].append(ep); cols["frame"].append(t); cols["eef"].append(eef); cols["obj"].append(o); cols["tgt"].append(tg)
                cols["chan"].append(np.r_[rel, role]); cols["role"].append(role); cols["subtask"].append(cur); cols["grasp"].append(grasp); cols["demo"].append(di)
                if first_succ < 0 and cur >= len(sub) - 1 and env._check_success():
                    first_succ = t
            meta[-1]["first_success"] = int(first_succ)
        if env is not None:
            env.close()
        A = {k: np.asarray(v) for k, v in cols.items()}
        for k in ("eef", "obj", "tgt", "chan"):
            A[k] = A[k].astype(np.float32)
        A["rel_o2t"] = A["tgt"] - A["obj"]
        A["rel_e2o"] = A["obj"] - A["eef"]
        A["rel_e2t"] = A["tgt"] - A["eef"]
        A["role"] = A["role"].astype(np.int8); A["subtask"] = A["subtask"].astype(np.int8)
        np.savez_compressed(dst + ".tmp.npz", **A)
        os.replace(dst + ".tmp.npz", dst)
        json.dump(meta, open(dst[:-4] + ".json", "w"))
        print(f"replay {task}: {len(rec['demos'])} eps, {len(A['ep'])} frames, {time.time() - t0:.0f}s, {os.path.getsize(dst) / 1e6:.2f} MB", flush=True)



# ------------------------------------------------------------------ map / vocab / index (numpy only)
def _align(exp, got):
    """Longest common subsequence of two length sequences (exact equality) -> list of (i_exp, j_got)."""
    n, m = len(exp), len(got)
    D = np.zeros((n + 1, m + 1), np.int32)
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            D[i, j] = D[i + 1, j + 1] + 1 if exp[i] == got[j] else max(D[i + 1, j], D[i, j + 1])
    i = j = 0
    out = []
    while i < n and j < m:
        if exp[i] == got[j]:
            out.append((i, j)); i += 1; j += 1
        elif D[i + 1, j] >= D[i, j + 1]:
            i += 1
        else:
            j += 1
    return out


def map_stage(a):
    """hdf5 demo -> lerobot episode_index.  LeRobot kept <= 100 demos per task, in order, truncated at the first success frame
    (remove_done_frames), so a demo's lerobot length is first_success + 1 + off (off chosen per task); aligned by LCS within the lerobot episodes of the same instruction text."""
    LL = lerobot_lengths()
    byText = {}
    for i in range(len(LL)):
        byText.setdefault(LL[i][1], []).append(i)
    index, report = {}, {}
    claimed = {}
    for p in sorted(glob.glob(OUT + "/mesa70/*.json")):
        task = os.path.basename(p)[:-5]
        meta = json.load(open(p))
        cand = byText.get(meta[0]["instruction"], [])
        got = [LL[i][0] for i in cand]
        best = (-1, None, None)
        for off in (0, 1, 2, -1, 3):
            exp = [(m["first_success"] + 1 + off) if m["first_success"] >= 0 else m["length"] for m in meta]
            pr = _align(exp, got)
            if len(pr) > best[0]:
                best = (len(pr), off, pr)
        n, off, pr = best
        for i, j in pr:
            e = cand[j]
            claimed.setdefault(e, []).append((task, n))
            meta[i]["episode_index"] = e
        json.dump(meta, open(p, "w"))
        report[task] = dict(n_demos=len(meta), n_lerobot=len(got), matched=n, off=off)
    # an episode claimed by two files (same instruction text): keep the file with the larger match count
    dup = {e: v for e, v in claimed.items() if len(v) > 1}
    json.dump(dict(report=report, n_dup_claims=len(dup)), open(OUT + "/map_report.json", "w"), indent=1)
    tot = sum(r["matched"] for r in report.values())
    print("matched", tot, "of lerobot", len(LL), "dup claims", len(dup))
    for t, r in report.items():
        if r["matched"] < min(r["n_demos"], r["n_lerobot"]):
            print("  partial", t, r)


def vocab(a):
    objs, recs, fines, index = {}, {}, {}, {}
    for p in sorted(glob.glob(OUT + "/mesa70/*.json")):
        for m in json.load(open(p)):
            for tab, k in ((objs, m["object_cat"]), (recs, m["receptacle_cat"]), (fines, m["receptacle_fine"])):
                if k is not None:
                    tab.setdefault(k, len(tab))
            if m["episode_index"] >= 0:
                index[m["episode_index"]] = dict(file=os.path.basename(p)[:-5], demo=m["demo"], hdf5_key=m["hdf5_key"], object_id=objs.get(m["object_cat"]),
                                                 receptacle_id=recs.get(m["receptacle_cat"]), receptacle_fine_id=fines.get(m["receptacle_fine"]), instruction=m["instruction"])
    json.dump(dict(object=objs, receptacle=recs, receptacle_fine=fines), open(OUT + "/vocab.json", "w"), indent=1)
    json.dump(index, open(OUT + "/index.json", "w"))
    print("episodes with a lerobot index", len(index), "objects", len(objs), "receptacles", len(recs), "fine", len(fines))




def table(a):
    """Frame-aligned lookup for training: <out>/xr_table.npz with rows in the LeRobot global `index` order (episode e starts at sum of lengths[:e]).
    chan[N,4] = [rel xyz (entity minus end effector, world frame, m), role], o2t[N,3] object-to-receptacle, e2o, e2t, subtask, grasp, valid[N] (False for episodes not replayed),
    and per episode the instructed object / receptacle / fine-receptacle ids (vocab.json) and n_frames."""
    LL = lerobot_lengths()
    L = np.array([LL[i][0] for i in range(len(LL))])
    start = np.r_[0, np.cumsum(L)]
    N = int(start[-1])
    T = dict(chan=np.zeros((N, 4), np.float32), o2t=np.zeros((N, 3), np.float32), e2o=np.zeros((N, 3), np.float32), e2t=np.zeros((N, 3), np.float32),
             subtask=np.zeros(N, np.int8), grasp=np.zeros(N, bool), valid=np.zeros(N, bool))
    ep_obj = -np.ones(len(L), np.int32); ep_rec = -np.ones(len(L), np.int32); ep_fine = -np.ones(len(L), np.int32)
    index = json.load(open(OUT + "/index.json"))
    for p in sorted(glob.glob(OUT + "/mesa70/*.npz")):
        A = np.load(p)
        name = os.path.basename(p)[:-4]
        meta = json.load(open(OUT + f"/mesa70/{name}.json"))
        for m in meta:
            e = m["episode_index"]
            if e < 0:
                continue
            sel = np.where(A["demo"] == m["demo"])[0]
            n = min(len(sel), int(L[e]))
            r = slice(int(start[e]), int(start[e]) + n)
            for k, src in (("chan", "chan"), ("o2t", "rel_o2t"), ("e2o", "rel_e2o"), ("e2t", "rel_e2t"), ("subtask", "subtask"), ("grasp", "grasp")):
                T[k][r] = A[src][sel[:n]]
            T["valid"][r] = True
            ie = index[str(e)]
            ep_obj[e], ep_rec[e], ep_fine[e] = ie["object_id"], ie["receptacle_id"], ie["receptacle_fine_id"]
    np.savez_compressed(OUT + "/xr_table.npz", ep_start=start, ep_len=L, ep_object=ep_obj, ep_receptacle=ep_rec, ep_receptacle_fine=ep_fine, **T)
    print("table rows", N, "valid frames", int(T["valid"].sum()), "episodes with labels", int((ep_obj >= 0).sum()), "of", len(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["extract", "replay", "map", "vocab", "table"])
    ap.add_argument("--tasks", default="all")
    ap.add_argument("--max-eps", type=int, default=None, help="extract only: first N demos (tests)")
    ap.add_argument("--demos", default=None, help="replay only: demo range a:b of the hdf5 order (output suffix .da-b)")
    ap.add_argument("--src-sfx", default="", help="replay only: suffix of the src file (tests use .n10)")
    ap.add_argument("--shard", default="0/1", help="i/n: every n-th task from i")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    OUT = a.out
    if os.environ.get("XR_PROFILE") and a.stage == "replay":
        import cProfile, pstats
        cProfile.run("replay(a)", "/coc/flash10/ksingara3/.claude_tmp/xr_prof.out")
        pstats.Stats("/coc/flash10/ksingara3/.claude_tmp/xr_prof.out").sort_stats("cumtime").print_stats(18)
    else:
        {"extract": extract, "replay": replay, "map": map_stage, "vocab": vocab, "table": table}[a.stage](a)
