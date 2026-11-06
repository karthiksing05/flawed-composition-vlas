"""T16 (docs/SCAFFOLD_RESULTS.md 33; m3_prereg sc_t16): structure inside the VLA's routine.  pi0.5, T8b's model, demos and budget; each frame's prompt names one role, chosen by the hold event; the event chooses again at test time.

    python experiments/sc_t16.py lex                -> results/flawed/sc_t16_lex.json; prints every instruction's two prompts and the event per task
    python experiments/sc_t16.py convert            LeRobot dataset lol/libero_goal8_roles
    python experiments/sc_t16.py norm
    python experiments/sc_t16.py serve --port P --ckpt DIR [--seed 13]
    python experiments/sc_t16.py read               -> results/flawed/sc_t16_read.json
"""
import argparse
import json
import logging
import os
import pathlib
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT, "/coc/flash10/ksingara3/robot/openpi/src"]
LEROBOT_HOME = "/coc/flash10/ksingara3/robot/data/hf/lerobot"
OUT = os.path.join(ROOT, "results", "flawed")
OD = os.path.join(OUT, "sc_t16")
LEX = os.path.join(OUT, "sc_t16_lex.json")
MODES = os.path.join(OUT, "sc_t12_modes.json")
KNOWN = tuple(f"K{i}" for i in range(1, 9))
READ_C = tuple(f"K{i}" for i in range(1, 7)) + tuple(f"N{i}" for i in range(1, 7))
B, MARGIN = 2000, 0.10
SUP = bool(os.environ.get("SC_SUP"))                                        # SCAFFOLD_RESULTS 33.1: the supervised stage: role phrases given by the task definitions
if SUP:
    OD = os.path.join(OUT, "sc_t16s")
NAME = "pi05_goal8_rsup" if SUP else "pi05_goal8_roles"
TAG = "sc_t16s" if SUP else "sc_t16"
PHRASE = dict(akita_black_bowl_1="the bowl", cream_cheese_1="the cream cheese", wine_bottle_1="the wine bottle", plate_1="the plate",
              flat_stove_1_cook_region="the stove", wooden_cabinet_1_top_side="the cabinet", wine_rack_1_top_region="the rack")
_GIVEN = {}


def given(instruction):
    """The supervised stage's prompts: the task definition's goal atom named by hand.  None for the drawer and push tasks, whose instructions stay whole."""
    if not _GIVEN:
        for t in n2()["tasks"]:
            _GIVEN[t["language"]] = None if t["compound"] in ("K7", "K8", "N7", "N8") else ("object: " + PHRASE[t["object"]], "target: " + PHRASE[t["target"]])
    return _GIVEN[instruction]


def n2():
    return json.load(open(os.path.join(OUT, "n2_tasks.json"), encoding="utf-8"))


def lang():
    return {t["compound"]: t["language"] for t in n2()["tasks"]}


def prompts(lex, words, instruction):
    """-> (object prompt, target prompt), or None for an instruction this routing leaves whole: no parse, a second clause ("and"), or a mode other than close."""
    from lifelong_ood.policy import parts_flow as PF
    if SUP:
        return given(instruction)
    toks = instruction.lower().split()
    r = PF.role_parse(lex, instruction)
    if r is None or "and" in toks or PF.verb_parse(words, instruction) != "close":
        return None
    o = [w for w in toks if lex["ref"].get(w) == r[0]]
    t = [w for w in toks if lex["ref"].get(w) == r[1]]
    return "object: the " + " ".join(o), "target: the " + " ".join(t)


def held_rows(state, actions, tw, lift):
    """The hold event over one episode's rows, from the dataset's own state and the previous action's gripper command."""
    from lifelong_ood.policy import parts_flow as PF
    det, prev, out = PF.ProprioHeld(tw, lift), -1.0, []
    for s, a in zip(state, actions):
        out.append(bool(det.step(s[6:8], prev, s[2])))
        prev = float(a[6])
    return np.array(out)


def lex():
    import fa1_common as FC
    import pp_p_train as PT
    from lifelong_ood.policy import data as D
    from lifelong_ood.policy import parts_flow as PF
    Dm = dict(np.load(os.path.join(FC.DATA, "demos.npz")))
    names = list(FC.KNOWN)
    H = PT.pgate_held(Dm, PT.pgate_fit_tasks(names))
    lx = PF.role_lexicon(PT.build_situations(names, Dm, H, None, motion=True), rarest=True)
    d = D.Demos()
    L = lang()
    st, ac = [], []
    for c in KNOWN:
        for e in d.episodes(d.task_of_text[L[c]]):
            ep = d.load(e)
            st.append(ep["state"])
            ac.append(ep["actions"])
    grip = np.concatenate(st)[:, 6:8]
    cmd = np.concatenate([np.r_[-1.0, a[:-1, 6]] for a in ac])
    pg = PF.pgate_fit(grip, cmd)
    modes = json.load(open(MODES, encoding="utf-8"))
    words = PF.verb_words(modes, {w for w, p in lx["pg"].items() if p > 0})
    out = dict(lex=lx, words=words, tw=pg["tw"], lift=pg["lift"])
    with open(LEX, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print("[t16] tw", pg, flush=True)
    for c in sorted(L):
        print("[t16]", c, L[c], "->", prompts(lx, words, L[c]), flush=True)
    i = 0
    for c in KNOWN:
        n = len(d.episodes(d.task_of_text[L[c]]))
        hs = [held_rows(st[j], ac[j], pg["tw"], pg["lift"]) for j in range(i, i + n)]
        i += n
        print("[t16]", c, "episodes", n, "with a hold", sum(h.any() for h in hs), "rows held %.3f" % np.mean(np.concatenate(hs)), flush=True)


def convert():
    os.environ["HF_LEROBOT_HOME"] = LEROBOT_HOME
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    from lifelong_ood.policy import data as D
    from lifelong_ood.policy import l90c_config as LC
    X = json.load(open(LEX, encoding="utf-8"))
    L = lang()
    d = D.Demos()
    REPO = LC.RSUP_REPO if SUP else LC.ROLES_REPO
    root = pathlib.Path(LEROBOT_HOME) / REPO
    assert not (root / "meta" / "info.json").exists(), f"{root} exists"
    features = {"image": {"dtype": "image", "shape": (256, 256, 3), "names": ["height", "width", "channel"]},
                "wrist_image": {"dtype": "image", "shape": (256, 256, 3), "names": ["height", "width", "channel"]},
                "state": {"dtype": "float32", "shape": (8,), "names": ["state"]},
                "actions": {"dtype": "float32", "shape": (7,), "names": ["actions"]}}
    ds = LeRobotDataset.create(repo_id=REPO, root=root, robot_type="panda", fps=10, features=features, image_writer_threads=8, image_writer_processes=4)
    k, counts = 0, {}
    for c in KNOWN:
        text = L[c]
        P = prompts(X["lex"], X["words"], text)
        for e in d.episodes(d.task_of_text[text]):
            ep = d.load(e, images=True)
            assert ep["task"] == text, (c, e)
            H = held_rows(ep["state"], ep["actions"], X["tw"], X["lift"])
            img, wrist = [D.Demos.decode(x) for x in ep["image"]], [D.Demos.decode(x) for x in ep["wrist_image"]]
            for i in range(len(ep["actions"])):
                task = text if P is None else P[int(H[i])]
                counts[task] = counts.get(task, 0) + 1
                ds.add_frame({"image": np.asarray(img[i]), "wrist_image": np.asarray(wrist[i]), "state": np.asarray(ep["state"][i], np.float32), "actions": np.asarray(ep["actions"][i], np.float32), "task": task})
            ds.save_episode()
            k += 1
            if k % 20 == 0:
                print(f"[t16] {k} episodes, at {c} ep {e}", flush=True)
    ds.stop_image_writer()
    with open(os.path.join(OUT, f"{TAG}_manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(repo_id=REPO, episodes=k, frames=int(ds.meta.total_frames), frames_per_prompt=counts), fh, indent=1)
    print("[t16] done", k, "episodes, frames per prompt", json.dumps(counts), flush=True)


def norm():
    import pyarrow.parquet as pq
    import openpi.shared.normalize as normalize
    from lifelong_ood.policy import data as D
    from lifelong_ood.policy import l90c_config as LC
    cfg = LC.train_config(NAME)
    REPO = LC.RSUP_REPO if SUP else LC.ROLES_REPO
    root = pathlib.Path(LEROBOT_HOME) / REPO
    files = sorted((root / "data").rglob("episode_*.parquet"))
    stats = {k: normalize.RunningStats() for k in ("state", "actions")}
    n, tasks = 0, set()
    for f in files:
        t = pq.read_table(f, columns=["state", "actions", "task_index"]).to_pydict()
        st, ac = np.asarray(t["state"], np.float32), np.asarray(t["actions"], np.float32)
        idx, _ = D.chunk_indices(len(ac), cfg.model.action_horizon)
        stats["state"].update(st)
        stats["actions"].update(ac[idx])
        tasks.update(t["task_index"])
        n += len(ac)
    normalize.save(cfg.assets_dirs / REPO, {k: s.get_statistics() for k, s in stats.items()})
    print(f"[t16] norm over {n} frames of {len(files)} episodes; distinct per-frame task indices {len(tasks)}", flush=True)


class Routed:
    """The hold event picks the prompt at each query: object prompt until the hold, target prompt during it.  Everything else is the wrapped policy's."""

    def __init__(self, policy, X, replan=5):
        from lifelong_ood.policy import parts_flow as PF
        self.PF, self.p, self.X, self.replan = PF, policy, X, replan
        self.det, self.prev, self.metadata = None, -1.0, getattr(policy, "metadata", {})

    def infer(self, obs):
        o = dict(obs)
        if o.get("meta/episode_start") or self.det is None:
            self.det, self.prev = self.PF.ProprioHeld(self.X["tw"], self.X["lift"]), -1.0
        st = np.asarray(o["observation/state"])
        on = bool(self.det.step(st[6:8], self.prev, st[2]))
        P = prompts(self.X["lex"], self.X["words"], str(o["prompt"]))
        if P is not None:
            o["prompt"] = P[int(on)]
        r = self.p.infer(o)
        self.prev = float(np.asarray(r["actions"])[self.replan - 1, 6])
        return r


def serve_cmd(a):
    from lifelong_ood.policy.backbone import PolicyBackbone
    from lifelong_ood.policy.serve import GuidedPolicy, serve, warm
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    bk = PolicyBackbone(NAME, checkpoint_dir=a.ckpt, device="cuda")
    pol = GuidedPolicy(bk, hooks=None, num_steps=10, seed=a.seed)
    warm(pol)
    serve(Routed(pol, json.load(open(LEX, encoding="utf-8"))), a.port)


def read():
    meta = n2()
    ks = meta["states"]["main"]

    def tid(c, l):
        return next(t["tid"] for t in meta["tasks"] if t["compound"] == c and t["layout"] == l)

    def client(path, t):
        by = {int(e["trial"]): e for e in json.load(open(path, encoding="utf-8"))["tasks"][str(t)]["episodes"]}
        return np.array([float(by[k]["success"]) for k in ks])

    def s_arm(c, l):
        return client(os.path.join(OD, f"s_main_{tid(c, l)}.json"), tid(c, l))

    def vla(c, l):
        return client(os.path.join(OUT, "sc_t8v", f"pi05g8_main_{tid(c, l)}.json"), tid(c, l))

    def diff(a, b):
        rng = np.random.default_rng(0)
        v = []
        for _ in range(B):
            t = rng.choice(a.shape[0], a.shape[0])
            s = rng.choice(a.shape[1], a.shape[1])
            v.append(float((a[t][:, s] - b[t][:, s]).mean()))
        return dict(a=float(a.mean()), b=float(b.mean()), delta=float((a - b).mean()), ci=[float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))])

    G = dict(known=READ_C[:6], new_pairings=READ_C[6:])
    res = dict(s_minus_vla={}, per_task={})
    for g, cs in G.items():
        for l in ("familiar", "new"):
            res["s_minus_vla"][f"{g}|{l}"] = diff(np.array([s_arm(c, l) for c in cs]), np.array([vla(c, l) for c in cs]))
    for c in list(READ_C) + ["K7", "K8", "N7", "N8"]:
        res["per_task"][c] = {l: float(s_arm(c, l).mean()) for l in ("familiar", "new")}
    x, kn = res["s_minus_vla"]["new_pairings|familiar"], res["s_minus_vla"]["known|familiar"]
    res["composition"] = "STRUCTURE WINS" if x["ci"][0] > MARGIN else ("STRUCTURE LOSES" if x["ci"][1] < -MARGIN else "NO CLEAR DIFFERENCE")
    res["cost"] = "NO COST" if kn["ci"][0] >= -MARGIN else "COST"
    print("[t16]", json.dumps(res), flush=True)
    with open(os.path.join(OUT, f"{TAG}_read.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("lex", "convert", "norm", "serve", "read"))
    ap.add_argument("--port", type=int)
    ap.add_argument("--ckpt")
    ap.add_argument("--seed", type=int, default=13)
    a = ap.parse_args()
    if a.what == "serve":
        serve_cmd(a)
    else:
        dict(lex=lex, convert=convert, norm=norm, read=read)[a.what]()
