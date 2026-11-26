"""XB rollout harness for RoboCasa365 pick-and-place cells (runs in bench/robocasa/venv_sim).  Client side only (policy servers are separate).

  --check                       placeability + forced-category check (policy free): per family and cell type, walk the pre-registered draw and keep the first K cells whose
                                object spawns (object category forced by obj_groups; up to 3 resets; spawn = reset without error and the episode instruction names the object).
                                Writes results/bench/xb/cells_robocasa_final.json (refuses to overwrite).
  --model M --cell-type T       rollouts: 20 episodes per cell, reset seeds 0-19, horizon = official task horizon, replan 5 (openpi servers), per-episode logs.
Per episode json: instruction, reset positions (objects + fixtures with sizes), success, steps; npz: object / eef / gripper tracks every 2 steps.
"""
import argparse, collections, json, os, sys, time
import numpy as np

ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
XB = ROOT + "/results/bench/xb"


def make_env(task, cat):
    import gymnasium as gym, robocasa  # noqa: F401
    return gym.make(f"robocasa/{task}", split="pretrain", seed=0, obj_groups=cat.replace(" ", "_"))


def scene_snapshot(env):
    e = env.unwrapped.env
    pos = {}
    for n, o in e.objects.items():
        try:
            pos[n] = np.asarray(e.sim.data.body_xpos[e.obj_body_id[n]], float).tolist()
        except Exception:
            pass
    fx = {}
    for n, f in getattr(e, "fixtures", {}).items():
        try:
            fx[n] = dict(pos=np.asarray(f.pos, float).tolist(), size=np.asarray(getattr(f, "size", [0, 0, 0]), float).tolist(), cls=type(f).__name__)
        except Exception:
            pass
    return pos, fx


def eef(env):
    try:
        sid = env.unwrapped.env.robots[0].eef_site_id
        sid = sid["right"] if isinstance(sid, dict) else sid
        return np.asarray(env.unwrapped.env.sim.data.site_xpos[sid], float)
    except Exception:
        return np.zeros(3)


def check():
    cells = json.load(open(f"{XB}/cells_robocasa.json"))
    prog_path = f"{XB}/rc_check_progress.json"
    prog = json.load(open(prog_path)) if os.path.exists(prog_path) else {}
    out = {"bench": "robocasa", "rule": "first 10 draws that spawn (3 resets); replaced by the next draw otherwise; logged", "families": {}}
    for fam, d in cells["families"].items():
        out["families"][fam] = {"task": d["task"]}
        for ctype, draw in (("heldout", d["heldout_draw"]), ("partner", d["partner_draw"])):
            keep, log = [], []
            for obj in draw:
                if len(keep) >= 10:
                    break
                key = f"{fam}|{ctype}|{obj}"
                if key not in prog:
                    ok, why = False, []
                    env = None
                    try:
                        env = make_env(d["task"], obj)
                        for t in range(3):
                            try:
                                o, _ = env.reset(seed=1000 + t)
                                lang = str(o["annotation.human.task_description"]).lower()
                                pos, fx = scene_snapshot(env)
                                if obj.lower() in lang:
                                    ok = True; break
                                why.append("instruction does not name object: " + lang)
                            except Exception as ex:      # noqa
                                why.append(type(ex).__name__ + ": " + str(ex)[:120])
                    except Exception as ex:              # noqa
                        why.append("make: " + type(ex).__name__ + ": " + str(ex)[:120])
                    finally:
                        try:
                            env.unwrapped.env.close()
                        except Exception:
                            pass
                    prog[key] = dict(object=obj, placeable=ok, why=why)
                    json.dump(prog, open(prog_path, "w"))
                    print(f"[check] {fam} {ctype} {obj}: {'OK' if ok else 'FAIL'} {why[:1]}", flush=True)
                log.append(prog[key])
                if prog[key]["placeable"]:
                    keep.append(obj)
            out["families"][fam][ctype] = dict(cells=keep, log=log, n_short=max(0, 10 - len(keep)))
    with open(f"{XB}/cells_robocasa_final.json", "x") as fh:
        json.dump(out, fh, indent=1)


class OpenPI:
    def __init__(self, port):
        from openpi_client import websocket_client_policy as wcp, image_tools
        self.c = wcp.WebsocketClientPolicy("127.0.0.1", port); self.it = image_tools

    def act(self, obs, lang):
        rs = lambda k: self.it.convert_to_uint8(self.it.resize_with_pad(np.ascontiguousarray(obs[k]), 224, 224))
        state = np.concatenate((obs["state.end_effector_position_relative"], obs["state.end_effector_rotation_relative"], obs["state.base_position"],
                                obs["state.base_rotation"], obs["state.gripper_qpos"]), axis=0)
        el = {"observation/image": rs("video.robot0_agentview_left"), "observation/wrist_image": rs("video.robot0_eye_in_hand"),
              "observation/right_image": rs("video.robot0_agentview_right"), "observation/state": state, "prompt": lang}
        return np.asarray(self.c.infer(el)["actions"])[:5]


class Groot15:
    kind = "dict"

    def __init__(self, port):
        import json_numpy, requests
        json_numpy.patch(); self.rq = requests; self.url = f"http://127.0.0.1:{port}/act"

    def act(self, obs, lang):
        o = {k: np.asarray(v)[None] for k, v in obs.items() if k.startswith("video.") or k.startswith("state.")}
        o["annotation.human.task_description"] = [lang]
        r = self.rq.post(self.url, json={"observation": o}); r.raise_for_status()
        a = r.json(); n = len(next(iter(a.values())))
        return [{k: np.asarray(v)[t] for k, v in a.items()} for t in range(min(n, 16))]


def roll(a):
    from robocasa.utils.dataset_registry_utils import get_task_horizon
    from robocasa.utils.env_utils import convert_action
    final = json.load(open(f"{XB}/cells_robocasa_final.json"))
    cl = []
    for fam, d in final["families"].items():
        for o in d[a.cell_type]["cells"]:
            cl.append((fam, d["task"], o))
    ci, cn = [int(x) for x in a.chunk.split("/")]
    cl = [c for k, c in enumerate(cl) if k % cn == ci]
    si, sn = [int(x) for x in a.shard.split("/")]
    pol = None if a.resetonly else {"pi0": OpenPI, "pi05": OpenPI, "groot15": Groot15}[a.model](a.port)
    t0 = time.time()
    for fam, task, obj in cl:
        cname = f"{fam.replace('->', '_to_')}__{obj.replace(' ', '_')}"
        cdir = f"{XB}/reset/robocasa/{cname}" if a.resetonly else f"{XB}/roll/robocasa/{a.model}/{cname}"
        os.makedirs(cdir, exist_ok=True)
        todo = [e for e in range(a.episodes) if e % sn == si and not os.path.exists(f"{cdir}/ep{e}.json")]
        if a.smoke:
            todo = todo[:1]
        if not todo:
            continue
        env = make_env(task, obj); horizon = get_task_horizon(task)
        for e in todo:
            obs, _ = env.reset(seed=e)
            lang = str(obs["annotation.human.task_description"])
            pos0, fx = scene_snapshot(env)
            if a.resetonly:
                json.dump(dict(task=task, family=fam, object=obj, episode=e, instruction=lang, reset_pos=pos0, fixtures=fx), open(f"{cdir}/ep{e}.json", "w")); continue
            plan = collections.deque(); done = False; t = 0; tr = []
            while t < horizon:
                if not plan:
                    plan.extend(pol.act(obs, lang))
                act = plan.popleft()
                obs, r, d, trunc, info = env.step(act if getattr(pol, "kind", "vec") == "dict" else convert_action(act))
                t += 1
                if t % 2 == 0:
                    pp, _ = scene_snapshot(env)
                    tr.append((t, pp, np.asarray(obs["state.gripper_qpos"], float).ravel(), eef(env)))
                if info["success"]:
                    done = True; break
            names = sorted(pos0)
            np.savez_compressed(f"{cdir}/ep{e}.npz", t=np.asarray([x[0] for x in tr]), names=np.asarray(names),
                                pos=np.asarray([[x[1].get(n, [np.nan] * 3) for n in names] for x in tr], float), grip=np.asarray([x[2] for x in tr]), eef=np.asarray([x[3] for x in tr]))
            json.dump(dict(bench="robocasa", model=a.model, cell_type=a.cell_type, family=fam, task=task, object=obj, episode=e, seed=e, instruction=lang, success=bool(done), steps=t,
                           horizon=horizon, reset_pos=pos0, fixtures=fx, final_pos=tr[-1][1] if tr else {}, wall_s=time.time() - t0), open(f"{cdir}/ep{e}.json", "w"))
            print(f"[robocasa/{a.model}/{a.cell_type}] {fam} {obj} ep{e} success={done} steps={t} ({time.time() - t0:.0f}s)", flush=True)
        env.unwrapped.env.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true"); ap.add_argument("--model", default="pi05"); ap.add_argument("--cell-type", default="partner", choices=["heldout", "partner"])
    ap.add_argument("--chunk", default="0/1"); ap.add_argument("--shard", default="0/1"); ap.add_argument("--port", type=int, default=8002)
    ap.add_argument("--episodes", type=int, default=20); ap.add_argument("--smoke", action="store_true"); ap.add_argument("--resetonly", action="store_true")
    a = ap.parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")
    check() if a.check else roll(a)
