"""XB rollout harness for RoboTwin 2.0 place_a2b_{left,right} (runs in bench/robotwin/env).  Wraps RoboTwin's own eval loop (scripts/eval_policy_xpolicylab.py, imported
unchanged) and COPIES eval_remote_policy with per-episode logging.  The task class is patched by exec of a transformed copy of envs/<task>.py (the upstream clone is never
edited): the four np.random.choice calls that pick A's and B's model name / id are replaced by FORCED values; the patched source is written to results/bench/xb/rt_patched/.

Modes (all from the repo root; cwd is switched to the RoboTwin clone):
  --mode check   forced-pair check: 20 forced resets for random draws, the requested (name, id) must come back (policy free).  Writes results/bench/xb/rt_check_<task>.json
  --mode prep    policy-free: per cell scan seeds from 100000 upward, forced pair, run the scripted expert (the pre-registered expert-seed filter), keep the first 10 seeds the
                 expert solves, store episode_info (for the language instruction) and A/B reset poses.  Writes results/bench/xb/rt_seeds/<task>/<A>__<B>.json
  --mode roll    policy rollouts on the stored seeds (no expert re-run).  Writes results/bench/xb/roll/robotwin/<model>/<task>/<cell>/ep<k>.json + npz
"""
import argparse, importlib, json, os, re, sys, time, types
import numpy as np

ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
RT = "/coc/flash10/ksingara3/robot/bench/robotwin/RoboTwin"
XB = ROOT + "/results/bench/xb"
os.chdir(RT)
for p in (RT, RT + "/scripts", RT + "/description/utils"):
    if p not in sys.path:
        sys.path.insert(0, p)
E = importlib.import_module("eval_policy_xpolicylab")
import envs  # noqa: E402

OBJECT_RX = r"np\.random\.choice\((object_list(?:_np)?)\)"


def install_patched(task):
    src = open(f"{RT}/envs/{task}.py", encoding="utf-8").read()
    n0 = len(src)
    src = src.replace("\nclass ", "\nFORCED = {}\n\n\nclass ", 1)
    src, nA = re.subn(r"(self\.selected_modelname_A = )" + OBJECT_RX, r'\1FORCED["A_name"]', src)
    src, nB = re.subn(r"(self\.selected_modelname_B = )" + OBJECT_RX, r'\1FORCED["B_name"]', src)
    src, nB2 = re.subn(r"(\n\s+while self\.selected_modelname_B == self\.selected_modelname_A:\n\s+)self\.selected_modelname_B = " + OBJECT_RX, r"\1break", src)
    src, nAi = re.subn(r"(self\.selected_model_id_A = )np\.random\.choice\(available_model_ids\)", r'\1FORCED["A_id"]', src)
    src, nBi = re.subn(r"(self\.selected_model_id_B = )np\.random\.choice\(available_model_ids\)", r'\1FORCED["B_id"]', src)
    assert (nA, nB, nAi, nBi) == (1, 2, 1, 1) or (nA, nAi, nBi) == (1, 1, 1), (task, nA, nB, nAi, nBi)
    os.makedirs(XB + "/rt_patched", exist_ok=True)
    pp = f"{XB}/rt_patched/{task}.py"
    if not os.path.exists(pp):
        open(pp, "w", encoding="utf-8").write(src)
    mod = types.ModuleType(f"envs.{task}"); mod.__package__ = "envs"; mod.__file__ = pp
    exec(compile(src, pp, "exec"), mod.__dict__)
    sys.modules[f"envs.{task}"] = mod
    return mod


def split_id(s):
    name, base = s.split("/")
    return name, int(base.replace("base", ""))


def make_ctx(task, policy="X_VLA", port=0, action_type="ee"):
    usr = dict(bench_name="RoboTwin", task_name=task, env_cfg_type="aloha_agilex", policy_name=policy, host="localhost", port=port, protocol="ws", root_dir=RT, device_id="0",
               seed=0, task_config="demo_clean", instruction_type=None, xpolicylab_root=RT + "/XPolicyLab", eval_batch=False, action_type=action_type, ckpt_name="xb")
    args, emb = E.load_task_args(usr)
    itype = E.resolve_eval_instruction(usr, args)
    args["eval_instruction"] = itype; usr["instruction_type"] = itype
    args["eval_video_log"] = False; args["eval_mode"] = True; args["render_freq"] = 0
    mod = install_patched(task)
    env = E.class_decorator(task)
    usr["left_arm_dim"] = len(args["left_embodiment_config"]["arm_joints_name"][0]); usr["right_arm_dim"] = len(args["right_embodiment_config"]["arm_joints_name"][1])
    return usr, args, itype, mod, env


def pose(o):
    return np.asarray(o.get_pose().p, float).tolist()


def set_forced(mod, A, B):
    (an, ai), (bn, bi) = split_id(A), split_id(B)
    mod.FORCED.update(A_name=an, A_id=ai, B_name=bn, B_id=bi)


def check_forced(env, A, B):
    return (f"{env.selected_modelname_A}/base{int(env.selected_model_id_A)}" == A) and (f"{env.selected_modelname_B}/base{int(env.selected_model_id_B)}" == B)


def cellkey(A, B):
    return A.replace("/", "-") + "__" + B.replace("/", "-")


def mode_check(a):
    cells = json.load(open(f"{XB}/cells_robotwin.json"))["tasks"][a.task]
    usr, args, itype, mod, env = make_ctx(a.task)
    out, ok = [], 0
    for A, B in cells["heldout_draw"][:10] + cells["partner_draw"][:10]:
        set_forced(mod, A, B)
        try:
            env.setup_demo(now_ep_num=0, seed=100000, is_test=True, **args)
            good = check_forced(env, A, B); res = dict(A=A, B=B, returned=[f"{env.selected_modelname_A}/base{int(env.selected_model_id_A)}", f"{env.selected_modelname_B}/base{int(env.selected_model_id_B)}"], ok=bool(good))
        except Exception as ex:  # noqa
            res = dict(A=A, B=B, ok=False, error=type(ex).__name__ + ": " + str(ex)[:200])
        E.safe_close_env(env)
        ok += int(res["ok"]); out.append(res); print(res, flush=True)
    json.dump(dict(task=a.task, n=len(out), n_ok=ok, items=out), open(f"{XB}/rt_check_{a.task}.json", "x"), indent=1)


def mode_prep(a):
    cells = json.load(open(f"{XB}/cells_robotwin.json"))["tasks"][a.task]
    ci, cn = [int(x) for x in a.chunk.split("/")]
    todo = [("heldout", p) for p in cells["heldout_draw"][:a.ndraw]] + [("partner", p) for p in cells["partner_draw"][:a.ndraw]]
    todo = [t for k, t in enumerate(todo) if k % cn == ci]
    usr, args, itype, mod, env = make_ctx(a.task)
    os.makedirs(f"{XB}/rt_seeds/{a.task}", exist_ok=True)
    for ctype, (A, B) in todo:
        fp = f"{XB}/rt_seeds/{a.task}/{cellkey(A, B)}.json"
        if os.path.exists(fp):
            continue
        set_forced(mod, A, B)
        seeds, skipped, now = [], [], 100000
        t0 = time.time()
        while len(seeds) < a.n and now < 100000 + a.max_scan:
            args["render_freq"] = 0
            rec = None
            try:
                env.setup_demo(now_ep_num=len(seeds), seed=now, is_test=True, **args)
                forced_ok = check_forced(env, A, B)
                reset = dict(A=pose(env.object), B=pose(env.target_object))
                info = env.play_once()
                good = bool(env.plan_success and env.check_success()) and forced_ok
                E.safe_close_env(env)
                if good:
                    rec = dict(seed=now, episode_info=info, reset_pos=reset)
                else:
                    skipped.append(dict(seed=now, why="expert fail" if forced_ok else "forced pair mismatch"))
            except Exception as ex:  # noqa
                E.safe_close_env(env); skipped.append(dict(seed=now, why=type(ex).__name__ + ": " + str(ex)[:120]))
            if rec:
                seeds.append(rec)
            now += 1
        json.dump(dict(task=a.task, cell_type=ctype, A=A, B=B, n=len(seeds), feasible=len(seeds) >= a.n, seeds=seeds, skipped=skipped, wall_s=time.time() - t0), open(fp, "w"), default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
        print(f"[prep {a.task}] {ctype} {A} {B}: {len(seeds)} seeds, skipped {len(skipped)} ({time.time() - t0:.0f}s)", flush=True)


def mode_roll(a):
    final = json.load(open(f"{XB}/cells_robotwin_final.json"))
    cl = [(a.task, c) for c in final["tasks"][a.task][a.cell_type]]
    ci, cn = [int(x) for x in a.chunk.split("/")]
    cl = [c for k, c in enumerate(cl) if k % cn == ci]
    usr, args, itype, mod, env = make_ctx(a.task, policy=a.policy, port=a.port, action_type=a.action_type)
    client = E.build_policy_client(usr)
    t0 = time.time()
    try:
        for task, (A, B) in cl:
            seeds = json.load(open(f"{XB}/rt_seeds/{task}/{cellkey(A, B)}.json"))["seeds"][:a.n]
            cdir = f"{XB}/roll/robotwin/{a.model}/{task}/{cellkey(A, B)}"
            os.makedirs(cdir, exist_ok=True)
            set_forced(mod, A, B)
            for k, s in enumerate(seeds):
                if os.path.exists(f"{cdir}/ep{k}.json") or (a.smoke and k >= 1):
                    continue
                now = s["seed"]
                args["render_freq"] = 0
                env.setup_demo(now_ep_num=k, seed=now, is_test=True, **args)
                forced_ok = check_forced(env, A, B)
                instruction = E.build_instruction(args, s["episode_info"], itype, a.n, now)
                env.set_instruction(instruction=instruction)
                reset = dict(A=pose(env.object), B=pose(env.target_object))
                E.prepare_policy_case(client, task, now, instruction, a.action_type); E.reset_policy(client)
                succ, steps, err, tr = False, 0, None, []
                try:
                    while not E.is_episode_end(env):
                        obs = env.get_obs()
                        client.call(func_name="update_obs", obs=E.robotwin_obs_to_xpolicylab(obs, instruction=env.get_instruction(), env_idx=0, frequency=30, task_env=env))
                        chunk = E.normalize_action_chunk(client.call(func_name="get_action"))
                        for ai, act in enumerate(chunk):
                            fa, at = E.xpolicylab_action_to_robotwin(act, action_type=a.action_type, current_observation=obs)
                            env.take_action(fa, action_type=at); steps += 1
                            try:
                                tr.append([steps] + pose(env.object) + pose(env.target_object) + [float(env.robot.get_left_gripper_val()), float(env.robot.get_right_gripper_val())])
                            except Exception:
                                pass
                            if env.eval_success:
                                succ = True; break
                            if E.is_episode_end(env) or ai + 1 == len(chunk):
                                break
                            obs = env.get_obs()
                            client.call(func_name="update_obs", obs=E.robotwin_obs_to_xpolicylab(obs, instruction=env.get_instruction(), env_idx=0, frequency=30, task_env=env))
                        if succ:
                            break
                except Exception as ex:  # noqa
                    err = type(ex).__name__ + ": " + str(ex)[:200]
                E.notify_trial_end(client, task, now, succ)
                E.safe_close_env(env)
                np.savez_compressed(f"{cdir}/ep{k}.npz", track=np.asarray(tr, float))
                json.dump(dict(bench="robotwin", model=a.model, cell_type=a.cell_type, task=task, A=A, B=B, episode=k, seed=now, instruction=instruction, forced_ok=forced_ok, reset_pos=reset,
                               success=bool(succ), steps=steps, error=err, step_lim=env.step_lim, wall_s=time.time() - t0), open(f"{cdir}/ep{k}.json", "w"))
                print(f"[robotwin/{a.model}/{a.cell_type}] {task} {A} {B} ep{k} seed={now} success={succ} steps={steps} err={err} ({time.time() - t0:.0f}s)", flush=True)
    finally:
        try:
            E.close_policy_client(client)
        except Exception:
            pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["check", "prep", "roll"]); ap.add_argument("--task", required=True)
    ap.add_argument("--chunk", default="0/1"); ap.add_argument("--n", type=int, default=10); ap.add_argument("--ndraw", type=int, default=24); ap.add_argument("--max-scan", type=int, default=60)
    ap.add_argument("--model", default="xvla"); ap.add_argument("--policy", default="X_VLA"); ap.add_argument("--cell-type", default="partner", choices=["heldout", "partner"])
    ap.add_argument("--port", type=int, default=6000); ap.add_argument("--smoke", action="store_true"); ap.add_argument("--action-type", default="ee")
    a = ap.parse_args()
    {"check": mode_check, "prep": mode_prep, "roll": mode_roll}[a.mode](a)
