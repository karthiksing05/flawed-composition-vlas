"""XB rollout harness for MESA / BiMESA (runs in the mesa uv venv).  Adapted from mesa-env/scripts/eval_server.py (same env construction, replan, step budget,
success rule); adds per-episode logging for the pre-registered failure classes:  reset positions of every object and fixture, instruction, success, and
object / end-effector / gripper tracks.  Resumable: an episode whose json exists is skipped.

  python xb_roll_mesa.py --bench mesa|bimesa --model <name> --cell-type heldout|partner --shard i/n --port P [--episodes 50] [--tasks a,b] [--smoke]
Output: results/bench/xb/roll/<bench>/<model>/<task>/ep<k>.json + ep<k>.npz
"""
import argparse, collections, json, os, sys, time
import numpy as np

ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
OUTR = ROOT + "/results/bench/xb/roll"


def eef_names(bench):
    return ["robot0_eef_pos"] if bench == "mesa" else ["robot0_eef_pos", "robot1_eef_pos"]


def snapshot(env):
    pos = {}
    for n, bid in env.obj_body_id.items():
        pos[n] = np.asarray(env.sim.data.body_xpos[bid], float).copy()
    return pos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True, choices=["mesa", "bimesa"]); ap.add_argument("--model", required=True)
    ap.add_argument("--cell-type", required=True, choices=["heldout", "partner"]); ap.add_argument("--shard", default="0/1")
    ap.add_argument("--port", type=int, default=8001); ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--tasks", default=""); ap.add_argument("--smoke", action="store_true"); ap.add_argument("--out", default=OUTR)
    ap.add_argument("--resetonly", action="store_true"); ap.add_argument("--stride", type=int, default=2); ap.add_argument("--chunk", default="0/1")
    a = ap.parse_args()
    from mesa import make_env
    from mesa.task_suites.eval_set import EvalSet
    from mesa.utils.eval_helpers import EvalArgs, ensure_not_scalar
    from openpi_client import websocket_client_policy as wcp

    cells = json.load(open(f"{ROOT}/results/bench/xb/cells_{a.bench}.json"))
    set_name = cells["eval_set_heldout"] if a.cell_type == "heldout" else cells["eval_set_partner"]
    task_ids = cells["heldout_tasks"] if a.cell_type == "heldout" else cells["partner_tasks"]
    if a.tasks:
        task_ids = [t for t in task_ids if t in a.tasks.split(",")]
    ci, cn = [int(x) for x in a.chunk.split("/")]
    task_ids = [t for k, t in enumerate(task_ids) if k % cn == ci]
    args = EvalArgs()
    args.controller_type = "joint_pos"
    if a.bench == "bimesa":
        args.robots = ["ReverseMountedYam", "ReverseMountedYam"]
        args.camera_names = ["egocentric", "robot0_eye_in_hand", "robot1_eye_in_hand"]
        args.state_keys = ["robot0_joint_pos", "robot0_gripper_jaw_width", "robot1_joint_pos", "robot1_gripper_jaw_width"]
    eval_set = EvalSet(set_name, split="eval")
    names = eval_set.get_task_names()
    idx = {n: i for i, n in enumerate(names)}
    si, sn = [int(x) for x in a.shard.split("/")]
    work = [(t, e) for t in task_ids for e in range(a.episodes)]
    work = [w for k, w in enumerate(work) if (k % sn) == si]
    if a.smoke:
        work = work[:2]
    client = None if a.resetonly else wcp.WebsocketClientPolicy("127.0.0.1", a.port)
    t0, nok, ndone = time.time(), 0, 0
    for tname, ep in work:
        od = f"{a.out}/{a.bench}/{a.model}/{tname}"
        jp = f"{od}/ep{ep}.json"
        if (a.resetonly and os.path.exists(f"{ROOT}/results/bench/xb/reset/{a.bench}/{tname}/ep{ep}.json")) or (not a.resetonly and os.path.exists(jp)):
            continue
        os.makedirs(od, exist_ok=True) if not a.resetonly else None
        tid = idx[tname]
        instr = str(eval_set.get_task_instruction(tid))
        inst_idx = ep % max(1, sum(1 for f in os.listdir(os.path.join(eval_set.task_suite_path, tname, eval_set.split)) if f.endswith(".json") and not f.endswith(".dex.json")))
        parsed = eval_set.get_parsed_problem(tid, inst_idx)
        init_state = eval_set.get_init_state(tid, inst_idx)
        env = make_env(parsed_problem=parsed, controller_type=args.controller_type, control_delta=args.control_delta, camera_heights=args.camera_height,
                       camera_widths=args.camera_width, camera_names=args.camera_names, has_renderer=False, deterministic_reset=init_state is not None,
                       **({"robots": args.robots} if args.robots is not None else {}))
        try:
            obs = env.stable_reset() if init_state is None else env.reset_to(init_state)
            cat = {n: getattr(o, "category_name", "") for n, o in list(env.objects_dict.items()) + list(env.fixtures_dict.items())}
            reset_pos = {n: p.tolist() for n, p in snapshot(env).items()}
            if a.resetonly:       # policy-free reset-state pass (features for the predictor)
                rd = f"{ROOT}/results/bench/xb/reset/{a.bench}/{tname}"; os.makedirs(rd, exist_ok=True)
                json.dump(dict(task=tname, episode=ep, instruction=instr, objects_categories=cat, reset_pos=reset_pos), open(f"{rd}/ep{ep}.json", "w")); continue
            plan = collections.deque(); t = 0; success = False; replay = 0; last = None
            done_sub, done_dist, nsub_prev = set(), set(), 0
            max_steps = args.max_steps
            tr_t, tr_pos, tr_eef, tr_grip = [], [], [], []
            first_success = None
            while t < max_steps:
                if t % a.stride == 0:
                    pos = snapshot(env)
                    tr_t.append(t); tr_pos.append(np.stack([pos[n] for n in cat]));
                    tr_eef.append(np.concatenate([np.asarray(obs[k], float).ravel()[:3] for k in eef_names(a.bench) if k in obs] or [np.zeros(3)]))
                    tr_grip.append(np.concatenate([ensure_not_scalar(np.asarray(obs[k], float)).ravel() for k in args.state_keys if "gripper" in k]))
                if not success and not plan:
                    images = {k: obs[f"{k}_image"] for k in args.camera_names}
                    state = np.concatenate([ensure_not_scalar(obs[k]) for k in args.state_keys], axis=-1)
                    chunk = client.infer({"images": images, "state": state, "prompt": instr})["actions"]
                    plan.extend(np.asarray(x, dtype=np.float32) for x in chunk[: args.replan_steps])
                action = last if success else plan.popleft()
                last = action
                obs, _, _, info = env.step(action.tolist()); t += 1
                if info.get("subtask_complete") is not None:
                    done_sub.update(info["subtask_complete"])
                if info.get("distractor_complete") is not None:
                    done_dist.update(info["distractor_complete"])
                if len(done_sub) > nsub_prev:
                    max_steps += (len(done_sub) - nsub_prev) * args.per_subtask_extra_steps; nsub_prev = len(done_sub)
                if bool(info.get("success", False)) and not success:
                    success, replay, first_success = True, 20, t
                if success:
                    replay -= 1
                    if replay <= 0:
                        break
            fin = snapshot(env)
            rec = dict(bench=a.bench, model=a.model, cell_type=a.cell_type, task=tname, episode=ep, init_state_idx=inst_idx, instruction=instr, success=bool(success),
                       first_success_step=first_success, steps=t, objects_categories=cat, reset_pos=reset_pos, final_pos={n: p.tolist() for n, p in fin.items()},
                       completed_subtasks=sorted(map(str, done_sub)), distractor_completed=len(done_dist) > 0, stride=a.stride, wall_s=time.time() - t0)
            np.savez_compressed(f"{od}/ep{ep}.npz", t=np.asarray(tr_t), names=np.asarray(list(cat)), pos=np.stack(tr_pos), eef=np.stack(tr_eef), grip=np.stack(tr_grip))
            json.dump(rec, open(jp, "w"))
            nok += int(success); ndone += 1
            print(f"[{a.bench}/{a.model}/{a.cell_type}] {tname} ep{ep} success={success} steps={t}  ({nok}/{ndone}; {time.time() - t0:.0f}s)", flush=True)
        finally:
            env.close()


if __name__ == "__main__":
    main()
