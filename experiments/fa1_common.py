"""FLAWED_COMPOSITION prototype A, stage 1 (section 5.5): the LIBERO-side helpers shared by fa1_data.py and fa1_roll.py.

ORACLE STATE.  Every arm sees the simulator's ground truth: the poses of LIBERO-Goal's movable objects and target sites (slots), the
top drawer's opening, the end effector and the gripper.  Roles (object, target) come from a template parse of the instruction
(`roles`): oracle grounding, the same for every arm.

Environment: LIBERO's ControlEnv without cameras (no renderer: the physics is the same simulator and controller as N2's
OffScreenRenderEnv; fa1_data.py's `equiv` mode checks it step for step on a GPU node).  LIBERO venv (py3.8).
"""
import math
import os
import re

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = "/coc/flash10/ksingara3/robot/data/fa1"                      # demo-derived arrays and checkpoints: outside git
RAW = "/coc/flash10/ksingara3/robot/data/libero_raw/libero_goal"
LIBBDDL = "/coc/flash10/ksingara3/robot/openpi/third_party/libero/libero/libero/bddl_files/libero_goal"
DUMMY = [0.0] * 6 + [-1.0]

# the slots: the scene's 4 movable objects (bodies) and 5 target sites (fixture / table regions)
ENT = ["akita_black_bowl_1", "cream_cheese_1", "wine_bottle_1", "plate_1",
       "flat_stove_1_cook_region", "wooden_cabinet_1_top_side", "wine_rack_1_top_region", "wooden_cabinet_1_top_region",
       "main_table_stove_front_region"]
MOVABLE = ENT[:4]
# simple shape features (radius, height in m; hand-set from b1_3_gen footprints and the meshes), the same for every arm
SIZE = {"akita_black_bowl_1": (0.06, 0.05), "cream_cheese_1": (0.045, 0.03), "wine_bottle_1": (0.045, 0.28), "plate_1": (0.10, 0.02),
        "flat_stove_1_cook_region": (0.09, 0.0), "wooden_cabinet_1_top_side": (0.10, 0.0), "wine_rack_1_top_region": (0.10, 0.0),
        "wooden_cabinet_1_top_region": (0.10, 0.0), "main_table_stove_front_region": (0.04, 0.0)}

# the 8 known LIBERO-Goal tasks (N2's K1-K8) and their raw demo files
KNOWN = {"K1": "put_the_bowl_on_the_plate", "K2": "put_the_bowl_on_the_stove", "K3": "put_the_bowl_on_top_of_the_cabinet",
         "K4": "put_the_wine_bottle_on_top_of_the_cabinet", "K5": "put_the_wine_bottle_on_the_rack",
         "K6": "put_the_cream_cheese_in_the_bowl", "K7": "open_the_top_drawer_and_put_the_bowl_inside",
         "K8": "push_the_plate_to_the_front_of_the_stove"}

NOUN = {"bowl": "akita_black_bowl_1", "cream cheese": "cream_cheese_1", "wine bottle": "wine_bottle_1", "plate": "plate_1",
        "stove": "flat_stove_1_cook_region", "cabinet": "wooden_cabinet_1_top_side", "rack": "wine_rack_1_top_region"}


def roles(instr):
    """Template parse of a LIBERO-Goal / N2 instruction -> (object entity, target entity).  ORACLE grounding (a rule)."""
    s = " ".join(instr.lower().split())
    m = re.fullmatch(r"open the top drawer and put the (.+) inside", s)
    if m:
        return NOUN[m.group(1)], "wooden_cabinet_1_top_region"
    m = re.fullmatch(r"push the (.+) to the front of the stove", s)
    if m:
        return NOUN[m.group(1)], "main_table_stove_front_region"
    m = re.fullmatch(r"put the (.+?) (?:on top of the|on the|in the) (.+)", s)
    if m:
        return NOUN[m.group(1)], NOUN[m.group(2)]
    raise ValueError(f"no template matches {instr!r}")


def make_env(bddl):
    from libero.libero.envs.env_wrapper import ControlEnv
    return ControlEnv(bddl_file_name=bddl, use_camera_obs=False, has_renderer=False, has_offscreen_renderer=False)


def fresh_env(bddl, seed=7):
    """N2's branch path: a fresh env, numpy and the env re-seeded, per episode (libero_client.py, resets = 0)."""
    np.random.seed(seed)
    env = make_env(bddl)
    env.seed(seed)
    return env


def drawer_joint(env):
    m = env.env.sim.model
    names = [m.joint_id2name(i) for i in range(m.njnt)]
    js = [n for n in names if n and "wooden_cabinet_1" in n and "top" in n]
    assert len(js) == 1, names
    return js[0]


def _yaw_of_mat(R):
    v = R[:2, 0] if np.linalg.norm(R[:2, 0]) > 0.3 else R[:2, 1]
    return float(math.atan2(v[1], v[0]))


def _quat_mat(q_xyzw):
    x, y, z, w = q_xyzw
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def scene(env, obs, djoint):
    """The oracle readout: ee_pos [3], ee_rot [3, 3], grip [2], pos [9, 3], yaw [9], drawer (top drawer joint qpos)."""
    inner, sim = env.env, env.env.sim
    pos, yaw = np.zeros((len(ENT), 3)), np.zeros(len(ENT))
    for i, n in enumerate(ENT):
        if n in MOVABLE:
            bid = inner.obj_body_id[n]
            pos[i] = sim.data.body_xpos[bid]
            yaw[i] = _yaw_of_mat(np.array(sim.data.body_xmat[bid]).reshape(3, 3))
        else:
            sid = sim.model.site_name2id(n)
            pos[i] = sim.data.site_xpos[sid]
            yaw[i] = _yaw_of_mat(np.array(sim.data.site_xmat[sid]).reshape(3, 3))
    jid = sim.model.joint_name2id(djoint)
    return dict(ee_pos=np.asarray(obs["robot0_eef_pos"], np.float64), ee_rot=_quat_mat(np.asarray(obs["robot0_eef_quat"], np.float64)),
                grip=np.asarray(obs["robot0_gripper_qpos"], np.float64), pos=pos, yaw=yaw,
                drawer=float(sim.data.qpos[sim.model.jnt_qposadr[jid]]))


def movable_positions(env):
    """libero_client.movable_positions (every movable object of the scene, objects_dict order)."""
    names = list(env.env.objects_dict.keys())
    return names, np.array([np.array(env.env.sim.data.body_xpos[env.env.obj_body_id[n]]) for n in names])
