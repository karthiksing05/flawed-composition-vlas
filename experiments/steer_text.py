"""SR 74 steerability probe: instruction templates, swap candidates and entity positions shared by the client (LIBERO venv, py3.8),
the server wrapper (openpi / GR00T / SmolVLA venvs) and the read (CPU).  No heavy imports at module level; must parse under python 3.8.

Instruction template (the n2 tasks' own, results/flawed/n2_tasks.json; scene 2's, experiments/s2_tasks.py):
    "put " + OBJ[object] + " " + TGT[target]          e.g. "put the bowl on top of the cabinet", "put the white mug on the left plate"
Swaps replace the object phrase (grasp phase) or the target phrase (place phase) and keep everything else.
"""
import numpy as np

SCENES = {
    1: dict(
        obj={"akita_black_bowl_1": "the bowl", "cream_cheese_1": "the cream cheese", "wine_bottle_1": "the wine bottle",
             "plate_1": "the plate"},
        tgt={"plate_1": "on the plate", "flat_stove_1_cook_region": "on the stove", "wooden_cabinet_1_top_side": "on top of the cabinet",
             "akita_black_bowl_1": "in the bowl", "wine_rack_1_top_region": "on the rack"},
        # sc_t16.PHRASE (R1's role phrases), the names a role-phrase front end writes
        phrase={"akita_black_bowl_1": "the bowl", "cream_cheese_1": "the cream cheese", "wine_bottle_1": "the wine bottle",
                "plate_1": "the plate", "flat_stove_1_cook_region": "the stove", "wooden_cabinet_1_top_side": "the cabinet",
                "wine_rack_1_top_region": "the rack"},
        absent_obj=["the ketchup", "the alphabet soup", "the tomato sauce", "the milk"],
        absent_tgt=["on the tray", "in the basket"],
        nonsense_obj="the glorp", nonsense_tgt="on the zibble",
        probe_tids=[0, 2, 4, 6, 10],                                  # K1 K2 K3 K4 K6 (the K set of Tables A and B)
    ),
    2: dict(
        obj={"porcelain_mug_1": "the white mug", "white_yellow_mug_1": "the yellow and white mug", "red_coffee_mug_1": "the red mug"},
        tgt={"plate_1": "on the left plate", "plate_2": "on the right plate"},
        phrase={"porcelain_mug_1": "the white mug", "white_yellow_mug_1": "the yellow and white mug", "red_coffee_mug_1": "the red mug",
                "plate_1": "the left plate", "plate_2": "the right plate"},
        absent_obj=["the green mug", "the blue mug"],
        absent_tgt=["on the middle plate", "on the tray"],
        nonsense_obj="the glorp", nonsense_tgt="on the zibble",
        probe_tids=[0, 1],                                            # K1 K2
    ),
}


def lang(scene, o, t):
    S = SCENES[scene]
    return "put " + S["obj"][o] + " " + S["tgt"][t]


def cands(scene, phase, o, t):
    """Swap candidates in a fixed order: the other movable objects (grasp; not the trained target entity) or the other targets
    present (place; not the trained one and not the held object)."""
    S = SCENES[scene]
    if phase == "grasp":
        return [x for x in S["obj"] if x != o and x != t]
    return [x for x in S["tgt"] if x != t and x != o]


def cf_text(scene, phase, o, t, cand):
    if phase == "grasp":
        return lang(scene, cand, t)
    return lang(scene, o, cand)


def word_text(scene, phase, o, t, kind, i):
    """The absent-word and nonsense-word instruction of the i-th such query: (full text, slot phrase)."""
    S = SCENES[scene]
    if phase == "grasp":
        w = S["absent_obj"][i % len(S["absent_obj"])] if kind == "absent" else S["nonsense_obj"]
        return "put " + w + " " + S["tgt"][t], w
    w = S["absent_tgt"][i % len(S["absent_tgt"])] if kind == "absent" else S["nonsense_tgt"]
    return "put " + S["obj"][o] + " " + w, w


def phrases(scene, o, t):
    S = SCENES[scene]
    return ["object: " + S["phrase"][o], "target: " + S["phrase"][t]]


def phrases_word(scene, phase, o, t, w):
    """Role phrases with the slot's phrase replaced by a word phrase (absent / nonsense word through a role-phrase front end)."""
    S = SCENES[scene]
    wp = w
    for pre in ("on top of ", "on ", "in "):
        if wp.startswith(pre):
            wp = wp[len(pre):]
            break
    if phase == "grasp":
        return ["object: " + w, "target: " + S["phrase"][t]]
    return ["object: " + S["phrase"][o], "target: " + wp]


def entity_pos(raw_env, name):
    """Simulator position of an entity (measurement only): a movable object's body, a region / side's site."""
    inner = raw_env.env
    sim = inner.sim
    if name in inner.obj_body_id:
        return np.array(sim.data.body_xpos[inner.obj_body_id[name]], np.float64)
    return np.array(sim.data.site_xpos[sim.model.site_name2id(name)], np.float64)


def chunk_delta(actions, eef, absolute, n):
    """Displacement a chunk commands over its first n steps: the sum of the xyz deltas (delta controllers) or the n-th target minus
    the hand (absolute controllers: X-VLA)."""
    a = np.asarray(actions, np.float64)
    n = min(int(n), len(a))
    if absolute:
        return a[n - 1, :3] - np.asarray(eef, np.float64)
    return a[:n, :3].sum(0)


def angle(u, v):
    u, v = np.asarray(u, np.float64), np.asarray(v, np.float64)
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu < 1e-9 or nv < 1e-9:
        return float("nan")
    return float(np.arccos(np.clip(np.dot(u, v) / (nu * nv), -1.0, 1.0)))
