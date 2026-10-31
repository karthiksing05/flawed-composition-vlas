"""FLAWED_COMPOSITION prototype A, stage 1b (section 5.6): the PHYSICAL-CONTACT phase signal shared by fa1b_data.py and fa1b_roll.py.
Stage 1's helpers (fa1_common.py) are reused unchanged; this file only adds the held test and the grasp offset.

Read from simulator state only (oracle, like stage 1's poses), never from the commanded gripper:
  touching(o) = (left, right): o is in contact with the gripper's left / right finger (MuJoCo contacts of the current state; a finger
                = any geom of the gripper's finger bodies, pads included; an object geom = any geom whose top-level body is o's)
  lift(o)     = o's height minus its height at the episode's first step (demos: the demo's first kept step; rollouts: the first
                policy step, after the settle steps)
HELD (the phase signal; a two-state machine stepped once per simulator step, `Held`):
  turns ON  when o touches BOTH fingers and lift(o) >= LIFT (0.01 m): the object is lifted in the gripper;
  stays ON  while o touches at least one finger;
  turns OFF when o touches neither finger.
The grasp offset is o's position minus the end effector's, world frame, captured at the step HELD turns on and kept constant while it
stays on (0 while off).
"""
import numpy as np

LIFT = 0.01


def finger_geoms(env):
    """(left geom ids, right geom ids) of the robot's gripper fingers."""
    m = env.env.sim.model
    left, right = set(), set()
    for g in range(m.ngeom):
        b = m.body_id2name(int(m.geom_bodyid[g])) or ""
        if not b.startswith("gripper0_"):
            continue
        if "leftfinger" in b or "finger_joint1_tip" in b:
            left.add(g)
        elif "rightfinger" in b or "finger_joint2_tip" in b:
            right.add(g)
    assert left and right, "no finger geoms found"
    return left, right


def object_geoms(env, name):
    m = env.env.sim.model
    root = int(m.body_rootid[env.env.obj_body_id[name]])
    return {g for g in range(m.ngeom) if int(m.body_rootid[int(m.geom_bodyid[g])]) == root}


class Contact:
    """Per env: the finger and object geom sets, then touching(name) on the current simulator state."""

    def __init__(self, env, names):
        self.env = env
        self.left, self.right = finger_geoms(env)
        self.obj = {n: object_geoms(env, n) for n in names}

    def touching(self, name):
        d = self.env.env.sim.data
        og, lt, rt = self.obj[name], False, False
        for i in range(d.ncon):
            c = d.contact[i]
            a, b = int(c.geom1), int(c.geom2)
            for x, y in ((a, b), (b, a)):
                if y in og:
                    lt = lt or x in self.left
                    rt = rt or x in self.right
        return lt, rt

    def pos(self, name):
        return np.array(self.env.env.sim.data.body_xpos[self.env.env.obj_body_id[name]], np.float64)


class Held:
    """The phase signal and the grasp offset for one object over one episode (step once per simulator step)."""

    def __init__(self):
        self.on, self.off, self.z0 = False, np.zeros(3), None

    def step(self, lt, rt, obj_pos, ee_pos):
        obj_pos = np.asarray(obj_pos, np.float64)
        if self.z0 is None:
            self.z0 = float(obj_pos[2])
        if not self.on and lt and rt and obj_pos[2] - self.z0 >= LIFT:
            self.on, self.off = True, obj_pos - np.asarray(ee_pos, np.float64)
        elif self.on and not (lt or rt):
            self.on, self.off = False, np.zeros(3)
        return self.on, self.off.copy()
