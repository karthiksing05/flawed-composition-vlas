"""FLAWED_COMPOSITION prototype A, stage 1c (section 5.7): the canonical place frame's two inputs, computed the same way from demo rows
(training) and online (rollouts), from simulator state only.
  zrest  the held object's rest height: the minimum of its height over every not-held step so far in the episode, frozen when the
         hold (fa1b_common.Held) turns on; 0 while not held
  cyaw   the bearing, frozen at the hold's onset, from the held object's foot (its position with z - zrest + z_table) to the target
         reference, as a yaw angle (the place frame's x-axis points from the foot at onset toward the target); 0 while not held, and
         0 if the foot is within 1 cm of the target horizontally at the onset
"""
import math

import numpy as np

TABLE = 8                                                       # slot whose height is the table surface (factored_flow.TABLE)


class Canon:
    def __init__(self):
        self.zmin, self.on, self.zrest, self.cyaw = math.inf, False, 0.0, 0.0

    def step(self, held, obj_pos, tgt_pos_fn):
        """held: this step's held flag; obj_pos: the object's position; tgt_pos_fn(): (target position, table height), called only
        at the onset."""
        if not held:
            self.zmin = min(self.zmin, float(obj_pos[2]))
            self.on, self.zrest, self.cyaw = False, 0.0, 0.0
            return self.zrest, self.cyaw
        if not self.on:
            zr = self.zmin if math.isfinite(self.zmin) else float(obj_pos[2])
            pt, ztab = tgt_pos_fn()
            foot = np.array([obj_pos[0], obj_pos[1], obj_pos[2] - zr + ztab])
            u = np.asarray(pt[:2]) - foot[:2]
            self.zrest = zr
            self.cyaw = float(math.atan2(u[1], u[0])) if np.linalg.norm(u) > 0.01 else 0.0
            self.on = True
        return self.zrest, self.cyaw


def canon_rows(D, held, oi, ti):
    """Per training row (demos.npz order): zrest, cyaw from Canon stepped over each demo's kept steps."""
    n = len(held)
    key = D["task"].astype(np.int64) * 1000 + D["demo"]
    zr, cy = np.zeros(n, np.float32), np.zeros(n, np.float32)
    c = None
    for i in range(n):
        if i == 0 or key[i] != key[i - 1]:
            c = Canon()
        zr[i], cy[i] = c.step(bool(held[i] > 0.5), D["pos"][i, oi[i]], lambda i=i: (D["pos"][i, ti[i]], float(D["pos"][i, TABLE, 2])))
    return zr, cy
