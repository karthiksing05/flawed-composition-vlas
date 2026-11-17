"""RAA place-step instrumentation (EXPLORATORY diagnostic; plan docs/notes/raa_instr_plan.md).  LIBERO venv (py3.8).

A pure observer around the LIBERO env.  It is installed UNDER raa.py's and rg.py's own env wrappers, so each sees the executed action
(after ReGuide's override for rgp) and nothing it does changes any action: only reads of sim.data after each env.step, no render.

    python experiments/raa_instr.py raa client --arm raa ...        (raa.py client's arguments)
    python experiments/raa_instr.py rg  client --arm rgp ...        (rg.py client's arguments)

Per episode one file  <dir of --out>/instr_<arm>_<tid>_t<trial>.npz  with one row per control step (n = env steps since set_init_state,
the first 10 are libero_client's settle steps; n = -1 is the state right after set_init_state):
  a(7) executed action | eef(3) eefq(4 xyzw) grip(2) gripper qpos | obj(3) objq(4 wxyz) the instructed object's body pose
  tgt(3) target xyz (body origin, or site) | z_ok contact xy under dsite(3)  the terms of LIBERO's On(obj, tgt) = tgt.check_ontop(obj):
    body target:  tgt_z <= obj_z, contact(tgt, obj), body-origin xy distance < 0.03   (z_ok, contact, xy)
    site target:  site.under(site_pos, site_mat, obj_pos) and contact(parent, obj)    (under, contact; dsite = site-frame offset)
  on_terms = the product of the terms recomputed here | on_full = tgt_state.check_ontop(obj_state) | succ = env._check_success() | done
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]
KEYS = ("n", "a", "eef", "eefq", "grip", "obj", "objq", "tgt", "z_ok", "contact", "xy", "under", "dsite", "on_terms", "on_full", "succ",
        "done")
WIDTH = dict(n=1, a=7, eef=3, eefq=4, grip=2, obj=3, objq=4, tgt=3, z_ok=1, contact=1, xy=1, under=1, dsite=3, on_terms=1, on_full=1,
             succ=1, done=1)


def install(arm, tid, outdir, branches):
    import fc_n2_client as FN
    LC = FN.LC
    TASK = FN.N2["tasks"][tid]
    OE, TE = TASK["object"], TASK["target"]
    br = json.load(open(branches, encoding="utf-8"))
    ORIG = LC.OffScreenRenderEnv
    COUNT = dict(ep=0, err=0)
    os.makedirs(outdir, exist_ok=True)

    class LogEnv:
        def __init__(self, env):
            self._e = env
            self._rows = None

        def __getattr__(self, k):
            return getattr(self._e, k)

        def set_init_state(self, s):
            o = self._e.set_init_state(s)
            self._rows = {k: [] for k in KEYS}
            self._n = -1
            self._snap(o, None, False)
            return o

        def step(self, a):
            r = self._e.step(a)
            if self._rows is not None:
                self._n += 1
                self._snap(r[0], a, bool(r[2]))
            return r

        def _snap(self, obs, a, done):
            E = self._e.env
            nan = float("nan")
            try:
                sim = E.sim
                bid = E.obj_body_id[OE]
                op = np.array(sim.data.body_xpos[bid], np.float64)
                oq = np.array(sim.data.body_xquat[bid], np.float64)
                st_t, st_o = E.object_states_dict[TE], E.object_states_dict[OE]
                tp = np.array(st_t.get_geom_state()["pos"], np.float64)
                z_ok = contact = xy = under = on_terms = nan
                ds = np.full(3, nan)
                if getattr(st_t, "object_state_type", "") == "site":
                    site = E.object_sites_dict[TE]
                    mat = np.array(sim.data.get_site_xmat(TE), np.float64)
                    under = float(bool(site.under(tp, mat, op)))
                    contact = float(bool(E.check_contact(E.get_object(st_t.parent_name), E.get_object(OE))))
                    ds = mat @ (op - tp)
                    xy = float(np.linalg.norm(op[:2] - tp[:2]))
                    on_terms = float(under and contact)
                else:
                    z_ok = float(tp[2] <= op[2])
                    contact = float(bool(st_t.check_contact(st_o)))
                    xy = float(np.linalg.norm(op[:2] - tp[:2]))
                    on_terms = float(bool(z_ok) and bool(contact) and xy < 0.03)
                on_full = float(bool(st_t.check_ontop(st_o)))
                succ = float(bool(E._check_success()))
                row = dict(n=self._n, a=np.full(7, nan) if a is None else np.asarray(a, np.float64), eef=np.asarray(obs["robot0_eef_pos"], np.float64),
                           eefq=np.asarray(obs["robot0_eef_quat"], np.float64), grip=np.asarray(obs["robot0_gripper_qpos"], np.float64),
                           obj=op, objq=oq, tgt=tp, z_ok=z_ok, contact=contact, xy=xy, under=under, dsite=ds, on_terms=on_terms,
                           on_full=on_full, succ=succ, done=float(done))
            except Exception as ex:                                         # noqa: BLE001  (logged, never hidden, never raised)
                COUNT["err"] += 1
                if COUNT["err"] <= 3:
                    print("[instr] snapshot failed: %r" % (ex,), flush=True)
                row = {k: np.full(WIDTH[k], float("nan")) for k in KEYS}
                row["n"] = self._n
            for k in KEYS:
                self._rows[k].append(np.asarray(row[k], np.float64).reshape(-1))

        def close(self):
            if self._rows is not None and len(self._rows["n"]) > 1:
                i = COUNT["ep"]
                COUNT["ep"] += 1
                trial = int(br[i]["trial"]) if i < len(br) else -1 - i
                path = os.path.join(outdir, "instr_%s_%d_t%d.npz" % (arm, tid, trial))
                d = {k: np.stack(self._rows[k]).astype(np.float64) for k in KEYS}
                np.savez_compressed(path + ".tmp.npz", trial=trial, tid=tid, arm=arm, oe=OE, te=TE, errs=COUNT["err"],
                                    site=getattr(E_state(self._e, TE), "object_state_type", "") == "site", site_size=site_size(self._e), **d)
                os.replace(path + ".tmp.npz", path)
                print("[instr] wrote %s (%d rows, snapshot errors so far %d)" % (path, len(d["n"]), COUNT["err"]), flush=True)
            self._rows = None
            self._e.close()

    def site_size(env):
        try:
            return np.array(env.env.object_sites_dict[TE].size, np.float64)
        except Exception:                                                   # noqa: BLE001
            return np.full(3, float("nan"))

    def E_state(env, name):
        return env.env.object_states_dict[name]

    LC.OffScreenRenderEnv = lambda *a, **k: LogEnv(ORIG(*a, **k))


def main():
    which = sys.argv[1]
    assert which in ("raa", "rg"), which
    assert sys.argv[2] == "client", sys.argv
    sys.argv.pop(1)
    av = sys.argv
    arm = av[av.index("--arm") + 1]
    tid = int(av[av.index("--tasks") + 1])
    out = av[av.index("--out") + 1]
    br = av[av.index("--branches") + 1]
    install(arm, tid, os.path.dirname(os.path.abspath(out)), br)
    if which == "raa":
        import raa
        raa.client_main()
    else:
        import rg
        rg.client_main()


if __name__ == "__main__":
    main()
