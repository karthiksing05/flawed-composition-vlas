"""Display-frame rollouts for the figures (docs/notes/fig_plan.md 3, package a).  LIBERO venv (py3.8), EGL.

A thin wrapper around any of the repo's clients (pp_atlas_client, rg.py, rg_comp.py, f1m_cue.py, mt_client.py, raa.py / raa_w.py, ...).
It puts a recording layer UNDER the client's own env wrappers (libero.libero.envs.OffScreenRenderEnv is wrapped before libero_client is
imported).  The layer only reads the simulator, for DISPLAY: a 512 px agentview frame every REC_EVERY steps (and at the last step) and the
EE pixel at every step (pp_f1_frames.project).  The policy's 256 px inputs, prompts, seeds, states and the episode protocol are the wrapped
client's, unchanged.  A subclass of pp_atlas.Rec is not used because rg_comp's env does not derive from it; one layer serves every harness.

    python experiments/fig_roll_client.py [--fig-first N] [--fig-skip M] [--fig-install-rec] CLIENT.py [CLIENT ARGS incl. --out OUT.json]
        --fig-first N / --fig-skip M   keep entries [M, M+N) of --branches (file order), as a copy under dirname(--out)
        --fig-install-rec              raa.py / raa_w.py clients have no track of their own: install pp_atlas.Rec as mt_client.py does
Output (all under dirname(--out), which must be under results/flawed/fig_frames): <out>_disp.npz (per episode, in the order the client's
episodes run: fr_buf / fr_off JPEG frames, fr_step row of each frame, pix EE pixel (row, col) per row, cam_*), <out>_figcfg.json, plus the
client's own <out>.json and <out>_track.npz.  Row k = the k-th step from the WAIT-th (10th) on, the row convention of pp_atlas.Rec.
"""
import io
import json
import os
import runpy
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

REC_EVERY, DISP, WAIT = 5, 512, 10
STATE = dict(out=None, eps=[])


def take(name, default):
    if name in sys.argv:
        i = sys.argv.index(name)
        v = sys.argv[i + 1]
        del sys.argv[i:i + 2]
        return int(v)
    return default


def encode(img):
    from PIL import Image
    bio = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(img)).save(bio, format="JPEG", quality=88)
    return np.frombuffer(bio.getvalue(), np.uint8)


class DispEnv:
    """Reads the wrapped OffScreenRenderEnv after each step; never alters it."""

    def __init__(self, env):
        self._e = env
        self._start()

    def _start(self):
        self.n, self.fr, self.fs, self.pix = 0, [], [], []

    def __getattr__(self, k):
        return getattr(self._e, k)

    def _sim(self):
        return self._e.env.sim

    def _render(self, row):
        self.fr.append(encode(self._sim().render(camera_name="agentview", height=DISP, width=DISP)[::-1]))   # upright
        self.fs.append(row)

    def step(self, a):
        r = self._e.step(a)
        self.n += 1
        if self.n >= WAIT:
            import pp_f1_frames as F1
            row = self.n - WAIT
            self.pix.append(F1.project(self._sim(), np.asarray(r[0]["robot0_eef_pos"], np.float64)[None, :], DISP, DISP)[0])
            if row % REC_EVERY == 0:
                self._render(row)
        return r

    def set_init_state(self, s):
        self._finish()
        self._start()
        return self._e.set_init_state(s)

    def _finish(self):
        if self.n <= WAIT:
            return
        row = self.n - WAIT
        if not self.fs or self.fs[-1] != row:
            self._render(row)                                                                   # the last step
        sim = self._sim()
        cid = sim.model.camera_name2id("agentview")
        STATE["eps"].append(dict(
            fr_off=np.cumsum([0] + [len(f) for f in self.fr]).astype(np.int64), fr_buf=np.concatenate(self.fr),
            fr_step=np.array(self.fs, np.int32), pix=np.array(self.pix, np.float32).reshape(-1, 2),
            cam_xpos=np.array(sim.data.cam_xpos[cid], np.float64), cam_xmat=np.array(sim.data.cam_xmat[cid], np.float64).reshape(3, 3),
            cam_fovy=np.array(sim.model.cam_fovy[cid], np.float64)))
        z = dict(n=np.array(len(STATE["eps"])))
        for i, d in enumerate(STATE["eps"]):
            z.update({"e%d_%s" % (i, k): v for k, v in d.items()})
        np.savez(STATE["out"].replace(".json", "_disp.npz"), **z)
        self.n = 0

    def close(self):
        self._finish()
        self._e.close()


def main():
    n_first, n_skip = take("--fig-first", 0), take("--fig-skip", 0)
    install_rec = "--fig-install-rec" in sys.argv
    if install_rec:
        sys.argv.remove("--fig-install-rec")
    client = sys.argv[1]
    sys.argv = sys.argv[1:]
    out = sys.argv[sys.argv.index("--out") + 1]
    assert "results/flawed/fig_frames" in os.path.abspath(out), "outputs go under results/flawed/fig_frames"
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    STATE["out"] = out
    if "--branches" in sys.argv and n_first:
        i = sys.argv.index("--branches")
        br = json.load(open(sys.argv[i + 1], encoding="utf-8"))
        nb = os.path.join(os.path.dirname(os.path.abspath(out)), "branches_%s_%d_%d.json" % (
            os.path.basename(sys.argv[i + 1]).replace(".json", ""), n_skip, n_first))
        with open(nb, "w", encoding="utf-8") as f:
            json.dump(br[n_skip:n_skip + n_first], f)
        sys.argv[i + 1] = nb
    import libero.libero.envs as LE
    _orig = LE.OffScreenRenderEnv
    LE.OffScreenRenderEnv = lambda *a, **k: DispEnv(_orig(*a, **k))                           # BEFORE libero_client imports it
    if install_rec:
        import fc_n2_client as FN
        import mt_client as MT
        MT.install_rec(FN)
    try:
        runpy.run_path(client if os.path.isabs(client) else os.path.join(HERE, client), run_name="__main__")
    finally:
        with open(out.replace(".json", "_figcfg.json"), "w", encoding="utf-8") as f:
            json.dump(dict(client=client, argv=sys.argv, fig_first=n_first, fig_skip=n_skip, rec_every=REC_EVERY, disp=DISP,
                           env={k: os.environ.get(k, "") for k in ("BACKBONE_CONFIG", "BACKBONE_CKPT", "SLURM_JOB_ID")}), f)


if __name__ == "__main__":
    main()
