"""RAA-F Route B (docs/notes/raaf_plan.md section 4): the target-centred crop, shared by the training builder (raafb_rv.py) and the
serving side (wa_sb.py).  Pure numpy (+ PIL for the resize); must parse under python 3.8 and 3.11.

Conventions (checked by raafb_proj.py against the rendered segmentation):
  client image  the policy's 256 x 256 image = robosuite's raw render [::-1, ::-1] (sc_rv.flip).  Row index runs top-down in the
                OpenCV pinhole convention, the column axis is mirrored: where.project(points, K, E) -> (row_cv, W - 1 - col_cv).
  K, E          robosuite camera_utils.get_camera_intrinsic_matrix / get_camera_extrinsic_matrix(sim, 'agentview', 256, 256).
  crop          CROP px square of the client image centred on (row, col) (rounded to the pixel), zero-padded outside the image,
                resized bilinear to OUT x OUT (= openpi's resize_with_pad for a square input).
"""
import os

import numpy as np

PIX = 256
CROP = 96
OUT = 224


def project_client(points, K, E, W=PIX):
    """World points [..., 3] -> (row, col) float in the client's 256 image.  Same formula as lifelong_ood/policy/where.project."""
    P = np.asarray(points, np.float64)
    Rt = np.linalg.inv(np.asarray(E, np.float64))
    Kx = np.eye(4)
    Kx[:3, :3] = np.asarray(K, np.float64)
    M = Kx @ Rt
    hom = np.concatenate([P, np.ones(P.shape[:-1] + (1,))], -1)
    q = hom @ M.T
    q = q[..., :2] / q[..., 2:3]
    return np.stack([q[..., 1], (W - 1) - q[..., 0]], -1)


def crop_square(img, rc, crop=CROP):
    """img [H, W, 3] uint8, rc = (row, col) centre (float) -> [crop, crop, 3] uint8 centred on round(rc); outside pixels are 0."""
    img = np.asarray(img)
    H, W = img.shape[:2]
    r0, c0 = int(round(float(rc[0]))) - crop // 2, int(round(float(rc[1]))) - crop // 2
    out = np.zeros((crop, crop, img.shape[2]), img.dtype)
    ra, rb, ca, cb = max(r0, 0), min(r0 + crop, H), max(c0, 0), min(c0 + crop, W)
    if ra < rb and ca < cb:
        out[ra - r0:rb - r0, ca - c0:cb - c0] = img[ra:rb, ca:cb]
    return out


def resize_out(x, out=OUT):
    from PIL import Image
    return np.asarray(Image.fromarray(np.asarray(x, np.uint8)).resize((out, out), Image.BILINEAR))


def make_crop(img, rc, crop=CROP, out=OUT):
    """The model's third image: the crop of the (painted) client agentview centred on rc, [out, out, 3] uint8."""
    return resize_out(crop_square(img, rc, crop), out)


def crop_centre_px(crop_rc, rc_true, crop=CROP, out=OUT):
    """Where the true point lands inside the OUT x OUT crop built around crop_rc: -> (row, col) float, pixel-centre coordinates (the crop centre itself is (OUT - 1) / 2 = 111.5; rounding the window to whole pixels shifts it by up to 0.5 client px per axis)."""
    s = out / float(crop)
    r0 = np.round(np.asarray(crop_rc, np.float64)) - crop // 2                 # window origin, as crop_square
    return (np.asarray(rc_true, np.float64) - r0 + 0.5) * s - 0.5            # PIL pixel-centre convention


# ---------------------------------------------------------------- client / server glue (Route B serving)
def client_uv(raw_env, pos_obj, pos_tgt, true_obj=None, true_tgt=None, oe=None, te=None, cache={}):
    """CLIENT (LIBERO venv): the crop centres (row, col, client 256 px) of the Estimator's object / target estimates, and of the
    simulator-truth positions for the error log.  K / E of the fixed agentview are read from the sim once per task and cached by sim id.
    -> dict of meta/raa_crop_uv_{obj,tgt} and meta/raa_crop_tuv_{obj,tgt} (lists of 2 floats)."""
    from f0b_firstframes_seg import CU                                       # robosuite.utils.camera_utils inlined (it imports h5py, absent from the LIBERO venv)
    sim = raw_env.env.sim
    key = id(sim)
    if key not in cache:
        cache.clear()
        cache[key] = (CU.get_camera_intrinsic_matrix(sim, "agentview", PIX, PIX), CU.get_camera_extrinsic_matrix(sim, "agentview"))
    K, E = cache[key]
    lo, lt = lift_of(oe), lift_of(te)                                       # the crop centre is the visible geom centre: Estimator position + class offset
    out = {"meta/raa_crop_uv_obj": project_client(np.asarray(pos_obj) + lo, K, E).tolist(), "meta/raa_crop_uv_tgt": project_client(np.asarray(pos_tgt) + lt, K, E).tolist()}
    if true_obj is not None:
        out["meta/raa_crop_tuv_obj"] = project_client(np.asarray(true_obj) + lo, K, E).tolist()
        out["meta/raa_crop_tuv_tgt"] = project_client(np.asarray(true_tgt) + lt, K, E).tolist()
    return out


def attach_crop(o, painted_agent256, raa, on, skip=False, crop=CROP, out=OUT):
    """SERVER: set o['observation/crop_image'] = the target-centred crop of the painted 256 agentview.  The centre is the client's projection of
    the Estimator's (not the simulator's) estimate of the active role entity: the target while held, else the object.  `skip` (noM: the
    active entity is hidden and its channel is withheld) or a missing centre gives an all-zero crop, which the model sees as a black third
    image with the mask still true (the training set has no such frame; it is reported by the caller's counter).  -> centre used or None."""
    uv = raa.get("crop_uv_tgt" if on else "crop_uv_obj")
    if skip or uv is None:
        o["observation/crop_image"] = np.zeros((out, out, 3), np.uint8)
        return None
    o["observation/crop_image"] = make_crop(painted_agent256, uv, crop, out)
    return np.asarray(uv, np.float64)


# ---------------------------------------------------------------- per-class centre offset (orchestrator deviation, docs/notes/raafb_build.md section 8)
LIFT_PATH = os.environ.get("RAAFB_LIFT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "flawed", "raafb_lift.json"))
_LIFT = {}


def lift_of(slot):
    """Fixed world-axes offset (m) from a movable entity's body origin to the mean of its visible geom centres (raafb_lift.py, read from the simulator's
    geoms at reset); zeros for fixture sites and unknown slots."""
    if not _LIFT:
        import json
        _LIFT.update({k: np.asarray(v, np.float64) for k, v in json.load(open(LIFT_PATH, encoding="utf-8"))["lift"].items()})
    return _LIFT.get(slot, np.zeros(3))
