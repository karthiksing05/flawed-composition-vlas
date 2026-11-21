# RAA-F Route B build: code and checks (DRAFT of the config / trainer edits; nothing here is applied)

Written 2026-11-20 against docs/notes/raaf_plan.md section 4 (Route B, target crop in pi0.5's third image slot).  No training was run.  Python ran only
through sbatch (jobs 4818630 failed on h5py, 4818641 / 4818652 projection check, 4818642 builder, 4818657 / 4818662 smoke test).  Deviation to
record: two Bash attempts that contained `python3` on the login node were issued by mistake and were blocked by the permission classifier before
running; nothing executed.

## 1. Files written (all new)

| file | role |
|---|---|
| experiments/raafb_crop.py | shared crop code: `project_client` (K / E projection into the client 256 image), `crop_square`, `make_crop` (96 px -> 224), `crop_centre_px`, client `client_uv`, server `attach_crop` |
| experiments/raafb_proj.py | S-B1 projection check (LIBERO venv, GPU); also writes `camKE.npz` (K / E per task) |
| experiments/raafb_rv.py | training-data builder: sc_rv raac arm + `crop_image` feature (imports sc_rv, edits nothing) |
| experiments/wa_sb.py | serving copy of wa_s.py (server crop in `WARoleView.infer`, client `channel()` sends `meta/raa_crop_uv_*`) |
| experiments/raa_b.py | serving copy of raa.py (plan: "serving copies raa_b.py and wa_sb.py"), same additions, use `--arm raac` |
| experiments/raafb_smoke_test.py | model-free smoke test of the server crop against ground truth on recorded frames |
| /coc/flash10/ksingara3/.claude_tmp/pp/raafb_{proj,conv,smoke}.sh | sbatch scripts (new copies, no existing .sh touched) |
| /coc/flash10/ksingara3/.claude_tmp/raaf/b_test/ | camKE.npz, proj_check.json (and v1), ds/ (3-demo LeRobot test set), png/ (crop PNGs) |

## 2. Projection check S-B1 (results, 1,200 rows: K1-K6, 5 demos x 20 frames per task, object and target roles)

Reference named by the plan: centroid of the role entity's cleaned segmentation mask (client orientation).  Ground truth: the demo's own slot
position (chan_XX.npz `pos`).  Error = |projected GT - mask centroid| in client 256 px.

| set | n | median px | p90 px |
|---|---|---|---|
| movable roles (KILL SET) | 800 | **9.54** | 16.5 |
| movable object role | 600 | 9.89 | 18.3 |
| movable target role (plate, bowl) | 200 | 5.96 | 9.97 |
| per slot: cheese / plate / bowl / bottle | 100 / 100 / 400 / 200 | 1.8 / 2.7 / 9.5 / 15.8 | 6.0 / 5.1 / 10.0 / 22.8 |
| fixture sites (stove / rack / cabinet top), not like-for-like | 400 | 27.3 (4.2 / 18.4 / 32.9) | 33.6 |

Literal reading of the plan's kill (median > 8 px on the role entity): **it fires** on the movable set (9.54 px).  The error is NOT the camera model:
- `sc_pu.cam_model` + `sc_pu.project` (the plan's named code), converted to client orientation (col -> 255 - col), equals the robosuite K / E route to
  1.6e-13 px on every row (K / E orientation and mirroring are right: `hit` rate is 100% on the bowl).
- The residual is a reference mismatch.  The slot position is the body origin (object base); the mask centroid is the visible silhouette centre.  It is a
  fixed vertical offset: mean (proj - centroid) row = +8.0 px for the bowl, +13.9 px for the bottle, -2.1 px for the cheese, +0.9 px for the plate (small
  objects match).  A like-for-like 3D reference (mean of the entity's visible geom centres, projected) gives median **3.69 px** (p90 7.04) on the movable set
  (bowl 3.7, cheese 1.8, plate 1.0, bottle 5.5).
- Per-slot constant world lift (median geom centre minus slot position) applied to the GT point: movable median 3.7 px, p90 8.3 (bowl 3.7 / 5.8, cheese 1.8 / 6.0,
  plate 1.0 / 6.7, bottle 6.4 / 23.0: the bottle tilts when carried, a constant lift does not follow it).  In-sample diagnostic only; not wired in.
- Fixture targets (stove, cabinet top, rack top): a site vs the whole fixture's silhouette is not a valid pixel reference (the cabinet top is 33 px from the
  cabinet's silhouette centroid by construction).  Checked visually instead: the green cross sits on the cabinet top surface (png K4_d00_held_f054).

Decision for the plan owner: under the plan's literal rule Route B stops (k1).  Under the like-for-like reference it passes with margin (3.7 px).  The
offset is deterministic and identical in training and test crops (same projection of the same slot), so the crop window (96 px) is consistently placed
a fixed 8-14 px above the bowl / bottle centres.  A per-object lift constant in `raafb_rv.py` and `raafb_crop.client_uv` would remove it; I did not wire it
because the plan fixes the centre at the slot position.

## 3. Crop builder and visual check

`raafb_rv.py --task K1+K4+K6 --ndemo 1 --suffix _btest --out .../b_test/ds`: 3 episodes, 274 frames, crop_image 224 x 224 x 3, token / prompts / images
byte-identical in construction to sc_rv's raac (same functions), jitter N(0, 2 px) (median 2.2-2.5 px measured).  PNGs: `png/<task>_d00_<phase>_f<frame>_full_crop.png`
(left: painted agentview, green cross = projected ground truth, white box = the jittered 96 px window; right: the 224 crop).  Viewed K1 pre, K1 held,
K4 pre, K4 held, K6 held.  Centred: plate target (K1 held) centred within a few px; cheese / plate small objects centred.  Bowl (K1 pre, K6 held) and bottle
(K4 pre): inside the crop, horizontally centred, 10-15 source px above the crop centre (the base-origin offset).  Cabinet top (K4 held): the site is the crop
centre, but the cabinet sits at the right image edge so about a third of the crop is zero padding (black).  Unrouted K7 / K8 frames get an all-zero crop.

Builder numbers per phase (mask centroid vs projected centre, median px; from the manifest raafb_rv_btest_manifest.json): K1 pre 9.5, K1 held 2.7, K4 pre 14.0,
K6 pre 0.3 (cheese), K6 held 9.3; K4 held 32.9 (fixture, n/a).

## 4. Smoke test (job 4818662, ALL OK)

`raafb_smoke_test.py`: T1 synthetic markers land at the crop centre (+-2 crop px) and `on` selects the target key; T2 on 30 recorded K1 / K4 / K6 frames with a
0.6 cm Estimator error the true position lands exactly where the estimator error says (within the pixel rounding of the window) and a perfect centre puts it
in the middle; the mask centroid is median 17.5 crop px (7.5 client px) from the crop centre, max 46.7, against 8 px = 18.7 crop px (the same
base-origin offset as section 2); T3 skip (noM) / no centre -> all-zero uint8 crop and `None`; T4 border zero padding; T5 the new files parse.

## 5. Third image slot: does openpi's input pipeline pass a third image?  NO, it must be changed

`/coc/flash10/ksingara3/robot/openpi/src/openpi/policies/libero_policy.py`, class `LiberoInputs` (line 30), `__call__`:
- line 62: `"right_wrist_0_rgb": np.zeros_like(base_image),` (the slot is hard-coded to zeros; `observation/crop_image` is never read);
- line 68: `"right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,` (mask True only for pi0-FAST; for pi0.5 it is False).
So the stock pipeline gives `image_mask["right_wrist_0_rgb"] = False`, and the model drops those tokens.  The model itself uses the third image when the mask is
true: `openpi/models/pi0.py:106-125` (`embed_prefix` iterates `obs.images`, repeats `obs.image_masks[name]` over the image tokens); `models/model.py:173`
applies spatial augmentation only to keys without "wrist" in the name, so `right_wrist_0_rgb` gets colour jitter only and the crop stays centred in training.
Serving goes through `policy._input_transform` (backbone.py `observation()`), i.e. the config's data transforms, so the same replacement covers serving.

## 6. DRAFT edits (apply at merge; I did not touch l90c_config.py or c_train_pi05.py)

### 6.1 NEW file lifelong_ood/policy/raafb_inputs.py
```python
"""RAA-F Route B: LiberoInputs with the target crop in the third image slot (right_wrist_0_rgb), mask True."""
import dataclasses

import numpy as np
from openpi.policies import libero_policy


@dataclasses.dataclass(frozen=True)
class LiberoInputsB(libero_policy.LiberoInputs):
    def __call__(self, data: dict) -> dict:
        out = super().__call__(data)
        crop = libero_policy._parse_image(data["observation/crop_image"])        # KeyError if absent: never a silent masked slot
        out["image"]["right_wrist_0_rgb"] = crop
        out["image_mask"]["right_wrist_0_rgb"] = np.True_
        return out
```

### 6.2 lifelong_ood/policy/l90c_config.py (additive)
```python
RAAFB_REPO = "lol/libero_g8v_raacb"                        # RAA-F Route B (experiments/raafb_rv.py): raac + crop_image
# NAMES: append "pi05_g8v_raas_lb_ae"
# AE:    "pi05_g8v_raas_lb_ae": ("pi05_g8v_raac", "lol/libero_g8v_raac"),      # RAA-c assets / norm stats kept (plan section 1)


def _crop_data(C, data):
    """data (a LeRobotLiberoDataConfig) -> same config whose repack maps crop_image and whose input transform is LiberoInputsB."""
    import openpi.transforms as T
    from . import raafb_inputs as RI
    from openpi.policies import libero_policy

    @dataclasses.dataclass(frozen=True)
    class CropLiberoDataConfig(C.LeRobotLiberoDataConfig):
        def create(self, assets_dirs, model_config):
            dc = super().create(assets_dirs, model_config)
            repack = T.Group(inputs=[T.RepackTransform({"observation/image": "image", "observation/wrist_image": "wrist_image",
                                                        "observation/crop_image": "crop_image", "observation/state": "state",
                                                        "actions": "actions", "prompt": "prompt"})])
            ins = [RI.LiberoInputsB(model_type=model_config.model_type) if isinstance(t, libero_policy.LiberoInputs) else t
                   for t in dc.data_transforms.inputs]
            assert any(isinstance(t, RI.LiberoInputsB) for t in ins)
            return dataclasses.replace(dc, repack_transforms=repack, data_transforms=T.Group(inputs=ins, outputs=dc.data_transforms.outputs))
    return CropLiberoDataConfig(**{f.name: getattr(data, f.name) for f in dataclasses.fields(data)})

# inside train_config(), next to pi05_g8v_raas_l_ae:
        if name == "pi05_g8v_raas_lb_ae":                       # RAA-F Route B: RAA-sL exactly (token + servo, weights from pi05_libero) + the crop in the 3rd image slot
            from . import raac_model as RM
            from . import raas_model as RS
            m = base.model
            model = RS.Pi0ConfigS(pi05=True, action_horizon=m.action_horizon, discrete_state_input=False, **RS.servo_fields())
            data = _crop_data(C, dataclasses.replace(data, repo_id=RAAFB_REPO))
            return dataclasses.replace(base, name=name, ema_decay=None, freeze_filter=frozen, data=data, model=model,
                                       weight_loader=RM.RaacLoader(LIBERO_PARAMS), **over)
```
The model class needs no change: the third image already exists in `Pi0Config.inputs_spec` (pi0_config.py:73-78) and `embed_prefix` already loops over all three
images.  No new parameter (the crop uses the same frozen SigLIP), so `RaacLoader(LIBERO_PARAMS)` loads as for RAA-sL.  No norm step: assets dir `pi05_g8v_raac`
is reused.

### 6.3 experiments/c_train_pi05.py (additive)
```python
# line 131: add "raas_lb" to the tuple ("raac", "raas", "raas_l", "raas_ls2", "raac_lf", "raas_lb")  (export through raac_export.py: same tensors)
# line 154: add "raas_lb" to the --g8v choices
```
Training smoke (plan k2) additions: in the first batch assert `batch.image_masks["right_wrist_0_rgb"].all()` and that `batch.images["right_wrist_0_rgb"]` is not
all zero; step-0 loss must be <= 2x RAA-sL's .0556.  The data repo has to be built first: `raafb_rv.py` without --task / --suffix, `--out` omitted (default
`<HF_LEROBOT_HOME>/lol/libero_g8v_raacb`), `--cam .../b_test/camKE.npz` (K / E are constants of the 6 task cameras; copy to results/flawed/ when registering).

### 6.4 Serving
`wa_sb.py serve` (default config pi05_g8v_raas_lb_ae) and the matching client (`wa_sb.py client`, same arguments as wa_s.py); `raa_b.py serve|client --arm raac`.
Client sends `meta/raa_crop_uv_{obj,tgt}` (Estimator estimate projected with the sim camera, client 256 px) and `meta/raa_crop_tuv_*` (truth, log only);
server crops the PAINTED 256 agentview (the same image the training crop is cut from), target key while held, object key otherwise.  Open item for the a2
probe patch group `in.crop`: the crop's 256 prefix tokens are the third image block of the prefix (after base and left wrist).

## 7. Open points for the plan owner
1. The 8 px kill fires literally (movable median 9.54 px) but only because the body origin is not the silhouette centre; like-for-like it is 3.7 px.  Decide: accept
   the like-for-like reading, add a per-object lift constant, or stop Route B.
2. Fixture target sites cannot be checked against a pixel reference; visual check only.
3. Training draws the crop from GT; at serving the centre inherits the Estimator error (0.6 cm = about 2 px, the jitter sigma used).  The jitter is per frame
   and independent; no per-chunk consistency is imposed.
4. K7 / K8 and no-channel / noM queries use an all-zero crop with the mask True; the training set contains zero crops only for K7 / K8.

## 8. DEVIATIONS (orchestrator decisions, 2026-11-20, after section 2's literal kill) and the S-B1 re-check

**(a) Reference change (orchestrator decision).**  The crop is centred on the entity's VISIBLE GEOM CENTRE, not its body origin.  Reason: the body origin sits at the object's base
(the segmentation centroid is at the silhouette centre), and the camera model matches robosuite to 1e-13 px (section 2), so the 9.54 px median was a reference mismatch.  The offset is
a fixed per-class 3D offset read from the simulator's own geoms at the task's reset state (experiments/raafb_lift.py, job 4818969, nothing fitted to the check): bowl (+0.0003, +0.0004, +0.0284) m,
cheese ~0, bottle (0, +0.0065, +0.0769), plate (0, 0, +0.0083); identical in all 6 task envs (spread 0.0000).  Fixture sites keep the site (offset 0).  Applied in raafb_rv.py (training) and
raafb_crop.client_uv (serving: Estimator position + class offset); stored in results/flawed/raafb_lift.json.

**(b) Base model (orchestrator decision).**  Servo off: RAA-c-from-libero exactly as Route A (config pi05_g8v_raac_lb_ae, `RS.servo_fields(k=0.0)`), instead of the plan's RAA-sL model
(pi05_g8v_raas_lb_ae).  The only change vs Route A's model is the data repo and the crop.  wa_sb.py / raa_b.py default config renamed to pi05_g8v_raac_lb_ae.  Section 6.2's snippet is
superseded by the applied edit (config name, `k=0.0`).

**S-B1 re-check (job 4818984, 1,200 rows, same frames as section 2).**  Movable roles, reference = class-offset geom centre: **median 3.71 px, p90 8.35 px, KILL: no** (origin reference:
9.54 / 16.5).  By slot (median / p90): bowl 3.71 / 5.81 (was 9.54 / 10.0), cheese 1.80 / 5.99, plate 1.01 / 6.66 (was 2.69 / 5.05), bottle 6.49 / 23.0 (was 15.8 / 22.8; the p90 is the bottle tilting
in the carry, a constant world offset does not follow it).  Movable object role 3.74 / 10.6, movable target role 3.17 / 5.64.  Fixture sites unchanged and still not like-for-like (stove 4.2,
rack 18.4, cabinet 32.9; visual check only).  Files: results/flawed/raafb_s_b1.json.  Re-built 3-demo test set and smoke test (jobs 4819022, 4819023): ALL OK, mask-centroid offset in crop px
median 6.5 (was 17.5), max 24.4; PNGs re-viewed (K4 pre: bottle mid-body at the crop centre; K6 held: bowl centred horizontally, slightly high under the gripper).

## 9. LAUNCH RECORD (2026-11-20)

Applied (additive) edits: lifelong_ood/policy/raafb_inputs.py (new: LiberoInputsB), lifelong_ood/policy/l90c_config.py (RAAFB_REPO, `_crop_data`, NAMES / AE entry and the
`pi05_g8v_raac_lb_ae` block: Route A's model, servo_k = 0, weights from pi05_libero, RAA-c norm stats, data lol/libero_g8v_raacb), experiments/c_train_pi05.py (`raac_lb` in the export tuple and
the --g8v choices).  The committed copies of these two shared files also carry the Route A agent's still-uncommitted `raac_lf` lines.

- Dataset lol/libero_g8v_raacb (job 4819195, second attempt; the first, 4819021, died at K7 on a missing K / E key and was fixed): 400 episodes, 52,181 frames (= lol/libero_g8v_raac).
  Checker raafb_check.py: PASS (state / actions / prompts row-for-row equal to raac, crop_image 224 x 224 x 3, 0 all-zero crops on 3,471 decoded routed frames, 0 non-zero crops on
  1,716 decoded unrouted frames).  Manifest results/flawed/raafb_rv_manifest.json.  Per-phase mask-centroid offsets (median px, class-offset centre): K1 pre 3.6, K6 held 3.7 (more in the manifest).
- Real-pipeline check (job 4819196, raafb_batch_check.py): repack + LiberoInputsB + normalise on 200 frames: `image_mask["right_wrist_0_rgb"]` True on all, crop non-zero on 131 routed frames,
  zero on 69 unrouted.  (plan k2: slot unmasked: PASS)
- 20-step smoke (exp smoke, tail-lab job 4819374 after a long overcap queue wait; overcap job 4819197 cancelled): step-0 loss **.0403** (RAA-sL .0556; k2 limit .1112), state_proj in the
  parameter list, RAA-c norm stats loaded, exit 0.
- Prereg row (before any main job): ledger/m3_prereg/m3_prereg__libero_goal__raafb__s0__ac741443.json, tag **raafb**, experiments/prereg/raafb.py + raafb_pred.txt (job 4819395).
- Chains (copies raafb_train.sh / raafb_chain.sh in .claude_tmp/pp; ids in raafb_chain_ids.txt): overcap exp `main` jobs 4819403-4819408 (6 x 4 h, afterany); tail-lab exp `tl` jobs
  4819409-4819414.  tail-lab held 2 running of this user's jobs at launch (raasls2_tl3, raaf_tl1), so the tl chain makes 3 (cap 4).  Both chains write
  /coc/flash10/ksingara3/robot/checkpoints_l90c/jax/pi05_g8v_raac_lb_ae/{main,tl}; cancel the second chain when the first reaches step 6000 (as for RAA-sL).
- Open: kill k3 (dev K) and k4 (probe) are read after training; the probe's `in.crop` patch group (a2_probe_f.py) is not written yet.
