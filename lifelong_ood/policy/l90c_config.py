"""openpi TrainConfigs for LIBERO-90-C (docs/L90C.md), registered at runtime so openpi's own config.py is never edited.

  pi05_l90c        pi05_libero's recipe (same model: pi0.5, action horizon 10, continuous state dropped; same LIBERO transforms;
                   AdamW, clip 1.0, EMA 0.999, cosine schedule) from pi05_base, on lol/libero_l90c (the frozen split's
                   training tasks only, experiments/c_data.py), with its own norm stats.  FSDP over 4 GPUs.  Batch size and
                   steps are set by experiments/c_train_pi05.py (the 200-step timing job decides them) and recorded there.
  pi05_l90c_lora   the same with openpi's LoRA variants (gemma_2b_lora + gemma_300m_lora), no EMA (openpi's low-mem recipe).
  pi05_l90c_fac    pi05_l90c plus FACTORED CONDITIONING (docs/L90C.md 3.1): per sample, with probability 0.3 the instruction is
                   replaced by one of its part prompts (object-only / goal-only / verb-only, the plain variant of
                   lifelong_ood/protos/store.py's rule), with probability 0.1 by null (""); the action target is unchanged
                   (Factored Diffusion Policies 2605.22596's per-factor dropout; CRAFT 2610.00524's counterfactual alignment).
  pi05_base_libero pi05_base itself under the LIBERO transforms and pi05_libero's norm stats: the contamination audit's handle.

`register()` puts them into openpi's registry (openpi.training.config._CONFIGS_DICT) so get_config / create_trained_policy /
PolicyBackbone find them by name.
"""
from __future__ import annotations

import dataclasses
import os
import sys

OPENPI = "/coc/flash10/ksingara3/robot/openpi"
CACHE = "/coc/flash10/ksingara3/robot/openpi_cache/openpi-assets/checkpoints"
L90C = "/coc/flash10/ksingara3/robot/checkpoints_l90c"          # our checkpoints, assets and converted PyTorch weights
REPO_ID = "lol/libero_l90c"
S2_REPO = "lol/libero_g8v_raacs2"                          # SR 89: kitchen raac + scene-2 cycles (experiments/s2_rv.py convert)
RAAF_REPO = "lol/libero_g8v_raacf"                          # RAA-F (docs/notes/raaf_plan.md): kitchen raac + relabelled held copies (experiments/raaf_rv.py)
BASE_PARAMS = f"{CACHE}/pi05_base/params"
LIBERO_PARAMS = f"{CACHE}/pi05_libero/params"                  # openpi pi05_libero (the frozen "I" / pi05 of SR 45 / 52 / 70): RAA-L (SR 80) initialises from it
NAMES = ("pi05_l90c", "pi05_l90c_lora", "pi05_base_libero", "pi05_l90c_fac", "pi05_goal8", "pi05_goal8_roles", "pi05_goal8_rsup",
         "pi05_goal8_ae", "pi05_goal8_rsup_ae", "pi05_goal8_roles_ae", "pi05_g8v_plain_ae", "pi05_g8v_rsup_ae", "pi05_g8v_rview_ae",
         "pi05_g8v_rview2_ae", "pi05_g8v_rbox_ae", "pi05_g8v_rrel_ae", "pi05_g8v_raa_ae", "pi05_g8v_raao_ae", "pi05_g8v_raac_ae", "pi05_g8v_raas_ae", "pi05_g8v_raa_l_ae", "pi05_g8v_raas_l_ae", "pi05_g8v_raas_ls2_ae", "pi05_g8v_raac_lf_ae")
# SCAFFOLD_RESULTS 34: the single-GPU ladder.  "_ae" = the same data and norm stats as its parent, with the VLM (SigLIP and
# the PaliGemma LLM) frozen and only the action expert and its projections trained; no EMA, as openpi's frozen-param recipes.
AE = {"pi05_goal8_ae": ("pi05_goal8", "lol/libero_goal8"), "pi05_goal8_rsup_ae": ("pi05_goal8_rsup", "lol/libero_goal8_rsup"),
      "pi05_goal8_roles_ae": ("pi05_goal8_roles", "lol/libero_goal8_roles"),
      # the ladder's re-rendered frames (experiments/sc_rv.py): R0 plain, R1 role phrases, R2 role views, from the SAME renders.
      # No full-fine-tune parent exists; the first entry only names the assets dir that sc_rv.py norm writes.  rview = the first
      # painting rule (kept as its record); rview2 / rbox = the second rule, exact masks / bounding boxes
      "pi05_g8v_plain_ae": ("pi05_g8v_plain", "lol/libero_g8v_plain"), "pi05_g8v_rsup_ae": ("pi05_g8v_rsup", "lol/libero_g8v_rsup"),
      "pi05_g8v_rview_ae": ("pi05_g8v_rview", "lol/libero_g8v_rview"),
      "pi05_g8v_rview2_ae": ("pi05_g8v_rview2", "lol/libero_g8v_rview2"), "pi05_g8v_rbox_ae": ("pi05_g8v_rbox", "lol/libero_g8v_rbox"),
      # SCAFFOLD_RESULTS 73 (RAA-min): the same frames as plain / rview2, each routed frame with the role channel as a text suffix
      "pi05_g8v_rrel_ae": ("pi05_g8v_rrel", "lol/libero_g8v_rrel"), "pi05_g8v_raa_ae": ("pi05_g8v_raa", "lol/libero_g8v_raa"),
      # SCAFFOLD_RESULTS 76 (RAA-o): RAA with the target-role channel = target minus the held object (eef + in-hand offset fixed at the grasp event)
      "pi05_g8v_raao_ae": ("pi05_g8v_raao", "lol/libero_g8v_raao"),
      # SCAFFOLD_RESULTS 77 (RAA-c): RAA-o's frames and channel, the channel as a CONTINUOUS state token (raac_model.py), no text suffix
      "pi05_g8v_raac_ae": ("pi05_g8v_raac", "lol/libero_g8v_raac"),
      # SCAFFOLD_RESULTS 78 (RAA-s): RAA-c's data, norm stats and state token (its assets / repo), plus the servo residual action head (raas_model.py)
      "pi05_g8v_raas_ae": ("pi05_g8v_raac", "lol/libero_g8v_raac"),
      # SCAFFOLD_RESULTS 80 (RAA-L): RAA data, norm stats (its own assets) and recipe; weights from the LIBERO-fine-tuned pi05_libero instead of pi05_base
      "pi05_g8v_raa_l_ae": ("pi05_g8v_raa", "lol/libero_g8v_raa"),
      # SCAFFOLD_RESULTS 82 (RAA-sL): RAA-s exactly (pi05_g8v_raas_ae: raac data / norm stats / state token + servo), weights from pi05_libero instead of pi05_base
      "pi05_g8v_raas_l_ae": ("pi05_g8v_raac", "lol/libero_g8v_raac"),
      # SCAFFOLD_RESULTS 89 (RAA-sL-S2): RAA-sL exactly; assets (norm stats, servo constants) stay RAA-c's, only the data repo differs (kitchen raac + the scene-2 K1 / K2 cycles, experiments/s2_rv.py)
      "pi05_g8v_raas_ls2_ae": ("pi05_g8v_raac", "lol/libero_g8v_raac"),
      # RAA-F Route A (docs/notes/raaf_plan.md 1): RAA-c exactly (state token, servo_k = 0) from pi05_libero; RAA-c norm stats / assets, data repo = RAAF_REPO
      "pi05_g8v_raac_lf_ae": ("pi05_g8v_raac", "lol/libero_g8v_raac")}
RSUP_REPO = "lol/libero_goal8_rsup"                             # SCAFFOLD_RESULTS 33.1: the supervised stage, role phrases given by the task definitions
GOAL8_REPO = "lol/libero_goal8"                                 # SCAFFOLD_RESULTS 30: the eight known LIBERO-Goal tasks only (experiments/sc_t8v.py)
ROLES_REPO = "lol/libero_goal8_roles"                           # SCAFFOLD_RESULTS 33: the same frames, each with its role's prompt (experiments/sc_t16.py)
FAC_PROMPTS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "results", "l90c",
                           "fac_prompts.json")
P_PART, P_NULL = 0.3, 0.1


def _openpi():
    if OPENPI + "/src" not in sys.path:
        sys.path.insert(0, OPENPI + "/src")
    import openpi.training.config as C
    return C


def train_config(name, **over):
    C = _openpi()
    from openpi.models import pi0_config
    from openpi.training import optimizer as O
    from openpi.training import weight_loaders as W
    lib = C.get_config("pi05_libero")
    if name == "pi05_base_libero":
        cfg = dataclasses.replace(lib, name=name, weight_loader=W.CheckpointWeightLoader(BASE_PARAMS),
                                  data=dataclasses.replace(lib.data, assets=C.AssetsConfig(
                                      assets_dir=f"{CACHE}/pi05_libero/assets", asset_id="physical-intelligence/libero")))
        return dataclasses.replace(cfg, **over)
    if name in AE:                                              # frozen VLM, trainable action expert; parent's data and norm stats
        import flax.nnx as nnx
        from openpi.shared import nnx_utils
        full, repo = AE[name]
        base = train_config("pi05_l90c")
        frozen = nnx.Any(nnx_utils.PathRegex(".*img.*"),
                         nnx.All(nnx_utils.PathRegex(".*llm.*"), nnx.Not(nnx_utils.PathRegex(".*llm.*_1.*"))))
        data = dataclasses.replace(base.data, repo_id=repo,
                                   assets=C.AssetsConfig(assets_dir=os.path.join(L90C, "assets", full), asset_id=repo))
        if name == "pi05_g8v_raac_ae":                          # SR 77: + one state_proj (the channel token); base weights, state_proj random init
            from . import raac_model as RM
            m = base.model
            model = RM.Pi0ConfigC(pi05=True, action_horizon=m.action_horizon, discrete_state_input=False)
            return dataclasses.replace(base, name=name, ema_decay=None, freeze_filter=frozen, data=data, model=model,
                                       weight_loader=RM.RaacLoader(BASE_PARAMS), **over)
        if name == "pi05_g8v_raas_ae":                          # SR 78: RAA-c + the servo residual action head; K, max_step from results/flawed/raas_gate.json
            from . import raac_model as RM
            from . import raas_model as RS
            m = base.model
            model = RS.Pi0ConfigS(pi05=True, action_horizon=m.action_horizon, discrete_state_input=False, **RS.servo_fields())
            return dataclasses.replace(base, name=name, ema_decay=None, freeze_filter=frozen, data=data, model=model,
                                       weight_loader=RM.RaacLoader(BASE_PARAMS), **over)
        if name == "pi05_g8v_raas_l_ae":                        # SR 82: pi05_g8v_raas_ae exactly (model, servo constants, data), weights from pi05_libero's params; state_proj stays at its random init
            from . import raac_model as RM
            from . import raas_model as RS
            m = base.model
            model = RS.Pi0ConfigS(pi05=True, action_horizon=m.action_horizon, discrete_state_input=False, **RS.servo_fields())
            return dataclasses.replace(base, name=name, ema_decay=None, freeze_filter=frozen, data=data, model=model,
                                       weight_loader=RM.RaacLoader(LIBERO_PARAMS), **over)
        if name == "pi05_g8v_raas_ls2_ae":                      # SR 89: pi05_g8v_raas_l_ae exactly (model, servo constants, RAA-c norm stats / assets, weights from pi05_libero); ONLY the data repo differs
            from . import raac_model as RM
            from . import raas_model as RS
            m = base.model
            model = RS.Pi0ConfigS(pi05=True, action_horizon=m.action_horizon, discrete_state_input=False, **RS.servo_fields())
            data = dataclasses.replace(data, repo_id=S2_REPO)
            return dataclasses.replace(base, name=name, ema_decay=None, freeze_filter=frozen, data=data, model=model,
                                       weight_loader=RM.RaacLoader(LIBERO_PARAMS), **over)
        if name == "pi05_g8v_raac_lf_ae":                       # RAA-F Route A: RAA-c's model (state token, servo_k = 0: Pi0ConfigS reproduces Pi0C), weights from pi05_libero; ONLY the data repo differs from RAA-c's
            from . import raac_model as RM
            from . import raas_model as RS
            m = base.model
            model = RS.Pi0ConfigS(pi05=True, action_horizon=m.action_horizon, discrete_state_input=False, **RS.servo_fields(k=0.0))
            data = dataclasses.replace(data, repo_id=RAAF_REPO)
            return dataclasses.replace(base, name=name, ema_decay=None, freeze_filter=frozen, data=data, model=model,
                                       weight_loader=RM.RaacLoader(LIBERO_PARAMS), **over)
        if name == "pi05_g8v_raa_l_ae":                         # SR 80: RAA's config exactly (pi05_g8v_raa_ae), weights from pi05_libero's params
            return dataclasses.replace(base, name=name, ema_decay=None, freeze_filter=frozen, data=data,
                                       weight_loader=W.CheckpointWeightLoader(LIBERO_PARAMS), **over)
        return dataclasses.replace(base, name=name, ema_decay=None, freeze_filter=frozen, data=data, **over)
    if name == "pi05_goal8":                                  # pi05_l90c's recipe on the eight known tasks, its own norm stats
        base = train_config("pi05_l90c")
        return dataclasses.replace(base, name=name, data=dataclasses.replace(base.data, repo_id=GOAL8_REPO), **over)
    if name == "pi05_goal8_roles":                              # pi05_goal8 with per-frame role prompts
        base = train_config("pi05_l90c")
        return dataclasses.replace(base, name=name, data=dataclasses.replace(base.data, repo_id=ROLES_REPO), **over)
    if name == "pi05_goal8_rsup":                               # pi05_goal8_roles' routing with given role phrases
        base = train_config("pi05_l90c")
        return dataclasses.replace(base, name=name, data=dataclasses.replace(base.data, repo_id=RSUP_REPO), **over)
    if name == "pi05_l90c_fac":                                 # same data, steps and norm stats as pi05_l90c
        base = train_config("pi05_l90c")
        cfg = dataclasses.replace(base, name=name, data=_fac_data(C, base.data))
        return dataclasses.replace(cfg, **over)
    lora = name == "pi05_l90c_lora"
    model = (pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False,
                                  paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora")
             if lora else pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False))
    cfg = C.TrainConfig(
        name=name, model=model,
        data=C.LeRobotLiberoDataConfig(repo_id=REPO_ID, base_config=C.DataConfig(prompt_from_task=True),
                                       extra_delta_transform=False),
        batch_size=32,
        lr_schedule=O.CosineDecaySchedule(warmup_steps=1_000, peak_lr=5e-5, decay_steps=1_000_000, decay_lr=5e-5),   # pi05_libero's (constant
        # 5e-5 after warmup); warmup 1k instead of 10k for the smaller batch
        optimizer=O.AdamW(clip_gradient_norm=1.0),
        ema_decay=None if lora else 0.999,
        freeze_filter=model.get_freeze_filter() if lora else C.TrainConfig.__dataclass_fields__["freeze_filter"].default_factory(),
        weight_loader=W.CheckpointWeightLoader(BASE_PARAMS),
        num_train_steps=30_000, fsdp_devices=1 if lora else 4,
        assets_base_dir=os.path.join(L90C, "assets"), checkpoint_base_dir=os.path.join(L90C, "jax"),
        save_interval=1000, keep_period=5000, log_interval=50, num_workers=8, wandb_enabled=False)
    return dataclasses.replace(cfg, **over)


def fac_prompts(rebuild=False):
    """instruction -> [object-only, goal-only, verb-only] plain part prompts of every training task (store.parts_of's rule; a
    fixture task has no goal part).  Cached in results/l90c/fac_prompts.json."""
    import json
    if os.path.exists(FAC_PROMPTS) and not rebuild:
        return json.load(open(FAC_PROMPTS, encoding="utf-8"))
    from .. import l90c
    from ..protos import store as S
    from . import factors as F
    tasks = {t.tid: t for t in F.load(("libero_90", "libero_10", "libero_spatial", "libero_object", "libero_goal"))}
    phrases = F.object_phrases(list(tasks.values()))
    out = {}
    for tid in l90c.load_split()["train"]:
        t = tasks[tid]
        pt = S.parts_of(t, phrases)
        parts = [f"the {pt['obj']}"] if pt["obj"] else []
        parts += [pt["goal"]] if pt["goal"] else []
        parts += [pt["verb"]] if pt["verb"] else []
        out[t.language] = parts
    os.makedirs(os.path.dirname(FAC_PROMPTS), exist_ok=True)
    with open(FAC_PROMPTS, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    return out


class FactoredPrompt:
    """openpi data transform: the factored-conditioning prompt dropout (training only; inference never sees it).
    Its RNG is created lazily per process, so forked loader workers do not share a stream."""

    def __init__(self, table, p_part=P_PART, p_null=P_NULL, seed=0):
        self.table, self.p_part, self.p_null, self.seed = dict(table), p_part, p_null, seed
        self._pid, self._rng = None, None

    def __call__(self, data):
        import numpy as np
        if self._pid != os.getpid():
            self._pid, self._rng = os.getpid(), np.random.default_rng([self.seed, os.getpid()])
        p = data.get("prompt")
        text = p.decode() if isinstance(p, bytes) else str(np.asarray(p).item() if not isinstance(p, str) else p)
        u = self._rng.random()
        parts = self.table.get(text, [])
        if u < self.p_null:
            text = ""
        elif u < self.p_null + self.p_part and parts:
            text = parts[int(self._rng.integers(len(parts)))]
        return {**data, "prompt": text}


def _fac_data(C, base):
    """A LeRobotLiberoDataConfig whose data transforms start with FactoredPrompt (after the repack, before tokenisation)."""
    import openpi.transforms as T

    @dataclasses.dataclass(frozen=True)
    class FactoredLiberoDataConfig(C.LeRobotLiberoDataConfig):
        def create(self, assets_dirs, model_config):
            dc = super().create(assets_dirs, model_config)
            inputs = [FactoredPrompt(fac_prompts())] + list(dc.data_transforms.inputs)
            return dataclasses.replace(dc, data_transforms=T.Group(inputs=inputs, outputs=dc.data_transforms.outputs))
    return FactoredLiberoDataConfig(repo_id=base.repo_id, base_config=base.base_config,
                                    extra_delta_transform=base.extra_delta_transform,
                                    assets=C.AssetsConfig(assets_dir=os.path.join(L90C, "assets", "pi05_l90c"), asset_id=REPO_ID))


def register(names=NAMES):
    C = _openpi()
    for n in names:
        if n not in C._CONFIGS_DICT:
            C._CONFIGS_DICT[n] = train_config(n)
    return C


def pytorch_dir(name):
    """Where the converted PyTorch weights of a config live (PolicyBackbone's checkpoint_dir)."""
    return f"{CACHE}/pi05_base_pytorch" if name == "pi05_base_libero" else os.path.join(L90C, "pytorch", name)
