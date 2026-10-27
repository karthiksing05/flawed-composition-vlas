"""BindVLABackbone: a trained BindVLA checkpoint (experiments/bv_train.py) behind SmolVLABackbone's interface, so serve_guided.py
MODE=ours serves it unchanged (BACKBONE_CONFIG=smolvla..., BACKBONE_CKPT=<results/bindvla/<exp>/ckpt/step_N.pt or the exp dir>;
SmolVLABackbone dispatches here when the checkpoint is a BindVLA one).  The sampler, the prefix cache and the velocity are
SmolVLA's (VLAFlowMatching.denoise_step); only the prefix (lifelong_ood/policy/bindvla.py) and the normalisation (the training
cache's MEAN_STD stats, stored in the checkpoint) differ.  Observations are the openpi LIBERO client's (224 px, already flipped,
8-D state), which is what the training cache stores.
"""
from __future__ import annotations

import glob
import os
import time

import numpy as np
import torch

from .backbone import Prefix
from .backbone_smolvla import SmolVLABackbone


def resolve(path):
    if os.path.isdir(path):
        have = sorted(glob.glob(os.path.join(path, "ckpt", "step_*.pt"))) or sorted(glob.glob(os.path.join(path, "step_*.pt")))
        assert have, f"no BindVLA checkpoint under {path}"
        return have[-1]
    return path


def is_bindvla(path):
    if not path:
        return False
    if os.path.isdir(path):
        return bool(glob.glob(os.path.join(path, "ckpt", "step_*.pt")) or glob.glob(os.path.join(path, "step_*.pt")))
    return path.endswith(".pt") and "bindvla" in path


class BindVLABackbone(SmolVLABackbone):
    kind = "bindvla"

    def __init__(self, checkpoint=None, device="cuda", num_steps=None):
        from . import bindvla as BV
        t0 = time.time()
        self.checkpoint_dir = resolve(checkpoint)
        ck = torch.load(self.checkpoint_dir, map_location="cpu", weights_only=False)
        cfg = BV.BVConfig(**ck["cfg"])
        self.bv = BV.BindVLA(cfg, device=device)
        res = self.bv.load_state_dict(ck["model"], strict=False)
        assert not res.unexpected_keys, res.unexpected_keys
        self.bv = self.bv.to(device).eval()
        for q in self.bv.parameters():
            q.requires_grad_(False)
        self.cfg, self.bvcfg = self.bv.scfg, cfg
        self.cond = BV.Conditioner(self.bv.tokenizer, cfg)
        self.model = self.bv.flow
        self.policy = None
        self.device = device
        self.H, self.A, self.a = cfg.chunk, cfg.max_action_dim, 7
        self.dim, self.dim_full = self.H * self.a, self.H * self.A
        self.real = np.concatenate([np.arange(self.a) + h * self.A for h in range(self.H)])
        self.num_steps = int(num_steps or 10)
        self.clip = (-1e9, 1e9)
        self.layout = "chunk"
        self.img = None
        self.n_params = int(sum(p.numel() for p in self.bv.parameters()))
        self.expert = self.model.vlm_with_expert.lm_expert
        st = ck["stats"]
        self.s_mu, self.s_sd = np.asarray(st["state"]["mean"], np.float32), np.asarray(st["state"]["std"], np.float32)
        self.action_stats = {k: np.asarray(st["actions"][k], np.float32) for k in ("mean", "std")}
        self.eps = 1e-8
        self.step = int(ck["step"])
        self.setting = ck.get("setting")
        self.load_secs = time.time() - t0

    @torch.no_grad()
    def observations(self, obs_dicts):
        img = torch.stack([torch.from_numpy(np.ascontiguousarray(o["observation/image"])).permute(2, 0, 1) for o in obs_dicts]).float() / 255.0
        wr = torch.stack([torch.from_numpy(np.ascontiguousarray(o["observation/wrist_image"])).permute(2, 0, 1) for o in obs_dicts]).float() / 255.0
        if img.shape[-1] != 224:
            img = torch.nn.functional.interpolate(img, size=(224, 224), mode="bilinear", align_corners=False)
            wr = torch.nn.functional.interpolate(wr, size=(224, 224), mode="bilinear", align_corners=False)
        st = torch.zeros(len(obs_dicts), self.bvcfg.max_state_dim)
        s = np.stack([(np.asarray(o["observation/state"], np.float32) - self.s_mu) / (self.s_sd + self.eps) for o in obs_dicts])
        st[:, :8] = torch.from_numpy(s)
        cs = [self.cond(str(o["prompt"])) for o in obs_dicts]
        b = dict(img=img, wrist=wr, state=st, lang_tokens=torch.from_numpy(np.stack([c[0] for c in cs])),
                 lang_masks=torch.from_numpy(np.stack([c[1] for c in cs])), unit_tokens=torch.from_numpy(np.stack([c[2] for c in cs])),
                 unit_masks=torch.from_numpy(np.stack([c[3] for c in cs])))
        b = {k: v.to(self.device) for k, v in b.items()}
        b["state_vec"] = b["state"]
        return _Obs(b)

    def observation(self, obs_dict):
        return self.observations([obs_dict])

    @torch.no_grad()
    def prefix(self, observation):
        pp, kv, diag = self.bv.prefix_cache(observation.batch)
        self.last_diag = diag
        return Prefix(state=observation.state, pad_masks=pp, kv=kv, n=int(pp.shape[0]))

    def text_span(self, observation):
        raise NotImplementedError("BindVLA has no instruction span in its bottleneck prefix")

    def cond_layers(self):
        return list(self.model.vlm_with_expert.get_vlm_model().text_model.layers)

    def unnormalize(self, rows, state=None):
        R = np.asarray(rows, np.float32).reshape(-1, self.H, self.a)
        return R * (self.action_stats["std"] + self.eps) + self.action_stats["mean"]

    def describe(self):
        return (f"BindVLABackbone({self.checkpoint_dir}: setting {self.setting}, step {self.step}, H={self.H}, A={self.A}, "
                f"steps={self.num_steps}, n_params={self.n_params:,}, load {self.load_secs:.1f}s)")


class _Obs:
    def __init__(self, batch):
        self.batch = batch
        self.state = batch["state"]
