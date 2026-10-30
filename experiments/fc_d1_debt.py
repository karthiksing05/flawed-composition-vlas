"""FLAWED_COMPOSITION D1: compositional debt per interface in a trained BindVLA-family checkpoint, by interventions.  No training.
SmolVLA venv, one GPU.  Estimators are fixed in docs/FLAWED_COMPOSITION.md section 4.5 and the pre-registration row (m3_prereg
tag pd01); this script only writes per-frame records, experiments/fc_d1_read.py aggregates them.

Frame sets (both use the training-referent event rule, fc_d1_frames.frames_of: frames e-10, e-5, e around the first two
grasp / release events):
  IN   the held-out episodes of the 76 training tasks (bv_data.py: 2 per task, never trained on), from the training cache
  DEV  the 29 dev tasks' own demos (fc_d1_frames.py; scored only, never trained on, never executed)

Every prediction is the model's 10-step Euler sample (its serving sampler) from 2 fixed noise draws, seeded by the frame's
identity, so every model and every variant sees the same noise.  Action error = MSE over the first 10 chunk steps x 7 dims in
the cache's MEAN_STD-normalised units (pad steps masked), averaged over the 2 draws.  The output change of a variant =
the same MSE between the variant's and the intact sample.

Interfaces.  BindVLA (main2, C1): role messages = the binder's slot tokens (verb, object, goal per step); bypass = the global
token (main2 only; C1 has none) and the state token.  vanilla: role messages = the instruction tokens and the agentview connector
cells within Chebyshev 1 (8 x 8 grid) of the step-0 contact-event referent tokens (the cache's refs; the episode's frame-0
refs when a frame has none); bypass = the other agentview cells, the wrist cells and the state token.

Variants on IN frames (partner = a frame of another training task of the SAME BDDL scene, chosen by a fixed hash, at the
same phase fraction of its first held-out episode):
  role   the partner's instruction on this frame's images (BindVLA: slot tokens recomputed with it; vanilla: its tokens)
  byp    the bypass inputs taken from the partner frame, role messages kept
  st     the state token alone taken from the partner frame
  glob   main2 only: the global token alone from the partner frame
  cut    the bypass mean-ablated over the IN frames (main2: global token; vanilla: non-interface agentview cells and wrist
         cells; C1: nothing to cut, cut = intact)
  cutS   cut and the state token from the IN-frame mean normalised state
Purity (IN frames, image and scene fixed): the instruction with step 0's GOAL phrase replaced by another training task's goal
phrase from the same scene (other-role swap for the object message, own-role swap for the goal message) and with step 0's
OBJECT phrase replaced by the object phrase of another transport task of the same scene (the reverse); a frame is scored only
when both swaps exist and each replaced phrase occurs once in the instruction.  Messages: the binder's step-0 slot token (BindVLA only), and the value states the
expert reads (16 layers, concatenated) at the slot position (BindVLA) or averaged over the role phrase's instruction tokens
(vanilla).  Probe (descriptive): nearest-centroid decoding, across held-out episodes, of the other role's atom from this role's
intact message, within groups of training tasks that share this role's atom.

    python experiments/fc_d1_debt.py --ckpt results/bindvla/main2/ckpt/step_060000.pt --name main2 [--limit N]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "experiments")]
CACHE = "/coc/flash10/ksingara3/robot/data/bindvla_cache"
OUT = os.path.join(ROOT, "results", "flawed")
HE = 10
DRAWS = 2
BS = 16
T_FLOW = (0.25, 0.5, 0.75)


def hseed(*k):
    return int(hashlib.sha256("|".join(map(str, k)).encode()).hexdigest()[:8], 16)


def step0_phrases(text):
    import s1_labels as SL
    from lifelong_ood.policy import primitives as P
    st = P.steps_of(text.lower().strip().rstrip("."))
    if not st:
        return "", ""
    o, g = SL.split_step(" ".join(st[0].strip().lower().split()))
    import lifelong_ood.policy.bindvla as BV
    o = BV.VERB.sub("", o.strip())
    return o.strip(), g.strip()


def replace_once(text, a, b):
    if not a or text.count(a) != 1 or a == b:
        return None
    return text.replace(a, b, 1)


def cells_of(tokens):
    """16 x 16 referent tokens -> the 8 x 8 connector cells within Chebyshev 1."""
    out = set()
    for t in tokens:
        if t < 0:
            continue
        r, c = divmod(int(t), 16)
        r, c = r // 2, c // 2
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if 0 <= r + dr < 8 and 0 <= c + dc < 8:
                    out.add((r + dr) * 8 + c + dc)
    return sorted(out)


class Model:
    def __init__(self, ckpt):
        from lifelong_ood.policy.backbone_bindvla import BindVLABackbone
        self.bk = BindVLABackbone(ckpt, device="cuda")
        self.bv, self.cfg, self.cond = self.bk.bv, self.bk.bvcfg, self.bk.cond
        self.kind = "vanilla" if self.bv.binder is None else ("bind_glob" if self.cfg.glob else "bind_noglob")
        self.dt = self.bv.flow.vlm_with_expert.get_vlm_model().text_model.layers[0].self_attn.q_proj.weight.dtype
        self.a_mu = torch.tensor(self.bk.action_stats["mean"], device="cuda")
        self.a_sd = torch.tensor(self.bk.action_stats["std"], device="cuda")

    # inputs -------------------------------------------------------------------------------------------------------
    def vision(self, agent, wrist):
        img = torch.from_numpy(np.stack(agent)).permute(0, 3, 1, 2).float().cuda() / 255.0
        wr = torch.from_numpy(np.stack(wrist)).permute(0, 3, 1, 2).float().cuda() / 255.0
        gA, cA = self.bv.vision(img)
        gW, cW = self.bv.vision(wr)
        return dict(gA=gA, cA=cA, gW=gW, cW=cW, img=img, wrist=wr)

    def state(self, raw):
        s = np.stack([(np.asarray(x, np.float32) - self.bk.s_mu) / (self.bk.s_sd + 1e-8) for x in raw])
        st = torch.zeros(len(raw), self.cfg.max_state_dim, device="cuda")
        st[:, :8] = torch.from_numpy(s).cuda()
        return st

    def lang(self, texts):
        cs = [self.cond(t) for t in texts]
        T = lambda i: torch.from_numpy(np.stack([c[i] for c in cs])).cuda()          # noqa: E731
        return T(0), T(1), T(2), T(3)

    def roles(self, V, texts):
        """-> dict of role-message tensors for these instructions on these images."""
        lt, lm, ut, um = self.lang(texts)
        if self.bv.binder is None:
            return dict(lang=self.bv.embed_tokens(lt) * math.sqrt(self.bv.d_tok), lmask=lm.bool())
        toks, valid, g, _ = self.bv.binder(V["gA"], V["gW"], torch.stack([V["cA"].mean(1), V["cW"].mean(1)], 1),
                                           self.bv.embed_tokens(ut), um.bool(), full_emb=self.bv.embed_tokens(lt), full_mask=lm.bool())
        return dict(toks=toks, valid=valid, g=g)

    def prefix(self, R, V, st, cA=None, cW=None, g=None):
        """assemble the prefix the expert reads; optional overrides of the agentview / wrist cells (vanilla) or the global token."""
        B = st.shape[0]
        one = lambda n: torch.ones(B, n, dtype=torch.bool, device="cuda")              # noqa: E731
        stok = self.bv.flow.state_proj(st.float())[:, None]
        if self.bv.binder is None:
            parts = [V["cA"] if cA is None else cA, V["cW"] if cW is None else cW, R["lang"], stok]
            pads = [one(64 + 64), R["lmask"], one(1)]
        else:
            parts, pads = [R["toks"]], [R["valid"]]
            if self.cfg.glob:
                parts.append(R["g"] if g is None else g)
                pads.append(one(1))
            parts.append(stok)
            pads.append(one(1))
        embs = torch.cat([p.to(self.dt) for p in parts], 1)
        pad = torch.cat(pads, 1)
        return embs, pad

    @torch.no_grad()
    def kv(self, embs, pad):
        from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks
        att = torch.zeros(pad.shape, dtype=torch.bool, device="cuda")
        att[:, -1] = True
        att2 = make_att_2d_masks(pad, att)
        pos = torch.cumsum(pad, dim=1) - 1
        _, kv = self.bv.flow.vlm_with_expert.forward(attention_mask=att2, position_ids=pos, past_key_values=None,
                                                     inputs_embeds=[embs, None], use_cache=True, fill_kv_cache=True)
        return kv

    @torch.no_grad()
    def sample(self, embs, pad, noises):
        kv = self.kv(embs, pad)
        outs = []
        for nz in noises:
            x = nz.clone()
            dt = -1.0 / 10
            for s in range(10):
                t = torch.full((x.shape[0],), 1.0 + s * dt, device="cuda")
                x = x + dt * self.bv.flow.denoise_step(prefix_pad_masks=pad, past_key_values=kv, x_t=x, timestep=t)
            outs.append(x[:, :, :7].float())
        return torch.stack(outs, 0), kv                                               # [D, B, H, 7]

    def values_at(self, kv, idx):
        """value states the expert reads, concatenated over layers: idx [B, n] positions -> mean over n -> [B, L * heads * hd]."""
        out = []
        for l in sorted(kv):
            v = kv[l]["value_states"].float()                                        # [B, T, h, hd]
            g = torch.stack([v[b, idx[b]].mean(0) for b in range(v.shape[0])])
            out.append(g.reshape(g.shape[0], -1))
        return torch.cat(out, -1)


def in_frames(store, limit=0):
    """held-out training episodes -> frame list with partners (fixed hash) and purity swap texts."""
    from fc_d1_frames import frames_of
    from lifelong_ood.policy import factors as F
    import c_split as CS
    tasks = {t.tid: t for t in F.load(CS.SUITES)}
    ho = {}
    for i in store.held_idx:
        ho.setdefault(int(store.ep[i]), []).append(int(i))
    by_task = {}
    for e in sorted(ho):
        by_task.setdefault(store.eps[e]["tid"], []).append(e)
    scene_of = {t: tasks[t].scene for t in by_task}
    text_of = {t: store.eps[by_task[t][0]]["text"] for t in by_task}
    atoms = {}
    for t in by_task:
        st = CS.steps(tasks[t])
        atoms[t] = (st[0][1], st[0][2]) if st else (None, None)
    rows = []
    for t in sorted(by_task):
        o, g = step0_phrases(text_of[t])
        cands = sorted([u for u in by_task if u != t and scene_of[u] == scene_of[t] and text_of[u] != text_of[t]],
                       key=lambda u: hseed("d1-partner", t, u))
        part = cands[0] if cands else None
        gs = os_ = None
        for u in cands:
            ou, gu = step0_phrases(text_of[u])
            if gs is None and gu and g and gu != g:
                gs = (u, replace_once(text_of[t], g, gu))
            if os_ is None and ou and o and g and gu and ou != o:     # object swaps only between transport tasks (movable objects)
                os_ = (u, replace_once(text_of[t], o, ou))
        for e in by_task[t]:
            idx = np.array(ho[e])
            acts, state = store.actions[idx], store.state[idx]
            for f, ph, k in frames_of(acts, state):
                i = int(idx[f])
                ref = [int(x) for x in store.ref[i][0]]
                if max(ref) < 0:
                    ref = [int(x) for x in store.ref[idx[0]][0]]
                r = dict(set="IN", tid=t, ep=e, frame=f, phase=ph, transport=k, i=i, text=text_of[t], scene=scene_of[t],
                         frac=f / max(len(idx) - 1, 1), ref=ref, atoms=atoms[t], obj_phrase=o, goal_phrase=g,
                         partner=part, goalswap=gs, objswap=os_)
                if part is not None:
                    pe = by_task[part][0]
                    pidx = np.array(ho[pe])
                    r["partner_i"] = int(pidx[int(round(r["frac"] * (len(pidx) - 1)))])
                    r["partner_text"] = text_of[part]
                rows.append(r)
    return rows[:limit] if limit else rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--limit", type=int, default=0, help="smoke: first N IN frames and N DEV frames only")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    t0 = time.time()
    torch.backends.cuda.matmul.allow_tf32 = True
    import bv_train as BT
    store = BT.Store()
    M = Model(a.ckpt)
    assert np.allclose(M.bk.action_stats["mean"], store.a_mu) and np.allclose(M.bk.s_mu, store.s_mu), "stats differ from the cache"
    IN = in_frames(store, a.limit)
    z0 = np.load(os.path.join(CACHE, "d1_dev.npz"))
    z = {k: z0[k] for k in z0.files}
    nd = len(z["tid"]) if not a.limit else min(a.limit, len(z["tid"]))
    from lifelong_ood import l90c
    for t in set(z["tid"].tolist()):
        l90c.check_task(t)
    lang_of = json.load(open(os.path.join(OUT, "d1_frames.json"), encoding="utf-8"))["per_task"]

    def tgt(acts, pad):
        x = (torch.from_numpy(np.stack(acts)).cuda() - M.a_mu) / (M.a_sd + 1e-8)
        return x, torch.from_numpy(np.stack(pad)).cuda()

    def err(pred, target, pad):
        """pred [D, B, H, 7] vs target [B, H, 7] -> per-frame MSE over the first HE unpadded steps, mean over draws."""
        m = (~pad[:, :HE]).float()
        d = ((pred[:, :, :HE] - target[None, :, :HE]) ** 2).mean(-1)                 # [D, B, HE]
        return ((d * m).sum(-1) / m.sum(-1).clamp(min=1)).mean(0)

    def delta(p, q):
        return ((p[:, :, :HE] - q[:, :, :HE]) ** 2).mean((-1, -2)).mean(0)

    def site(pred, acts, pad):
        raw = pred * (M.a_sd + 1e-8) + M.a_mu
        tr = torch.from_numpy(np.stack(acts)).cuda()
        m = (~pad[:, :HE]).float()[..., None]
        ps = (raw[:, :, :HE, :3] * m[None]).sum(2)
        ts = (tr[:, :HE, :3] * m).sum(1)
        return (ps - ts[None]).norm(dim=-1).mean(0)

    def noises(keys):
        out = []
        for d in range(DRAWS):
            out.append(torch.stack([torch.from_numpy(np.random.default_rng(hseed("d1-noise", *k, d)).standard_normal(
                (M.cfg.chunk, M.cfg.max_action_dim)).astype(np.float32)) for k in keys]).cuda())
        return out

    @torch.no_grad()
    def flow_loss(V, R, st, target, pad, keys):
        """fixed-(t, noise) flow MSE over the 7 real dims and unpadded steps, mean over T_FLOW."""
        embs, pp = M.prefix(R, V, st)
        from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks
        f = M.bv.flow
        act = torch.zeros(target.shape[0], M.cfg.chunk, M.cfg.max_action_dim, device="cuda")
        act[:, :, :7] = target
        tot = 0
        for j, tv in enumerate(T_FLOW):
            nz = torch.stack([torch.from_numpy(np.random.default_rng(hseed("d1-flow", *k, j)).standard_normal(
                (M.cfg.chunk, M.cfg.max_action_dim)).astype(np.float32)) for k in keys]).cuda()
            time_ = torch.full((target.shape[0],), tv, device="cuda")
            x_t = tv * nz + (1 - tv) * act
            se, sp, sa = f.embed_suffix(x_t, time_)
            att = torch.zeros(pp.shape, dtype=torch.bool, device="cuda")
            att[:, -1] = True
            pad2 = torch.cat([pp, sp], 1)
            att2 = make_att_2d_masks(pad2, torch.cat([att.to(sa.dtype), sa], 1))
            pos = torch.cumsum(pad2, dim=1) - 1
            (_, out), _ = f.vlm_with_expert.forward(attention_mask=att2, position_ids=pos, past_key_values=None,
                                                    inputs_embeds=[embs, se], use_cache=False, fill_kv_cache=False)
            v = f.action_out_proj(out[:, -M.cfg.chunk:].float())[:, :, :7]
            m = (~pad).float()
            tot = tot + (((v - (nz[:, :, :7] - target)) ** 2).mean(-1) * m).sum(-1) / m.sum(-1).clamp(min=1)
        return tot / len(T_FLOW)

    # ── IN pass 1: means for the cut variants ─────────────────────────────────────────────────────────────────────
    sumg, sumA, sumW, sums, n = 0, 0, 0, 0, 0
    for b in range(0, len(IN), BS):
        rs = IN[b:b + BS]
        its = [store.item(r["i"]) for r in rs]
        with torch.no_grad():
            V = M.vision([x["img"] for x in its], [x["wrist"] for x in its])
            st = M.state([store.state[r["i"]] for r in rs])
            if M.bv.binder is not None:
                sumg = sumg + M.roles(V, [r["text"] for r in rs])["g"].float().sum(0)
            sumA, sumW, sums, n = sumA + V["cA"].sum(0), sumW + V["cW"].sum(0), sums + st.sum(0), n + len(rs)
    mean_g = (sumg / n) if M.bv.binder is not None else None
    mean_A, mean_W, mean_s = sumA / n, sumW / n, sums / n

    rec, pur = [], []
    msgs = {"obj_tok": [], "goal_tok": [], "obj_kv": [], "goal_kv": [], "row": []}

    def vanilla_positions(texts, phrase_of):
        """instruction-token positions (in the prefix) of a phrase of each text; None if not found."""
        out = []
        for t, ph in zip(texts, phrase_of):
            s = t.strip() + "\n"
            e = M.bv.tokenizer(s, max_length=M.cfg.tok_len, padding="max_length", truncation=True, return_offsets_mapping=True)
            c0 = s.find(ph) if ph else -1
            if c0 < 0:
                out.append(None)
                continue
            c1 = c0 + len(ph)
            p = [128 + j for j, (x, y) in enumerate(e["offset_mapping"]) if e["attention_mask"][j] and y > c0 and x < c1]
            out.append(p or None)
        return out

    # ── IN pass 2: variants, purity, messages ─────────────────────────────────────────────────────────────────────
    for b in range(0, len(IN), BS):
        rs = IN[b:b + BS]
        B = len(rs)
        its = [store.item(r["i"]) for r in rs]
        target, pad = tgt([store.actions[np.minimum(np.arange(r["i"], r["i"] + 50), store.end[r["i"]] - 1)] for r in rs],
                          [np.arange(r["i"], r["i"] + 50) >= store.end[r["i"]] for r in rs])
        raw_acts = [store.actions[np.minimum(np.arange(r["i"], r["i"] + 50), store.end[r["i"]] - 1)] for r in rs]
        keys = [(r["tid"], r["ep"], r["frame"]) for r in rs]
        nz = noises(keys)
        with torch.no_grad():
            V = M.vision([x["img"] for x in its], [x["wrist"] for x in its])
            st = M.state([store.state[r["i"]] for r in rs])
            R = M.roles(V, [r["text"] for r in rs])
            p_int, kv_int = M.sample(*M.prefix(R, V, st), nz)
            e_int = err(p_int, target, pad)
            fl = flow_loss(V, R, st, target, pad, keys)
            sit = site(p_int, raw_acts, pad)
            res = {"intact": (p_int, e_int)}
            # partner-based variants
            hp = [r.get("partner_i") is not None for r in rs]
            pi = [r["partner_i"] if r.get("partner_i") is not None else r["i"] for r in rs]
            pits = [store.item(i) for i in pi]
            Vp = M.vision([x["img"] for x in pits], [x["wrist"] for x in pits])
            stp = M.state([store.state[i] for i in pi])
            Rrole = M.roles(V, [r.get("partner_text") or r["text"] for r in rs])
            res["role"] = M.sample(*M.prefix(Rrole, V, st), nz)[0]
            res["st"] = M.sample(*M.prefix(R, V, stp), nz)[0]
            if M.bv.binder is None:
                keep = torch.zeros(B, 64, 1, device="cuda")
                for j, r in enumerate(rs):
                    keep[j, cells_of(r["ref"])] = 1
                cA_byp = keep * V["cA"] + (1 - keep) * Vp["cA"]
                res["byp"] = M.sample(*M.prefix(R, V, stp, cA=cA_byp, cW=Vp["cW"]), nz)[0]
                cA_cut = keep * V["cA"] + (1 - keep) * mean_A[None]
                cW_cut = mean_W[None].expand_as(V["cW"])
                res["cut"] = M.sample(*M.prefix(R, V, st, cA=cA_cut, cW=cW_cut), nz)[0]
                res["cutS"] = M.sample(*M.prefix(R, V, mean_s[None].expand_as(st), cA=cA_cut, cW=cW_cut), nz)[0]
            elif M.cfg.glob:
                gp = M.roles(Vp, [r["text"] for r in rs])["g"]
                res["glob"] = M.sample(*M.prefix(R, V, st, g=gp), nz)[0]
                res["byp"] = M.sample(*M.prefix(R, V, stp, g=gp), nz)[0]
                gm = mean_g[None].expand_as(R["g"]).to(R["g"].dtype)
                res["cut"] = M.sample(*M.prefix(R, V, st, g=gm), nz)[0]
                res["cutS"] = M.sample(*M.prefix(R, V, mean_s[None].expand_as(st), g=gm), nz)[0]
            else:
                res["byp"] = res["st"]
                res["cut"] = p_int
                res["cutS"] = M.sample(*M.prefix(R, V, mean_s[None].expand_as(st)), nz)[0]
            for j, r in enumerate(rs):
                row = dict(set="IN", tid=r["tid"], ep=r["ep"], frame=r["frame"], phase=r["phase"], transport=r["transport"],
                           scene=r["scene"], partner=r["partner"] if hp[j] else None, flow=float(fl[j]), site=float(sit[j]),
                           err={k: float(err(v[:, j:j + 1], target[j:j + 1], pad[j:j + 1])[0]) for k, v in
                                [("intact", p_int)] + [(k, v) for k, v in res.items() if k != "intact"]},
                           delta={k: float(delta(v[:, j:j + 1], p_int[:, j:j + 1])[0]) for k, v in res.items() if k != "intact"})
                if not hp[j]:
                    for k in ("role", "st", "byp", "glob"):
                        row["delta"].pop(k, None)
                        row["err"].pop(k, None)
                rec.append(row)
            # purity: the same images, instruction swaps
            texts0 = [r["text"] for r in rs]
            tg = [r["goalswap"][1] if r["goalswap"] and r["goalswap"][1] else None for r in rs]
            to = [r["objswap"][1] if r["objswap"] and r["objswap"][1] else None for r in rs]
            ok = [x is not None and y is not None for x, y in zip(tg, to)]
            if any(ok):
                J = [j for j in range(B) if ok[j]]
                sub = lambda X: {k: v[J] for k, v in X.items()}                       # noqa: E731
                Vs = sub({k: V[k] for k in ("gA", "cA", "gW", "cW")})
                sts = st[J]
                variants = {"intact": [texts0[j] for j in J], "goalswap": [tg[j] for j in J], "objswap": [to[j] for j in J]}
                tok_m, kv_m = {}, {}
                for vn, tx in variants.items():
                    Rv = M.roles(Vs, tx)
                    kvv = M.kv(*M.prefix(Rv, Vs, sts))
                    if M.bv.binder is not None:
                        tok_m[vn] = (Rv["toks"][:, 1].float(), Rv["toks"][:, 2].float())
                        idx_o = torch.full((len(J), 1), 1, device="cuda", dtype=torch.long)
                        idx_g = torch.full((len(J), 1), 2, device="cuda", dtype=torch.long)
                        kv_m[vn] = (M.values_at(kvv, idx_o), M.values_at(kvv, idx_g))
                    else:
                        # role phrase of this variant's text: the object phrase is unchanged by the goal swap and vice versa
                        ph_o = [step0_phrases(t)[0] for t in tx]
                        ph_g = [step0_phrases(t)[1] for t in tx]
                        po, pg = vanilla_positions(tx, ph_o), vanilla_positions(tx, ph_g)
                        if any(x is None for x in po + pg):
                            kv_m = None
                            break
                        kv_m[vn] = (M.values_at(kvv, [torch.tensor(x, device="cuda") for x in po]),
                                    M.values_at(kvv, [torch.tensor(x, device="cuda") for x in pg]))
                for jj, j in enumerate(J):
                    r = rs[j]
                    pr = dict(tid=r["tid"], ep=r["ep"], frame=r["frame"], goal_partner=r["goalswap"][0], obj_partner=r["objswap"][0])
                    for lvl, mm in (("tok", tok_m), ("kv", kv_m)):
                        if not mm:
                            continue
                        I, G, O = mm["intact"], mm["goalswap"], mm["objswap"]
                        pr[f"{lvl}_obj_other"] = float((G[0][jj] - I[0][jj]).norm())    # object message, goal changed
                        pr[f"{lvl}_obj_own"] = float((O[0][jj] - I[0][jj]).norm())      # object message, object changed
                        pr[f"{lvl}_goal_other"] = float((O[1][jj] - I[1][jj]).norm())
                        pr[f"{lvl}_goal_own"] = float((G[1][jj] - I[1][jj]).norm())
                        pr[f"{lvl}_obj_norm"] = float(I[0][jj].norm())
                        pr[f"{lvl}_goal_norm"] = float(I[1][jj].norm())
                    pur.append(pr)
            # messages for the probe (intact)
            if M.bv.binder is not None:
                msgs["obj_tok"].append(R["toks"][:, 1].float().cpu().numpy())
                msgs["goal_tok"].append(R["toks"][:, 2].float().cpu().numpy())
            idx_ok = []
            if M.bv.binder is not None:
                vo = M.values_at(kv_int, torch.full((B, 1), 1, device="cuda", dtype=torch.long))
                vg = M.values_at(kv_int, torch.full((B, 1), 2, device="cuda", dtype=torch.long))
                idx_ok = list(range(B))
            else:
                po = vanilla_positions(texts0, [r["obj_phrase"] for r in rs])
                pg = vanilla_positions(texts0, [r["goal_phrase"] for r in rs])
                idx_ok = [j for j in range(B) if po[j] is not None and pg[j] is not None]
                if idx_ok:
                    sub_kv = {l: {k: v[idx_ok] for k, v in d.items()} for l, d in kv_int.items()}
                    vo = M.values_at(sub_kv, [torch.tensor(po[j], device="cuda") for j in idx_ok])
                    vg = M.values_at(sub_kv, [torch.tensor(pg[j], device="cuda") for j in idx_ok])
            if idx_ok:
                msgs["obj_kv"].append(vo.cpu().numpy().astype(np.float16))
                msgs["goal_kv"].append(vg.cpu().numpy().astype(np.float16))
                msgs["row"] += [dict(tid=rs[j]["tid"], ep=rs[j]["ep"], atoms=rs[j]["atoms"]) for j in idx_ok]
        print(f"[d1] IN {b + B}/{len(IN)} {time.time() - t0:.0f}s", flush=True)

    # ── DEV: intact ─────────────────────────────────────────────────────────────────────────────────────────────────
    for b in range(0, nd, BS):
        J = list(range(b, min(b + BS, nd)))
        target, pad = tgt([z["actions"][j] for j in J], [z["pad"][j] for j in J])
        keys = [(str(z["tid"][j]), int(z["ep"][j]), int(z["frame"][j])) for j in J]
        texts = [lang_of[str(z["tid"][j])]["language"] for j in J]
        with torch.no_grad():
            V = M.vision([z["agent"][j] for j in J], [z["wrist"][j] for j in J])
            st = M.state([z["state"][j] for j in J])
            R = M.roles(V, texts)
            p, _ = M.sample(*M.prefix(R, V, st), noises(keys))
            e = err(p, target, pad)
            fl = flow_loss(V, R, st, target, pad, keys)
            sit = site(p, [z["actions"][j] for j in J], pad)
        for jj, j in enumerate(J):
            rec.append(dict(set="DEV", tid=str(z["tid"][j]), bin=str(z["bin"][j]), ep=int(z["ep"][j]), frame=int(z["frame"][j]),
                            phase=str(z["phase"][j]), transport=int(z["transport"][j]), err={"intact": float(e[jj])},
                            flow=float(fl[jj]), site=float(sit[jj])))
        print(f"[d1] DEV {J[-1] + 1}/{nd} {time.time() - t0:.0f}s", flush=True)

    # ── probe (descriptive): other role's atom from this role's intact message, within same-atom groups, across episodes ──
    probe = {}
    rows = msgs["row"]
    for lvl in ("tok", "kv"):
        for role, other, key in (("obj", 1, 0), ("goal", 0, 1)):
            X = msgs.get(f"{role}_{lvl}")
            if not X:
                continue
            X = np.concatenate(X).astype(np.float32) if lvl == "kv" else np.concatenate(X)
            if lvl == "tok":
                X = X[[k for k in range(len(X))]] if len(X) == len(rows) else None
            if X is None or len(X) != len(rows):
                continue
            Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
            groups = {}
            for k, r in enumerate(rows):
                if r["atoms"][key] is None or r["atoms"][other] is None:
                    continue
                groups.setdefault(r["atoms"][key], []).append(k)
            hits, chance, used = [], [], []
            for gname, ks in groups.items():
                cls = sorted({rows[k]["atoms"][other] for k in ks})
                if len(cls) < 2:
                    continue
                used.append(gname)
                for k in ks:
                    # centroids from the OTHER held-out episodes of each class (leave this frame's episode out)
                    cents = []
                    for c in cls:
                        m = [q for q in ks if rows[q]["atoms"][other] == c and rows[q]["ep"] != rows[k]["ep"]]
                        cents.append(Xn[m].mean(0) if m else None)
                    if any(c is None for c in cents):
                        continue
                    pred = cls[int(np.argmax([Xn[k] @ c for c in cents]))]
                    hits.append(float(pred == rows[k]["atoms"][other]))
                    chance.append(1.0 / len(cls))
            probe[f"{role}_{lvl}"] = dict(groups=used, n=len(hits), acc=float(np.mean(hits)) if hits else None,
                                          chance=float(np.mean(chance)) if chance else None)
    os.makedirs(a.out, exist_ok=True)
    out = dict(name=a.name, ckpt=os.path.relpath(os.path.abspath(a.ckpt), ROOT), kind=M.kind, step=M.bk.step, setting=M.bk.setting,
               n_in=len(IN), n_dev=nd, records=rec, purity=pur, probe=probe, secs=time.time() - t0, limit=a.limit)
    fn = os.path.join(a.out, f"d1_{a.name}.json" if not a.limit else f"d1_{a.name}_smoke.json")
    with open(fn, "w", encoding="utf-8") as fh:
        json.dump(out, fh)
    print(f"[d1] wrote {fn} ({len(rec)} records, {len(pur)} purity rows, {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
