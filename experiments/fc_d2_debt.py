"""FLAWED_COMPOSITION D2 (section 4.7): per-task compositional debt of an archive VLA on its OOD tasks, by interventions on the
prefix the action expert reads.  No training.  One GPU, the model's own venv (pi0.5: openpi; X-VLA: SmolVLA venv; GR00T: groot).

Interface (section 4.1, restricted; see 4.7): the role messages are the expert-read prefix features at the instruction's role
phrases.  pi0.5: the value states of all 18 PaliGemma layers (the KV cache the expert attends); X-VLA: the Florence-2 encoder
features the denoiser reads (kv["vlm"]); GR00T: the Eagle backbone features the DiT cross-attends (kv["vl"]).  The role-phrase
positions are the token span where the instruction's tokenisation differs from that of the phrase-swapped instruction
(longest common prefix and suffix; tokenizer-agnostic).  Bypass = every input outside the instruction (both views, state).

Per task (frames: 10 per task, frame 0 of 10 demos for L90, the 10 rendered start states for LI):
  purity    instruction with step 0's goal phrase replaced by another task's goal phrase of the same scene / suite, and with its
            object phrase replaced (transport tasks only); per role, sum ||dm|| under the other-role swap / sum under the own-role swap;
            the task's value = mean over the two roles (a task with only one valid role uses that one)
  bypass    partner = another task of the same scene (L90, within the block) / suite (LI), fixed hash, same frame index;
            role swap = the partner's instruction on this frame; bypass swap = the partner's views and state with this instruction;
            share = sum d_byp / (sum d_byp + sum d_role); d = MSE between samples over the first min(10, H) steps of the real dims,
            in the model's own normalised rows, mean over 2 seeds shared by all variants of a task
  combination (L90, pi0.5 only): action error on the event frames of the task's demos (scored only), unnormalised actions divided by
            the pooled per-dim std of the scored demo chunks, first 10 steps x 7 dims
-> results/flawed/d2_<model>.json

    python experiments/fc_d2_debt.py --model pi05|xvla|groot [--limit N --out DIR]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "experiments")]
CACHE = "/coc/flash10/ksingara3/robot/data/bindvla_cache"
OUT = os.path.join(ROOT, "results", "flawed")
SEEDS = (0, 1)
HE = 10


def hseed(*k):
    return int(hashlib.sha256("|".join(map(str, k)).encode()).hexdigest()[:8], 16)


def step0_phrases(text):
    import re
    import s1_labels as SL
    from lifelong_ood.policy import primitives as P
    st = P.steps_of(text.lower().strip().rstrip("."))
    if not st:
        return "", ""
    o, g = SL.split_step(" ".join(st[0].strip().lower().split()))
    o = re.sub(r"^(open|close|turn on|turn off|pick up|push|put|place|stack)\s+", "", o.strip())
    return o.strip(), g.strip()


def replace_once(text, a, b):
    if not a or text.count(a) != 1 or a == b:
        return None
    return text.replace(a, b, 1)


def load(model):
    if model == "pi05":
        from lifelong_ood.policy.backbone import PolicyBackbone
        return PolicyBackbone("pi05_libero", device="cuda")
    if model == "xvla":
        from lifelong_ood.policy.backbone_xvla import XVLABackbone
        return XVLABackbone(None, device="cuda")
    from lifelong_ood.policy.backbone_groot import GrootBackbone
    return GrootBackbone(None, device="cuda")


def row_ids(model, bk, o, b):
    """-> (instruction-region token ids of row b, base: sequence position of ids[0])."""
    if model == "pi05":
        m = o.tokenized_prompt_mask[b].bool()
        return o.tokenized_prompt[b][m].tolist(), 256 * len(o.images)
    if model == "xvla":
        off, nv = bk.text_span(o)
        return o.input_ids[b][: int(nv[b])].tolist(), int(off)
    m = o.eagle["eagle_attention_mask"][b].bool()
    first = int(torch.nonzero(m)[0])
    return o.eagle["eagle_input_ids"][b][m].tolist(), first


def feats(model, kv):
    """-> list of [B, T, D] expert-read feature sequences."""
    if model == "pi05":
        if hasattr(kv, "value_cache"):
            vs = list(kv.value_cache)
        elif hasattr(kv, "layers"):
            vs = [l.values for l in kv.layers]
        else:
            vs = [x[1] for x in kv]
        return [v.float().permute(0, 2, 1, 3).reshape(v.shape[0], v.shape[2], -1) for v in vs]
    return [(kv["vlm"] if model == "xvla" else kv["vl"]).float()]


def diff_span(a, b):
    """common prefix p and suffix s of token lists a, b -> (p, s)."""
    p = 0
    while p < min(len(a), len(b)) and a[p] == b[p]:
        p += 1
    s = 0
    while s < min(len(a), len(b)) - p and a[-1 - s] == b[-1 - s]:
        s += 1
    return p, s


def message(F_, base, span):
    return torch.cat([f[:, base + span[0]: base + span[1]].mean(1) for f in F_], -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=("pi05", "xvla", "groot"))
    ap.add_argument("--limit", type=int, default=0, help="smoke: first N tasks per block")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    t0 = time.time()
    from lifelong_ood.policy import factors as F
    bk = load(a.model)
    H = min(HE, bk.H)
    fr = json.load(open(os.path.join(OUT, "d2_frames.json"), encoding="utf-8"))
    l90 = {t.tid: t for t in F.load(("libero_90",))}
    ood = {f"{t.suite.split('_')[1]}:{t.index}": t for t in F.load(F.OOD_SUITES)}
    blocks = {"LI": {k: dict(text=ood[k].language, scene=k.split(":")[0]) for k in fr["li"]}}
    if a.model == "pi05":
        blocks["L90"] = {t: dict(text=l90[t].language, scene=l90[t].scene) for t in fr["l90"]}
        z0 = np.load(os.path.join(CACHE, "d2_l90.npz"))
        Z = {k: z0[k] for k in z0.files}
    li = {}
    for s, pre in (("libero_goal_ood", "goal"), ("libero_spatial_ood", "spatial")):
        for i in range(10):
            z = np.load(os.path.join(ROOT, "results", "lang", "frames", f"{s}_{i}.npz"))
            li[f"{pre}:{i}"] = dict(img=z["img"], wimg=z["wimg"], state=z["state8"])

    def frames0(blk, key):
        if blk == "LI":
            d = li[key]
            return [dict(img=d["img"][j], wimg=d["wimg"][j], state=d["state"][j]) for j in range(len(d["img"]))]
        idx = np.flatnonzero((Z["tid"] == key) & (Z["kind"] == "f0"))
        return [dict(img=Z["agent"][j], wimg=Z["wrist"][j], state=Z["state"][j]) for j in idx]

    def obs(fs, text):
        return bk.observations([{"observation/image": f["img"], "observation/wrist_image": f["wimg"], "observation/state": f["state"],
                                 "prompt": text} for f in fs])

    def rows_of(pre, n, seed):
        return np.asarray(bk.sample(n, pre, seed=seed), np.float32).reshape(n, bk.H, -1)

    out = {"model": a.model, "blocks": {}}
    for blk, T in blocks.items():
        keys = sorted(T)[: a.limit or None]
        res = {}
        for key in keys:
            text, scene = T[key]["text"], T[key]["scene"]
            o_, g_ = step0_phrases(text)
            cands = sorted([u for u in T if u != key and T[u]["scene"] == scene and T[u]["text"] != text],
                           key=lambda u: hseed("d2-partner", key, u))
            gs = os_ = None
            for u in cands:
                ou, gu = step0_phrases(T[u]["text"])
                if gs is None and gu and g_ and gu != g_:
                    gs = (u, replace_once(text, g_, gu))
                if os_ is None and ou and o_ and g_ and gu and ou != o_:
                    os_ = (u, replace_once(text, o_, ou))
            part = cands[0] if cands else None
            fs = frames0(blk, key)
            n = len(fs)
            r = dict(text=text, partner=part, goalswap=gs, objswap=os_, n=n)
            with torch.no_grad():
                o_int = obs(fs, text)
                p_int = bk.prefix(o_int)
                seeds = [hseed("d2-seed", key, s) for s in SEEDS]
                y_int = [rows_of(p_int, n, s) for s in seeds]
                # purity
                if gs and gs[1] and os_ and os_[1]:
                    ids_i, base_i = row_ids(a.model, bk, o_int, 0)
                    Fi = feats(a.model, p_int.kv)
                    msg = {}
                    for vn, tx in (("goalswap", gs[1]), ("objswap", os_[1])):
                        o_v = obs(fs, tx)
                        p_v = bk.prefix(o_v)
                        ids_v, base_v = row_ids(a.model, bk, o_v, 0)
                        p, s = diff_span(ids_i, ids_v)
                        sp_i, sp_v = (p, len(ids_i) - s), (p, len(ids_v) - s)
                        if sp_i[1] <= sp_i[0] or sp_v[1] <= sp_v[0]:
                            msg = None
                            break
                        msg[vn] = dict(Fv=feats(a.model, p_v.kv), base_v=base_v, sp_i=sp_i, sp_v=sp_v, shift=len(ids_v) - len(ids_i))
                    if msg:
                        G, O = msg["goalswap"], msg["objswap"]
                        obj_sp, goal_sp = O["sp_i"], G["sp_i"]          # the object / goal phrase spans in the intact instruction
                        if obj_sp[1] <= goal_sp[0]:                      # object before goal (every template here)
                            m_obj_i, m_goal_i = message(Fi, base_i, obj_sp), message(Fi, base_i, goal_sp)
                            m_obj_g = message(G["Fv"], G["base_v"], obj_sp)                         # object message, goal swapped
                            m_goal_g = message(G["Fv"], G["base_v"], G["sp_v"])                     # goal message, goal swapped (own)
                            m_obj_o = message(O["Fv"], O["base_v"], O["sp_v"])                      # object message, object swapped (own)
                            m_goal_o = message(O["Fv"], O["base_v"], (goal_sp[0] + O["shift"], goal_sp[1] + O["shift"]))
                            nr = lambda x, y: float((x - y).norm(dim=-1).sum())                     # noqa: E731
                            r["purity"] = dict(obj_other=nr(m_obj_g, m_obj_i), obj_own=nr(m_obj_o, m_obj_i),
                                               goal_other=nr(m_goal_o, m_goal_i), goal_own=nr(m_goal_g, m_goal_i),
                                               obj_tokens=obj_sp[1] - obj_sp[0], goal_tokens=goal_sp[1] - goal_sp[0])
                # bypass
                if part is not None:
                    fp = frames0(blk, part)[:n]
                    if len(fp) == n:
                        y_role = [rows_of(bk.prefix(obs(fs, T[part]["text"])), n, s) for s in seeds]
                        y_byp = [rows_of(bk.prefix(obs(fp, text)), n, s) for s in seeds]
                        d = lambda Y: float(np.mean([((y[:, :H] - x[:, :H]) ** 2).mean() for y, x in zip(Y, y_int)]))  # noqa: E731
                        r["bypass"] = dict(d_role=d(y_role), d_byp=d(y_byp))
                # combination (pi0.5 on L90)
                if blk == "L90":
                    idx = np.flatnonzero((Z["tid"] == key) & (Z["kind"] == "ev"))
                    errs = []
                    for b in range(0, len(idx), 16):
                        J = idx[b:b + 16]
                        fe = [dict(img=Z["agent"][j], wimg=Z["wrist"][j], state=Z["state"][j]) for j in J]
                        pe = bk.prefix(obs(fe, text))
                        for s in seeds:
                            raw = bk.unnormalize(rows_of(pe, len(J), s).reshape(len(J), -1))[:, :HE, :7]
                            errs.append((raw, Z["actions"][J][:, :HE, :7], Z["pad"][J][:, :HE]))
                    r["comb_raw"] = errs
            res[key] = r
            print(f"[d2] {a.model} {blk} {key} {time.time() - t0:.0f}s", flush=True)
        if blk == "L90":                                 # pooled per-dim std of the scored demo chunks, then the per-task error
            allt = np.concatenate([np.concatenate([e[1][~e[2]] for e in r["comb_raw"]]) for r in res.values() if r.get("comb_raw")])
            sd = allt.std(0) + 1e-6
            for r in res.values():
                if r.get("comb_raw"):
                    ev = [((((p - t) / sd) ** 2).mean(-1) * ~m).sum() / max((~m).sum(), 1) for p, t, m in r["comb_raw"]]
                    r["comb_err"] = float(np.mean(ev))
                    r["comb_frames"] = int(sum(len(e[0]) for e in r["comb_raw"]) // len(SEEDS))
                r.pop("comb_raw", None)
            out["l90_action_sd"] = sd.tolist()
        out["blocks"][blk] = res
    out["secs"] = time.time() - t0
    out["describe"] = bk.describe()
    os.makedirs(a.out, exist_ok=True)
    fn = os.path.join(a.out, f"d2_{a.model}{'_smoke' if a.limit else ''}.json")
    with open(fn, "w", encoding="utf-8") as fh:
        json.dump(out, fh, default=str)
    print(f"[d2] wrote {fn} {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
