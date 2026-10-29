"""PLAN_BINDER B0 (docs/PLAN_BINDER.md section 3): pretrained vision-language backbones as referring-expression binders on S1's 240
dev units (results/bind/s1/labels.json, the frames S1 / S1b / S1c / S3b scored).  Offline; no rollout.

Run with the B0 venv in ISOLATED mode (downloads live in /coc/flash10/ksingara3/b0_ext; nothing is imported from there except the
installed packages, plus Molmo's own remote code, which transformers loads from the weights directory):
    /coc/flash10/ksingara3/b0_ext/venv/bin/python -I experiments/b0_vlm.py --arm ARM [--max N]
        -> results/bind/b0/<ARM>__<config>.npz  (keys, peak, mask [U, 256] bool, meta json)   (--max N: the first N TRAINING units,
           written to results/bind/b0/smoke/; no dev unit is seen before the pre-registration pb0)

ARMS and prompts (pre-registered in experiments/prereg/pb0.py).  {phrase} = the role phrase exactly as S1's parse gives it
(labels.json "phrase"); {lang} = the dev task's full instruction (results/bind/s1/tasks.json "lang").
  pg_mix     PaliGemma mix (not the policy's base), google/paligemma-3b-mix-224, native prompts  detect: "detect {phrase}"  segment: "segment {phrase}"
             NOT RUN: gated, no local weights (as is google/paligemma-3b-pt-224, the policy's base); code kept for completeness
  pi05_base  the PaliGemma backbone of openpi's pi05_base (the init of pi05_l90c; openpi_cache pi05_base_pytorch), loaded into the
             same HF PaliGemma class (config = openpi gemma_pytorch.py) with a processor rebuilt from big_vision's tokenizer
             (robot/data/paligemma_tok, sha256 = the Hub's) + SigLIP 224 preprocessing; the same prompts.  Smoke (6 TRAINING
             units): fluent English captions / yes-answers, no <loc> token, under every prompt -> recorded as found, no prompt change
  pi05_l90c  supplementary: the same, from our fine-tuned pi05_l90c@5000 (robot/checkpoints_l90c/pytorch/pi05_l90c_5000); detect_phrase only
  qwen       Qwen/Qwen2.5-VL-7B-Instruct, bbox JSON:  'Locate "{phrase}" in the image and output its bbox coordinates in JSON format.'
  molmo      allenai/Molmo-7B-D-0924, pointing:      'Point to "{phrase}".'
  context variant (every arm): PaliGemma  'detect {phrase} (from: {lang})' / 'segment {phrase} (from: {lang})';
             Qwen / Molmo prefix 'The robot is told: "{lang}". ' before the phrase prompt.
Images: the 224 x 224 agentview frame (client orientation) for every arm; Qwen gets a 2x bicubic upsample (448, i.e. 256 merged
visual tokens like PaliGemma-224), because its processor keeps native resolution; its coordinates are divided by 2.
Greedy decoding.  max_new_tokens: PaliGemma 64, Qwen 128, Molmo 200.

Output -> S1 scoring format (as B1a / S3b did):
  box (detect / Qwen):  the FIRST box; mask = tokens whose box coverage >= 0.5; peak = the token holding the box centre (B1a's rule)
  segment:              the first <loc>x4 <seg>x16 group, decoded by big_vision's VQ-VAE decoder (vae-oid.npz) to a 64 x 64 mask
                        resized into the box; mask = tokens with mask coverage >= 0.5; peak = the token of maximum coverage (ties: the
                        one nearest the mask centroid); an empty decoded mask falls back to the box (recorded in meta)
  points (Molmo):       S3b's rule: peak = the token holding the most points (ties: the first point's); mask = tokens with a point
  nothing parseable:    peak -1 (a miss); the raw answer is kept in meta.
"""
import argparse
import json
import os
import re
import time

import numpy as np

ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
EXT = "/coc/flash10/ksingara3/b0_ext"
W = os.path.join(EXT, "weights")
OD = os.path.join(ROOT, "results", "bind", "b0")
CKPT = {"pi05_base": "/coc/flash10/ksingara3/robot/openpi_cache/openpi-assets/checkpoints/pi05_base_pytorch/model.safetensors",
        "pi05_l90c": "/coc/flash10/ksingara3/robot/checkpoints_l90c/pytorch/pi05_l90c_5000/model.safetensors"}
PFX = "paligemma_with_expert.paligemma."
TOK = "/coc/flash10/ksingara3/robot/data/paligemma_tok/paligemma-3b-pt-224"
LOC = re.compile(r"<loc(\d{4})><loc(\d{4})><loc(\d{4})><loc(\d{4})>")
SEG = re.compile(r"<loc(\d{4})><loc(\d{4})><loc(\d{4})><loc(\d{4})>((?:<seg\d{3}>){16})")


def tok_of(px):                      # = s1b_train.tok_of / s3b_robopoint.tok_of (copied: repo modules are not importable under -I)
    x, y = np.clip(px[..., 0], 0, 223), np.clip(px[..., 1], 0, 223)
    return (y // 14).astype(int) * 16 + (x // 14).astype(int)


def box_out(x0, y0, x1, y1):
    """box in 224 pixels -> (peak, mask) by B1a's rule."""
    x0, y0, x1, y1 = max(0.0, x0), max(0.0, y0), min(223.0, x1), min(223.0, y1)
    if x1 < x0 or y1 < y0:
        return -1, np.zeros(256, bool)
    cov = np.zeros((224, 224), np.float32)
    cov[int(y0):int(np.ceil(y1)) + 1, int(x0):int(np.ceil(x1)) + 1] = 1.0
    cy, cx = (y0 + y1) / 2, (x0 + x1) / 2
    pk = int(min(15, cy // 14)) * 16 + int(min(15, cx // 14))
    return pk, cov.reshape(16, 14, 16, 14).mean((1, 3)).reshape(-1) >= 0.5


def mask_out(m224):
    """binary 224 x 224 mask -> (peak, mask): peak = max-coverage token, ties -> nearest the mask centroid."""
    cov = m224.astype(np.float32).reshape(16, 14, 16, 14).mean((1, 3)).reshape(-1)
    if cov.max() <= 0:
        return -1, np.zeros(256, bool)
    ys, xs = np.nonzero(m224)
    cy, cx = ys.mean() / 14, xs.mean() / 14
    best = np.flatnonzero(cov == cov.max())
    d = [(t // 16 + 0.5 - cy) ** 2 + (t % 16 + 0.5 - cx) ** 2 for t in best]
    return int(best[int(np.argmin(d))]), cov >= 0.5


def points_out(pts):
    """[(x, y) in 224 pixels] -> (peak, mask) by S3b's rule."""
    msk = np.zeros(256, bool)
    if not pts:
        return -1, msk
    tk = tok_of(np.array(pts, dtype=np.float64))
    msk[tk] = True
    cnt = np.bincount(tk, minlength=256)
    return int(next(t for t in tk if cnt[t] == cnt.max())), msk


# ---------------------------------------------------------------- PaliGemma segment decoder (big_vision VQ-VAE, vae-oid.npz)
class SegDecoder:
    def __init__(self, path, device):
        import torch
        self.t = torch
        z = np.load(path)
        self.dev = device
        self.p = {k: torch.tensor(z[k], dtype=torch.float32, device=device) for k in z.files}
        self.book = self.p["_vq_vae._embedding"]

        # the checkpoint is a PyTorch VQ-VAE (decoder.N keys); the HF space big-vision/paligemma-hf converts the same keys to an
        # equivalent flax decoder (Conv1x1 512->128, ReLU, 2 ResBlocks, 4 x [ConvTranspose 4x4 s2 + ReLU] 128->64->32->16, Conv1x1 -> 1)
        nn = torch.nn

        class Res(nn.Module):
            def __init__(s):
                super().__init__()
                s.net = nn.Sequential(nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(), nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
                                      nn.Conv2d(128, 128, 1))

            def forward(s, x):
                return x + s.net(x)
        self.dec = nn.Sequential(nn.Conv2d(512, 128, 1), nn.ReLU(), Res(), Res(),
                                 nn.ConvTranspose2d(128, 128, 4, 2, 1), nn.ReLU(), nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),
                                 nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(), nn.ConvTranspose2d(32, 16, 4, 2, 1), nn.ReLU(),
                                 nn.Conv2d(16, 1, 1))
        sd = {k[len("decoder."):]: v for k, v in self.p.items() if k.startswith("decoder.")}
        self.dec.load_state_dict(sd, strict=True)
        self.dec.to(device).eval()

    def mask64(self, codes):
        t = self.t
        with t.no_grad():
            q = self.book[t.tensor(codes, device=self.dev)].reshape(1, 4, 4, 512).permute(0, 3, 1, 2)
            m = self.dec(q)[0, 0].float().cpu().numpy()
        return np.clip(m * 0.5 + 0.5, 0, 1)                    # the space's [-1, 1] -> [0, 1]

    def paste(self, codes, box):
        """codes (16) + box (y0, x0, y1, x1) in 224 px -> binary 224 x 224 mask (the space's extract_objs, threshold 0.5)."""
        from PIL import Image
        y0, x0, y1, x1 = map(round, box)
        y0, x0, y1, x1 = max(0, y0), max(0, x0), min(224, y1), min(224, x1)
        out = np.zeros((224, 224), bool)
        if y1 > y0 and x1 > x0:
            m = Image.fromarray((self.mask64(codes) * 255).astype("uint8")).resize([x1 - x0, y1 - y0])
            out[y0:y1, x0:x1] = np.asarray(m) / 255.0 >= 0.5
        return out


def build(arm, device):
    import torch
    if arm in ("pg_mix", "pi05_base", "pi05_l90c"):
        from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor
        mp = os.path.join(W, "paligemma-3b-mix-224")
        if arm == "pg_mix":
            proc = PaliGemmaProcessor.from_pretrained(mp)
            model = PaliGemmaForConditionalGeneration.from_pretrained(mp, torch_dtype=torch.bfloat16)
            info = dict(source=mp)
        else:
            # google/paligemma-* is gated for this account, so the processor is rebuilt from big_vision's tokenizer (sha256-identical
            # to the Hub's; robot/data/paligemma_tok) and SigLIP's 224 preprocessing, and the config mirrors openpi's gemma_pytorch.py
            from safetensors.torch import load_file
            from transformers import GemmaTokenizer, SiglipImageProcessor
            from transformers.models.auto import CONFIG_MAPPING
            tk = GemmaTokenizer.from_pretrained(TOK)
            ip = SiglipImageProcessor(do_resize=True, size={"height": 224, "width": 224}, resample=3, do_rescale=True,
                                      rescale_factor=1 / 255, do_normalize=True, image_mean=[0.5] * 3, image_std=[0.5] * 3)
            ip.image_seq_length = 256
            proc = PaliGemmaProcessor(image_processor=ip, tokenizer=tk)
            sd = load_file(CKPT[arm])
            sd = {k[len(PFX):]: v for k, v in sd.items() if k.startswith(PFX)}
            # openpi saved the tied Gemma embedder once, as lm_head.weight (safetensors save_model drops the alias)
            sd.setdefault("model.language_model.embed_tokens.weight", sd.get("lm_head.weight"))
            nv = sd["model.language_model.embed_tokens.weight"].shape[0]
            cfg = CONFIG_MAPPING["paligemma"]()
            cfg._vocab_size = nv
            cfg.image_token_index = 257152
            t = cfg.text_config
            t.hidden_size, t.intermediate_size, t.num_attention_heads, t.head_dim = 2048, 16384, 8, 256
            t.num_hidden_layers, t.num_key_value_heads, t.hidden_activation, t.vocab_size = 18, 1, "gelu_pytorch_tanh", nv
            cfg.vision_config.intermediate_size = 4304
            cfg.vision_config.projection_dim = 2048
            cfg.vision_config.projector_hidden_act = "gelu_fast"
            model = PaliGemmaForConditionalGeneration(cfg)
            if "lm_head.weight" not in sd:
                sd["lm_head.weight"] = sd["model.language_model.embed_tokens.weight"]        # Gemma ties the head to the embedder
            res = model.load_state_dict(sd, strict=False)
            model = model.to(torch.bfloat16)
            info = dict(source=CKPT[arm], vocab=nv, missing=list(res.missing_keys), unexpected=list(res.unexpected_keys))
            print(f"[b0] {arm}: vocab {nv}; missing {res.missing_keys[:8]} ({len(res.missing_keys)}); unexpected "
                  f"{res.unexpected_keys[:8]} ({len(res.unexpected_keys)})", flush=True)
        return model.to(device).eval(), proc, info
    if arm == "qwen":
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        mp = os.path.join(W, "Qwen2.5-VL-7B-Instruct")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(mp, torch_dtype=torch.bfloat16).to(device).eval()
        return model, AutoProcessor.from_pretrained(mp), dict(source=mp)
    if arm == "molmo":
        from transformers import AutoModelForCausalLM, AutoProcessor
        mp = os.path.join(W, "Molmo-7B-D-0924")
        proc = AutoProcessor.from_pretrained(mp, trust_remote_code=True, torch_dtype="auto")
        model = AutoModelForCausalLM.from_pretrained(mp, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
        return model, proc, dict(source=mp)
    raise ValueError(arm)


def configs(arm):
    if arm == "pi05_l90c":                       # supplementary, detect only (budget)
        return ["detect_phrase"]
    if arm in ("pg_mix", "pi05_base"):
        return ["detect_phrase", "detect_context", "segment_phrase", "segment_context"]
    if arm == "qwen":
        return ["bbox_phrase", "bbox_context"]
    return ["point_phrase", "point_context"]


def prompt(arm, cfg, phrase, lang):
    kind, var = cfg.split("_")
    if arm in ("pg_mix", "pi05_base", "pi05_l90c"):
        return f"{kind} {phrase}" if var == "phrase" else f"{kind} {phrase} (from: {lang})"
    pre = "" if var == "phrase" else f'The robot is told: "{lang}". '
    if arm == "qwen":
        return pre + f'Locate "{phrase}" in the image and output its bbox coordinates in JSON format.'
    return pre + f'Point to "{phrase}".'


def run_one(arm, cfg, model, proc, im, q, segdec):
    import torch
    from PIL import Image
    meta = dict(query=q)
    if arm in ("pg_mix", "pi05_base", "pi05_l90c"):
        inp = proc(text="<image>" + q, images=im, return_tensors="pt").to(model.device)
        inp["pixel_values"] = inp["pixel_values"].to(torch.bfloat16)
        n = inp["input_ids"].shape[1]
        with torch.inference_mode():
            out = model.generate(**inp, max_new_tokens=64, do_sample=False)
        txt = proc.tokenizer.decode(out[0, n:], skip_special_tokens=False).replace("<eos>", "").replace("<pad>", "").strip()
        meta["answer"] = txt[:400]
        if cfg.startswith("segment"):
            m = SEG.search(txt)
            if m:
                y0, x0, y1, x1 = [int(v) / 1024 * 224 for v in m.groups()[:4]]
                codes = [int(c) for c in re.findall(r"<seg(\d{3})>", m.group(5))]
                m224 = segdec.paste(codes, (y0, x0, y1, x1)) if segdec is not None else None
                meta.update(parsed="seg", box=[x0, y0, x1, y1])
                if m224 is not None and m224.any():
                    pk, msk = mask_out(m224)
                    return pk, msk, meta
                meta["parsed"] = "seg_empty_mask_box_fallback"
                pk, msk = box_out(x0, y0, x1, y1)
                return pk, msk, meta
        m = LOC.search(txt)
        if not m:
            meta["parsed"] = "none"
            return -1, np.zeros(256, bool), meta
        y0, x0, y1, x1 = [int(v) / 1024 * 224 for v in m.groups()]
        meta.update(parsed="box", box=[x0, y0, x1, y1])
        pk, msk = box_out(x0, y0, x1, y1)
        return pk, msk, meta
    if arm == "qwen":
        im2 = im.resize((448, 448), Image.BICUBIC)
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = proc(text=[text], images=[im2], return_tensors="pt").to(model.device)
        n = inp["input_ids"].shape[1]
        with torch.inference_mode():
            out = model.generate(**inp, max_new_tokens=128, do_sample=False)
        txt = proc.batch_decode(out[:, n:], skip_special_tokens=True)[0].strip()
        meta["answer"] = txt[:400]
        m = re.search(r'"bbox_2d"\s*:\s*\[\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\]', txt) or \
            re.search(r"\[\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\]", txt)
        if not m:
            meta["parsed"] = "none"
            return -1, np.zeros(256, bool), meta
        x0, y0, x1, y1 = [float(v) / 2 for v in m.groups()]
        meta.update(parsed="box", box=[x0, y0, x1, y1])
        pk, msk = box_out(x0, y0, x1, y1)
        return pk, msk, meta
    from transformers import GenerationConfig
    inp = proc.process(images=[im], text=q)
    inp = {k: v.to(model.device).unsqueeze(0) for k, v in inp.items()}
    inp["images"] = inp["images"].to(torch.bfloat16)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = model.generate_from_batch(inp, GenerationConfig(max_new_tokens=200, stop_strings="<|endoftext|>"),
                                        tokenizer=proc.tokenizer)
    txt = proc.tokenizer.decode(out[0, inp["input_ids"].size(1):], skip_special_tokens=True).strip()
    meta["answer"] = txt[:400]
    pts = [(float(x), float(y)) for x, y in re.findall(r'x\d*="([\d.]+)"\s+y\d*="([\d.]+)"', txt)]
    pts = [(x * 224 / 100, y * 224 / 100) for x, y in pts if 0 <= x <= 100 and 0 <= y <= 100]
    meta.update(parsed="points" if pts else "none", n_points=len(pts))
    pk, msk = points_out(pts)
    return pk, msk, meta


def main():
    import torch
    from PIL import Image
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["pg_mix", "pi05_base", "pi05_l90c", "qwen", "molmo"])
    ap.add_argument("--max", type=int, default=0)     # smoke: the first N TRAINING units (no dev unit is seen before pb0)
    ap.add_argument("--configs", default="")          # colon-separated subset (default: all of the arm's)
    a = ap.parse_args()
    assert torch.cuda.is_available(), "refusing to run on CPU"
    dev = torch.device("cuda")
    lab = json.load(open(os.path.join(ROOT, "results", "bind", "s1", "labels.json"), encoding="utf-8"))
    tasks = json.load(open(os.path.join(ROOT, "results", "bind", "s1", "tasks.json"), encoding="utf-8"))
    units = lab["train"][:a.max] if a.max else lab["dev"]
    od = os.path.join(OD, "smoke") if a.max else OD
    os.makedirs(od, exist_ok=True)
    t0 = time.time()
    model, proc, info = build(a.arm, dev)
    cf = a.configs.split(":") if a.configs else configs(a.arm)
    segdec = SegDecoder(os.path.join(EXT, "vae", "vae-oid.npz"), dev) if any(c.startswith("segment") for c in cf) else None
    print(f"[b0] {a.arm} loaded in {time.time() - t0:.0f}s on {torch.cuda.get_device_name(0)}; configs {cf}; {len(units)} units", flush=True)
    for c in cf:
        keys, peaks, masks, metas = [], [], [], []
        t1 = time.time()
        for i, e in enumerate(units):
            img = np.load(os.path.join(ROOT, e["frame"]), allow_pickle=True)["img"]
            im = Image.fromarray(img).convert("RGB")
            q = prompt(a.arm, c, e["phrase"], tasks[e["tid"]]["lang"])
            pk, msk, meta = run_one(a.arm, c, model, proc, im, q, segdec)
            keys.append(f"{e['role']}|{e['frame']}")
            peaks.append(pk)
            masks.append(msk)
            metas.append(json.dumps(meta))
            if i < 12 or i % 40 == 0:
                print(f"[b0] {a.arm}/{c} {i + 1}/{len(units)} {e['role']} {e['phrase']!r}: peak {pk} | {meta.get('parsed')} | "
                      f"{meta.get('answer', '')[:140]!r} ({time.time() - t1:.0f}s)", flush=True)
        np.savez(os.path.join(od, f"{a.arm}__{c}.npz"), keys=np.array(keys), peak=np.array(peaks), mask=np.stack(masks),
                 meta=np.array(metas), info=np.array(json.dumps(info)))
        npars = sum(json.loads(m).get("parsed") != "none" for m in metas)
        print(f"[b0] {a.arm}/{c} done: {len(keys)} units, parsed {npars}, {time.time() - t1:.0f}s", flush=True)
    print(f"[b0] {a.arm} all done {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
