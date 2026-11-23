# Literature TODOs in paper/DRAFT.md (resolved 2026-11-23)

## 1. Grant et al., arXiv 2603.19233
Sources: https://arxiv.org/abs/2603.19233 , https://arxiv.org/html/2603.19233v1 , https://arxiv.org/html/2603.19233v6
- Versions: v1 19 Mar 2026; v2 12 Oct 2026; up to v6 (revised 2 Nov 2026). Comments: "NeurIPS 2026 Poster, ICLR Multimodal Intelligence Workshop Oral". The NeurIPS poster page was not fetched.
- v1 and v6 both report 99.6% (n = 1,968 pi0.5 pairs) of episodes closer to the source task, with destination success 2.6% (v6 Table 3, v1 Table 4). Metric is tie-band-conditional in v6. Per-model override: X-VLA 99.8, OFT 96.4 (v6), GR00T 57.0, SmolVLA 52.1 (v6; 15.8 in v1 expert pathway), ACT 0.0.
- "Absolute workspace coordinates" appears in both (Sec. 4.3 "Motor Programs Bind to Absolute Spatial Coordinates"); physical UR10e: 50/50 moved-source trials go to recorded coordinates.
- The "0% cross-task transfer" hint: the only 0% figures found are destination-task success for X-VLA, SmolVLA, GR00T (and Table 2 zero-layer success), not pi0.5 transfer. No version contradicts the draft's 99.6%. Only the first 100k characters of v1 and v6 were read, and v2 was not read separately, so a changed number in the unread tail is not excluded.
- Change: cite 2603.19233v6; state 99.6% of 1,968 as tie-band conditional, 2.6% destination success, weaker on GR00T/SmolVLA; removed both TODOs (Related work para and moved-plate paragraph).

## 2. w2VLA (arXiv 2606.21496)
Source: https://arxiv.org/abs/2606.21496 , https://arxiv.org/html/2606.21496v2 ("Decoupling the Declarative from the Procedural in Vision-Language-Action Models", Tsagkas et al.; v2 9 Oct 2026). Read the first ~100k characters of the HTML.
- Atomic command = skill string + object string, "independently encoded by the frozen VLM text encoder". Object embedding dotted with VLM patch embeddings gives per-camera heatmaps -> FiLM conditioning block; skill embedding -> MLP -> second FiLM block, acting on robot-state tokens. Multi-step tasks decomposed zero-shot by Gemini 3.1 Pro; experiment instructions were already atomic.
- Draft was accurate in spirit; rewrote to describe the mechanism and removed the abstract-only caveat.

## 3. Wulff and Cangelosi (arXiv 2610.10178)
Source: https://arxiv.org/abs/2610.10178 , https://arxiv.org/html/2610.10178v1
- Five corruptions of LIBERO instructions; activation and attribution patching on action modules. Abstract: both models respond strongly to empty instructions and especially directional language; localisation differs (GR00T N1.7 early periodic cross-attention layers; pi0.5 earliest plus some later layers). The draft's "in pi0.5 and GR00T" matches the abstract; the strongest directional effect is in GR00T.
- Measure: change in flow-matching loss on 10,000 LIBERO-10 steps against the original demonstration; no success rate, no predicted-chunk change.
- Change: one sentence reconciling: different perturbation setting and measure (loss under patching vs. chunk change under a left-right swap); no claim that either result is wrong.
