# PLAN_SOLVE S4: a noise-robust binding adapter (pre-registered 2026-10-26, before the adapter is trained)

Goal: the binding adapter should degrade gracefully when the binder is wrong, not damage the policy (S2: -0.18 on training cells).
EXPLORATORY. There is no confirmatory read and no sealed set. Ledger: m3_prereg tag `ps4`; read row `s4_robust` tag read.

## 1. Training (experiments/s4_train.py)

S0's recipe is unchanged:
- the same 56 training cells and frames;
- the policy's flow loss, with the policy frozen;
- AdamW 3e-4 / wd 0.01, batch 16, 3000 steps, seed 0;
- P_CF 0.5, cf examples keep oracle masks;
- the same held-out frames.

**One change: OWN-prompt examples draw each role's mask independently from a mixture.**

| share | mask source |
|---|---|
| 0.35 | oracle (S0's masks) |
| 0.35 | synthetic, uniform over 6 kinds (below) |
| 0.30 | the deployable binder's actual mask on that training frame (S1b or S1c, 50 / 50) |

The six synthetic kinds:
- one_in: one token of the oracle mask;
- one_out: one token outside it;
- shift: the mask moved by up to 3 tokens;
- dilate: the mask dilated by 1-2 steps;
- sibling: S1's competitor (a same-type instance or same-body region), or else another visible entry;
- empty.

**Binder masks** come from experiments/s4_bmask.py, which ran before this pre-registration. It is training data only, not an
evaluation.
- Scope: 54 of the 56 cells. For 2 cells the parse does not align with the goal atoms; their binder share uses the synthetic
  corruptions.
- Phrases: experiments/s4_phrases.py, the same split as S2.

**Target**: the target stays the demo chunk. A mask that disagrees with the demo raises the loss only if the adapter follows it,
so the adapter should learn to use a mask only when it agrees with the scene evidence.

**Held-out diagnostic** (reported, no bar): on the held-out frames with the own prompt, the flow loss of no edit vs the S0
adapter vs the S4 adapter, for each mask source.

## 2. Closed loop (seeds 4-5; never used for any selection)

Arms:

| arm | runs |
|---|---|
| robust + S1c masks per replan | s2_serve `--binder s1c`, S2_ADAPTER = results/bind/s4/adapter.pt |
| robust + oracle masks | s2_sandbox `--arm oracle`, the live simulator masks of S0's rule |
| zs | reused: S0's runs at seeds 4-5 |
| S0 adapter + oracle | reused: S0's runs at seeds 4-5, for reference |

Units, fitted to the budget:
- **dev**: the 72 units with fresh state 0-2 of the 24 dev tasks. This is a deviation from all 120 units, disclosed.
- **train**: S0's 50 damage units, robust + S1c arm only.

## 3. Bars (point estimates; PASS iff all three; experiments/s4_read.py)

- **B1**: training-cell damage, success(robust + S1c) - success(zs), >= -0.05.
- **B2**: dev binding error(robust + S1c) < zs. A CI that excludes 0 is not required; the CI is reported.
- **B3**: dev binding error(robust + oracle) <= 0.10, i.e. robustness kept the oracle gain.

Also reported: success and motion failure for every arm, closure vs the robust-oracle arm, and the first-call S1c mask diagnostic.

## 4. Cost (planned, of about 4.7 A40-h)

| step | A40-h |
|---|---|
| binder masks | about 0.2 |
| training | about 1.0 |
| closed loop: train arm | about 0.5 |
| closed loop: dev S1c arm | about 0.9 |
| closed loop: dev oracle arm | about 0.6 |
| total | about 3.2 |

## 5. Result (read 2026-10-27; ledger s4_robust tag read; results/bind/s4.json): FAIL (all three bars)

Seeds 4-5. Dev: 72 units (24 tasks, states 0-2). Train: 50 units. Arms are paired on states, seeds and noise.

| dev | success | binding error | motion failure |
|---|---|---|---|
| zs | 0.285 [0.125, 0.458] | 0.229 [0.083, 0.396] | 0.486 |
| robust adapter + S1c masks | 0.354 [0.188, 0.535] | 0.236 [0.111, 0.382] | 0.410 |
| robust adapter + oracle masks | 0.347 [0.181, 0.528] | 0.118 [0.028, 0.229] | 0.535 |
| S0 adapter + oracle masks (reference) | 0.354 [0.181, 0.535] | 0.042 [0.000, 0.118] | 0.604 |

**Bars:**
- **B1 fails.** Training-cell damage with S1c masks is **-0.07 [-0.14, -0.02]** (success 0.98 -> 0.91, all motion).
- **B2 fails.** Dev binding error with S1c masks is 0.236 vs zs 0.229: **+0.007 [-0.125, +0.118]**.
- **B3 fails.** Binding error with oracle masks is **0.118**, above 0.10. Robustness halved the oracle gain: -0.111 [-0.257, +0.035] vs zs,
  against S0's -0.188 [-0.326, -0.063] at the same seeds.

**Other results:**
- With S1c masks, success is +0.069 [-0.021, +0.181] over zs. The gain comes from fewer motion failures (-0.076 [-0.174, +0.021]),
  not from binding.
- Closure vs the robust-oracle arm: binding -0.06.
- First-call S1c masks: object mostly inside the target in 0.74 of units; goal 0.54; one token 0.24 / 0.33.

**Held-out frames** (flow loss, own prompt; training cells' held-out frames):

| masks | no edit | S0 adapter | robust adapter |
|---|---|---|---|
| one wrong token | 0.0201 | 0.0255 | 0.0199 |
| sibling | 0.0201 | 0.0277 | 0.0204 |
| S1c | 0.0199 | 0.0175 | 0.0166 |
| oracle | 0.0201 | 0.0151 | 0.0163 |

So offline the robust adapter learned to ignore wrong masks, at some cost to the oracle gain. Closed loop, that did not carry over
to undamaged training cells or to a binding gain from S1c masks.

**Reading:**
1. **Noise training trades the oracle gain for robustness.** It did not separate useful from harmful masks well enough: binding
   with S1c masks is unchanged, and binding with oracle masks is weaker.
2. **The damage shrank but did not vanish.** It is -0.07 here vs -0.18 for S0 + S1b in S2. Those used different seeds and binders,
   so this is not a paired comparison.
3. **The success gain with S1c masks is a motion effect.** The adapter's edit of the deep image K/V changes motion as well as
   binding.

**Disclosures:**
- Dev used 72 of 120 units, states 0-2, to fit the budget (pre-registered).
- Two cells had no aligned parse, so their binder share used synthetic corruptions.
- S1c's head was trained on these training cells, so its masks on training frames are in-sample.
- The binder-mask pass cost 0.72 A40-h, against 0.2 planned.
- **Port collision**: running 4 instances per job made the last port overlap the next job ID's first port. One dev shard's server
  failed, and that shard was rerun alone (job 4257505).
- **Preemption**: overcap preempted and requeued four jobs. Units already written were skipped, but the requeued job and the rerun
  may both have run dev shard 3's remaining units; saves are atomic.
- The trainer smoke (3 steps) ran before the prereg row. It wrote no adapter or checkpoint.

**Cost:**

| step | A40-h |
|---|---|
| binder masks | 0.72 |
| smoke | 0.05 |
| training | 1.30 |
| closed loop, including preempted segments | 2.67 |
| **S4 total** | **about 4.74** |

This uses the remaining ~4.7, so the approved 20 A40-h are spent (about 0.05 over, within the rounding of the running estimate).
