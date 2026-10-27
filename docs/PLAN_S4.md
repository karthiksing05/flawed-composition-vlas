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
