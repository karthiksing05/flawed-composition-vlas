# TAIL baseline on the R1 stream, registered 2026-11-23 before any TAIL job was launched

Arm: one rank-8 LoRA adapter per stage (HB, OW, TC) on all attention (qkv, proj) and MLP (fc1, fc2) linear layers of every
block, base frozen, trained on that stage's 16 demos only (same seeds, demo draws, adapt_steps 3000, lora_lr 2e-3 as r1s.py).
Eval: own-stage and single new-value cells use that stage's adapter (task id given); panel cells use no adapter; pair cells
have no TAIL rule: TAIL-select applies one adapter (mean over choosing either stage's adapter, and oracle best-of-two per
cell), TAIL-merge adds both adapters' weight deltas. Same eval cells, episodes (5 singles/panel, 10 pairs/triple), eval seed.

- TL-P1: TAIL-select pairs (mean over choice) <= .35 mean over 3 seeds, below routed smax (.489) in every seed.
- TL-P2: TAIL-merge pairs below routed smax by >= .08 in the 3-seed mean.
- TL-P3: TAIL panel >= .80 (base frozen, adapters routed by task id).
- TL-P4: TAIL single new-value cells within .10 of routed parts (both are routed adapters; TAIL has more capacity).

Added 2026-11-23 (before any parts-wmerge job was launched; TAIL training job 4847910 was already running):
Arm PARTS-WMERGE (PSEC-style): our saved routed rank-1 parts (parts/{HB,OW,TC}.pt per seed), at pair cells both parts' LoRA
deltas are added in weight space (W + dW_i + dW_j, one forward per Euler step) instead of smax on velocities. Pairs only.
- WM-P1: parts-wmerge pairs below routed smax (.489) by >= .05 in the 3-seed mean.

## Results (2026-11-23; jobs: train 4847910, eval 4847977-4848000, report 4848092; report_tail.json)
Adapters trained without divergence (rank 8, lr 2e-3, 3000 steps, final loss ~.26-.28; 131072 params each). Pairs 30 eps/type.
| | seed0 | seed1 | seed2 | mean |
|---|---|---|---|---|
| TAIL-select pairs (mean over choice) | .133 | .200 | .250 | .194 |
| TAIL-select pairs (oracle best-of-two) | .267 | .400 | .489 | .385 |
| TAIL-merge pairs (adapter weight sum) | .011 | .011 | .000 | .007 |
| parts-wmerge pairs (PSEC-style) | .033 | .111 | .122 | .089 |
| routed smax pairs | .567 | .478 | .422 | .489 |
| TAIL singles (own adapter) | .430 | .519 | .578 | .509 |
| routed smax singles | .556 | .578 | .711 | .615 |
| TAIL panel (base) | .85 | .85 | .85 | .85 |

- TL-P1: PASS (.194 <= .35; below smax in every seed).
- TL-P2: PASS (merge .007 vs .489, gap .482 >= .08).
- TL-P3: PASS (.85 >= .80; panel is the frozen base, identical to routed).
- TL-P4: FAIL (singles .509 vs .615, gap .106 > .10; TAIL is lower, not higher, despite more capacity; seed gaps .126/.059/.133).
- WM-P1: PASS (parts-wmerge .089 vs .489, gap .400 >= .05).
