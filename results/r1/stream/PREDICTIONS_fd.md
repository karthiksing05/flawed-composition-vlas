# Factor-dropout (Factored Diffusion Policies style) baseline on the R1 stream, registered 2026-11-24 before any FD job was launched

Arm: one monolith trained from scratch (72k steps, lr 3e-4, bs 256: base recipe + 12k, as joint_scratch) on all data at the end of
the stream (base demos + 16 demos per new value; r1s.py joint weighting; same seeds/demo draws). Factor identity: the free policy
sees no indicator (stripped from the state), so FD gets the 16-dim factor multi-hot (robot/object/obstacle/objective blocks of 4)
through task_in plus a keep mask; dropping a factor zeroes its 4 entries and adds a learned null vector; p_drop = 0.25 per factor,
independent. Test on pair/triple cells: fd-sum v = v_null + sum_f (v_f - v_null); fd-full = all factors on. Singles and panel: fd-full.
Same eval cells, episode counts, eval seed as r1s.py.

- FD-P1: fd-sum pairs below routed smax (.489) in the 3-seed mean.
- FD-P2: fd-full pairs >= joint (.337) in the 3-seed mean (dropout regularises composition).
- FD-P3: fd-sum pairs >= PoCo-style sum of our parts (.259) in the 3-seed mean.

## Results (2026-11-24; jobs: train 4851953/4851954/4851955, eval 4852717-4852740, report 4852863; report_fd.json)
Training: 72k steps, final loss ~.231-.236 for all seeds. Pairs 30 eps/type (90 per seed), same eval seed as r1s.py.
| pairs | seed0 | seed1 | seed2 | mean |
|---|---|---|---|---|
| fd-sum | .333 | .256 | .211 | .267 |
| fd-full | .333 | .244 | .311 | .296 |
| routed smax | .567 | .478 | .422 | .489 |
| joint | .378 | .344 | .289 | .337 |
| PoCo sum (routed_sum) | .244 | .233 | .300 | .259 |
Other (fd-full): singles .363/.422/.400 (mean .395; routed .615); panel .40/.30/.50 (mean .40; routed .867, joint .683);
triple fd-sum .30/.00/.20 (.167), fd-full .10/.00/.20 (.100).
- FD-P1: PASS (.267 < .489, below in every seed).
- FD-P2: FAIL (.296 < .337; below joint in seeds 0 and 1, above in seed 2).
- FD-P3: PASS, marginal (.267 >= .259; seed-level mixed: .333/.256/.211 vs .244/.233/.300).
Implementation: the free policy gets no factor identity (indicator stripped from state), so FD uses the 16-dim multi-hot via task_in
(cond oracle) + 4-dim keep mask; dropped factor f: its 4 multi-hot entries zeroed and learned null[f] (256-d) added; p=.25 independent.
