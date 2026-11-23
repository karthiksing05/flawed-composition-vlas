# Remaining TODOs for the ICML build (not part of the submission)

Citations
- RESOLVED 2026-11-23: POT-VLA (potvla2607) and w2VLA (w2vla2606) titles and author lists match the arXiv abs pages; the papers call themselves POT-VLA and w^2VLA (text now writes w$^{2}$VLA). All other refs_extra entries were re-checked the same way and match; every references.bib entry has authors.
- STAGE (stage2609): on the abs page STAGE is the paper title, the interface is VISA and the benchmark is SAT-Bench; the text cites it only as "STAGE (sensitivity probes)", which is acceptable; revisit if a reviewer wants the benchmark named.
- "Nouns, Not Modifiers" (ICML 2026) and the MEM / pi0.7 memory mentions in the draft have no arXiv id; the first was dropped from the text (it sat in the Appendix L.6 list of shortened numbers), the others stay as plain-text mentions in Appendix I without a citation.
- Entries in refs_extra.bib (Wulff and Cangelosi, Jiang et al. audit, RoboMME, LIBERO-Mem, de Haan et al.) were taken from the arXiv abstract pages on 2026-11-23. de Haan et al. is cited as the NeurIPS 2019 paper (arXiv 1905.11979).

Figures
- Figure A1 has a small in-image caption line ("held object (rows) on target (columns)") whose first letter is clipped at the left edge; cosmetic, not an anonymity issue.
- RESOLVED 2026-11-23: draft TODO(check) items. Fig 5 SR 91 effects verified in docs/notes/sr91.md (+.408 [+.350, +.463], +.367 [+.217, +.517]; intervals added). Figure A1 caption corrected (grids are K1-K6/N1-N6 in scene 1 and K1, K2, N1, N2 in scene 2, not K1-K8/N1-N8). Figure A10 numbers verified against SR 61 (salient .974 of 151, nearer .752 of 141; intervals added). Figure A13 caption now describes panels a-d as drawn by fig_probe.py (panel d is pi0.5 only).
- Appendix D: the "targeted registered patching test of instruction-token versus cue-token keys and values at layer 14" was marked TODO(pending: not run) in the draft; the text now says we have not run it.

Layout
- Figure 3 of the draft (fig_atlas, the measured break by model) moved to Appendix B, so the main text has five figures; Table B1 in the appendix carries the same numbers.
- Table 4 of the draft (repair matrix) lost its "New layout" row (it said "not applicable" in every cell); the caption states that the new layout is a limit of the task.

Dropped from the appendix as internal process notes (no numbers lost): the "Deviations recorded by the agent" paragraph of the ReGuide-with-resolver subsection (scratch wrapper scripts), commit hashes, job ids and file paths of the pre-registration rows.

Status
- 2026-11-23: main.tex and parts/*.tex are now the canonical paper (revised after docs/notes/review_atlas_adv.md). paper/DRAFT.md is no longer kept in sync.
