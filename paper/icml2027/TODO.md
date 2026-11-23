# Remaining TODOs for the ICML build (not part of the submission)

Citations
- POT-VLA (potvla2607) and w2VLA (w2vla2606): the short names come from the draft, not from the arXiv titles; confirm them in the PDFs. The bib entries carry the arXiv titles.
- STAGE (stage2609): STAGE is the framework name; confirm that the SAT-Bench benchmark name is not needed.
- "Nouns, Not Modifiers" (ICML 2026) and the MEM / pi0.7 memory mentions in the draft have no arXiv id; the first was dropped from the text (it sat in the Appendix L.6 list of shortened numbers), the others stay as plain-text mentions in Appendix I without a citation.
- Entries in refs_extra.bib (Wulff and Cangelosi, Jiang et al. audit, RoboMME, LIBERO-Mem, de Haan et al.) were taken from the arXiv abstract pages on 2026-11-23. de Haan et al. is cited as the NeurIPS 2019 paper (arXiv 1905.11979).

Figures
- Figure A1 has a small in-image caption line ("held object (rows) on target (columns)") whose first letter is clipped at the left edge; cosmetic, not an anonymity issue.
- Draft TODO(check) items that were stripped from the captions and still need a check: Fig 5 (SR 91 effect sizes +.41 and +.37 are as printed by the figure script; confirm against the logged results), Figure A1 and Figure A13 panel contents against the figure scripts, Figure A10 (salient and nearer rates .974 and .752 against the Section 4 wording).
- Appendix D: the "targeted registered patching test of instruction-token versus cue-token keys and values at layer 14" was marked TODO(pending: not run) in the draft; the text now says we have not run it.

Layout
- Figure 3 of the draft (fig_atlas, the measured break by model) moved to Appendix B, so the main text has five figures; Table B1 in the appendix carries the same numbers.
- Table 4 of the draft (repair matrix) lost its "New layout" row (it said "not applicable" in every cell); the caption states that the new layout is a limit of the task.

Dropped from the appendix as internal process notes (no numbers lost): the "Deviations recorded by the agent" paragraph of the ReGuide-with-resolver subsection (scratch wrapper scripts), commit hashes, job ids and file paths of the pre-registration rows.
