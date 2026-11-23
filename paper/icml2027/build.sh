#!/bin/bash
#SBATCH -p overcap
#SBATCH -A overcap
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 00:20:00
#SBATCH --exclude=sky1,hal,tars,t1000,vicki,johnny5,c3po,olivaw,megabot,ripl-s1,gundam,kitt,nestor,samantha,voltron,jill,gideon,walle,xaea-12,spot,cyborg,heistotron,major,baymax
#SBATCH -o /coc/flash10/ksingara3/.claude_tmp/pp/icml_build_%j.log
# Build the ICML submission on a compute node. usage: sbatch build.sh  (NPAGES = pages rendered to PNG, default 30)
set -e
D=/coc/flash10/ksingara3/flawed-composition-vlas/paper
C=/coc/flash10/ksingara3/compositionality-theory/notes/audit/figcheck
PY=~/flash/miniconda3/envs/chunks/bin/python
cd $D/icml2027
# references.bib = paper/refs.bib without the working notes + the entries added for this build
grep -v '^  note = ' $D/refs.bib > references.bib
cat refs_extra.bib >> references.bib
~/flash/miniconda3/envs/tex/bin/tectonic --keep-logs main.tex > tectonic.out 2>&1 || true
grep -v -E 'Underfull|^Running|^Rerunning|^Skipped' tectonic.out | tail -50
$PY $C/icml_check.py $D/icml2027/main.pdf /coc/flash10/ksingara3/.claude_tmp/pp/pages ${NPAGES:-30} 2>&1 | grep -v deprecated
$PY - <<'EOF'
import fitz
d = fitz.open("/coc/flash10/ksingara3/flawed-composition-vlas/paper/icml2027/main.pdf")
for i, p in enumerate(d):
    t = p.get_text()
    for k in ("The repair matrix. N is", "Conclusion","Impact Statement", "AI Use Statement", "References", "Appendix", "A. Setting details"):
        if k in t:
            print(k, "on page", i + 1)
EOF
$PY - <<'EOF'
import fitz, re
d = fitz.open("/coc/flash10/ksingara3/flawed-composition-vlas/paper/icml2027/main.pdf")
print("PDF metadata:", d.metadata)
bad = re.compile(r"ksingara|karthik|georgia|gatech|/coc|flash10|github|\bSR ?\d|docs/|ledger|prereg|fresh", re.I)
for i, p in enumerate(d):
    for m in bad.finditer(p.get_text()):
        print("ANON HIT page", i + 1, repr(p.get_text()[max(0, m.start() - 30):m.end() + 30]))
print("pages:", len(d))
EOF
$PY $C/fontlist.py $D/icml2027/main.pdf 2>&1 | grep -v deprecated | awk '{print $3}' | sed -E 's/^[A-Z]{6}\+//' | sort -u | tr '\n' ' '
echo
grep -E 'Warning.*(undefined|multiply)' main.log | sort -u | head -20 || true
