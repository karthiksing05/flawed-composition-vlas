#!/bin/bash
# NPHP resubmission (2026-11-25) after the 2026-11-24 disk-full failure.  Same design as nphp_jobs/nphp_pipeline.sh.  Already COMPLETED and kept (not resubmitted): checks, states, disjoint, geom, all pqsL s42/s43 rollouts + E1, pqA0 s42 K and N 16+18+20.
# New: 9 trainings (from scratch; no checkpoint survived), exports, pcheck, readgate, remaining rollouts, readmain, E1.  keep-period 6000 (final + rolling save).  H* on tail-lab (4 GPUs), the rest on overcap.
PP=/coc/flash10/ksingara3/.claude_tmp/pp; REPO=/coc/flash10/ksingara3/flawed-composition-vlas; JD=$REPO/docs/notes/nphp_jobs_r2
EX=sky1,hal,tars,t1000,vicki,johnny5,c3po,olivaw,megabot,ripl-s1,gundam,kitt,nestor,samantha,voltron,jill,gideon,walle,xaea-12,spot,cyborg,heistotron,major,baymax
OV="-p overcap -A overcap --exclude=$EX"
ids=$PP/nphp_pipeline_ids_r2.txt
sb() { lab=$1; shift; j=$(sbatch --parsable "$@"); echo "$lab $j" >> $ids; echo "$lab $j" >&2; echo $j; }
G=$JD/nphp_gated.sh
declare -A CFG=([pqH]=pi05_g8v_raah0_l_ae [pqA0]=pi05_g8v_raac_l_ae [pqsL]=pi05_g8v_raas_l_ae)
declare -A LAST EXPJ
for spec in pqH:42:tl pqH:43:tl pqH:44:tl pqH:45:tl pqA0:43:main pqA0:44:main pqA0:45:main pqsL:44:main pqsL:45:main; do
  IFS=: read a s c <<< "$spec"
  LAST[$a$s]=$(CHAIN=$c CFG=${CFG[$a]} SEED=$s LBL=$a$s bash $JD/nphp_chain.sh 2>>$ids)
  echo "train_${a}_$s ${LAST[$a$s]}" >> $ids
  EXPJ[$a$s]=$(sb export_${a}_$s $OV -J nphp_export_${a}$s --dependency=afterany:${LAST[$a$s]} -c 8 --mem=64G -t 02:00:00 --export=ALL,KIND=export,ALAB=$a,SEED=$s $G)
  if [ $a = pqH ]; then EXPJ[$a$s]=$(sb pcheck_$s $OV -J nphp_pcheck$s --dependency=afterany:${EXPJ[$a$s]} --gres=gpu:a40:1 -c 8 --mem=64G -t 00:40:00 --export=ALL,KIND=pcheck,ALAB=$a,SEED=$s $G); fi
done
declare -A N_JOBS
ALLR=""
roll() { # arm seed kind tids dep -> id
  local dep=""; [ -n "$5" ] && dep="--dependency=afterany:$5"
  sb roll_$1_$2_$3_$4 $OV -J nphp_a_$1_$2_$4 $dep --gres=gpu:a40:1 -c 8 --mem=64G -t 02:30:00 --export=ALL,KIND=$3,ALAB=$1,SEED=$2,TIDS=$4 $G
}
# the one missing existing-checkpoint cell: pqA0 s42 N 22+24+26 (cells 22, 24 exist; 26 missing; finished cells skipped)
n2=$(roll pqA0 42 rollN 22+24+26 ""); ALLR="$ALLR:$n2"; N_JOBS[pqA042]=$n2
for s in 42 43 44 45; do
  for a in pqH pqA0 pqsL; do
    case "$a$s" in pqA042|pqsL42|pqsL43) continue;; esac
    d="${EXPJ[$a$s]:-}"
    k1=$(roll $a $s rollK 0+2+4 "$d"); k2=$(roll $a $s rollK 6+8+10 "$d"); ALLR="$ALLR:$k1:$k2"
    if [ $a = pqH ]; then
      rg=$(sb readgate_$s $OV -J nphp_readgate$s --dependency=afterany:$k1:$k2 -c 4 --mem=24G -t 00:20:00 --export=ALL,KIND=readgate,ALAB=pqH,SEED=$s $G); ALLR="$ALLR:$rg"; nd=$rg
    else nd="$d"; fi
    n1=$(roll $a $s rollN 16+18+20 "$nd"); n2=$(roll $a $s rollN 22+24+26 "$nd"); ALLR="$ALLR:$n1:$n2"
    N_JOBS[$a$s]="$n1:$n2"
  done
done
sb readmain $OV -J nphp_readmain --dependency=afterany$ALLR -c 4 --mem=24G -t 00:30:00 --export=ALL,KIND=readmain $G > /dev/null
for a in pqH pqA0 pqsL; do
  e=""
  for s in 42 43 44 45; do
    [ $a = pqsL ] && [ $s -le 43 ] && continue   # e1 for pqsL s42/s43 COMPLETED 2026-11-24
    e="$e:$(sb e1_${a}_$s $OV -J nphp_e1_$a$s --dependency=afterany:${N_JOBS[$a$s]} -c 4 --mem=24G -t 00:20:00 --export=ALL,KIND=e1arm,ALAB=$a,SEED=$s $G)"
  done
  sb e1pool_$a $OV -J nphp_e1pool_$a --dependency=afterany$e -c 4 --mem=24G -t 00:20:00 --export=ALL,KIND=e1pool,ALAB=$a $G > /dev/null
done
