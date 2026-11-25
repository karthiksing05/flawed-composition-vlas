#!/bin/bash
# NPHJ resubmission (2026-11-25) after the 2026-11-24 disk-full failure.  Same design as nphj_jobs/nphj_pipeline.sh (checks marker exists, check job not repeated).
# Trained from scratch (no checkpoint survived).  keep-period = final step.  Everything on overcap.  A0-24k seed 43 is NOT queued: it would push the projected total past the 110 GPU-h cap (see results note).
PP=/coc/flash10/ksingara3/.claude_tmp/pp; REPO=/coc/flash10/ksingara3/flawed-composition-vlas; JD=$REPO/docs/notes/nphj_jobs_r2
EX=sky1,hal,tars,t1000,vicki,johnny5,c3po,olivaw,megabot,ripl-s1,gundam,kitt,nestor,samantha,voltron,jill,gideon,walle,xaea-12,spot,cyborg,heistotron,major,baymax
OV="-p overcap -A overcap --exclude=$EX"
ids=$PP/nphj_pipeline_ids_r2.txt
sb() { lab=$1; shift; j=$(sbatch --parsable "$@"); echo "$lab $j" >> $ids; echo "$lab $j" >&2; echo $j; }
G=$JD/nphj_gated.sh
declare -A LAST EXPJ N_JOBS
roll() { local dep=""; [ -n "$5" ] && dep="--dependency=afterany:$5"
  sb roll_$1_$2_$3_$4 $OV -J nphj_a_$1_$2_$4 $dep --gres=gpu:a40:1 -c 8 --mem=64G -t 01:45:00 --export=ALL,KIND=$3,ALAB=$1,SEED=$2,TIDS=$4 $G; }
for spec in pzJ:42 pzJ:43 pzJ:44 pzJ:45 pzA24:42; do
  IFS=: read a s <<< "$spec"
  if [ $a = pzA24 ]; then
    LAST[$a$s]=$(CFG=pi0_g8v_raac_l_ae SEED=$s LBL=${a}_$s EXP=t24k_s$s STEPS=24001 KEEP=24000 NL=10 bash $JD/nphj_chain.sh 2>>$ids)
  else
    LAST[$a$s]=$(CFG=pi0_g8v_raaj_l_ae SEED=$s LBL=${a}_$s EXP=tj_s$s STEPS=6001 KEEP=6000 NL=4 bash $JD/nphj_chain.sh 2>>$ids)
  fi
  echo "train_${a}_$s ${LAST[$a$s]}" >> $ids
  ex=$(sb export_${a}_$s $OV -J nphj_export_${a}$s --dependency=afterany:${LAST[$a$s]} -c 8 --mem=64G -t 02:00:00 --export=ALL,KIND=export,ALAB=$a,SEED=$s $G)
  k1=$(roll $a $s rollK 0+2+4 "$ex"); k2=$(roll $a $s rollK 6+8+10 "$ex")
  rg=$(sb readgate_${a}_$s $OV -J nphj_readgate_${a}$s --dependency=afterany:$k1:$k2 -c 4 --mem=24G -t 00:20:00 --export=ALL,KIND=readgate,ALAB=$a,SEED=$s $G)
  n1=$(roll $a $s rollN 16+18+20 "$rg"); n2=$(roll $a $s rollN 22+24+26 "$rg")
  N_JOBS[$a$s]="$n1:$n2"
done
for a in pzJ pzA24; do
  e=""; seeds="42 43 44 45"; [ $a = pzA24 ] && seeds="42"
  for s in $seeds; do e="$e:$(sb e1_${a}_$s $OV -J nphj_e1_$a$s --dependency=afterany:${N_JOBS[$a$s]} -c 4 --mem=24G -t 00:20:00 --export=ALL,KIND=e1arm,ALAB=$a,SEED=$s $G)"; done
  sb e1pool_$a $OV -J nphj_e1pool_$a --dependency=afterany$e -c 4 --mem=24G -t 00:20:00 --export=ALL,KIND=e1pool,ALAB=$a $G > /dev/null
done
