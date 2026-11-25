#!/bin/bash
# NPHP unattended pipeline: login node, run ONCE after the nphp prereg row is committed (>= 60 s) and nphp_checks.sh has been submitted (job ids in nphp_pipeline_ids.txt).  Only sbatch runs here.
# Tail-lab: the 4 H* training chains (4 GPUs).  Overcap: everything else (requeue-safe, resumable from the newest checkpoint).
PP=/coc/flash10/ksingara3/.claude_tmp/pp; REPO=/coc/flash10/ksingara3/flawed-composition-vlas; JD=$REPO/docs/notes/nphp_jobs
EX=sky1,hal,tars,t1000,vicki,johnny5,c3po,olivaw,megabot,ripl-s1,gundam,kitt,nestor,samantha,voltron,jill,gideon,walle,xaea-12,spot,cyborg,heistotron,major,baymax
OV="-p overcap -A overcap --exclude=$EX"
ids=$PP/nphp_pipeline_ids.txt
sb() { lab=$1; shift; j=$(sbatch --parsable "$@"); echo "$lab $j" >> $ids; echo "$lab $j" >&2; echo $j; }
G=$JD/nphp_gated.sh; CPU=$PP/pz_cpu.sh
OKJ=$(awk '$1=="checks_ok"{print $2}' $ids | tail -1); [ -n "$OKJ" ] || { echo "no checks_ok job id"; exit 2; }
# 1. eighth set (afterwards: disjointness -> geometry); rollouts depend on the disjointness job
st=$(sb states $OV -J nphp_states --gres=gpu:a40:1 -c 4 --mem=32G -t 02:00:00 --export=ALL,WHAT=n2 $JD/nphp_states.sh)
dj=$(sb disjoint $OV -J nphp_disjoint --dependency=afterok:$st --export=ALL,VENV=openpi,CMD=experiments/nphp_disjoint.py $CPU)
ge=$(sb geom $OV -J nphp_geom --dependency=afterok:$dj --export=ALL,VENV=libero,CMD=experiments/nphp_geom.py $CPU)
# 2. trainings (9): H* seeds 42-45 on tail-lab, A0 43-45 and RAA-sL 44-45 on overcap; the first link of each needs afterok:checks_ok
declare -A CFG=([pqH]=pi05_g8v_raah0_l_ae [pqA0]=pi05_g8v_raac_l_ae [pqsL]=pi05_g8v_raas_l_ae)
declare -A LAST EXPJ
for spec in pqH:42:tl pqH:43:tl pqH:44:tl pqH:45:tl pqA0:43:main pqA0:44:main pqA0:45:main pqsL:44:main pqsL:45:main; do
  IFS=: read a s c <<< "$spec"
  LAST[$a$s]=$(DEP0=$OKJ CHAIN=$c CFG=${CFG[$a]} SEED=$s LBL=$a$s bash $JD/nphp_chain.sh 2>>$ids)
  echo "train_${a}_$s ${LAST[$a$s]}" >> $ids
  EXPJ[$a$s]=$(sb export_${a}_$s $OV -J nphp_export_${a}$s --dependency=afterany:${LAST[$a$s]} -c 8 --mem=64G -t 02:00:00 --export=ALL,KIND=export,ALAB=$a,SEED=$s $G)
  if [ $a = pqH ]; then EXPJ[$a$s]=$(sb pcheck_$s $OV -J nphp_pcheck$s --dependency=afterany:${EXPJ[$a$s]} --gres=gpu:a40:1 -c 8 --mem=64G -t 00:40:00 --export=ALL,KIND=pcheck,ALAB=$a,SEED=$s $G); fi
done
# 3. rollouts (existing checkpoints: A0 s42, sL s42 and s43 start as soon as the set is checked)
declare -A N_JOBS
ALLR=""
roll() { # arm seed kind tids dep -> id
  local dep=afterok:$dj; [ -n "$5" ] && dep=$dep,afterany:$5
  sb roll_$1_$2_$3_$4 $OV -J nphp_a_$1_$2_$4 --dependency=$dep --gres=gpu:a40:1 -c 8 --mem=64G -t 02:30:00 --export=ALL,KIND=$3,ALAB=$1,SEED=$2,TIDS=$4 $G
}
for s in 42 43 44 45; do
  for a in pqH pqA0 pqsL; do
    d="${EXPJ[$a$s]:-}"
    k1=$(roll $a $s rollK 0+2+4 "$d"); k2=$(roll $a $s rollK 6+8+10 "$d"); ALLR="$ALLR:$k1:$k2"
    if [ $a = pqH ]; then
      rg=$(sb readgate_$s $OV -J nphp_readgate$s --dependency=afterany:$k1:$k2 -c 4 --mem=24G -t 00:20:00 --export=ALL,KIND=readgate,ALAB=pqH,SEED=$s $G); ALLR="$ALLR:$rg"; nd=$rg
    else nd="$d"; fi
    n1=$(roll $a $s rollN 16+18+20 "$nd"); n2=$(roll $a $s rollN 22+24+26 "$nd"); ALLR="$ALLR:$n1:$n2"
    N_JOBS[$a$s]="$n1:$n2"
  done
done
rm=$(sb readmain $OV -J nphp_readmain --dependency=afterany$ALLR -c 4 --mem=24G -t 00:30:00 --export=ALL,KIND=readmain $G)
# 4. E1 per (arm, seed), then pooled per arm
for a in pqH pqA0 pqsL; do
  e=""
  for s in 42 43 44 45; do e="$e:$(sb e1_${a}_$s $OV -J nphp_e1_$a$s --dependency=afterany:${N_JOBS[$a$s]},afterok:$ge -c 4 --mem=24G -t 00:20:00 --export=ALL,KIND=e1arm,ALAB=$a,SEED=$s $G)"; done
  sb e1pool_$a $OV -J nphp_e1pool_$a --dependency=afterany$e -c 4 --mem=24G -t 00:20:00 --export=ALL,KIND=e1pool,ALAB=$a $G
done
