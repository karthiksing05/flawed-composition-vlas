#!/bin/bash
# one-off driver (run once from the login node): resubmits the stage-1 jobs that vanished from the queue (sacct shows PENDING, squeue none; cause unknown).  No dependencies (afterok is rejected, afterany was not honoured).
cd /coc/flash10/ksingara3/flawed-composition-vlas; D=docs/notes/xb_jobs
add(){ echo "$1	$2	$3	$4" >> $D/xb_jobids.tsv; }
for t in place_a2b_left place_a2b_right; do for c in 0 1 2 3; do [ $t = place_a2b_left ] && [ $c = 0 ] && continue; n=xb_rt_prep_${t#place_a2b_}_$c; J=$(sbatch --parsable -J $n --export=ALL,MODE=prep,TASK=$t,CHUNK=$c/4 $D/xb_roll_rt.sh); add $J $n prep "resubmit robotwin prep $t $c/4"; done; done
for m in pi0 pi0fast; do for c in 0 1 2 3; do n=xb_mesa_${m}_par_$c; J=$(sbatch --parsable -J $n --export=ALL,BENCH=mesa,MODEL=$m,CELLTYPE=partner,CHUNK=$c/4,NSHARD=4 $D/xb_roll_mesa.sh); add $J $n partner "resubmit mesa $m $c/4"; done; done
for c in 0 1 2 3; do n=xb_bimesa_pi05_par_$c; J=$(sbatch --parsable -J $n --export=ALL,BENCH=bimesa,MODEL=pi05,CELLTYPE=partner,CHUNK=$c/4,NSHARD=4 $D/xb_roll_mesa.sh); add $J $n partner "resubmit bimesa pi05 $c/4"; done
J=$(sbatch --parsable -J xb_smoke_mesa_groot -t 00:40:00 --export=ALL,BENCH=mesa,MODEL=groot,CELLTYPE=partner,CHUNK=0/20,NSHARD=1,SMOKE=1 $D/xb_roll_mesa.sh); add $J xb_smoke_mesa_groot smoke "resubmit"
