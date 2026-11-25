#!/bin/bash
# NPHP pre-flight (login node, ONCE, before the prereg row; submits only checks + smoke): identity check (GPU), smoke training of 30 steps -> scratch export -> pcheck on the smoke checkpoint -> marker nphp_checks_ok.
# Training links refuse without the marker.  Appends "<label> <jobid>" to nphp_pipeline_ids.txt.
PP=/coc/flash10/ksingara3/.claude_tmp/pp; REPO=/coc/flash10/ksingara3/flawed-composition-vlas; JD=$REPO/docs/notes/nphp_jobs; CK=/coc/flash10/ksingara3/robot/checkpoints_l90c
EX=sky1,hal,tars,t1000,vicki,johnny5,c3po,olivaw,megabot,ripl-s1,gundam,kitt,nestor,samantha,voltron,jill,gideon,walle,xaea-12,spot,cyborg,heistotron,major,baymax
OV="-p overcap -A overcap --exclude=$EX"
ids=$PP/nphp_pipeline_ids.txt
sb() { lab=$1; shift; j=$(sbatch --parsable "$@"); echo "$lab $j" >> $ids; echo "$lab $j" >&2; echo $j; }
id=$(sb ident -p tail-lab -A tail-lab --exclude=$EX -J nphp_ident --export=ALL $JD/nphp_check.sh)
tr=$(sb smoke_train $OV -J nphp_smoke_train -t 01:30:00 --export=ALL,ARGS=--name~pi05_g8v_raah0_l_ae~--exp~smoke~--seed~42~--steps~31~--batch~32~--save-interval~10~--keep-period~10 $JD/nphp_train.sh)
ex=$(sb smoke_export $OV -J nphp_smoke_export --dependency=afterok:$tr -c 8 --mem=64G -t 01:00:00 --export=ALL,SRC=$CK/jax/pi05_g8v_raah0_l_ae/smoke/30,DST=$PP/nphp_smoke_pt,NAME=smoke $JD/nphp_export.sh)
pc=$(sb smoke_pcheck $OV -J nphp_smoke_pcheck --dependency=afterok:$ex:$id --gres=gpu:a40:1 -c 8 --mem=64G -t 01:00:00 --export=ALL,EXTRA=--torch-ckpt~$PP/nphp_smoke_pt $JD/nphp_check.sh)
sb checks_ok $OV -J nphp_checks_ok --dependency=afterok:$pc -c 1 --mem=2G -t 00:05:00 --export=ALL,KIND=smokeok $JD/nphp_gated.sh
