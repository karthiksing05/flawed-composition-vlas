#!/bin/bash
#SBATCH -p overcap
#SBATCH -A overcap
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -t 03:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --exclude=sky1,hal,tars,t1000,vicki,johnny5,c3po,olivaw,megabot,ripl-s1,gundam,kitt,nestor,samantha,voltron,jill,gideon,walle,xaea-12,spot,cyborg,heistotron,major,baymax
#SBATCH -o /coc/flash10/ksingara3/.claude_tmp/pp/xb_%x_%j.out
# XB setup: RoboCasa365 checkpoints (pi0, GR00T N1.5 official, X-VLA community).  CPU only.
R=/coc/flash10/ksingara3/robot; B=$R/bench; export HF_HOME=$B/hf PYTHONUNBUFFERED=1
$R/openpi/.venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
B="/coc/flash10/ksingara3/robot/bench"
def go(repo, local, **kw):
    try: p=snapshot_download(repo, local_dir=local, **kw); print("downloaded", repo, p, flush=True)
    except Exception as e: print("FAILED", repo, repr(e), flush=True)
go("robocasa/robocasa365_checkpoints", f"{B}/robocasa/ckpt", allow_patterns=["pi0/pi0_robocasa_pretrain_human300/multitask_learning/75000/params/*","pi0/pi0_robocasa_pretrain_human300/multitask_learning/75000/assets/*","pi0/pi0_robocasa_pretrain_human300/multitask_learning/75000/_CHECKPOINT_METADATA"])
go("robocasa/robocasa365_checkpoints", f"{B}/robocasa/ckpt", allow_patterns=["gr00t_n1-5/multitask_learning/checkpoint-120000/*"], ignore_patterns=["*optimizer*","*rng_state*"])
go("deksprime/xvla-base-robocasa-human300-ri", f"{B}/robocasa/ckpt_xvla")
PY
du -sh $B/robocasa/ckpt/* $B/robocasa/ckpt_xvla 2>&1; ls $B/robocasa/ckpt/pi0/pi0_robocasa_pretrain_human300/multitask_learning/75000 $B/robocasa/ckpt/gr00t_n1-5/multitask_learning/checkpoint-120000 $B/robocasa/ckpt_xvla 2>&1 | head -40
echo SETUP_RC_DONE
