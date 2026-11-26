#!/bin/bash
# XB setup: RoboTwin pi0.5 cotrain checkpoint (12.4 GB tar) + SmolVLA checkpoint and its VLM.  CPU only.
#SBATCH -p overcap
#SBATCH -A overcap
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -t 03:00:00
#SBATCH --requeue
#SBATCH --exclude=sky1,hal,tars,t1000,vicki,johnny5,c3po,olivaw,megabot,ripl-s1,gundam,kitt,nestor,samantha,voltron,jill,gideon,walle,xaea-12,spot,cyborg,heistotron,major,baymax
#SBATCH -o /coc/flash10/ksingara3/.claude_tmp/pp/xb_%x_%j.out
R=/coc/flash10/ksingara3/robot; B=$R/bench; RT=$B/robotwin/RoboTwin; export HF_HOME=$B/hf PYTHONUNBUFFERED=1
$R/openpi/.venv/bin/python - <<'PY'
from huggingface_hub import hf_hub_download, snapshot_download
B="/coc/flash10/ksingara3/robot/bench"; P=f"{B}/robotwin/RoboTwin/XPolicyLab/policy"
def go(f, *a, **k):
    try: print("ok", f(*a, **k), flush=True)
    except Exception as e: print("FAILED", a, repr(e), flush=True)
go(hf_hub_download, "TianxingChen/RoboTwin2.0", "cotrain_ckpt/Pi_05/RoboTwin-lerobot_v30-aloha_agilex-joint-0.tar.gz", repo_type="dataset", local_dir=f"{B}/robotwin/ckpt_dl")
go(snapshot_download, "lerobot/smolvla_robotwin", local_dir=f"{P}/SmolVLA/checkpoints/smolvla_robotwin")
go(snapshot_download, "HuggingFaceTB/SmolVLM2-500M-Video-Instruct", local_dir=f"{P}/SmolVLA/checkpoints/shared/SmolVLM2-500M-Video-Instruct")
PY
mkdir -p $RT/XPolicyLab/policy/Pi_05/checkpoints && cd $RT/XPolicyLab/policy/Pi_05/checkpoints && tar xzf $B/robotwin/ckpt_dl/cotrain_ckpt/Pi_05/*.tar.gz && find . -maxdepth 3 | head -20
du -sh $RT/XPolicyLab/policy/Pi_05/checkpoints $RT/XPolicyLab/policy/SmolVLA/checkpoints
ls $RT/XPolicyLab/policy/SmolVLA/ $RT/XPolicyLab/policy/Pi_05/openpi | head -30
echo SETUP_RT_DONE
