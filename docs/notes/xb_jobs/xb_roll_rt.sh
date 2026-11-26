#!/bin/bash
# XB RoboTwin 2.0.  env: MODE=check|prep|roll  TASK=place_a2b_left|place_a2b_right  [prep: CHUNK=i/n]  [roll: MODEL=xvla|pi05|smolvla CELLTYPE CHUNK NSHARD(=1) SMOKE]
#SBATCH -p overcap
#SBATCH -A overcap
#SBATCH --gres=gpu:a40:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --exclude=sky1,hal,tars,t1000,vicki,johnny5,c3po,olivaw,megabot,ripl-s1,gundam,kitt,nestor,samantha,voltron,jill,gideon,walle,xaea-12,spot,cyborg,heistotron,major,baymax
#SBATCH -o /coc/flash10/ksingara3/.claude_tmp/pp/xb_%x_%j.out
set -uo pipefail
source /coc/flash10/ksingara3/flawed-composition-vlas/docs/notes/xb_jobs/xb_lib.sh
xb_guard
MODE=${MODE:?}; TASK=${TASK:?}; CHUNK=${CHUNK:-0/1}
RT=$B/robotwin/RoboTwin; E=$B/robotwin/env; X=$B/robotwin/xvla_env
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 TMPDIR=$B/tmp CUDA_HOME=$E LD_LIBRARY_PATH=$E/lib:${LD_LIBRARY_PATH:-} PYTHONWARNINGS=ignore::UserWarning
nvidia-smi -L
cd $REPO
if [ "$MODE" = check ]; then PATH=$E/bin:$PATH python experiments/xb_roll_robotwin.py --mode check --task $TASK 2>&1 | tail -40; xb_log_job rt check $TASK; exit 0; fi
if [ "$MODE" = prep ];  then PATH=$E/bin:$PATH python experiments/xb_roll_robotwin.py --mode prep --task $TASK --chunk $CHUNK --ndraw ${NDRAW:-24} 2>&1 | grep -v "step:" | tail -80; xb_log_job rt prep $TASK $CHUNK; exit 0; fi
[ -f $XBOUT/cells_robotwin_final.json ] || { echo "[xb] no cells_robotwin_final.json: refused"; exit 4; }
MODEL=${MODEL:?}; CELLTYPE=${CELLTYPE:?}; NSHARD=${NSHARD:-1}
if [ "$CELLTYPE" = heldout ]; then [ -f $XBOUT/pred_robotwin.json ] || { echo "[xb] no pred_robotwin.json: held-out refused"; exit 4; }; fi
PORT=$((20000 + SLURM_JOB_ID % 20000))
case $MODEL in
  xvla)    POL=X_VLA; AT=ee
           cd $RT/XPolicyLab; PATH=$X/bin:$PATH CUDA_VISIBLE_DEVICES=0 python setup_policy_server.py --config_path policy/X_VLA/deploy.yml --overrides port=$PORT host=localhost bench_name=RoboTwin task_name=$TASK ckpt_name=RoboTwin-cotrain-aloha_agilex-ee-0 env_cfg_type=aloha_agilex seed=0 policy_name=X_VLA action_type=ee action_dim=14 > $XBOUT/serve_rt_${MODEL}_$SLURM_JOB_ID.log 2>&1 &;;
  pi05)    POL=Pi_05; AT=joint
           cd $RT/XPolicyLab/policy/Pi_05; PYTHONPATH=$RT:$B/robotwin/extra_pylibs:$RT/XPolicyLab/policy/Pi_05/openpi/src:$RT/XPolicyLab/policy/Pi_05/openpi/packages/openpi-client/src OPENPI_DATA_HOME=$R/openpi_cache CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 $R/openpi/.venv/bin/python $RT/XPolicyLab/setup_policy_server.py --config_path deploy.yml --overrides port=$PORT host=localhost bench_name=RoboTwin task_name=$TASK ckpt_name=RoboTwin-lerobot_v30-aloha_agilex-joint-0 env_cfg_type=aloha_agilex seed=0 policy_name=Pi_05 action_type=joint action_dim=14 checkpoint_num=29999 train_config_name=pi05_base_aloha_full_sim_arx-x5_seed_0 repo_id=RoboTwin_lerobot_v30 > $XBOUT/serve_rt_${MODEL}_$SLURM_JOB_ID.log 2>&1 &;;
  smolvla) POL=SmolVLA; AT=joint
           cd $RT/XPolicyLab/policy/SmolVLA; PYTHONPATH=$RT:$B/robotwin/extra_pylibs $R/smolvla/.venv/bin/python $RT/XPolicyLab/setup_policy_server.py --config_path deploy.yml --overrides port=$PORT host=localhost bench_name=RoboTwin task_name=$TASK ckpt_name=smolvla_robotwin env_cfg_type=aloha_agilex seed=0 policy_name=SmolVLA action_type=joint action_dim=14 pretrained_path=$RT/XPolicyLab/policy/SmolVLA/checkpoints/smolvla_robotwin > $XBOUT/serve_rt_${MODEL}_$SLURM_JOB_ID.log 2>&1 &;;
  *) echo "model $MODEL not wired"; exit 2;;
esac
SP=$!; trap 'kill $SP 2>/dev/null' EXIT
xb_wait_port $PORT $SP 900 || { tail -30 $XBOUT/serve_rt_${MODEL}_$SLURM_JOB_ID.log; exit 3; }
echo "[xb] server up ($PORT)"
cd $REPO
PATH=$E/bin:$PATH python experiments/xb_roll_robotwin.py --mode roll --task $TASK --model $MODEL --policy $POL --action-type $AT --cell-type $CELLTYPE --chunk $CHUNK --port $PORT ${SMOKE:+--smoke} 2>&1 | grep -v "step:" | tail -60
xb_log_job rt $MODEL $TASK $CELLTYPE $CHUNK
