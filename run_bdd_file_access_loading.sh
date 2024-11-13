#!/bin/bash
#SBATCH --account=genai-ad
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --time=00:20:00
#SBATCH --gres=gpu:4
#SBATCH --partition=booster
#SBATCH --output=%j.out

# Without this, srun does not inherit cpus-per-task from sbatch. #128
echo "----------------------------------"
export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"
# so processes know who to talk to
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
# Allow communication over InfiniBand cells.
MASTER_ADDR="${MASTER_ADDR}i"
# Get IP for hostname.
export MASTER_ADDR="$(nslookup "$MASTER_ADDR" | grep -oP '(?<=Address: ).*')"
export MASTER_PORT=7010

echo "MASTER_ADDR:MASTER_PORT=""$MASTER_ADDR":"$MASTER_PORT"
echo "----------------------------------"

echo "Job id: $SLURM_JOB_ID"
export CUDA_VISIBLE_DEVICES=0,1,2,3

source env/activate.sh

srun --cpu_bind=none bash -c "torchrun \
    --nnodes=$SLURM_NNODES \
    --rdzv_backend c10d \
    --nproc_per_node=gpu \
    --rdzv_id $RANDOM \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    --rdzv_conf=is_host=\$(if ((SLURM_NODEID)); then echo 0; else echo 1; fi) \
    bdd_file_access_loading.py --batch_size 256 \
    "


