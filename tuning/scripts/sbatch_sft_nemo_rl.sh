#!/bin/bash
#SBATCH --job-name=sft-nemo-rl
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=96
#SBATCH --mem=0
#SBATCH --time=48:00:00
#SBATCH --output=logs/sft-nemo-rl-%j.out
#SBATCH --error=logs/sft-nemo-rl-%j.err

set -euxo pipefail

# Create logs directory if it doesn't exist
mkdir -p logs

# Print job information
echo "============================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "GPUs per node: ${SLURM_GPUS_PER_NODE:-8}"
echo "Start time: $(date)"
echo "============================================"

# Environment setup
# Adjust these paths according to your environment
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export WANDB_API_KEY="${WANDB_API_KEY:-$(cat ~/.wandb_api_key 2>/dev/null || echo '')}"
export TMPDIR="${TMPDIR:-/tmp}"

# CUDA settings
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export NCCL_DEBUG=INFO

# Change to tuning directory
cd "$(dirname "$0")/.."

# Parse command line arguments
CONFIG="${1:-configs/sft_nemo_rl.yaml}"
shift || true
OVERRIDES="$@"

echo "Config: ${CONFIG}"
echo "Overrides: ${OVERRIDES}"

# Run training with uv
# NeMo-RL uses Ray for distributed training, which handles multi-GPU automatically
uv run python train_sft_nemo_rl.py \
    --config "${CONFIG}" \
    cluster.gpus_per_node=${SLURM_GPUS_PER_NODE:-8} \
    cluster.num_nodes=${SLURM_NNODES:-1} \
    ${OVERRIDES}

echo "============================================"
echo "End time: $(date)"
echo "Job completed successfully!"
echo "============================================"
