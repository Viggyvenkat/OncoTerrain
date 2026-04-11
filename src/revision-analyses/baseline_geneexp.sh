#!/bin/bash
#SBATCH --partition=cgpu         # Partition (job queue)
#SBATCH --requeue                 # Return job to the queue if preempted
#SBATCH --job-name=baseline_geneexp      # Assign a short name to your job
#SBATCH --nodes=1                 # Number of nodes you require
#SBATCH --ntasks=1                # Total # of tasks across all nodes
#SBATCH --cpus-per-task=8          # Cores per task (>1 if multithread tasks)
#SBATCH --mem=256G
#SBATCH --gres=gpu:1              # Request 1 GPU
#SBATCH --time=3-00:00:00   # 3 days, 0 hours, 0 minutes, 0 seconds
#SBATCH --output=slurm.%N.%j.out  # STDOUT output file
#SBATCH --error=slurm.%N.%j.err   # STDERR output file (optional)

source /cache/home/vvv11/miniforge3/etc/profile.d/conda.sh
set +u
conda activate revision
set -u
python baseline_geneexp.py