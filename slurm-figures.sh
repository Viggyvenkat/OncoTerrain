#!/bin/bash
#SBATCH --partition=p_sd948         # Partition (job queue)
#SBATCH --requeue                 # Return job to the queue if preempted
#SBATCH --job-name=scRNA-Heavy       # Assign a short name to your job
#SBATCH --nodes=1                 # Number of nodes you require
#SBATCH --ntasks=1                # Total # of tasks across all nodes
#SBATCH --cpus-per-task=16         # Cores per task (>1 if multithread tasks)
#SBATCH --mem-per-cpu=16G
#SBATCH --time=3-00:00:00   # 3 days, 0 hours, 0 minutes, 0 seconds
#SBATCH --output=slurm.%N.%j.out  # STDOUT output file
#SBATCH --error=slurm.%N.%j.err   # STDERR output file (optional)

module load python/3.13.1
python src/fig-generation/figure-5.py