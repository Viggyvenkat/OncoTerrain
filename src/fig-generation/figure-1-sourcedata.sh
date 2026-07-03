#!/bin/bash
#SBATCH --partition=p_sd948         # Partition (job queue)
#SBATCH --requeue                 # Return job to the queue if preempted
#SBATCH --job-name=fig1_sourcedata
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G                # processed_data.h5ad is ~31 GB; needs plenty of RAM
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm.%N.%j.out
#SBATCH --error=slurm.%N.%j.err

# Submit from the repo root:  sbatch src/fig-generation/figure-1-sourcedata.sh
# (the figure scripts resolve data/ and figures/ relative to the working directory)
cd "${SLURM_SUBMIT_DIR:-$(pwd)}" || exit 1

source /cache/home/vvv11/miniforge3/etc/profile.d/conda.sh
set +u
conda activate revision
set -u

# Figure 1: only panel 1G (cell-type composition barplot) has source data; no statistical tests.
python src/fig-generation/figure-1.py --source-data

echo "Figure 1 source data written to src/fig-generation/source-data/figure_1/"
