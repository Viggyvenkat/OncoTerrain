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

set -euo pipefail

cd /Users/vigneshvenkat/Desktop/SJDLab/OncoTerrain || exit 1
mkdir -p .cache/matplotlib .cache/numba
export MPLCONFIGDIR="/Users/vigneshvenkat/Desktop/SJDLab/OncoTerrain/.cache/matplotlib"
export NUMBA_CACHE_DIR="/Users/vigneshvenkat/Desktop/SJDLab/OncoTerrain/.cache/numba"

if [ -f /cache/home/vvv11/miniforge3/etc/profile.d/conda.sh ]; then
    source /cache/home/vvv11/miniforge3/etc/profile.d/conda.sh
    set +u
    conda activate revision
    set -u
fi

# Figure 1: full figure render; panel 1G also writes source data.
python src/fig-generation/figure-1.py

echo "Figure 1 render complete; source data written to src/fig-generation/source-data/figure_1/"
