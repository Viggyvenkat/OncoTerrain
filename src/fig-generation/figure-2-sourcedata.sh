#!/bin/bash
#SBATCH --partition=p_sd948
#SBATCH --requeue
#SBATCH --job-name=fig2_sourcedata
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
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

# Figure 2: panels 2C and 2G. Exports per-cell pathway-score source data, exact pairwise
# Mann-Whitney U + FDR-BH p-values (with medians/direction), and per-pathway SPLIT boxplots
# (figure_2C_<Pathway>.png / figure_2G_<Pathway>.png) so each panel shows its own p-values.
python src/fig-generation/figure-2.py

echo "Figure 2 render complete; source data written to src/fig-generation/source-data/figure_2/"
