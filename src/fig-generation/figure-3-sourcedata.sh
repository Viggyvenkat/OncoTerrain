#!/bin/bash
#SBATCH --partition=p_sd948
#SBATCH --requeue
#SBATCH --job-name=fig3_sourcedata
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm.%N.%j.out
#SBATCH --error=slurm.%N.%j.err

set -euo pipefail

cd ../.. || exit 1
mkdir -p .cache/matplotlib .cache/numba
export MPLCONFIGDIR=".cache/matplotlib"
export NUMBA_CACHE_DIR=".cache/numba"

if [ -f /cache/home/vvv11/miniforge3/etc/profile.d/conda.sh ]; then
    source /cache/home/vvv11/miniforge3/etc/profile.d/conda.sh
    set +u
    conda activate revision
    set -u
fi

# Figure 3: box/violin panels 3G, 3G-endothelial, 3L (Mann-Whitney U -> exact p-values with
# medians/direction) and bar/composition panels 3H, 3I, 3J, 3K, endothelial barplot (source data
# only). Skips the pseudotime/UMAP/per-gene-grid image panels.
python src/fig-generation/figure-3.py

echo "Figure 3 render complete; source data written to src/fig-generation/source-data/figure_3/"
