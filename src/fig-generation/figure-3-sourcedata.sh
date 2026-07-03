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

# Submit from the repo root:  sbatch src/fig-generation/figure-3-sourcedata.sh
cd "${SLURM_SUBMIT_DIR:-$(pwd)}" || exit 1

source /cache/home/vvv11/miniforge3/etc/profile.d/conda.sh
set +u
conda activate revision
set -u

# Figure 3: box/violin panels 3G, 3G-endothelial, 3L (Mann-Whitney U -> exact p-values with
# medians/direction) and bar/composition panels 3H, 3I, 3J, 3K, endothelial barplot (source data
# only). Skips the pseudotime/UMAP/per-gene-grid image panels.
python src/fig-generation/figure-3.py --source-data

echo "Figure 3 source data written to src/fig-generation/source-data/figure_3/"
