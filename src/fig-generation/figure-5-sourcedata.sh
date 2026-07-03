#!/bin/bash
#SBATCH --partition=p_sd948
#SBATCH --requeue
#SBATCH --job-name=fig5_sourcedata
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=2-00:00:00
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

# --- Part 1 (Python): 5B stage Mann-Whitney (+FDR), 5E polar-bar heights, 5F similarity deltas ---
python src/fig-generation/figure-5.py

# --- Part 2 (Python): 5D violin source data (per-CAT malignant fraction) ---
# Requires figures/all_oncoterrain/OncoTerrain_annotated.h5ad from a prior OncoTerrain inference run.
python src/fig-generation/violin_plot_5D.py

# --- Part 3 (R): TCGA PCA / PLS-DA scores + loadings, and CopyKAT classification proportions ---
# TODO: set the R module you use for these scripts (Rscript must be on PATH). Adjust the name if needed.
module load R 2>/dev/null || true
if command -v Rscript >/dev/null 2>&1; then
    Rscript src/fig-generation/tcga-val.R
    Rscript src/fig-generation/copyKAT-val.R
else
    echo "ERROR: Rscript not found. Load your R module and rerun:"
    echo "       Rscript src/fig-generation/tcga-val.R"
    echo "       Rscript src/fig-generation/copyKAT-val.R"
    exit 1
fi

# Re-bundle the Figure 5 workbook so it also includes any R-produced TCGA/CopyKAT CSVs.
python -c "import sys; sys.path.insert(0, 'src/fig-generation'); import source_data_common as s; s.aggregate_to_excel(5)"

echo "Figure 5 render complete; source data written to src/fig-generation/source-data/figure_5/"
