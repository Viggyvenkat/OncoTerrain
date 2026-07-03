#!/bin/bash
#SBATCH --partition=p_sd948
#SBATCH --requeue
#SBATCH --job-name=fig4_sourcedata
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

# --- Part 1 (Python): spatial correlation source table + cancer-vs-non-cancer Mann-Whitney ---
python src/fig-generation/figure-4-spatial.py

# --- Part 2 (R): CellChat ligand-receptor interaction tables (communication prob + permutation p) ---
# TODO: set the R module you use for CellChat (Rscript must be on PATH). Adjust the name if needed.
module load R 2>/dev/null || true
if command -v Rscript >/dev/null 2>&1; then
    Rscript src/fig-generation/figure-4-cellchat.R
else
    echo "ERROR: Rscript not found. Load your R module and rerun:"
    echo "       Rscript src/fig-generation/figure-4-cellchat.R"
    exit 1
fi

# Re-bundle the Figure 4 workbook so it also includes any R-produced CellChat CSVs.
python -c "import sys; sys.path.insert(0, 'src/fig-generation'); import source_data_common as s; s.aggregate_to_excel(4)"

echo "Figure 4 render complete; source data written to src/fig-generation/source-data/figure_4/"
