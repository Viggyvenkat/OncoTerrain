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

# Submit from the repo root:  sbatch src/fig-generation/figure-5-sourcedata.sh
cd "${SLURM_SUBMIT_DIR:-$(pwd)}" || exit 1

source /cache/home/vvv11/miniforge3/etc/profile.d/conda.sh
set +u
conda activate revision
set -u

# --- Part 1 (Python): 5B stage Mann-Whitney (+FDR), 5E polar-bar heights, 5F similarity deltas ---
python src/fig-generation/figure-5.py --source-data

# --- Part 2 (Python): 5D violin source data (per-CAT malignant fraction) ---
# Requires figures/all_oncoterrain/OncoTerrain_annotated.h5ad from a prior OncoTerrain inference run.
python src/fig-generation/violin_plot_5D.py --source-data

# --- Part 3 (R): TCGA PCA / PLS-DA scores + loadings, and CopyKAT classification proportions ---
# TODO: activate the R environment used for these scripts before the Rscript lines, e.g.:
#   module load R/4.3.1            # or
#   conda activate <your-r-env>
Rscript src/fig-generation/tcga-val.R --source-data
Rscript src/fig-generation/copyKAT-val.R --source-data

# Re-bundle the Figure 5 workbook so it also includes the R-produced TCGA/CopyKAT CSVs.
python -c "import sys; sys.path.insert(0, 'src/fig-generation'); import source_data_common as s; s.aggregate_to_excel(5)"

echo "Figure 5 source data written to src/fig-generation/source-data/figure_5/"
