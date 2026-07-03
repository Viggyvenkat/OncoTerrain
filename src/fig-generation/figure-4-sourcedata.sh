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

# Submit from the repo root:  sbatch src/fig-generation/figure-4-sourcedata.sh
cd "${SLURM_SUBMIT_DIR:-$(pwd)}" || exit 1

source /cache/home/vvv11/miniforge3/etc/profile.d/conda.sh
set +u
conda activate revision
set -u

# --- Part 1 (Python): spatial correlation source table + cancer-vs-non-cancer Mann-Whitney ---
python src/fig-generation/figure-4-spatial.py --source-data

# --- Part 2 (R): CellChat ligand-receptor interaction tables (communication prob + permutation p) ---
# TODO: activate the R environment used for CellChat before this line, e.g.:
#   module load R/4.3.1            # or
#   conda activate <your-cellchat-r-env>
Rscript src/fig-generation/figure-4-cellchat.R --source-data

# Re-bundle the Figure 4 workbook so it also includes the R-produced CellChat CSVs.
python -c "import sys; sys.path.insert(0, 'src/fig-generation'); import source_data_common as s; s.aggregate_to_excel(4)"

echo "Figure 4 source data written to src/fig-generation/source-data/figure_4/"
