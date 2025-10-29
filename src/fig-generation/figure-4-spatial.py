import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scanorama
import os
import logging
import re
import matplotlib.colors as mcolors
from scipy.stats import pearsonr, mannwhitneyu
from scipy.spatial.distance import pdist, squareform
from itertools import combinations
import seaborn as sns
import squidpy as sq

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("debug_log.txt", mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def get_disease_status(sample_name):
    sample_upper = sample_name.upper()
    
    non_cancer_patterns = ['D1_1', 'D1_2', 'D2_1', 'D2_2']
    if any(pattern in sample_upper for pattern in non_cancer_patterns):
        return 'non-cancer'
    
    if 'B1' in sample_upper or 'B2' in sample_upper:
        return 'non-cancer'
    
    return 'cancer'

def rename_spatial_files(root_dir):
    logger.debug(f"Renaming spatial files in {root_dir}")
    for current_path, dirs, files in os.walk(root_dir, topdown=False):
        for file_name in files:
            if file_name.endswith(".h5"):
                old_file = os.path.join(current_path, file_name)
                new_file = os.path.join(current_path, "filtered_feature_bc_matrix.h5")
                if old_file != new_file and not os.path.exists(new_file):
                    os.rename(old_file, new_file)
                    logger.debug(f"Renamed {old_file} -> {new_file}")

    for current_path, dirs, files in os.walk(root_dir, topdown=False):
        for dir_name in dirs:
            if 'spatial' in dir_name:
                old_dir = os.path.join(current_path, dir_name)
                new_dir = os.path.join(current_path, 'spatial')
                if old_dir != new_dir and not os.path.exists(new_dir):
                    os.rename(old_dir, new_dir)
                    logger.debug(f"Renamed {old_dir} -> {new_dir}")

def load_and_preprocess_spatial_data(root_dir):
    adatas = []
    logger.debug(f"Loading spatial data from {root_dir}")
    for sample_folder in os.listdir(root_dir):
        sample_path = os.path.join(root_dir, sample_folder)
        if os.path.isdir(sample_path):
            spatial_dir = os.path.join(sample_path, "spatial")
            if os.path.exists(spatial_dir):
                # Use scanpy for Visium reading
                adata = sc.read_visium(sample_path)

                adata.var_names_make_unique()
                # mark mitochondrial genes (human convention "MT-")
                adata.var["mt"] = adata.var_names.str.upper().str.startswith("MT-")

                # QC + normalization in scanpy
                sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)
                sc.pp.normalize_total(adata, target_sum=1e4)  # optional target_sum
                sc.pp.log1p(adata)
                sc.pp.highly_variable_genes(
                    adata, flavor="seurat", n_top_genes=2000, inplace=True
                )

                disease_status = get_disease_status(sample_folder)
                adata.obs['disease_status'] = disease_status
                adata.obs['sample_name'] = sample_folder
                adata.uns['disease_status'] = disease_status
                adata.uns['sample_name'] = sample_folder

                adatas.append(adata)
                logger.debug(
                    f"Loaded spatial sample {sample_folder} with shape {adata.shape}, "
                    f"disease status: {disease_status}"
                )
    return adatas

def make_custom_cmap(color):
    return mcolors.LinearSegmentedColormap.from_list(
        "", 
        [(0.0, "gray"), (0.5, "gray"), (1.0, color)]
    )

def plot_spatial_gene_sets(adatas, output_dir, img=False, library_id=None, point_size=1.5):
    os.makedirs(output_dir, exist_ok=True)
    gene_sets = [
        ["MIF", "CD74", "CXCR4"],
        ["MIF", "CD74", "CD44"],
        ["ANXA1", "FPR1"],
        ["PPIA", "BSG"],
        ["FN1", "CD44"],
        ["FN1","ITGA3"],
        ["COL1A2","ITGA3"],
        ["APP", "CD74"],
        ["LAMC1", "ITGA3"],
        ["CD6","ALCAM"]
    ]

    for i, adata in enumerate(adatas):
        sample_name = adata.uns.get("sample_name", f"sample_{i}")
        disease_status = adata.uns.get("disease_status", "unknown")

        sample_dir = os.path.join(output_dir, f"{sample_name}_{disease_status}")
        os.makedirs(sample_dir, exist_ok=True)

        for set_idx, genes in enumerate(gene_sets):
            genes_present = [g for g in genes if g in adata.var_names]

            if genes_present:
                for gene in genes_present:
                    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

                    sq.pl.spatial_scatter(
                        adata,
                        color=gene,
                        library_id=library_id,
                        img=img,
                        ax=ax,
                        size=point_size,
                        na_color="white",
                        cmap="Spectral_r",
                        shape="hex",
                        linewidth=0,  
                    )

                    ax.set_title(f"{gene} - {sample_name} ({disease_status})")
                    plt.tight_layout()

                    gene_output_file = os.path.join(sample_dir, f"{gene}_spatial.png")
                    plt.savefig(gene_output_file, dpi=300, bbox_inches="tight")
                    plt.close()

                    logger.debug(f"Saved spatial plot for {gene} to {gene_output_file}")
            else:
                logger.warning(
                    f"Genes not found in {sample_name} ({disease_status}): {', '.join(genes)}"
                )

        logger.info(
            f"Completed spatial gene plots for sample {sample_name} ({disease_status})"
        )

def calculate_spatial_correlations(adatas, gene_sets, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    # consistent, colorblind-friendly palette you already chose
    palette = {'non-cancer': '#84A970', 'cancer': '#FF8C00'}

    # light, clean base style
    sns.set_theme(style="white", context="talk")  # larger fonts suitable for figures

    correlation_records = []

    for adata in adatas:
        sample_name = adata.uns.get("sample_name", "unknown")
        disease_status = adata.uns.get("disease_status", "unknown")

        logger.debug(f"Processing {sample_name} ({disease_status})")

        for set_idx, genes in enumerate(gene_sets):
            anchor_gene = genes[0]
            target_genes = genes[1:]

            if anchor_gene not in adata.var_names:
                logger.warning(f"Anchor gene {anchor_gene} missing in {sample_name}. Skipping.")
                continue

            anchor_expr = (
                adata[:, anchor_gene].X.toarray().flatten()
                if hasattr(adata[:, anchor_gene].X, "toarray")
                else adata[:, anchor_gene].X.flatten()
            )

            for target_gene in target_genes:
                if target_gene not in adata.var_names:
                    logger.warning(f"Target gene {target_gene} missing in {sample_name}. Skipping.")
                    continue

                target_expr = (
                    adata[:, target_gene].X.toarray().flatten()
                    if hasattr(adata[:, target_gene].X, "toarray")
                    else adata[:, target_gene].X.flatten()
                )
                corr, pval = pearsonr(anchor_expr, target_expr)

                correlation_records.append(
                    {
                        "sample_name": sample_name,
                        "disease_status": disease_status,
                        "gene_set": f"Set_{set_idx+1}",
                        "anchor_gene": anchor_gene,
                        "target_gene": target_gene,
                        "correlation": corr,
                    }
                )

    results_df = pd.DataFrame(correlation_records)
    if results_df.empty:
        logger.warning("No correlation data computed.")
        return results_df

    results_file = os.path.join(output_dir, "spatial_anchor_correlations.csv")
    results_df.to_csv(results_file, index=False)
    logger.info(f"Saved correlation data to {results_file}")

    # ----- Publication-quality violin + box overlays -----
    for gene_set in results_df["gene_set"].unique():
        set_df = results_df[results_df["gene_set"] == gene_set]
        anchor_gene = set_df["anchor_gene"].iloc[0]
        target_genes = set_df["target_gene"].unique()
        n_targets = len(target_genes)

        n_cols = min(3, n_targets)
        n_rows = (n_targets + n_cols - 1) // n_cols

        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(5 * n_cols, 4.2 * n_rows), squeeze=False, layout="constrained"
        )

        # ensure consistent order across panels
        order = ["cancer", "non-cancer"] if set(order := set_df["disease_status"].unique()) == set(["cancer", "non-cancer"]) else sorted(set_df["disease_status"].unique())

        for idx, target_gene in enumerate(target_genes):
            row, col = divmod(idx, n_cols)
            ax = axes[row][col]
            pair_df = set_df[set_df["target_gene"] == target_gene]

        # 1) Violin first (no inner box), with hue=x to satisfy seaborn>=0.14
        sns.violinplot(
            data=pair_df,
            x="disease_status",
            y="correlation",
            order=order,
            hue="disease_status",   # <-- add hue
            palette=palette,
            dodge=False,            # <-- draw one violin per x
            inner=None,             # <-- removes the gray bar
            cut=0,
            linewidth=1.2,
            ax=ax,
        )

        # 2) Slim boxplot on top (no palette => no warning)
        sns.boxplot(
            data=pair_df,
            x="disease_status",
            y="correlation",
            order=order,
            width=0.25,             # fits neatly within violin
            showcaps=True,
            showfliers=False,
            boxprops=dict(facecolor="white", edgecolor="black", linewidth=1.4),
            medianprops=dict(color="black", linewidth=1.6),
            whiskerprops=dict(color="black", linewidth=1.2),
            capprops=dict(color="black", linewidth=1.2),
            ax=ax,
            zorder=3,
        )

        # Cosmetics for a publication-ready panel
        ax.set_title(f"{anchor_gene} vs {target_gene}", fontsize=12)
        ax.set_xlabel("")
        ax.set_ylabel("Correlation")
        ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.4)  # subtle guide lines
        sns.despine(ax=ax)

        # hide any unused panels
        total = n_rows * n_cols
        for j in range(len(target_genes), total):
            r, c = divmod(j, n_cols)
            axes[r][c].axis("off")

        fig.suptitle(f"Spatial Correlations - {gene_set} (anchor: {anchor_gene})", fontsize=16, y=1.02)
        plot_file = os.path.join(output_dir, f"spatial_correlations_{gene_set}.png")
        fig.savefig(plot_file, dpi=300, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"Saved plot for {gene_set} to {plot_file}")

    # ----- Summary stats in logs -----
    logger.info("\nSpatial Correlation Summary:")
    logger.info("-" * 50)
    for gene_set in results_df["gene_set"].unique():
        set_df = results_df[results_df["gene_set"] == gene_set]
        logger.info(f"\n{gene_set}:")
        for disease in ["cancer", "non-cancer"]:
            disease_df = set_df[set_df["disease_status"] == disease]
            if len(disease_df) > 0:
                mean_corr = disease_df["correlation"].mean()
                std_corr = disease_df["correlation"].std()
                logger.info(f"  {disease}: mean={mean_corr:.3f} ± {std_corr:.3f} (n={len(disease_df)})")

    return results_df

def print_sample_metadata(adatas):
    logger.info("Sample Metadata Summary:")
    logger.info("-" * 50)
    
    cancer_count = 0
    non_cancer_count = 0
    
    for adata in adatas:
        sample_name = adata.uns.get("sample_name", "unknown")
        disease_status = adata.uns.get("disease_status", "unknown")
        n_spots = adata.n_obs
        n_genes = adata.n_vars
        
        logger.info(f"Sample: {sample_name}")
        logger.info(f"  Disease Status: {disease_status}")
        logger.info(f"  Spots: {n_spots}")
        logger.info(f"  Genes: {n_genes}")
        logger.info("-" * 30)
        
        if disease_status == "cancer":
            cancer_count += 1
        elif disease_status == "non-cancer":
            non_cancer_count += 1
    
    logger.info(f"Total samples: {len(adatas)}")
    logger.info(f"Cancer samples: {cancer_count}")
    logger.info(f"Non-cancer samples: {non_cancer_count}")

def _bh_fdr(pvals):
    """Benjamini–Hochberg correction. Returns array of adjusted p-values aligned to input."""
    pvals = np.asarray(pvals, dtype=float)
    n = pvals.size
    order = np.argsort(pvals)
    ranked = pvals[order]
    adj = np.empty(n, dtype=float)
    cummin = 1.0
    for i in range(n-1, -1, -1):
        rank = i + 1
        val = ranked[i] * n / rank
        cummin = min(cummin, val)
        adj[i] = cummin
    # re-order to original
    out = np.empty(n, dtype=float)
    out[order] = np.clip(adj, 0, 1)
    return out

def compute_figure_four_G_significance(results_df, output_dir, alpha=0.05):
    """
    For each (gene_set, anchor_gene, target_gene), compare correlation distributions
    between disease_status == 'cancer' vs 'non-cancer' using Mann–Whitney U (two-sided).
    Apply Benjamini–Hochberg FDR across all pairs and write figure_four_G_significance.csv.
    """
    os.makedirs(output_dir, exist_ok=True)
    if results_df is None or results_df.empty:
        raise ValueError("results_df is empty. Run calculate_spatial_correlations first.")

    required_cols = {'gene_set', 'anchor_gene', 'target_gene', 'disease_status', 'correlation'}
    missing = required_cols - set(results_df.columns)
    if missing:
        raise ValueError(f"results_df missing required columns: {missing}")

    # Ensure only the two groups we care about
    df = results_df.copy()
    df = df[df['disease_status'].isin(['cancer', 'non-cancer'])]

    records = []
    for (gset, anchor, target), g in df.groupby(['gene_set', 'anchor_gene', 'target_gene'], dropna=False):
        c_vals = g.loc[g['disease_status'] == 'cancer', 'correlation'].dropna().values
        n_vals = g.loc[g['disease_status'] == 'non-cancer', 'correlation'].dropna().values

        if len(c_vals) == 0 or len(n_vals) == 0:
            stat = np.nan
            pval = np.nan
        else:
            # two-sided test; SciPy auto chooses exact/asymptotic based on sizes
            stat, pval = mannwhitneyu(c_vals, n_vals, alternative='two-sided', method='auto')

        records.append({
            'gene_set': gset,
            'anchor_gene': anchor,
            'target_gene': target,
            'n_cancer': int(len(c_vals)),
            'n_non_cancer': int(len(n_vals)),
            'median_cancer': float(np.median(c_vals)) if len(c_vals) else np.nan,
            'median_non_cancer': float(np.median(n_vals)) if len(n_vals) else np.nan,
            'U_statistic': float(stat) if not np.isnan(stat) else np.nan,
            'p_value': float(pval) if not np.isnan(pval) else np.nan,
        })

    out = pd.DataFrame.from_records(records)

    # FDR across all non-NaN p-values
    mask = out['p_value'].notna().values
    if mask.sum() > 0:
        out.loc[mask, 'p_value_fdr'] = _bh_fdr(out.loc[mask, 'p_value'].values)
    else:
        out['p_value_fdr'] = np.nan

    out['significant_FDR_{}'.format(alpha)] = out['p_value_fdr'] < alpha

    # Sort nicely for readability
    out = out.sort_values(['gene_set', 'anchor_gene', 'target_gene']).reset_index(drop=True)

    # Write CSV
    csv_path = os.path.join(output_dir, "figure_four_G_significance.csv")
    out.to_csv(csv_path, index=False)
    logger.info(f"Saved Mann–Whitney + FDR results to {csv_path}")

    return out

if __name__ == "__main__":
    logger.debug("Starting spatial analytics with disease status annotation.")
    root_dir = "data/spatial-data/Zuanietal"
    rename_spatial_files(root_dir)
    adatas = load_and_preprocess_spatial_data("data/spatial-data")
    
    print_sample_metadata(adatas)
    
    gene_sets = [
        ["MIF", "CD74", "CXCR4"],
        ["MIF", "CD74", "CD44"],
        ["ANXA1", "FPR1"],
        ["PPIA", "BSG"],
        ["FN1", "CD44"],
        ["FN1","ITGA3"],
        ["COL1A2","ITGA3"],
        ["APP", "CD74"],
        ["LAMC1", "ITGA3"],
        ["CD6","ALCAM"]
    ]
    
    correlation_results = calculate_spatial_correlations(adatas, gene_sets, "figures/spatial_correlations")
        
    plot_spatial_gene_sets(adatas, img=False, output_dir="figures/spatial_gene_sets")
    logger.debug("Pipeline completed.")

    sig_df = compute_figure_four_G_significance(correlation_results, output_dir="figures", alpha=0.05)