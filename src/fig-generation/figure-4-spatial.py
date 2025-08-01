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
                adata = sc.read_visium(sample_path)
                adata.var_names_make_unique()
                adata.var["mt"] = adata.var_names.str.startswith("MT-")
                sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)
                sc.pp.normalize_total(adata, inplace=True)
                sc.pp.log1p(adata)
                sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=2000, inplace=True)
                
                disease_status = get_disease_status(sample_folder)
                adata.obs['disease_status'] = disease_status
                adata.obs['sample_name'] = sample_folder
                adata.uns['disease_status'] = disease_status
                adata.uns['sample_name'] = sample_folder
                
                adatas.append(adata)
                logger.debug(f"Loaded spatial sample {sample_folder} with shape {adata.shape}, disease status: {disease_status}")
    return adatas

def integrate_and_cluster_spatial(adatas, output_path):
    logger.debug("Integrating and clustering spatial data.")
    adatas_cor = scanorama.correct_scanpy(adatas, return_dimred=True)
    adata_spatial = sc.concat(
        adatas_cor, label="library_id", uns_merge="unique", index_unique="-"
    )
    sc.pp.neighbors(adata_spatial, use_rep="X_scanorama")
    sc.tl.umap(adata_spatial)
    sc.tl.leiden(adata_spatial, key_added="clusters", n_iterations=2, flavor="igraph")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    sc.pl.umap(
        adata_spatial,
        color="clusters",
        palette=sc.pl.palettes.default_20,
        ax=axes[0],
        show=False,
        title="Clusters"
    )
    
    sc.pl.umap(
        adata_spatial,
        color="library_id",
        palette=sc.pl.palettes.default_20,
        ax=axes[1],
        show=False,
        title="Library ID"
    )
    
    sc.pl.umap(
        adata_spatial,
        color="disease_status",
        palette={"cancer": "red", "non-cancer": "blue"},
        ax=axes[2],
        show=False,
        title="Disease Status"
    )
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.close()
    logger.debug(f"Spatial clustering plot saved to {output_path}")
    return adata_spatial

def make_custom_cmap(color):
    return mcolors.LinearSegmentedColormap.from_list(
        "", 
        [(0.0, "gray"), (0.5, "gray"), (1.0, color)]
    )

def plot_spatial_gene_sets(adatas, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    gene_sets = [
        ["MIF", "CD74", "CXCR4"],
        ["MIF", "CD74", "CD44"],
        ["ANXA1", "FPR1"],
        ["PPIA", "BSG"]
    ]
    
    gene_colors = {
        "MIF": make_custom_cmap("red"),
        "CD74": make_custom_cmap("blue"),
        "CXCR4": make_custom_cmap("green"),
        "CD44": make_custom_cmap("purple"),
        "ANXA1": make_custom_cmap("orange"),
        "FPR1": make_custom_cmap("magenta"),
        "PPIA": make_custom_cmap("darkblue"),
        "BSG": make_custom_cmap("darkred")
    }

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
                    cmap = gene_colors.get(gene, mcolors.LinearSegmentedColormap.from_list("", ["gray", "gray", "black"]))
                    
                    sc.pl.spatial(
                        adata,
                        color=gene,
                        ax=ax,
                        show=False,
                        size=1.2,
                        cmap=cmap,
                        img_key=None, 
                        na_color='white',
                        alpha_img=0.0 
                    )
                    
                    ax.set_title(f"{gene} - {sample_name} ({disease_status})")
                    plt.tight_layout()
                    
                    gene_output_file = os.path.join(sample_dir, f"{gene}_spatial.png")
                    plt.savefig(gene_output_file, dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    logger.debug(f"Saved spatial plot for {gene} to {gene_output_file}")
            
            else:
                logger.warning(f"Genes not found in {sample_name} ({disease_status}): {', '.join(genes)}")
        
        logger.info(f"Completed spatial gene plots for sample {sample_name} ({disease_status})")

def calculate_spatial_correlations(adatas, gene_sets, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    palette = {'non-cancer': '#84A970', 'cancer': '#FF8C00'}
    
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
            
            anchor_expr = adata[:, anchor_gene].X.toarray().flatten() if hasattr(adata[:, anchor_gene].X, 'toarray') else adata[:, anchor_gene].X.flatten()
            
            for target_gene in target_genes:
                if target_gene not in adata.var_names:
                    logger.warning(f"Target gene {target_gene} missing in {sample_name}. Skipping.")
                    continue
                
                target_expr = adata[:, target_gene].X.toarray().flatten() if hasattr(adata[:, target_gene].X, 'toarray') else adata[:, target_gene].X.flatten()
                corr, pval = pearsonr(anchor_expr, target_expr)
                
                correlation_records.append({
                    'sample_name': sample_name,
                    'disease_status': disease_status,
                    'gene_set': f"Set_{set_idx+1}",
                    'anchor_gene': anchor_gene,
                    'target_gene': target_gene,
                    'correlation': corr
                })
    
    results_df = pd.DataFrame(correlation_records)
    if results_df.empty:
        logger.warning("No correlation data computed.")
        return results_df
    
    results_file = os.path.join(output_dir, "spatial_anchor_correlations.csv")
    results_df.to_csv(results_file, index=False)
    logger.info(f"Saved correlation data to {results_file}")
    
    for gene_set in results_df['gene_set'].unique():
        set_df = results_df[results_df['gene_set'] == gene_set]
        anchor_gene = set_df['anchor_gene'].iloc[0]
        target_genes = set_df['target_gene'].unique()
        n_targets = len(target_genes)
        
        n_cols = min(3, n_targets)
        n_rows = (n_targets + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)
        
        for idx, target_gene in enumerate(target_genes):
            row, col = divmod(idx, n_cols)
            ax = axes[row][col]
            
            pair_df = set_df[set_df['target_gene'] == target_gene]
            
            sns.boxplot(
                data=pair_df,
                x='disease_status',
                y='correlation',
                ax=ax,
                showcaps=True,
                boxprops={'facecolor': 'white', 'edgecolor': 'black'},
                medianprops={'color': 'black'},
                whiskerprops={'color': 'black'},
                capprops={'color': 'black'},
                flierprops={'markeredgecolor': 'black'}
            )
            
            sns.stripplot(
                data=pair_df,
                x='disease_status',
                y='correlation',
                ax=ax,
                dodge=True,
                palette=palette,
                alpha=0.7,
                size=4
            )
            
            ax.set_title(f"{anchor_gene} vs {target_gene}", fontsize=12)
            ax.set_xlabel("")
            ax.set_ylabel("Correlation")
        
        for j in range(idx + 1, n_rows * n_cols):
            row, col = divmod(j, n_cols)
            axes[row][col].axis('off')
        
        plt.suptitle(f"Spatial Correlations - {gene_set} (anchor: {anchor_gene})", fontsize=16, y=1.02)
        plt.tight_layout()
        plot_file = os.path.join(output_dir, f"spatial_correlations_{gene_set}.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved plot for {gene_set} to {plot_file}")
    
    logger.info("\nSpatial Correlation Summary:")
    logger.info("-" * 50)
    for gene_set in results_df['gene_set'].unique():
        set_df = results_df[results_df['gene_set'] == gene_set]
        logger.info(f"\n{gene_set}:")
        for disease in ['cancer', 'non-cancer']:
            disease_df = set_df[set_df['disease_status'] == disease]
            if len(disease_df) > 0:
                mean_corr = disease_df['correlation'].mean()
                std_corr = disease_df['correlation'].std()
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
        ["PPIA", "BSG"]
    ]
    
    correlation_results = calculate_spatial_correlations(adatas, gene_sets, "figures/spatial_correlations")
    
    # adata_spatial = integrate_and_cluster_spatial(adatas, "figures/spatial_library_id.png")
    
    plot_spatial_gene_sets(adatas, "figures/spatial_gene_sets")
    logger.debug("Pipeline completed.")