import scanpy as sc
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import logging
import pandas as pd
from py_monocle import (learn_graph, order_cells, compute_cell_states, regression_analysis, 
                         differential_expression_genes,)
from matplotlib.patches import Patch
import matplotlib.cm as cm
from matplotlib.lines import Line2D
import sys
from scipy.stats import mannwhitneyu
import itertools
from statsmodels.stats.multitest import multipletests
import scipy.sparse as sps
import matplotlib.patches as mpatches

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s:%(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout) 
    ]
)

EPITHELIAL_CATS = [
    'AT2', 'AT1', 'Suprabasal', 'Basal resting',
    'Multiciliated (non-nasal)', 'Goblet (nasal)',
    'Club (nasal)', 'Ciliated (nasal)',
    'Club (non-nasal)', 'Multiciliated (nasal)',
    'Goblet (bronchial)', 'Transitional Club AT2',
    'AT2 proliferating', 'Goblet (subsegmental)'
]

TARGET_PROJECTS = [
    'SD2_raw_feature_bc_matrix','SD3_raw_feature_bc_matrix','SD6_raw_features_bc_matrix',
    'SD7_raw_features_bc_matrix','SD8_raw_features_bc_matrix','SD10_raw_features_bc_matrix',
    'SD12_raw_features_bc_matrix','SD15_raw_features_bc_matrix','SD16_raw_features_bc_matrix'
]

_target_cmap = plt.get_cmap('tab20')
TARGET_COLORS = {s: _target_cmap(i % 20) for i, s in enumerate(TARGET_PROJECTS)}

def __figure_three_A(adata, save_path=None):
    orig = adata.obs['leiden_res_20.00_celltype'].cat.add_categories(['Other'])
    
    adata.obs['highlight'] = (
        orig
        .where(~orig.isin(EPITHELIAL_CATS), other='Other')
        .cat
        .remove_unused_categories()
    )

    non_epi = [cat for cat in adata.obs['highlight'].cat.categories if cat != 'Other']

    base_pal = sc.pl.palettes.default_20
    if len(non_epi) > len(base_pal):
        palette_vals = plt.cm.tab20.colors * ((len(non_epi) // len(base_pal)) + 1)
    else:
        palette_vals = base_pal

    palette = {cat: palette_vals[i] for i, cat in enumerate(non_epi)}
    palette['Other'] = '#A9A9A9'
    
    plt.figure(figsize=(15, 8))
    sc.pl.umap(
        adata,
        color='highlight',
        palette=palette,
        size=0.2,
        legend_loc='right margin',
        title='Non-epithelial cell types (colored) vs epithelial (gray)',
        show=False
    )
    plt.tight_layout(rect=[0, 0, 0.8, 1]) 

    if save_path is None:
        figures_dir = Path.cwd() / 'figures'
        figures_dir.mkdir(parents=True, exist_ok=True)
        save_path = figures_dir / 'figure_3A.png'
    else:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.3)
    plt.close()

def __figure_three_C(adata, save_path=None):
    logging.debug("=== START __figure_three_C ===")
    t_cells = adata[
        adata.obs['leiden_res_20.00_celltype']
        .isin(['CD8 T cells', 'CD4 T cells', 'T cells proliferating', 'NK cells'])
    ].copy()
    logging.debug(f"Subset to T cells: {t_cells.n_obs} cells")

    t_cells.obs['tumor_stage'] = t_cells.obs['tumor_stage'].replace(['nan', 'NaN', 'NAN'], np.nan)

    dist_pre = t_cells.obs['tumor_stage'].value_counts(dropna=False)
    logging.debug(f"tumor_stage distribution before drop (incl NaN/strings):\n{dist_pre}")

    mask = t_cells.obs['tumor_stage'].notna()
    n_missing = (~mask).sum()
    logging.debug(f"Found {n_missing} cells with missing or string 'nan' tumor_stage")
    if n_missing > 0:
        logging.debug(f"Dropping {n_missing} cells with missing tumor_stage")
        t_cells = t_cells[mask, :].copy()

    dist_post = t_cells.obs['tumor_stage'].value_counts(dropna=False)
    logging.debug(f"tumor_stage distribution after drop (incl NaN):\n{dist_post}")

    sc.pp.highly_variable_genes(
        t_cells,
        n_top_genes=5000,
        flavor='seurat_v3',
        subset=False,
        n_bins=20
    )
    t_cells = t_cells[:, t_cells.var['highly_variable']].copy()
    logging.debug(f"After HVG subset: {t_cells.n_vars} genes")

    umap = t_cells.obsm['X_umap']

    if 'leiden' not in t_cells.obs:
        sc.tl.leiden(t_cells, key_added='leiden')
    clusters = pd.to_numeric(t_cells.obs['leiden'], errors='coerce').fillna(-1).astype(int).values

    projected_pts, mst, centroids = learn_graph(matrix=umap, clusters=clusters)
    pseudotime = order_cells(
        umap,
        centroids,
        mst=mst,
        projected_points=projected_pts,
        root_cells=0
    )
    t_cells.obs['pseudotime'] = pseudotime
    logging.debug("Pseudotime computed")

    cmap_gray_purple = LinearSegmentedColormap.from_list(
        'gray_purple',
        ['#A9A9A9', 'purple']
    )

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        umap[:, 0],
        umap[:, 1],
        c=pseudotime,
        s=0.1,
        cmap=cmap_gray_purple,
        rasterized=True
    )

    df_umap = pd.DataFrame(umap, columns=['UMAP1', 'UMAP2'], index=t_cells.obs_names)
    df_umap['sample'] = t_cells.obs['sample'].values
    df_umap['tumor_stage'] = t_cells.obs['tumor_stage'].values
    df_umap = df_umap.dropna(subset=['tumor_stage'])

    def mode_or_first(x):
        m = x.mode()
        return m.iloc[0] if len(m) > 0 else x.iloc[0]

    centroid_df = (
        df_umap
        .groupby('sample')
        .agg({'UMAP1': 'mean', 'UMAP2': 'mean', 'tumor_stage': mode_or_first})
        .reset_index()
    )

    stage_colors = {
        'non-cancer': '#84A970',
        'early':      '#E4C282',
        'advanced':   '#FF8C00'
    }

    ax = plt.gca()
    target_handles = []
    seen_targets = set()

    for _, row in centroid_df.iterrows():
        sname = row['sample']
        stg = row['tumor_stage']

        if sname in TARGET_PROJECTS:
            color = TARGET_COLORS[sname]
            ax.scatter(
                row['UMAP1'], row['UMAP2'],
                c=[color],
                s=90, marker='*', 
                edgecolors='black', linewidths=0.4,
                zorder=12
            )
            if sname not in seen_targets:
                target_handles.append(
                    Line2D([0], [0], marker='*', linestyle='',
                           markersize=10, markeredgewidth=0.8,
                           markeredgecolor='black', color=color, label=sname)
                )
                seen_targets.add(sname)
        else:
            ax.scatter(
                row['UMAP1'], row['UMAP2'],
                c=[stage_colors.get(stg, 'black')],
                s=30, marker='o',
                edgecolors='none',
                zorder=10
            )

    # Stage legend (small, inside)
    stage_handles = [
        mpatches.Patch(color=stage_colors['non-cancer'], label='non-cancer'),
        mpatches.Patch(color=stage_colors['early'], label='early'),
        mpatches.Patch(color=stage_colors['advanced'], label='advanced'),
    ]
    stage_legend = ax.legend(handles=stage_handles, title='Tumor stage',
                             loc='lower left', frameon=False)
    ax.add_artist(stage_legend)

    # Target legend (outside, right)
    if target_handles:
        ax.legend(handles=target_handles, title='TARGET PROJECTS',
                  loc='upper left', bbox_to_anchor=(1.02, 1),
                  borderaxespad=0., frameon=False)


    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.debug(f"Saved trajectory plot to {save_path}")
    logging.debug("=== END __figure_three_C ===")

    return t_cells

def __figure_three_D(adata, save_path=None):
    logging.debug("=== START __figure_three_E ===")
    mphage_cells = adata[
        adata.obs['leiden_res_20.00_celltype']
        .isin([
            'Alveolar macrophages',
            'Alveolar Mφ MT-positive',
            'Alveolar Mφ proliferating',
            'Interstitial Mφ perivascular',
            'Alveolar Mφ CCL3+',
            'Monocyte derived Mφ',
            'Mast cells',
            'Plasmacytoid DCs',
            'DC2',
            'Migratory DCs',
            'Classical Monocytes',
            'Non classical monocytes'
        ])
    ].copy()
    logging.debug(f"Subset to macrophages: {mphage_cells.n_obs} cells")

    mphage_cells.obs['tumor_stage'] = mphage_cells.obs['tumor_stage'].replace(['nan', 'NaN', 'NAN'], np.nan)

    dist_pre = mphage_cells.obs['tumor_stage'].value_counts(dropna=False)
    logging.debug(f"tumor_stage distribution before drop (incl NaN/strings):\n{dist_pre}")

    mask = mphage_cells.obs['tumor_stage'].notna()
    n_missing = (~mask).sum()
    logging.debug(f"Found {n_missing} cells with missing or string 'nan' tumor_stage")
    if n_missing > 0:
        logging.debug(f"Dropping {n_missing} cells with missing tumor_stage")
        mphage_cells = mphage_cells[mask, :].copy()

    dist_post = mphage_cells.obs['tumor_stage'].value_counts(dropna=False)
    logging.debug(f"tumor_stage distribution after drop (incl NaN):\n{dist_post}")

    sc.pp.highly_variable_genes(
        mphage_cells,
        n_top_genes=5000,
        flavor='seurat_v3',
        subset=False,
        n_bins=20
    )
    mphage_cells = mphage_cells[:, mphage_cells.var['highly_variable']].copy()
    logging.debug(f"After HVG subset: {mphage_cells.n_vars} genes")

    umap = mphage_cells.obsm['X_umap']

    if 'leiden' not in mphage_cells.obs:
        sc.tl.leiden(mphage_cells, key_added='leiden')
    clusters = pd.to_numeric(
        mphage_cells.obs['leiden'],
        errors='coerce'
    ).fillna(-1).astype(int).values

    projected_pts, mst, centroids = learn_graph(matrix=umap, clusters=clusters)
    pseudotime = order_cells(
        umap,
        centroids,
        mst=mst,
        projected_points=projected_pts,
        root_cells=0
    )
    mphage_cells.obs['pseudotime'] = pseudotime
    logging.debug("Pseudotime computed")

    cmap_gray_purple = LinearSegmentedColormap.from_list(
        'gray_purple',
        ['#A9A9A9', 'purple']
    )

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        umap[:, 0],
        umap[:, 1],
        c=pseudotime,
        s=0.1,
        cmap=cmap_gray_purple,
        rasterized=True
    )

    df_umap = pd.DataFrame(umap, columns=['UMAP1', 'UMAP2'], index=mphage_cells.obs_names)
    df_umap['sample'] = mphage_cells.obs['sample'].values
    df_umap['tumor_stage'] = mphage_cells.obs['tumor_stage'].values
    df_umap = df_umap.dropna(subset=['tumor_stage'])

    def mode_or_first(x):
        m = x.mode()
        return m.iloc[0] if len(m) > 0 else x.iloc[0]

    centroid_df = (
        df_umap
        .groupby('sample')
        .agg({
            'UMAP1': 'mean',
            'UMAP2': 'mean',
            'tumor_stage': mode_or_first
        })
        .reset_index()
    )

    stage_colors = {
        'non-cancer': '#84A970',
        'early':      '#E4C282',
        'advanced':   '#FF8C00'
    }

    ax = plt.gca()
    target_handles = []
    seen_targets = set()

    for _, row in centroid_df.iterrows():
        sname = row['sample']
        stg = row['tumor_stage']

        if sname in TARGET_PROJECTS:
            color = TARGET_COLORS[sname]
            ax.scatter(
                row['UMAP1'], row['UMAP2'],
                c=[color],
                s=90, marker='*', 
                edgecolors='black', linewidths=0.4,
                zorder=12
            )
            if sname not in seen_targets:
                target_handles.append(
                    Line2D([0], [0], marker='*', linestyle='',
                           markersize=10, markeredgewidth=0.8,
                           markeredgecolor='black', color=color, label=sname)
                )
                seen_targets.add(sname)
        else:
            ax.scatter(
                row['UMAP1'], row['UMAP2'],
                c=[stage_colors.get(stg, 'black')],
                s=30, marker='o',
                edgecolors='none',
                zorder=10
            )

    # Stage legend (small, inside)
    stage_handles = [
        mpatches.Patch(color=stage_colors['non-cancer'], label='non-cancer'),
        mpatches.Patch(color=stage_colors['early'], label='early'),
        mpatches.Patch(color=stage_colors['advanced'], label='advanced'),
    ]
    stage_legend = ax.legend(handles=stage_handles, title='Tumor stage',
                             loc='lower left', frameon=False)
    ax.add_artist(stage_legend)

    # Target legend (outside, right)
    if target_handles:
        ax.legend(handles=target_handles, title='TARGET PROJECTS',
                  loc='upper left', bbox_to_anchor=(1.02, 1),
                  borderaxespad=0., frameon=False)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.debug(f"Saved trajectory plot to {save_path}")
    logging.debug("=== END __figure_three_E ===")

    return mphage_cells

def __figure_three_E(adata, save_path=None):
    logging.debug("=== START __figure_three_G ===")
    fibroblast_cells = adata[
        adata.obs['leiden_res_20.00_celltype']
        .isin([
            "Peribronchial fibroblasts",
            "Adventitial fibroblasts",
            "Alveolar fibroblasts",
            "Subpleural fibroblasts",
            "Myofibroblasts",
            "Fibromyocytes"
        ])
    ].copy()
    logging.debug(f"Subset to fibroblasts: {fibroblast_cells.n_obs} cells")

    fibroblast_cells.obs['tumor_stage'] = fibroblast_cells.obs['tumor_stage'].replace(['nan', 'NaN', 'NAN'], np.nan)
    dist_pre = fibroblast_cells.obs['tumor_stage'].value_counts(dropna=False)
    logging.debug(f"tumor_stage distribution before drop (incl NaN/strings):\n{dist_pre}")
    mask = fibroblast_cells.obs['tumor_stage'].notna()
    n_missing = (~mask).sum()
    logging.debug(f"Found {n_missing} cells with missing or string 'nan' tumor_stage")
    if n_missing > 0:
        logging.debug(f"Dropping {n_missing} cells with missing tumor_stage")
        fibroblast_cells = fibroblast_cells[mask, :].copy()
    dist_post = fibroblast_cells.obs['tumor_stage'].value_counts(dropna=False)
    logging.debug(f"tumor_stage distribution after drop (incl NaN):\n{dist_post}")

    sc.pp.highly_variable_genes(
        fibroblast_cells,
        n_top_genes=5000,
        flavor='seurat_v3',
        subset=False,
        n_bins=20
    )
    fibroblast_cells = fibroblast_cells[:, fibroblast_cells.var['highly_variable']].copy()
    logging.debug(f"After HVG subset: {fibroblast_cells.n_vars} genes")

    umap = fibroblast_cells.obsm['X_umap']

    if 'leiden' not in fibroblast_cells.obs:
        sc.tl.leiden(fibroblast_cells, key_added='leiden')
    clusters = pd.to_numeric(
        fibroblast_cells.obs['leiden'],
        errors='coerce'
    ).fillna(-1).astype(int).values

    projected_pts, mst, centroids = learn_graph(matrix=umap, clusters=clusters)
    pseudotime = order_cells(
        umap,
        centroids,
        mst=mst,
        projected_points=projected_pts,
        root_cells=0
    )
    fibroblast_cells.obs['pseudotime'] = pseudotime
    logging.debug("Pseudotime computed")

    cmap_gray_purple = LinearSegmentedColormap.from_list(
        'gray_purple',
        ['#A9A9A9', 'purple']
    )

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        umap[:, 0],
        umap[:, 1],
        c=pseudotime,
        s=0.5,
        cmap=cmap_gray_purple,
        rasterized=True
    )

    df_umap = pd.DataFrame(umap, columns=['UMAP1', 'UMAP2'], index=fibroblast_cells.obs_names)
    df_umap['sample'] = fibroblast_cells.obs['sample'].values
    df_umap['tumor_stage'] = fibroblast_cells.obs['tumor_stage'].values
    df_umap = df_umap.dropna(subset=['tumor_stage'])

    def mode_or_first(x):
        m = x.mode()
        return m.iloc[0] if len(m) > 0 else x.iloc[0]

    centroid_df = (
        df_umap
        .groupby('sample')
        .agg({
            'UMAP1': 'mean',
            'UMAP2': 'mean',
            'tumor_stage': mode_or_first
        })
        .reset_index()
    )

    stage_colors = {
        'non-cancer': '#84A970',
        'early':      '#E4C282',
        'advanced':   '#FF8C00'
    }

    ax = plt.gca()
    target_handles = []
    seen_targets = set()

    for _, row in centroid_df.iterrows():
        sname = row['sample']
        stg = row['tumor_stage']

        if sname in TARGET_PROJECTS:
            color = TARGET_COLORS[sname]
            ax.scatter(
                row['UMAP1'], row['UMAP2'],
                c=[color],
                s=90, marker='*', 
                edgecolors='black', linewidths=0.4,
                zorder=12
            )
            if sname not in seen_targets:
                target_handles.append(
                    Line2D([0], [0], marker='*', linestyle='',
                           markersize=10, markeredgewidth=0.8,
                           markeredgecolor='black', color=color, label=sname)
                )
                seen_targets.add(sname)
        else:
            ax.scatter(
                row['UMAP1'], row['UMAP2'],
                c=[stage_colors.get(stg, 'black')],
                s=30, marker='o',
                edgecolors='none',
                zorder=10
            )

    # Stage legend (small, inside)
    stage_handles = [
        mpatches.Patch(color=stage_colors['non-cancer'], label='non-cancer'),
        mpatches.Patch(color=stage_colors['early'], label='early'),
        mpatches.Patch(color=stage_colors['advanced'], label='advanced'),
    ]
    stage_legend = ax.legend(handles=stage_handles, title='Tumor stage',
                             loc='lower left', frameon=False)
    ax.add_artist(stage_legend)

    # Target legend (outside, right)
    if target_handles:
        ax.legend(handles=target_handles, title='TARGET PROJECTS',
                  loc='upper left', bbox_to_anchor=(1.02, 1),
                  borderaxespad=0., frameon=False)
        
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.debug(f"Saved trajectory plot to {save_path}")
    logging.debug("=== END __figure_three_G ===")

    return fibroblast_cells

def __figure_three_F_1(adata, save_dir=None):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    t_cells = adata[
        adata.obs['leiden_res_20.00_celltype']
        .isin(['CD8 T cells', 'CD4 T cells', 'T cells proliferating'])
    ].copy()

    cmap_custom = LinearSegmentedColormap.from_list('gray_emerald', ['#A9A9A9', '#50C878'])

    genes = [
        'CD3D', 'CD3E', 'CD3G',
        'CD4', 'CD8A', 'CD8B',
        'FOXP3',
        'STAT4', 'TBX21', 'GATA3', 'RORC',
        'IL2RA', 'IL7R',
        'PDCD1', 'CTLA4', 'LAG3', 'TIGIT',
        'CD69', 'CD44', 'CCR7'
    ]

    umap = t_cells.obsm['X_umap']
    x, y = umap[:, 0], umap[:, 1]
    x_c, y_c = x.mean(), y.mean()
    zoom_frac = 0.8
    half_x = (x.max() - x.min()) * zoom_frac / 2
    half_y = (y.max() - y.min()) * zoom_frac / 2

    for gene in genes:
        if gene not in t_cells.var_names:
            print(f"[WARN] Gene '{gene}' not found, skipping.")
            continue

        expr = t_cells[:, gene].X
        expr_arr = expr.toarray().flatten() if hasattr(expr, 'toarray') else np.array(expr).flatten()
        sorted_idx = np.argsort(expr_arr)
        t_sorted = t_cells[sorted_idx].copy()

        fig, ax = plt.subplots(figsize=(6, 6))
        sc.pl.umap(
            t_sorted,
            color=gene,
            size=0.5,
            sort_order=False,
            cmap=cmap_custom,
            ax=ax,
            show=False,
            colorbar_loc=None
        )
        ax.set_xlim(x_c - half_x, x_c + half_x)
        ax.set_ylim(y_c - half_y, y_c + half_y)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel('X1')
        ax.set_ylabel('X2')
        ax.set_title(f'{gene} Expression')
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
        ax.tick_params(width=1.5)

        sm = plt.cm.ScalarMappable(cmap=cmap_custom)
        sm.set_array(expr_arr)
        sm.set_clim(expr_arr.min(), expr_arr.max())
        cbar = fig.colorbar(sm, ax=ax, fraction=0.02, pad=0.04)
        cbar.set_label('Expression')

        fig.savefig(save_dir / f"{gene}.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
    
def __figure_three_F_2(adata, save_dir=None):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    mphage_cells = adata[
        adata.obs['leiden_res_20.00_celltype']
        .isin([
            'Alveolar macrophages', 'Alveolar Mφ MT-positive',
            'Alveolar Mφ proliferating', 'Interstitial Mφ perivascular',
            'Alveolar Mφ CCL3+'
        ])
    ].copy()

    cmap_custom = LinearSegmentedColormap.from_list('gray_emerald', ['#A9A9A9', '#50C878'])

    genes = [
        'CD68', 'CD163', 'CD86',
        'MRC1', 'MARCO',
        'CSF1R', 'ITGAM',
        'CCL3', 'CCL4', 'CXCL10',
        'S100A8', 'S100A9',
        'FCGR3A', 'CD14',
        'IL1B', 'TNF', 'IL10',
        'MT1', 'MT2A'
    ]

    umap = mphage_cells.obsm['X_umap']
    x, y = umap[:, 0], umap[:, 1]
    x_c, y_c = x.mean(), y.mean()
    zoom_frac = 0.8
    half_x = (x.max() - x.min()) * zoom_frac / 2
    half_y = (y.max() - y.min()) * zoom_frac / 2

    for gene in genes:
        if gene not in mphage_cells.var_names:
            print(f"[WARN] Gene '{gene}' not found, skipping.")
            continue

        expr = mphage_cells[:, gene].X
        expr_arr = expr.toarray().flatten() if hasattr(expr, 'toarray') else np.array(expr).flatten()
        sorted_idx = np.argsort(expr_arr)
        t_sorted = mphage_cells[sorted_idx].copy()

        fig, ax = plt.subplots(figsize=(6, 6))
        sc.pl.umap(
            t_sorted,
            color=gene,
            size=0.5,
            sort_order=False,
            cmap=cmap_custom,
            ax=ax,
            show=False,
            colorbar_loc=None
        )
        ax.set_xlim(x_c - half_x, x_c + half_x)
        ax.set_ylim(y_c - half_y, y_c + half_y)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel('X1')
        ax.set_ylabel('X2')
        ax.set_title(f'{gene} Expression')
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
        ax.tick_params(width=1.5)

        sm = plt.cm.ScalarMappable(cmap=cmap_custom)
        sm.set_array(expr_arr)
        sm.set_clim(expr_arr.min(), expr_arr.max())
        cbar = fig.colorbar(sm, ax=ax, fraction=0.02, pad=0.04)
        cbar.set_label('Expression')

        fig.savefig(save_dir / f"{gene}.png", dpi=300, bbox_inches='tight')
        plt.close(fig)

def __figure_three_F_3(adata, save_dir=None):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    fibroblast_cells = adata[
        adata.obs['leiden_res_20.00_celltype']
        .isin([
            "Peribronchial fibroblasts",
            "Adventitial fibroblasts",
            "Alveolar fibroblasts",
            "Subpleural fibroblasts",
            "Myofibroblasts",
            "Fibromyocytes"
        ])
    ].copy()

    cmap_custom = LinearSegmentedColormap.from_list('gray_emerald', ['#A9A9A9', '#50C878'])

    genes = [
        'COL1A1', 'COL3A1', 'COL5A1',
        'ACTA2', 'TAGLN',
        'FAP', 'PDGFRA',
        'DCN', 'LUM',
        'MMP2', 'MMP9', 'TIMP1',
        'FN1', 'THY1',
        'POSTN', 'SPARC', 'VCAN',
        'S100A4',
        'PDPN'
    ]

    umap = fibroblast_cells.obsm['X_umap']
    x, y = umap[:, 0], umap[:, 1]
    x_c, y_c = x.mean(), y.mean()
    zoom_frac = 0.8
    half_x = (x.max() - x.min()) * zoom_frac / 2
    half_y = (y.max() - y.min()) * zoom_frac / 2

    for gene in genes:
        if gene not in fibroblast_cells.var_names:
            print(f"[WARN] Gene '{gene}' not found, skipping.")
            continue

        expr = fibroblast_cells[:, gene].X
        expr_arr = expr.toarray().flatten() if hasattr(expr, 'toarray') else np.array(expr).flatten()
        sorted_idx = np.argsort(expr_arr)
        t_sorted = fibroblast_cells[sorted_idx].copy()

        fig, ax = plt.subplots(figsize=(6, 6))
        sc.pl.umap(
            t_sorted,
            color=gene,
            size=0.5,
            sort_order=False,
            cmap=cmap_custom,
            ax=ax,
            show=False,
            colorbar_loc=None
        )
        ax.set_xlim(x_c - half_x, x_c + half_x)
        ax.set_ylim(y_c - half_y, y_c + half_y)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel('X1')
        ax.set_ylabel('X2')
        ax.set_title(f'{gene} Expression')
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
        ax.tick_params(width=1.5)

        sm = plt.cm.ScalarMappable(cmap=cmap_custom)
        sm.set_array(expr_arr)
        sm.set_clim(expr_arr.min(), expr_arr.max())
        cbar = fig.colorbar(sm, ax=ax, fraction=0.02, pad=0.04)
        cbar.set_label('Expression')

        fig.savefig(save_dir / f"{gene}.png", dpi=300, bbox_inches='tight')
        plt.close(fig)


def __figure_three_G(adata, save_path=None):
    # Filter for T cells
    t_cells = adata[
        adata.obs['leiden_res_20.00_celltype']
        .isin(['CD8 T cells', 'CD4 T cells', 'T cells proliferating'])
    ].copy()

    stages = ['non-cancer', 'early', 'advanced']
    stage_colors = {
        'non-cancer': '#84A970',
        'early':      '#E4C282',
        'advanced':   '#FF8C00'
    }

    df = (
        t_cells.obs
        .loc[:, ['tumor_stage', 'HALLMARK_IL6_JAK_STAT3_SIGNALING']]
        .dropna(subset=['tumor_stage'])
    )

    data_by_stage = [
        df.loc[df['tumor_stage'] == st, 'HALLMARK_IL6_JAK_STAT3_SIGNALING'].values
        for st in stages
    ]

    # Pairwise Mann–Whitney U tests with Bonferroni correction
    pairs = list(itertools.combinations(range(len(stages)), 2))
    results = []
    for (i, j) in pairs:
        stat, p = mannwhitneyu(data_by_stage[i], data_by_stage[j], alternative='two-sided')
        p_adj = p * len(pairs)  # Bonferroni correction
        results.append({
            "group1": stages[i],
            "group2": stages[j],
            "U_statistic": stat,
            "p_value_raw": p,
            "p_value_adj": p_adj
        })

    # Save results to /figures/
    results_df = pd.DataFrame(results)
    figures_dir = Path("figures")
    figures_dir.mkdir(parents=True, exist_ok=True)
    results_csv_path = figures_dir / "figure_three_G_significance.csv"
    results_df.to_csv(results_csv_path, index=False)
    print(f"Significance results saved to: {results_csv_path}")

    # Plot boxplots (no annotations)
    fig, ax = plt.subplots(figsize=(6, 6))
    bp = ax.boxplot(
        data_by_stage,
        patch_artist=True,
        widths=0.6,
        medianprops=dict(linewidth=1.5),
        showfliers=False
    )

    for patch, st in zip(bp['boxes'], stages):
        patch.set_facecolor(stage_colors[st])
        patch.set_edgecolor('black')

    ax.set_xticks(range(1, len(stages) + 1))
    ax.set_xticklabels(stages, rotation=30)
    ax.set_ylabel('HALLMARK_IL6_JAK_STAT score')
    ax.set_title('IL6/JAK/STAT pathway activity by tumor stage')

    legend_handles = [Patch(facecolor=stage_colors[st], label=st) for st in stages]
    ax.legend(handles=legend_handles, title='Tumor stage', loc='upper right', frameon=False)

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

def __figure_three_H(adata, save_path = None):
    fibroblast_cells = adata[
        adata.obs['leiden_res_20.00_celltype']
        .isin([
            "Peribronchial fibroblasts",
            "Adventitial fibroblasts",
            "Alveolar fibroblasts",
            "Subpleural fibroblasts",
            "Myofibroblasts",
            "Fibromyocytes"
        ])
    ].copy()

    stages = ['non-cancer', 'early', 'advanced']
    stage_colors = {
        'non-cancer': '#84A970',
        'early':      '#E4C282',
        'advanced':   '#FF8C00'
    }

    fap_raw = fibroblast_cells[:, 'FAP'].X
    try:
        fap_vals = fap_raw.toarray().flatten()
    except AttributeError:
        fap_vals = np.array(fap_raw).flatten()

    df = pd.DataFrame({
        'FAP': fap_vals,
        'tumor_stage': fibroblast_cells.obs['tumor_stage'].values
    }).dropna(subset=['tumor_stage'])

    pct_expressing = []
    for st in stages:
        sub = df[df['tumor_stage'] == st]
        if len(sub) == 0:
            pct = 0.0
        else:
            pct = (sub['FAP'] > 0).sum() / len(sub) * 100
        pct_expressing.append(pct)

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(
        stages,
        pct_expressing,
        color=[stage_colors[st] for st in stages],
        edgecolor='black'
    )
    ax.set_ylabel('Percent FAP+ fibroblasts')
    ax.set_xlabel('Tumor stage')
    ax.set_title('FAP expression in fibroblasts by tumor stage')
    ax.set_ylim(0, 100)
    for bar, pct in zip(bars, pct_expressing):
        ax.text(
            bar.get_x() + bar.get_width()/2,
            pct + 2,
            f"{pct:.1f}%",
            ha='center',
            va='bottom'
        )

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

def __figure_three_I(adata, save_path=None):
    t_cells = adata[
        adata.obs['leiden_res_20.00_celltype']
        .isin(['CD8 T cells', 'CD4 T cells', 'T cells proliferating'])
    ].copy()
    t_cells.obs['tumor_stage'] = t_cells.obs['tumor_stage'].replace(['nan', 'NaN', 'NAN'], np.nan)
    t_cells = t_cells[~t_cells.obs['tumor_stage'].isna()].copy()

    genes = ['STAT4', 'CCR7', 'LAG3']
    for gene in genes:
        if gene not in t_cells.var_names:
            raise ValueError(f"Gene '{gene}' not found in adata.")

    expr_data = {
        gene: t_cells[:, gene].X.toarray().flatten()
        if hasattr(t_cells[:, gene].X, 'toarray') else np.array(t_cells[:, gene].X).flatten()
        for gene in genes
    }
    medians = {gene: np.median(expr_data[gene]) for gene in genes}

    def get_phenotype(i):
        return ''.join([
            f"{g}{'+' if expr_data[g][i] > medians[g] else '-'}"
            for g in genes
        ])

    t_cells.obs['phenotype'] = [get_phenotype(i) for i in range(t_cells.n_obs)]

    stages = ['non-cancer', 'early', 'advanced']
    all_phenos = sorted(t_cells.obs['phenotype'].unique())
    stage_pct_df = pd.DataFrame(index=all_phenos, columns=stages)

    for stage in stages:
        sub = t_cells.obs[t_cells.obs['tumor_stage'] == stage]
        total = len(sub)
        counts = sub['phenotype'].value_counts()
        for pheno in all_phenos:
            stage_pct_df.loc[pheno, stage] = (counts.get(pheno, 0) / total) * 100 if total > 0 else 0

    stage_pct_df = stage_pct_df.fillna(0)
    logging.info("Phenotype breakdown for Figure 3I:\n%s", stage_pct_df.applymap(lambda x: f"{x:.1f}%").to_string())


    cmap = cm.get_cmap('tab20', len(all_phenos))
    colors = [cmap(i) for i in range(len(all_phenos))]

    fig, ax = plt.subplots(figsize=(8, 6))
    bottom = np.zeros(len(stages))

    for i, pheno in enumerate(all_phenos):
        vals = stage_pct_df.loc[pheno].values.astype(float)
        ax.bar(
            stages,
            vals,
            bottom=bottom,
            color=colors[i],
            edgecolor='black',
            linewidth=0.5,
            label=pheno
        )
        bottom += vals

    ax.set_ylabel('Percentage of T cells')
    ax.set_title('T-cell phenotypes by tumor stage\n(> median expression of STAT4, CCR7, LAG3)')
    ax.set_ylim(0, 100)

    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(axis='both', which='both', length=0)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)

    legend_handles = [
        Line2D(
            [0], [0],
            marker='o',
            color='none',
            markerfacecolor=colors[i],
            markeredgecolor='black',
            markeredgewidth=0.5,
            markersize=8,
            label=pheno
        )
        for i, pheno in enumerate(all_phenos)
    ]

    ax.legend(
        handles=legend_handles,
        title='Phenotype',
        bbox_to_anchor=(1.02, 1),
        loc='upper left',
        frameon=False,
        borderaxespad=0.
    )

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

def __figure_three_J(adata, save_path=None):

    myeloid = adata[
        adata.obs['leiden_res_20.00_celltype']
        .isin([
            'Alveolar macrophages',
            'Alveolar Mφ MT-positive',
            'Alveolar Mφ proliferating',
            'Interstitial Mφ perivascular',
            'Alveolar Mφ CCL3+',
            'Monocyte derived Mφ',
            'Mast cells',
            'Plasmacytoid DCs',
            'DC2',
            'Migratory DCs',
            'Classical Monocytes',
            'Non classical monocytes'
        ])
    ].copy()

    myeloid.obs['tumor_stage'] = myeloid.obs['tumor_stage'].replace(['nan', 'NaN', 'NAN'], np.nan)
    myeloid = myeloid[~myeloid.obs['tumor_stage'].isna()].copy()

    genes = ['CD68', 'CD86', 'CD163']
    for gene in genes:
        if gene not in myeloid.var_names:
            raise ValueError(f"Gene '{gene}' not found in adata.")

    expr_data = {
        gene: myeloid[:, gene].X.toarray().flatten()
        if hasattr(myeloid[:, gene].X, 'toarray') else np.array(myeloid[:, gene].X).flatten()
        for gene in genes
    }
    medians = {gene: np.median(expr_data[gene]) for gene in genes}

    def get_phenotype(i):
        return ''.join([
            f"{g}{'+' if expr_data[g][i] > medians[g] else '-'}"
            for g in genes
        ])
    myeloid.obs['phenotype'] = [get_phenotype(i) for i in range(myeloid.n_obs)]

    stages = ['non-cancer', 'early', 'advanced']
    all_phenos = sorted(myeloid.obs['phenotype'].unique())
    stage_pct_df = pd.DataFrame(index=all_phenos, columns=stages)

    for stage in stages:
        sub = myeloid.obs[myeloid.obs['tumor_stage'] == stage]
        total = len(sub)
        counts = sub['phenotype'].value_counts()
        for pheno in all_phenos:
            stage_pct_df.loc[pheno, stage] = (counts.get(pheno, 0) / total) * 100 if total > 0 else 0
    stage_pct_df = stage_pct_df.fillna(0)
    logging.info("Phenotype breakdown for Figure 3J:\n%s", stage_pct_df.applymap(lambda x: f"{x:.1f}%").to_string())

    cmap = cm.get_cmap('tab20c', len(all_phenos))
    colors = [cmap(i) for i in range(len(all_phenos))]

    fig, ax = plt.subplots(figsize=(8, 6))
    bottom = np.zeros(len(stages))

    for i, pheno in enumerate(all_phenos):
        vals = stage_pct_df.loc[pheno].values.astype(float)
        ax.bar(stages, vals, bottom=bottom, color=colors[i], edgecolor='black', linewidth=0.5, label=pheno)
        bottom += vals

    ax.set_ylabel('Percentage of myeloid cells')
    ax.set_title('Myeloid phenotypes by tumor stage\n(> median expression of CD86, CD86, CD163)')
    ax.set_ylim(0, 100)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(axis='both', which='both', length=0)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)

    legend_handles = [
        Line2D([0], [0], marker='o', color='none',
               markerfacecolor=colors[i], markeredgecolor='black',
               markeredgewidth=0.5, markersize=8, label=pheno)
        for i, pheno in enumerate(all_phenos)
    ]
    ax.legend(
        handles=legend_handles,
        title='Phenotype',
        bbox_to_anchor=(1.02, 1),
        loc='upper left',
        frameon=False,
        borderaxespad=0.
    )

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

def __figure_three_K(adata, save_path=None):
    fibro = adata[
        adata.obs['leiden_res_20.00_celltype']
        .isin([
            "Peribronchial fibroblasts",
            "Adventitial fibroblasts",
            "Alveolar fibroblasts",
            "Subpleural fibroblasts",
            "Myofibroblasts",
            "Fibromyocytes"
        ])
    ].copy()

    fibro.obs['tumor_stage'] = fibro.obs['tumor_stage'].replace(['nan', 'NaN', 'NAN'], np.nan)
    fibro = fibro[~fibro.obs['tumor_stage'].isna()].copy()

    genes = ['FAP', 'ACTA2', 'COL1A1']
    for gene in genes:
        if gene not in fibro.var_names:
            raise ValueError(f"Gene '{gene}' not found in adata.")

    expr_data = {
        gene: fibro[:, gene].X.toarray().flatten()
        if hasattr(fibro[:, gene].X, 'toarray') else np.array(fibro[:, gene].X).flatten()
        for gene in genes
    }
    medians = {gene: np.median(expr_data[gene]) for gene in genes}

    def get_phenotype(i):
        return ''.join([
            f"{g}{'+' if expr_data[g][i] > medians[g] else '-'}"
            for g in genes
        ])
    fibro.obs['phenotype'] = [get_phenotype(i) for i in range(fibro.n_obs)]

    stages = ['non-cancer', 'early', 'advanced']
    all_phenos = sorted(fibro.obs['phenotype'].unique())
    stage_pct_df = pd.DataFrame(index=all_phenos, columns=stages)

    for stage in stages:
        sub = fibro.obs[fibro.obs['tumor_stage'] == stage]
        total = len(sub)
        counts = sub['phenotype'].value_counts()
        for pheno in all_phenos:
            stage_pct_df.loc[pheno, stage] = (counts.get(pheno, 0) / total) * 100 if total > 0 else 0
    stage_pct_df = stage_pct_df.fillna(0)
    logging.info("Phenotype breakdown for Figure 3K:\n%s", stage_pct_df.applymap(lambda x: f"{x:.1f}%").to_string())

    cmap = cm.get_cmap('tab10', len(all_phenos))
    colors = [cmap(i) for i in range(len(all_phenos))]

    fig, ax = plt.subplots(figsize=(8, 6))
    bottom = np.zeros(len(stages))

    for i, pheno in enumerate(all_phenos):
        vals = stage_pct_df.loc[pheno].values.astype(float)
        ax.bar(stages, vals, bottom=bottom, color=colors[i], edgecolor='black', linewidth=0.5, label=pheno)
        bottom += vals

    ax.set_ylabel('Percentage of fibroblasts')
    ax.set_title('Fibroblast phenotypes by tumor stage\n(> median expression of FAP, ACTA2, COL1A1)')
    ax.set_ylim(0, 100)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(axis='both', which='both', length=0)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)

    legend_handles = [
        Line2D([0], [0], marker='o', color='none',
               markerfacecolor=colors[i], markeredgecolor='black',
               markeredgewidth=0.5, markersize=8, label=pheno)
        for i, pheno in enumerate(all_phenos)
    ]
    ax.legend(
        handles=legend_handles,
        title='Phenotype',
        bbox_to_anchor=(1.02, 1),
        loc='upper left',
        frameon=False,
        borderaxespad=0.
    )

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

def __figure_three_L(adata, save_path=None):
    kras_mean = adata.obs["HALLMARK_KRAS_SIGNALING_UP"].mean()
    egfr_mean = adata.obs["REACTOME_SIGNALING_BY_EGFR_IN_CANCER"].mean()
    adata.obs["KRAS_high"] = adata.obs["HALLMARK_KRAS_SIGNALING_UP"] > kras_mean
    adata.obs["EGFR_high"] = adata.obs["REACTOME_SIGNALING_BY_EGFR_IN_CANCER"] > egfr_mean

    print(f"HALLMARK_KRAS_SIGNALING_UP mean = {kras_mean:.3f}")
    print(adata.obs["KRAS_high"].value_counts())
    print(f"REACTOME_SIGNALING_BY_EGFR_IN_CANCER mean = {egfr_mean:.3f}")
    print(adata.obs["EGFR_high"].value_counts())

    conditions = [
        (~adata.obs["KRAS_high"]) & (~adata.obs["EGFR_high"]),
        ( adata.obs["KRAS_high"]) & (~adata.obs["EGFR_high"]),
        (~adata.obs["KRAS_high"]) & ( adata.obs["EGFR_high"]),
        ( adata.obs["KRAS_high"]) & ( adata.obs["EGFR_high"]),
    ]
    group_order = ["Neither", "KRAS only", "EGFR only", "Both"]
    adata.obs["KRAS_EGFR_group"] = np.select(conditions, group_order, default="Neither")

    subsets = {
        "Macrophages": (
            adata[adata.obs['leiden_res_20.00_celltype'].isin([
                'Alveolar macrophages', 'Alveolar Mφ MT-positive',
                'Alveolar Mφ proliferating', 'Interstitial Mφ perivascular',
                'Alveolar Mφ CCL3+'
            ])],
            "CD68"
        ),
        "Fibroblasts": (
            adata[adata.obs['leiden_res_20.00_celltype'].isin([
                "Peribronchial fibroblasts", "Adventitial fibroblasts",
                "Alveolar fibroblasts", "Subpleural fibroblasts",
                "Myofibroblasts", "Fibromyocytes"
            ])],
            "COL1A2"
        ),
    }

    # --------- Pairwise significance: Mann–Whitney U + FDR (BH) across all tests in the figure ---------
    def _values_by_group(subset, gene, group_key="KRAS_EGFR_group"):
        # Extract 1-D expression vector for 'gene' aligned to subset.obs rows
        Xg = subset[:, [gene]].X
        vals = Xg.toarray().ravel() if sps.issparse(Xg) else np.asarray(Xg).ravel()
        data = {}
        for grp in group_order:
            mask = (subset.obs[group_key] == grp).to_numpy()
            data[grp] = vals[mask]
        return data

    results = []
    pvals_for_fdr = []
    fdr_index_map = []

    for subset_name, (subset, gene) in subsets.items():
        data_by_grp = _values_by_group(subset, gene)
        for g1, g2 in itertools.combinations(group_order, 2):
            x, y = data_by_grp[g1], data_by_grp[g2]
            if len(x) == 0 or len(y) == 0:
                results.append({
                    "panel": subset_name,
                    "gene": gene,
                    "group1": g1, "n1": len(x),
                    "group2": g2, "n2": len(y),
                    "U_statistic": np.nan,
                    "p_value_raw": np.nan,
                    "p_value_adj_fdr_bh": np.nan,
                    "effect_size_rbc": np.nan,
                    "median_group1": np.nan,
                    "median_group2": np.nan
                })
                continue

            U, p = mannwhitneyu(x, y, alternative='two-sided')
            n1, n2 = len(x), len(y)
            rbc = 1.0 - (2.0 * U) / (n1 * n2)  # rank-biserial correlation
            results.append({
                "panel": subset_name,
                "gene": gene,
                "group1": g1, "n1": n1,
                "group2": g2, "n2": n2,
                "U_statistic": float(U),
                "p_value_raw": float(p),
                "p_value_adj_fdr_bh": None,  # to be filled after FDR
                "effect_size_rbc": float(rbc),
                "median_group1": float(np.median(x)),
                "median_group2": float(np.median(y))
            })
            pvals_for_fdr.append(p)
            fdr_index_map.append(len(results) - 1)

    if pvals_for_fdr:
        _, pvals_fdr, _, _ = multipletests(pvals_for_fdr, method='fdr_bh')
        for idx, adj_p in zip(fdr_index_map, pvals_fdr):
            results[idx]["p_value_adj_fdr_bh"] = float(adj_p)

    # Save CSV
    figures_dir = Path("figures")
    figures_dir.mkdir(parents=True, exist_ok=True)
    csv_path = figures_dir / "figure_three_L_significance.csv"
    pd.DataFrame(results).to_csv(csv_path, index=False)
    print(f"Saved KRAS/EGFR-group significance (both panels) to: {csv_path}")

    # -------------------- Plot (unchanged) --------------------
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=False)

    for ax, (title, (subset, gene)) in zip(axes, subsets.items()):
        sc.pl.violin(
            subset,
            keys=gene,
            groupby="KRAS_EGFR_group",
            order=group_order,
            stripplot=False,
            jitter=0.2,
            size=1,
            ax=ax,
            show=False
        )
        ax.set_title(f"{title}\n{gene} expression")
        ax.set_xlabel("")
        ax.set_ylabel("Expression")

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def _compute_stage_fractions(adata, celltype_key, wanted_celltypes, genes, stages, csv_path):
    # subset by requested cell types
    sub = adata[adata.obs[celltype_key].isin(wanted_celltypes)].copy()

    # clean stage and drop NAs
    sub.obs['tumor_stage'] = sub.obs['tumor_stage'].replace(['nan', 'NaN', 'NAN'], np.nan)
    sub = sub[~sub.obs['tumor_stage'].isna()].copy()

    # gene presence checks
    missing = [g for g in genes if g not in sub.var_names]
    if missing:
        raise ValueError(f"Genes not found in adata: {missing}")

    # pull expression vectors and compute medians (within this subset)
    expr = {
        g: (sub[:, g].X.toarray().flatten()
            if hasattr(sub[:, g].X, 'toarray') else np.array(sub[:, g].X).flatten())
        for g in genes
    }
    med = {g: np.median(expr[g]) for g in genes}

    # call phenotypes: e.g., "STAT4+CCR7-LAG3+"
    def call_pheno(i):
        return ''.join([f"{g}{'+' if expr[g][i] > med[g] else '-'}" for g in genes])

    sub.obs['phenotype'] = [call_pheno(i) for i in range(sub.n_obs)]
    all_phenos = sorted(sub.obs['phenotype'].unique())

    # build fraction table (0..1)
    frac_df = pd.DataFrame(index=all_phenos, columns=stages, dtype=float)
    for stage in stages:
        stage_mask = (sub.obs['tumor_stage'] == stage)
        total = int(stage_mask.sum())
        if total == 0:
            frac_df[stage] = 0.0
            continue
        counts = sub.obs.loc[stage_mask, 'phenotype'].value_counts()
        for ph in all_phenos:
            frac_df.loc[ph, stage] = counts.get(ph, 0) / total

    # stable ordering: phenotypes by name, stages in provided order
    frac_df = frac_df.loc[all_phenos, stages].fillna(0.0)

    # write CSV
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    frac_df.to_csv(csv_path, index_label='phenotype')

    return frac_df

def make_supplementary_tables_p1_p3(adata, outdir="."):
    """
    Creates three CSVs with phenotype fractions by tumor stage:
      - supplementary_table_p1.csv : T cells (STAT4, CCR7, LAG3)
      - supplementary_table_p2.csv : Myeloid (CD68, CD86, CD163)
      - supplementary_table_p3.csv : Fibroblasts (FAP, ACTA2, COL1A1)
    Fractions are 0..1 for each stage and sum to 1 across phenotypes within a stage (up to rounding).
    """
    celltype_key = 'leiden_res_20.00_celltype'
    stages = ['non-cancer', 'early', 'advanced']

    # P1: T cells
    p1 = _compute_stage_fractions(
        adata=adata,
        celltype_key=celltype_key,
        wanted_celltypes=['CD8 T cells', 'CD4 T cells', 'T cells proliferating'],
        genes=['STAT4', 'CCR7', 'LAG3'],
        stages=stages,
        csv_path=Path(outdir) / "supplementary_table_p1.csv"
    )

    # P2: Myeloid
    p2 = _compute_stage_fractions(
        adata=adata,
        celltype_key=celltype_key,
        wanted_celltypes=[
            'Alveolar macrophages',
            'Alveolar Mφ MT-positive',
            'Alveolar Mφ proliferating',
            'Interstitial Mφ perivascular',
            'Alveolar Mφ CCL3+'
        ],
        genes=['CD68', 'CD86', 'CD163'],
        stages=stages,
        csv_path=Path(outdir) / "supplementary_table_p2.csv"
    )

    # P3: Fibroblasts
    p3 = _compute_stage_fractions(
        adata=adata,
        celltype_key=celltype_key,
        wanted_celltypes=[
            "Peribronchial fibroblasts",
            "Adventitial fibroblasts",
            "Alveolar fibroblasts",
            "Subpleural fibroblasts",
            "Myofibroblasts",
            "Fibromyocytes"
        ],
        genes=['FAP', 'ACTA2', 'COL1A1'],
        stages=stages,
        csv_path=Path(outdir) / "supplementary_table_p3.csv"
    )

    return {"p1": p1, "p2": p2, "p3": p3}

def __figure_three_state_change(adata, save_dir=None):
    """
    Figure 3B: UMAP with layered expression (>= median pops, < median stays gray)
      - STAT4 (T/NK only, red)
      - COL1A1 (Fibro only, blue)
      - CD163 (Myeloid/DC/Mast only, green)
      - Hallmark EMT (epithelium only, purple)

    Uses precomputed adata.obsm['X_umap'] (no PCA/UMAP/integration recompute).
    Saves PNG (high DPI) and PDF (vector).
    """
    # ---- Configuration / groups ----
    CELLTYPE_KEY = 'leiden_res_20.00_celltype'
    t_nk = ['CD8 T cells', 'CD4 T cells', 'T cells proliferating', 'NK cells']
    fibro = ["Peribronchial fibroblasts", "Adventitial fibroblasts", "Alveolar fibroblasts",
             "Subpleural fibroblasts", "Myofibroblasts", "Fibromyocytes"]
    myeloid_dc_mast = ['Alveolar macrophages', 'Alveolar Mφ MT-positive', 'Alveolar Mφ proliferating',
                       'Interstitial Mφ perivascular', 'Alveolar Mφ CCL3+', 'Monocyte derived Mφ',
                       'Mast cells', 'Plasmacytoid DCs', 'DC2', 'Migratory DCs',
                       'Classical Monocytes', 'Non classical monocytes']
    EPITHELIAL_CATS = [
        'AT2', 'AT1', 'Suprabasal', 'Basal resting',
        'Multiciliated (non-nasal)', 'Goblet (nasal)',
        'Club (nasal)', 'Ciliated (nasal)',
        'Club (non-nasal)', 'Multiciliated (nasal)',
        'Goblet (bronchial)', 'Transitional Club AT2',
        'AT2 proliferating', 'Goblet (subsegmental)'
    ]

    gene_stat4 = 'STAT4'
    gene_col1a1 = 'COL1A1'
    gene_cd163  = 'CD163'
    emt_key     = 'HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION'

    # ---- Safety checks (no recompute) ----
    if 'X_umap' not in adata.obsm:
        raise RuntimeError("UMAP embedding not found in adata.obsm['X_umap'].")
    if CELLTYPE_KEY not in adata.obs.columns:
        raise KeyError(f"'{CELLTYPE_KEY}' not found in adata.obs.")
    if emt_key not in adata.obs.columns:
        raise KeyError(f"'{emt_key}' not in adata.obs. Add it first (score_genes).")

    # ---- Helpers ----
    def get_gene_expr(_adata, gene):
        if _adata.raw is not None and gene in _adata.raw.var_names:
            Xg = _adata.raw[:, gene].X
        elif gene in _adata.var_names:
            Xg = _adata[:, gene].X
        else:
            raise KeyError(f"Gene '{gene}' not found in var_names or raw.var_names")
        if sps.issparse(Xg):
            Xg = Xg.toarray()
        return np.asarray(Xg).ravel()

    def mask_to_groups(vals, groups, key):
        keep = adata.obs[key].isin(groups).values
        out = vals.astype(float).copy()
        out[~keep] = np.nan
        return out

    def threshold_to_median(v):
        """
        Keep only >= median as colored signal; set < median to NaN so they render as gray background.
        Remaining values are normalized from median..max -> 0..1.
        """
        vv = v.astype(float).copy()
        m = ~np.isnan(vv)
        if m.sum() == 0:
            return np.full_like(vv, np.nan, dtype=float)
        med = np.nanmedian(vv)
        # Grey out below-median
        vv[vv < med] = np.nan
        vmax = np.nanmax(vv)
        if not np.isfinite(vmax) or vmax == med:
            # All surviving values equal to median -> show as uniform strongest tint
            return np.where(np.isnan(vv), np.nan, 1.0)
        # Normalize surviving values to [0,1]
        vv = (vv - med) / (vmax - med)
        return vv

    def sort_by_value_for_front(x, y, v):
        # draw low/NaN first so high values sit on top
        order = np.argsort(np.nan_to_num(v, nan=-1.0))
        return x[order], y[order], v[order]

    # ---- Coordinates ----
    umap = adata.obsm['X_umap']
    x = umap[:, 0].astype(float).copy()
    y = umap[:, 1].astype(float).copy()

    # ---- Values & masking ----
    v_stat4 = mask_to_groups(get_gene_expr(adata, gene_stat4), t_nk, CELLTYPE_KEY)
    v_col1a1 = mask_to_groups(get_gene_expr(adata, gene_col1a1), fibro, CELLTYPE_KEY)
    v_cd163  = mask_to_groups(get_gene_expr(adata, gene_cd163),  myeloid_dc_mast, CELLTYPE_KEY)

    v_emt_all = adata.obs[emt_key].to_numpy(dtype=float)
    v_emt = mask_to_groups(v_emt_all, EPITHELIAL_CATS, CELLTYPE_KEY)

    # ---- Keep only >= median (others go gray) & normalize ----
    v_stat4_n = threshold_to_median(v_stat4)
    v_col1a1_n = threshold_to_median(v_col1a1)
    v_cd163_n  = threshold_to_median(v_cd163)
    v_emt_n    = threshold_to_median(v_emt)

    # ---- Colormaps ----
    GREY = "#A9A9A9"
    cmap_red    = LinearSegmentedColormap.from_list("grey_red",    [GREY, "#B22222"])
    cmap_blue   = LinearSegmentedColormap.from_list("grey_blue",   [GREY, "#1E90FF"])
    cmap_green  = LinearSegmentedColormap.from_list("grey_green",  [GREY, "#C11C84"])
    cmap_purple = LinearSegmentedColormap.from_list("grey_purple", [GREY, "#6A5ACD"])

    # ---- Sort so higher values plot last (foreground) ----
    xs1, ys1, cs1 = sort_by_value_for_front(x, y, v_stat4_n)
    xs2, ys2, cs2 = sort_by_value_for_front(x, y, v_col1a1_n)
    xs3, ys3, cs3 = sort_by_value_for_front(x, y, v_cd163_n)
    xs4, ys4, cs4 = sort_by_value_for_front(x, y, v_emt_n)

    # ---- Plot ----
    fig = plt.figure(figsize=(10, 6))
    ax = plt.gca()

    # Background (all cells in light gray)
    ax.scatter(x, y, s=0.2, c=GREY, alpha=0.35, linewidths=0, rasterized=True)

    # Overlays (only >= median carry non-NaN color values)
    scat1 = ax.scatter(xs1, ys1, s=0.2, c=cs1, cmap=cmap_red,    vmin=0, vmax=1, linewidths=0, rasterized=True)
    scat2 = ax.scatter(xs2, ys2, s=0.2, c=cs2, cmap=cmap_blue,   vmin=0, vmax=1, linewidths=0, rasterized=True)
    scat3 = ax.scatter(xs3, ys3, s=0.2, c=cs3, cmap=cmap_green,  vmin=0, vmax=1, linewidths=0, rasterized=True)
    scat4 = ax.scatter(xs4, ys4, s=0.2, c=cs4, cmap=cmap_purple, vmin=0, vmax=1, linewidths=0, rasterized=True)

    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel('UMAP1'); ax.set_ylabel('UMAP2')
    ax.set_title('UMAP with STAT4 (red), COL1A1 (blue), CD163 (green), EMT (purple)\n'
                 'Only ≥ median signal shown in color; < median remains gray')

    # Non-overlapping colorbars (interpretation: 0 = median, 1 = max)
    cax1 = fig.add_axes([0.92, 0.70, 0.015, 0.20])
    cax2 = fig.add_axes([0.92, 0.47, 0.015, 0.20])
    cax3 = fig.add_axes([0.92, 0.24, 0.015, 0.20])
    cax4 = fig.add_axes([0.92, 0.01,  0.015, 0.20])

    cbar1 = plt.colorbar(scat1, cax=cax1); cbar1.set_label('STAT4 (≥ median in T/NK)')
    cbar2 = plt.colorbar(scat2, cax=cax2); cbar2.set_label('COL1A1 (≥ median in Fibro)')
    cbar3 = plt.colorbar(scat3, cax=cax3); cbar3.set_label('CD163 (≥ median in Myeloid/DC/Mast)')
    cbar4 = plt.colorbar(scat4, cax=cax4); cbar4.set_label('EMT (≥ median in epithelium)')

    # ---- Save in great quality ----
    if save_dir is None:
        save_dir = Path.cwd() / "figures"
    else:
        save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    png_path = save_dir / "fig-3B.png"
    pdf_path = save_dir / "fig-3B.pdf"

    fig.savefig(png_path, dpi=800, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved Figure 3B to:\n  - {png_path}\n  - {pdf_path}")

def __figure_three_endothelial_barplot(adata, save_path=None):
    """
    Stacked barplot of endothelial 'phenotypes' by tumor stage based on
    PLVAP / FLT1 / KDR expression above/below median within endothelial cells.
    """

    # --- Endothelial celltype names seen in your marker_genes dict ---
    ec_types = [
        "EC venous pulmonary",
        "EC arterial",
        "EC venous systemic",
        "EC general capillary",
        "EC aerocyte capillary",
        "Lymphatic EC mature",
        "Lymphatic EC differentiating",
        "Lymphatic EC proliferating",
    ]

    # Subset to endothelial cells
    ec = adata[adata.obs['leiden_res_20.00_celltype'].isin(ec_types)].copy()
    if ec.n_obs == 0:
        raise ValueError("No endothelial cells found with the expected labels in 'leiden_res_20.00_celltype'.")

    # Clean tumor_stage and drop NAs
    ec.obs['tumor_stage'] = ec.obs['tumor_stage'].replace(['nan', 'NaN', 'NAN'], np.nan)
    ec = ec[~ec.obs['tumor_stage'].isna()].copy()

    # Gene list (be tolerant of "FTL1" typo; prefer FLT1 if present)
    requested = ['PLVAP', 'FLT1', 'KDR']
    # If the dataset actually has "FTL1" (typo), remap to that
    genes = []
    for g in requested:
        if g in ec.var_names:
            genes.append(g)
        elif g == 'FLT1' and 'FTL1' in ec.var_names:
            genes.append('FTL1')
        else:
            raise ValueError(f"Gene '{g}' not found in adata (also checked FTL1 for FLT1).")

    # Pull expression and compute per-gene medians (sparse-safe)
    expr_data = {
        gene: (ec[:, gene].X.toarray().flatten()
               if hasattr(ec[:, gene].X, 'toarray') else
               np.array(ec[:, gene].X).flatten())
        for gene in genes
    }
    medians = {gene: np.median(expr_data[gene]) for gene in genes}

    # Build +/- phenotype labels like "PLVAP+FLT1-KDR+"
    def get_phenotype(i):
        return ''.join([
            f"{g}{'+' if expr_data[g][i] > medians[g] else '-'}"
            for g in genes
        ])

    ec.obs['phenotype'] = [get_phenotype(i) for i in range(ec.n_obs)]

    # Stages and percentage table
    stages = ['non-cancer', 'early', 'advanced']
    all_phenos = sorted(ec.obs['phenotype'].unique())
    stage_pct_df = pd.DataFrame(index=all_phenos, columns=stages)

    for stage in stages:
        sub = ec.obs[ec.obs['tumor_stage'] == stage]
        total = len(sub)
        counts = sub['phenotype'].value_counts()
        for pheno in all_phenos:
            stage_pct_df.loc[pheno, stage] = (counts.get(pheno, 0) / total) * 100 if total > 0 else 0
    stage_pct_df = stage_pct_df.fillna(0)

    logging.info("Endothelial phenotype breakdown (Angiogenesis markers):\n%s",
                 stage_pct_df.applymap(lambda x: f"{x:.1f}%").to_string())

    # Plot
    # Use a larger qualitative colormap in case there are many phenotypes
    cmap = cm.get_cmap('tab20', len(all_phenos))
    colors = [cmap(i) for i in range(len(all_phenos))]

    fig, ax = plt.subplots(figsize=(8, 6))
    bottom = np.zeros(len(stages))

    for i, pheno in enumerate(all_phenos):
        vals = stage_pct_df.loc[pheno].values.astype(float)
        ax.bar(stages, vals, bottom=bottom, color=colors[i],
               edgecolor='black', linewidth=0.5, label=pheno)
        bottom += vals

    ax.set_ylabel('Percentage of endothelial cells')
    ax.set_title('Endothelial phenotypes by tumor stage\n(Angiogenesis markers: PLVAP, FLT1, KDR; threshold = per-gene median)')
    ax.set_ylim(0, 100)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(axis='both', which='both', length=0)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)

    legend_handles = [
        Line2D([0], [0], marker='o', color='none',
               markerfacecolor=colors[i], markeredgecolor='black',
               markeredgewidth=0.5, markersize=8, label=pheno)
        for i, pheno in enumerate(all_phenos)
    ]
    ax.legend(
        handles=legend_handles,
        title='Phenotype (median-split)',
        bbox_to_anchor=(1.02, 1),
        loc='upper left',
        frameon=False,
        borderaxespad=0.
    )

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

def __figure_three_C_endothelial(adata, save_path=None):
    logging.debug("=== START __figure_three_C_endothelial ===")

    # Endothelial/lymphatic EC labels (from your marker set)
    ec_types = [
        "EC venous pulmonary",
        "EC arterial",
        "EC venous systemic",
        "EC general capillary",
        "EC aerocyte capillary",
        "Lymphatic EC mature",
        "Lymphatic EC differentiating",
        "Lymphatic EC proliferating",
    ]

    # Subset to endothelial cells
    ec_cells = adata[adata.obs['leiden_res_20.00_celltype'].isin(ec_types)].copy()
    logging.debug(f"Subset to endothelial cells: {ec_cells.n_obs} cells")
    if ec_cells.n_obs == 0:
        raise ValueError("No endothelial cells found with the expected labels in 'leiden_res_20.00_celltype'.")

    # Clean tumor_stage and drop NA
    ec_cells.obs['tumor_stage'] = ec_cells.obs['tumor_stage'].replace(['nan', 'NaN', 'NAN'], np.nan)

    dist_pre = ec_cells.obs['tumor_stage'].value_counts(dropna=False)
    logging.debug(f"tumor_stage distribution before drop (incl NaN/strings):\n{dist_pre}")

    mask = ec_cells.obs['tumor_stage'].notna()
    n_missing = (~mask).sum()
    logging.debug(f"Found {n_missing} cells with missing or string 'nan' tumor_stage")
    if n_missing > 0:
        logging.debug(f"Dropping {n_missing} cells with missing tumor_stage")
        ec_cells = ec_cells[mask, :].copy()

    dist_post = ec_cells.obs['tumor_stage'].value_counts(dropna=False)
    logging.debug(f"tumor_stage distribution after drop (incl NaN):\n{dist_post}")

    # HVGs on EC subset (Seurat v3 flavor to match your pipeline)
    sc.pp.highly_variable_genes(
        ec_cells,
        n_top_genes=5000,
        flavor='seurat_v3',
        subset=False,
        n_bins=20
    )
    ec_cells = ec_cells[:, ec_cells.var['highly_variable']].copy()
    logging.debug(f"After HVG subset: {ec_cells.n_vars} genes")

    # Use precomputed UMAP
    if 'X_umap' not in ec_cells.obsm:
        raise KeyError("UMAP coordinates 'X_umap' not found in ec_cells.obsm. Compute UMAP before calling this function.")
    umap = ec_cells.obsm['X_umap']

    # Leiden if missing
    if 'leiden' not in ec_cells.obs:
        sc.tl.leiden(ec_cells, key_added='leiden')
    clusters = pd.to_numeric(ec_cells.obs['leiden'], errors='coerce').fillna(-1).astype(int).values

    # Learn graph & order cells (root at cluster 0 for consistency)
    projected_pts, mst, centroids = learn_graph(matrix=umap, clusters=clusters)
    pseudotime = order_cells(
        umap,
        centroids,
        mst=mst,
        projected_points=projected_pts,
        root_cells=0
    )
    ec_cells.obs['pseudotime'] = pseudotime
    logging.debug("Pseudotime computed for endothelial cells")

    # Colormap and scatter
    cmap_gray_purple = LinearSegmentedColormap.from_list('gray_purple', ['#A9A9A9', 'purple'])

    plt.figure(figsize=(8, 6))
    plt.scatter(
        umap[:, 0],
        umap[:, 1],
        c=pseudotime,
        s=0.1,
        cmap=cmap_gray_purple,
        rasterized=True
    )

    # Centroids per sample (and dominant tumor_stage per sample)
    df_umap = pd.DataFrame(umap, columns=['UMAP1', 'UMAP2'], index=ec_cells.obs_names)
    df_umap['sample'] = ec_cells.obs['sample'].values
    df_umap['tumor_stage'] = ec_cells.obs['tumor_stage'].values
    df_umap = df_umap.dropna(subset=['tumor_stage'])

    def mode_or_first(x):
        m = x.mode()
        return m.iloc[0] if len(m) > 0 else x.iloc[0]

    centroid_df = (
        df_umap
        .groupby('sample')
        .agg({'UMAP1': 'mean', 'UMAP2': 'mean', 'tumor_stage': mode_or_first})
        .reset_index()
    )

    stage_colors = {
        'non-cancer': '#84A970',
        'early':      '#E4C282',
        'advanced':   '#FF8C00'
    }

    ax = plt.gca()
    target_handles = []
    seen_targets = set()

    for _, row in centroid_df.iterrows():
        sname = row['sample']
        stg = row['tumor_stage']
        if sname in TARGET_PROJECTS:
            color = TARGET_COLORS[sname]
            ax.scatter(
                row['UMAP1'], row['UMAP2'],
                c=[color],
                s=90, marker='*',
                edgecolors='black', linewidths=0.4,
                zorder=12
            )
            if sname not in seen_targets:
                target_handles.append(
                    Line2D([0], [0], marker='*', linestyle='',
                           markersize=10, markeredgewidth=0.8,
                           markeredgecolor='black', color=color, label=sname)
                )
                seen_targets.add(sname)
        else:
            ax.scatter(
                row['UMAP1'], row['UMAP2'],
                c=[stage_colors.get(stg, 'black')],
                s=30, marker='o',
                edgecolors='none',
                zorder=10
            )

    # Stage legend (inside)
    stage_handles = [
        mpatches.Patch(color=stage_colors['non-cancer'], label='non-cancer'),
        mpatches.Patch(color=stage_colors['early'], label='early'),
        mpatches.Patch(color=stage_colors['advanced'], label='advanced'),
    ]
    stage_legend = ax.legend(handles=stage_handles, title='Tumor stage',
                             loc='lower left', frameon=False)
    ax.add_artist(stage_legend)

    # Target legend (outside)
    if target_handles:
        ax.legend(handles=target_handles, title='TARGET PROJECTS',
                  loc='upper left', bbox_to_anchor=(1.02, 1),
                  borderaxespad=0., frameon=False)

    # Aesthetics & save
    ax.set_title('Endothelial trajectory pseudotime')
    ax.set_xlabel('UMAP1'); ax.set_ylabel('UMAP2')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.debug(f"Saved endothelial trajectory plot to {save_path}")
    logging.debug("=== END __figure_three_C_endothelial ===")

    return ec_cells

def __figure_three_G_endothelial(adata, save_path=None):
    """
    Boxplots of HALLMARK_ANGIOGENESIS by tumor stage for endothelial cells,
    with pairwise Mann–Whitney U tests (Bonferroni-corrected).
    """
    # Endothelial / lymphatic EC labels
    ec_types = [
        "EC venous pulmonary",
        "EC arterial",
        "EC venous systemic",
        "EC general capillary",
        "EC aerocyte capillary",
        "Lymphatic EC mature",
        "Lymphatic EC differentiating",
        "Lymphatic EC proliferating",
    ]

    # Filter for endothelial cells
    ec_cells = adata[
        adata.obs['leiden_res_20.00_celltype'].isin(ec_types)
    ].copy()

    stages = ['non-cancer', 'early', 'advanced']
    stage_colors = {
        'non-cancer': '#84A970',
        'early':      '#E4C282',
        'advanced':   '#FF8C00'
    }

    # Make sure the score exists
    score_col = 'HALLMARK_ANGIOGENESIS'
    if score_col not in ec_cells.obs.columns:
        raise KeyError(f"'{score_col}' not found in adata.obs. "
                       f"Please compute and store it in .obs['{score_col}'].")

    # Clean tumor_stage and subset
    ec_cells.obs['tumor_stage'] = ec_cells.obs['tumor_stage'].replace(['nan', 'NaN', 'NAN'], np.nan)

    df = (
        ec_cells.obs
        .loc[:, ['tumor_stage', score_col]]
        .dropna(subset=['tumor_stage'])
    )

    # Collect per-stage arrays (empty arrays are fine; tests will handle via try/except)
    data_by_stage = [df.loc[df['tumor_stage'] == st, score_col].values for st in stages]

    # Pairwise Mann–Whitney U tests with Bonferroni correction
    pairs = list(itertools.combinations(range(len(stages)), 2))
    results = []
    for (i, j) in pairs:
        x, y = data_by_stage[i], data_by_stage[j]
        if len(x) > 0 and len(y) > 0:
            stat, p = mannwhitneyu(x, y, alternative='two-sided')
            p_adj = p * len(pairs)  # Bonferroni
        else:
            stat, p, p_adj = np.nan, np.nan, np.nan
        results.append({
            "group1": stages[i],
            "group2": stages[j],
            "U_statistic": stat,
            "p_value_raw": p,
            "p_value_adj": p_adj
        })

    # Save results to /figures/
    results_df = pd.DataFrame(results)
    figures_dir = Path("figures")
    figures_dir.mkdir(parents=True, exist_ok=True)
    results_csv_path = figures_dir / "figure_three_G_endothelial_significance.csv"
    results_df.to_csv(results_csv_path, index=False)
    print(f"Significance results saved to: {results_csv_path}")

    # Plot boxplots (no annotations)
    fig, ax = plt.subplots(figsize=(6, 6))
    bp = ax.boxplot(
        data_by_stage,
        patch_artist=True,
        widths=0.6,
        medianprops=dict(linewidth=1.5),
        showfliers=False
    )

    for patch, st in zip(bp['boxes'], stages):
        patch.set_facecolor(stage_colors[st])
        patch.set_edgecolor('black')

    ax.set_xticks(range(1, len(stages) + 1))
    ax.set_xticklabels(stages, rotation=30)
    ax.set_ylabel('HALLMARK_ANGIOGENESIS score')
    ax.set_title('Angiogenesis pathway activity by tumor stage (Endothelial cells)')

    legend_handles = [Patch(facecolor=stage_colors[st], label=st) for st in stages]
    ax.legend(handles=legend_handles, title='Tumor stage', loc='upper right', frameon=False)

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

def __figure_three_F_endothelial(adata, save_dir="figures/fig3F_endothelial"):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Endothelial / lymphatic EC labels
    ec_types = [
        "EC venous pulmonary",
        "EC arterial",
        "EC venous systemic",
        "EC general capillary",
        "EC aerocyte capillary",
        "Lymphatic EC mature",
        "Lymphatic EC differentiating",
        "Lymphatic EC proliferating",
    ]

    # Subset to endothelial cells
    ec = adata[adata.obs['leiden_res_20.00_celltype'].isin(ec_types)].copy()
    if ec.n_obs == 0:
        raise ValueError("No endothelial cells found with the expected labels in 'leiden_res_20.00_celltype'.")

    # Colormap (same style as your T-cell plots)
    cmap_custom = LinearSegmentedColormap.from_list('gray_emerald', ['#A9A9A9', '#50C878'])

    # 9 EC / angiogenesis markers
    requested_genes = ['PECAM1', 'VWF', 'KDR', 'FLT1', 'PLVAP', 'CLDN5', 'EMCN', 'EGFL7', 'SOX17']

    # Allow FLT1↔FTL1 typo
    genes = []
    for g in requested_genes:
        if g in ec.var_names:
            genes.append(g)
        elif g == 'FLT1' and 'FTL1' in ec.var_names:
            genes.append('FTL1')
        else:
            # still include but warn and skip at plotting time
            genes.append(g)

    # Ensure UMAP exists
    if 'X_umap' not in ec.obsm:
        raise KeyError("UMAP coordinates 'X_umap' not found in .obsm. Compute UMAP before calling this function.")
    umap = ec.obsm['X_umap']

    # Precompute zoom window to match your style
    x, y = umap[:, 0], umap[:, 1]
    x_c, y_c = x.mean(), y.mean()
    zoom_frac = 0.8
    half_x = (x.max() - x.min()) * zoom_frac / 2
    half_y = (y.max() - y.min()) * zoom_frac / 2

    for gene in genes:
        if gene not in ec.var_names:
            print(f"[WARN] Gene '{gene}' not found in endothelial subset, skipping.")
            continue

        # Sort by expression to reduce overplotting
        expr = ec[:, gene].X
        expr_arr = expr.toarray().flatten() if hasattr(expr, 'toarray') else np.array(expr).flatten()
        sorted_idx = np.argsort(expr_arr)
        ec_sorted = ec[sorted_idx].copy()

        fig, ax = plt.subplots(figsize=(6, 6))
        sc.pl.umap(
            ec_sorted,
            color=gene,
            size=0.5,
            sort_order=False,
            cmap=cmap_custom,
            ax=ax,
            show=False,
            colorbar_loc=None
        )
        ax.set_xlim(x_c - half_x, x_c + half_x)
        ax.set_ylim(y_c - half_y, y_c + half_y)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel('X1')
        ax.set_ylabel('X2')
        ax.set_title(f'{gene} Expression')
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
        ax.tick_params(width=1.5)

        # Colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap_custom)
        sm.set_array(expr_arr)
        sm.set_clim(expr_arr.min(), expr_arr.max())
        cbar = fig.colorbar(sm, ax=ax, fraction=0.02, pad=0.04)
        cbar.set_label('Expression')

        out_path = save_dir / f"{gene}.png"
        fig.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    BASE_DIR = Path.cwd()
    adata = sc.read_h5ad(filename=str(BASE_DIR / 'data/processed_data.h5ad'))
    # __figure_three_A(adata, save_path=str(BASE_DIR / 'figures/fig-3A.png'))
    # __figure_three_C(adata, save_path=str(BASE_DIR / 'figures/fig-3C.png'))
    # __figure_three_D(adata, save_path=str(BASE_DIR / 'figures/fig-3D.png'))
    # __figure_three_E(adata, save_path=str(BASE_DIR / 'figures/fig-3E.png'))
    # __figure_three_F_1(adata, save_dir=str(BASE_DIR / 'figures/fig-3F1'))
    # __figure_three_F_2(adata, save_dir=str(BASE_DIR / 'figures/fig-3F2'))
    # __figure_three_F_3(adata, save_dir=str(BASE_DIR / 'figures/fig-3F3'))
    # __figure_three_G(adata, save_path=str(BASE_DIR / 'figures/fig-3G.png'))
    # __figure_three_H(adata, save_path=str(BASE_DIR / 'figures/fig-3H.png'))
    # __figure_three_I(adata, save_path=str(BASE_DIR / 'figures/fig-3I.png'))
    # __figure_three_J(adata, save_path=str(BASE_DIR / 'figures/fig-3J.png'))
    # __figure_three_K(adata, save_path=str(BASE_DIR / 'figures/fig-3K.png'))
    # __figure_three_L(adata, save_path=str(BASE_DIR / 'figures/fig-3L.png'))
    # __figure_three_state_change(adata, save_dir=str(BASE_DIR / 'figures/fig-3_state_change'))
    __figure_three_C_endothelial(adata, save_path=str(BASE_DIR / 'figures/fig-3C_endothelial.png'))
    __figure_three_G_endothelial(adata, save_path=str(BASE_DIR / 'figures/fig-3G_endothelial.png'))
    __figure_three_endothelial_barplot(adata, save_path=str(BASE_DIR / 'figures/fig-3_endothelial_barplot.png'))
    __figure_three_F_endothelial(adata, save_dir=str(BASE_DIR / 'figures/fig-3F_endothelial'))

    make_supplementary_tables_p1_p3(adata, outdir="figures/")
