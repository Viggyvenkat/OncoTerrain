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

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s:%(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)  # or sys.stderr if preferred
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

def __figure_three_A(adata, save_path=None):
    """
    Plot UMAP embedding of adata, highlighting non-epithelial cell types in color,
    graying out epithelial types, and saving the figure as a high-quality image.
    """

    orig = adata.obs['leiden_res_20.00_celltype'].cat.add_categories(['Other'])
    
    # Build a new 'highlight' column: keep labels not in EPITHELIAL_CATS, else 'Other'
    adata.obs['highlight'] = (
        orig
        .where(~orig.isin(EPITHELIAL_CATS), other='Other')
        .cat
        .remove_unused_categories()
    )

    # Determine the actual non-epithelial categories to color
    non_epi = [cat for cat in adata.obs['highlight'].cat.categories if cat != 'Other']

    # Choose a base palette
    base_pal = sc.pl.palettes.default_20
    # If there are more categories than the palette size, tile it
    if len(non_epi) > len(base_pal):
        palette_vals = plt.cm.tab20.colors * ((len(non_epi) // len(base_pal)) + 1)
    else:
        palette_vals = base_pal

    # Map each non-epithelial category to a distinct color; 'Other' (epithelial) is light gray
    palette = {cat: palette_vals[i] for i, cat in enumerate(non_epi)}
    palette['Other'] = '#d3d3d3'
    
    plt.figure(figsize=(15, 8))
    # Plot UMAP
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

    # Handle save path
    if save_path is None:
        figures_dir = Path.cwd() / 'figures'
        figures_dir.mkdir(parents=True, exist_ok=True)
        save_path = figures_dir / 'figure_3A.png'
    else:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.3)
    plt.close()


# ensure learn_graph and order_cells are in scope:
# from your_monocle_module import learn_graph, order_cells

def __figure_three_C(adata, save_path=None):
    """
    Run Monocle pseudotime on T cells, plot UMAP colored by pseudotime (gray→purple),
    and overlay sample-centroids colored by tumor_stage.
    """
    logging.debug("=== START __figure_three_C ===")
    # 1) Subset to the three T‐cell clusters
    t_cells = adata[
        adata.obs['leiden_res_20.00_celltype']
        .isin(['CD8 T cells', 'CD4 T cells', 'T cells proliferating'])
    ].copy()
    logging.debug(f"Subset to T cells: {t_cells.n_obs} cells")

    t_cells.obs['tumor_stage'] = t_cells.obs['tumor_stage'].replace(['nan', 'NaN', 'NAN'], np.nan)

    # Thorough NaN investigation for tumor_stage
    dist_pre = t_cells.obs['tumor_stage'].value_counts(dropna=False)
    logging.debug(f"tumor_stage distribution before drop (incl NaN/strings):\n{dist_pre}")

    mask = t_cells.obs['tumor_stage'].notna()
    n_missing = (~mask).sum()
    logging.debug(f"Found {n_missing} cells with missing or string 'nan' tumor_stage")
    if n_missing > 0:
        logging.debug(f"Dropping {n_missing} cells with missing tumor_stage")
        t_cells = t_cells[mask, :].copy()

    # Check post-drop distribution
    dist_post = t_cells.obs['tumor_stage'].value_counts(dropna=False)
    logging.debug(f"tumor_stage distribution after drop (incl NaN):\n{dist_post}")

    # 2) HVG selection (top 5000)
    sc.pp.highly_variable_genes(
        t_cells,
        n_top_genes=5000,
        flavor='seurat_v3',
        subset=False,
        n_bins=20
    )
    t_cells = t_cells[:, t_cells.var['highly_variable']].copy()
    logging.debug(f"After HVG subset: {t_cells.n_vars} genes")

    # 3) Neighbors & UMAP
    umap = t_cells.obsm['X_umap']

    # 4) Ensure a clustering exists for learn_graph
    if 'leiden' not in t_cells.obs:
        sc.tl.leiden(t_cells, key_added='leiden')
    clusters = pd.to_numeric(t_cells.obs['leiden'], errors='coerce').fillna(-1).astype(int).values

    # 5) Learn principal graph & pseudotime
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

    # 6) Build gray→purple colormap
    cmap_gray_purple = LinearSegmentedColormap.from_list(
        'gray_purple',
        ['gray', 'purple']
    )

    # 7) Scatter UMAP colored by pseudotime
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        umap[:, 0],
        umap[:, 1],
        c=pseudotime,
        s=0.1,
        cmap=cmap_gray_purple,
        rasterized=True
    )

    # 8) Compute and overlay sample-centroids colored by tumor_stage
    # Build a DataFrame with UMAP coords + sample + tumor_stage
    df_umap = pd.DataFrame(umap, columns=['UMAP1', 'UMAP2'], index=t_cells.obs_names)
    df_umap['sample'] = t_cells.obs['sample'].values
    df_umap['tumor_stage'] = t_cells.obs['tumor_stage'].values
    # Drop cells missing tumor_stage
    df_umap = df_umap.dropna(subset=['tumor_stage'])

    # Function to choose the modal stage per sample
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
    for _, row in centroid_df.iterrows():
        stg = row['tumor_stage']
        plt.scatter(
            row['UMAP1'], row['UMAP2'],
            c=stage_colors.get(stg, 'black'),
            s=30,
            edgecolors='none',
            marker='o',
            zorder=10
        )

    # Final plot tweaks
    plt.xticks([])
    plt.yticks([])
    plt.title('T-cell trajectory pseudotime\n(with sample-centroids by tumor stage)')
    cbar = plt.colorbar(scatter, pad=0.02)
    cbar.set_label('Pseudotime')
    plt.tight_layout()

    # 9) Save figure
    if save_path is None:
        figures_dir = Path.cwd() / 'figures'
        figures_dir.mkdir(parents=True, exist_ok=True)
        save_path = figures_dir / 'figure_3C.png'
    else:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.debug(f"Saved trajectory plot to {save_path}")
    logging.debug("=== END __figure_three_C ===")

    return t_cells

def __figure_three_D(adata, save_path=None):
    """
    Run Monocle pseudotime on macrophages, plot UMAP colored by pseudotime (gray→purple),
    and overlay sample-centroids colored by tumor_stage.
    """
    logging.debug("=== START __figure_three_E ===")
    # 1) Subset to the specified macrophage clusters
    mphage_cells = adata[
        adata.obs['leiden_res_20.00_celltype']
        .isin([
            'Alveolar macrophages',
            'Alveolar Mφ MT-positive',
            'Alveolar Mφ proliferating',
            'Interstitial Mφ perivascular',
            'Alveolar Mφ CCL3+'
        ])
    ].copy()
    logging.debug(f"Subset to macrophages: {mphage_cells.n_obs} cells")

    mphage_cells.obs['tumor_stage'] = mphage_cells.obs['tumor_stage'].replace(['nan', 'NaN', 'NAN'], np.nan)

    # Thorough NaN investigation for tumor_stage
    dist_pre = mphage_cells.obs['tumor_stage'].value_counts(dropna=False)
    logging.debug(f"tumor_stage distribution before drop (incl NaN/strings):\n{dist_pre}")

    mask = mphage_cells.obs['tumor_stage'].notna()
    n_missing = (~mask).sum()
    logging.debug(f"Found {n_missing} cells with missing or string 'nan' tumor_stage")
    if n_missing > 0:
        logging.debug(f"Dropping {n_missing} cells with missing tumor_stage")
        mphage_cells = mphage_cells[mask, :].copy()

    # Check post-drop distribution
    dist_post = mphage_cells.obs['tumor_stage'].value_counts(dropna=False)
    logging.debug(f"tumor_stage distribution after drop (incl NaN):\n{dist_post}")

    # 2) HVG selection (top 5000)
    sc.pp.highly_variable_genes(
        mphage_cells,
        n_top_genes=5000,
        flavor='seurat_v3',
        subset=False,
        n_bins=20
    )
    mphage_cells = mphage_cells[:, mphage_cells.var['highly_variable']].copy()
    logging.debug(f"After HVG subset: {mphage_cells.n_vars} genes")

    # 3) Neighbors & UMAP
    umap = mphage_cells.obsm['X_umap']

    # 4) Ensure a clustering exists for learn_graph
    if 'leiden' not in mphage_cells.obs:
        sc.tl.leiden(mphage_cells, key_added='leiden')
    clusters = pd.to_numeric(
        mphage_cells.obs['leiden'],
        errors='coerce'
    ).fillna(-1).astype(int).values

    # 5) Learn principal graph & pseudotime
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

    # 6) Build gray→purple colormap
    cmap_gray_purple = LinearSegmentedColormap.from_list(
        'gray_purple',
        ['gray', 'purple']
    )

    # 7) Scatter UMAP colored by pseudotime
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        umap[:, 0],
        umap[:, 1],
        c=pseudotime,
        s=0.1,
        cmap=cmap_gray_purple,
        rasterized=True
    )

    # 8) Compute and overlay sample-centroids colored by tumor_stage
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
    for _, row in centroid_df.iterrows():
        plt.scatter(
            row['UMAP1'], row['UMAP2'],
            c=stage_colors.get(row['tumor_stage'], 'black'),
            s=30,
            edgecolors='none',
            marker='o',
            zorder=10
        )

    plt.xticks([])
    plt.yticks([])
    plt.title('Macrophage trajectory pseudotime\n(with sample-centroids by tumor stage)')
    cbar = plt.colorbar(scatter, pad=0.02)
    cbar.set_label('Pseudotime')
    plt.tight_layout()

    # 9) Save figure
    if save_path is None:
        figures_dir = Path.cwd() / 'figures'
        figures_dir.mkdir(parents=True, exist_ok=True)
        save_path = figures_dir / 'figure_3E.png'
    else:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.debug(f"Saved trajectory plot to {save_path}")
    logging.debug("=== END __figure_three_E ===")

    return mphage_cells

def __figure_three_E(adata, save_path=None):
    """
    Run Monocle pseudotime on fibroblasts, plot UMAP colored by pseudotime (gray→purple),
    and overlay sample-centroids colored by tumor_stage.
    """
    logging.debug("=== START __figure_three_G ===")
    # 1) Subset to the specified fibroblast clusters
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
    # Thorough NaN investigation for tumor_stage
    dist_pre = fibroblast_cells.obs['tumor_stage'].value_counts(dropna=False)
    logging.debug(f"tumor_stage distribution before drop (incl NaN/strings):\n{dist_pre}")
    mask = fibroblast_cells.obs['tumor_stage'].notna()
    n_missing = (~mask).sum()
    logging.debug(f"Found {n_missing} cells with missing or string 'nan' tumor_stage")
    if n_missing > 0:
        logging.debug(f"Dropping {n_missing} cells with missing tumor_stage")
        fibroblast_cells = fibroblast_cells[mask, :].copy()
    # Check post-drop distribution
    dist_post = fibroblast_cells.obs['tumor_stage'].value_counts(dropna=False)
    logging.debug(f"tumor_stage distribution after drop (incl NaN):\n{dist_post}")

    # 2) HVG selection (top 5000)
    sc.pp.highly_variable_genes(
        fibroblast_cells,
        n_top_genes=5000,
        flavor='seurat_v3',
        subset=False,
        n_bins=20
    )
    fibroblast_cells = fibroblast_cells[:, fibroblast_cells.var['highly_variable']].copy()
    logging.debug(f"After HVG subset: {fibroblast_cells.n_vars} genes")

    # 3) Neighbors & UMAP
    umap = fibroblast_cells.obsm['X_umap']

    # 4) Ensure a clustering exists for learn_graph
    if 'leiden' not in fibroblast_cells.obs:
        sc.tl.leiden(fibroblast_cells, key_added='leiden')
    clusters = pd.to_numeric(
        fibroblast_cells.obs['leiden'],
        errors='coerce'
    ).fillna(-1).astype(int).values

    # 5) Learn principal graph & pseudotime
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

    # 6) Build gray→purple colormap
    cmap_gray_purple = LinearSegmentedColormap.from_list(
        'gray_purple',
        ['gray', 'purple']
    )

    # 7) Scatter UMAP colored by pseudotime
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        umap[:, 0],
        umap[:, 1],
        c=pseudotime,
        s=0.1,
        cmap=cmap_gray_purple,
        rasterized=True
    )

    # 8) Compute and overlay sample-centroids colored by tumor_stage
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
    for _, row in centroid_df.iterrows():
        plt.scatter(
            row['UMAP1'], row['UMAP2'],
            c=stage_colors.get(row['tumor_stage'], 'black'),
            s=30,
            edgecolors='none',
            marker='o',
            zorder=10
        )

    # Final plot tweaks
    plt.xticks([])
    plt.yticks([])
    plt.title('Fibroblast trajectory pseudotime\n(with sample-centroids by tumor stage)')
    cbar = plt.colorbar(scatter, pad=0.02)
    cbar.set_label('Pseudotime')
    plt.tight_layout()

    # 9) Save figure
    if save_path is None:
        figures_dir = Path.cwd() / 'figures'
        figures_dir.mkdir(parents=True, exist_ok=True)
        save_path = figures_dir / 'figure_3G.png'
    else:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.debug(f"Saved trajectory plot to {save_path}")
    logging.debug("=== END __figure_three_G ===")

    return fibroblast_cells

def __figure_three_F_1(adata, save_dir=None):
    from pathlib import Path
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    t_cells = adata[
        adata.obs['leiden_res_20.00_celltype']
        .isin(['CD8 T cells', 'CD4 T cells', 'T cells proliferating'])
    ].copy()

    cmap_custom = LinearSegmentedColormap.from_list('gray_emerald', ['darkgray', '#50C878'])

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
    from pathlib import Path
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

    cmap_custom = LinearSegmentedColormap.from_list('gray_emerald', ['darkgray', '#50C878'])

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
    from pathlib import Path
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

    cmap_custom = LinearSegmentedColormap.from_list('gray_emerald', ['darkgray', '#50C878'])

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


def __figure_three_G(adata, save_path = None):
    stages = ['non-cancer', 'early', 'advanced']
    stage_colors = {
        'non-cancer': '#84A970',
        'early':      '#E4C282',
        'advanced':   '#FF8C00'
    }

    # Extract the relevant data and drop missing stages
    df = (
        adata.obs
        .loc[:, ['tumor_stage', 'HALLMARK_IL6_JAK_STAT3_SIGNALING']]
        .dropna(subset=['tumor_stage'])
    )

    # Build a list of arrays, one per stage (in the defined order)
    data_by_stage = [
        df.loc[df['tumor_stage'] == st, 'HALLMARK_IL6_JAK_STAT3_SIGNALING'].values
        for st in stages
    ]

    # Create the boxplot
    fig, ax = plt.subplots(figsize=(6, 6))
    bp = ax.boxplot(
        data_by_stage,
        patch_artist=True,
        widths=0.6,
        medianprops=dict(linewidth=1.5),
        showfliers=False
    )

    # Color each box
    for patch, st in zip(bp['boxes'], stages):
        patch.set_facecolor(stage_colors[st])
        patch.set_edgecolor('black')

    # Tidy up axes
    ax.set_xticks(range(1, len(stages) + 1))
    ax.set_xticklabels(stages, rotation=30)
    ax.set_ylabel('HALLMARK_IL6_JAK_STAT score')
    ax.set_title('IL6/JAK/STAT pathway activity by tumor stage')

    # Add a legend for colors
    legend_handles = [Patch(facecolor=stage_colors[st], label=st) for st in stages]
    ax.legend(
        handles=legend_handles,
        title='Tumor stage',
        loc='upper right',
        frameon=False
    )

    plt.tight_layout()

    # Save or show
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

    # Extract FAP expression and tumor_stage
    # handle sparse vs dense
    fap_raw = fibroblast_cells[:, 'FAP'].X
    try:
        fap_vals = fap_raw.toarray().flatten()
    except AttributeError:
        fap_vals = np.array(fap_raw).flatten()

    df = pd.DataFrame({
        'FAP': fap_vals,
        'tumor_stage': fibroblast_cells.obs['tumor_stage'].values
    }).dropna(subset=['tumor_stage'])

    # Compute percentages
    pct_expressing = []
    for st in stages:
        sub = df[df['tumor_stage'] == st]
        if len(sub) == 0:
            pct = 0.0
        else:
            pct = (sub['FAP'] > 0).sum() / len(sub) * 100
        pct_expressing.append(pct)

    # Plot
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

    # Save or show
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

def __figure_three_I(adata, save_path=None):
    # 1. Subset T cells
    t_cells = adata[
        adata.obs['leiden_res_20.00_celltype']
        .isin(['CD8 T cells', 'CD4 T cells', 'T cells proliferating'])
    ].copy()
    t_cells.obs['tumor_stage'] = t_cells.obs['tumor_stage'].replace(['nan', 'NaN', 'NAN'], np.nan)
    t_cells = t_cells[~t_cells.obs['tumor_stage'].isna()].copy()

    # 2. Define genes
    genes = ['STAT4', 'CCR7', 'LAG3']
    for gene in genes:
        if gene not in t_cells.var_names:
            raise ValueError(f"Gene '{gene}' not found in adata.")

    # 3. Extract expression and compute medians
    expr_data = {
        gene: t_cells[:, gene].X.toarray().flatten()
        if hasattr(t_cells[:, gene].X, 'toarray') else np.array(t_cells[:, gene].X).flatten()
        for gene in genes
    }
    medians = {gene: np.median(expr_data[gene]) for gene in genes}

    # 4. Generate phenotype strings per cell (with no underscores)
    def get_phenotype(i):
        return ''.join([
            f"{g}{'+' if expr_data[g][i] > medians[g] else '-'}"
            for g in genes
        ])

    t_cells.obs['phenotype'] = [get_phenotype(i) for i in range(t_cells.n_obs)]

    # 5. Compute percentages per tumor stage
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


    # 6. Plot stacked bars with enhancements
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

    # Axis styling
    ax.set_ylabel('Percentage of T cells')
    ax.set_title('T-cell phenotypes by tumor stage\n(> median expression of STAT4, CCR7, LAG3)')
    ax.set_ylim(0, 100)

    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(axis='both', which='both', length=0)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)

    # 7. Elegant circular legend without box
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

    # 8. Save or show
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

def __figure_three_J(adata, save_path=None):

    # 1. Subset myeloid cells
    myeloid = adata[
        adata.obs['leiden_res_20.00_celltype']
        .isin([
            'Alveolar macrophages',
            'Alveolar Mφ MT-positive',
            'Alveolar Mφ proliferating',
            'Interstitial Mφ perivascular',
            'Alveolar Mφ CCL3+'
        ])
    ].copy()

    myeloid.obs['tumor_stage'] = myeloid.obs['tumor_stage'].replace(['nan', 'NaN', 'NAN'], np.nan)
    myeloid = myeloid[~myeloid.obs['tumor_stage'].isna()].copy()

    # 2. Define genes
    genes = ['CD68', 'CD86', 'CD163']
    for gene in genes:
        if gene not in myeloid.var_names:
            raise ValueError(f"Gene '{gene}' not found in adata.")

    # 3. Extract expression and compute medians
    expr_data = {
        gene: myeloid[:, gene].X.toarray().flatten()
        if hasattr(myeloid[:, gene].X, 'toarray') else np.array(myeloid[:, gene].X).flatten()
        for gene in genes
    }
    medians = {gene: np.median(expr_data[gene]) for gene in genes}

    # 4. Phenotype strings
    def get_phenotype(i):
        return ''.join([
            f"{g}{'+' if expr_data[g][i] > medians[g] else '-'}"
            for g in genes
        ])
    myeloid.obs['phenotype'] = [get_phenotype(i) for i in range(myeloid.n_obs)]

    # 5. Percentages
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

    # 6. Plot
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

    # 1. Subset fibroblasts
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

    # 2. Define genes
    genes = ['FAP', 'ACTA2', 'COL1A1']
    for gene in genes:
        if gene not in fibro.var_names:
            raise ValueError(f"Gene '{gene}' not found in adata.")

    # 3. Extract expression and compute medians
    expr_data = {
        gene: fibro[:, gene].X.toarray().flatten()
        if hasattr(fibro[:, gene].X, 'toarray') else np.array(fibro[:, gene].X).flatten()
        for gene in genes
    }
    medians = {gene: np.median(expr_data[gene]) for gene in genes}

    # 4. Phenotype strings
    def get_phenotype(i):
        return ''.join([
            f"{g}{'+' if expr_data[g][i] > medians[g] else '-'}"
            for g in genes
        ])
    fibro.obs['phenotype'] = [get_phenotype(i) for i in range(fibro.n_obs)]

    # 5. Percentages
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

    # 6. Plot
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
    import matplotlib.pyplot as plt
    from pathlib import Path
    import numpy as np
    import scanpy as sc

    # 1. Compute mean scores and binarize
    kras_mean = adata.obs["HALLMARK_KRAS_SIGNALING_UP"].mean()
    egfr_mean = adata.obs["REACTOME_SIGNALING_BY_EGFR_IN_CANCER"].mean()
    adata.obs["KRAS_high"] = adata.obs["HALLMARK_KRAS_SIGNALING_UP"] > kras_mean
    adata.obs["EGFR_high"] = adata.obs["REACTOME_SIGNALING_BY_EGFR_IN_CANCER"] > egfr_mean

    print(f"HALLMARK_KRAS_SIGNALING_UP mean = {kras_mean:.3f}")
    print(adata.obs["KRAS_high"].value_counts())
    print(f"REACTOME_SIGNALING_BY_EGFR_IN_CANCER mean = {egfr_mean:.3f}")
    print(adata.obs["EGFR_high"].value_counts())

    # 2. Define group labels
    conditions = [
        (~adata.obs["KRAS_high"]) & (~adata.obs["EGFR_high"]),
        ( adata.obs["KRAS_high"]) & (~adata.obs["EGFR_high"]),
        (~adata.obs["KRAS_high"]) & ( adata.obs["EGFR_high"]),
        ( adata.obs["KRAS_high"]) & ( adata.obs["EGFR_high"]),
    ]
    labels = ["Neither", "KRAS only", "EGFR only", "Both"]
    adata.obs["KRAS_EGFR_group"] = np.select(conditions, labels, default="Neither")

    # 3. Subsets and genes
    subsets = {
        "Macrophages": (
            adata[adata.obs['leiden_res_20.00_celltype'].isin([
                'Alveolar macrophages', 'Alveolar Mφ MT-positive',
                'Alveolar Mφ proliferating', 'Interstitial Mφ perivascular',
                'Alveolar Mφ CCL3+'
            ])],
            "MARCO"
        ),
        "Fibroblasts": (
            adata[adata.obs['leiden_res_20.00_celltype'].isin([
                "Peribronchial fibroblasts", "Adventitial fibroblasts",
                "Alveolar fibroblasts", "Subpleural fibroblasts",
                "Myofibroblasts", "Fibromyocytes"
            ])],
            "FN1"
        ),
    }

    # 4. Plot 1x2 violins
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=False)

    for ax, (title, (subset, gene)) in zip(axes, subsets.items()):
        sc.pl.violin(
            subset,
            keys=gene,
            groupby="KRAS_EGFR_group",
            order=labels,
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

    # 5. Save or show
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

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
    __figure_three_I(adata, save_path=str(BASE_DIR / 'figures/fig-3I.png'))
    __figure_three_J(adata, save_path=str(BASE_DIR / 'figures/fig-3J.png'))
    __figure_three_K(adata, save_path=str(BASE_DIR / 'figures/fig-3K.png'))
    __figure_three_L(adata, save_path=str(BASE_DIR / 'figures/fig-3L.png'))
