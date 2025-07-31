import scanpy as sc
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

# Global list of epithelial cell type categories
EPITHELIAL_CATS = [
    'AT2', 'AT1', 'Suprabasal', 'Basal resting',
    'Multiciliated (non-nasal)', 'Goblet (nasal)',
    'Club (nasal)', 'Ciliated (nasal)',
    'Club (non-nasal)', 'Multiciliated (nasal)',
    'Goblet (bronchial)', 'Transitional Club AT2',
    'AT2 proliferating', 'Goblet (subsegmental)'
]

def __figure_two_A(adata, save_path=None):
    """
    Plot UMAP embedding of adata, highlighting epithelial cell types in color, graying out others,
    and saving the figure as a high-quality image.
    """
    # Mark non-epithelial cells as 'Other'
    orig = adata.obs['leiden_res_20.00_celltype'].cat.add_categories(['Other'])

    adata.obs['highlight'] = (
        orig
        .where(orig.isin(EPITHELIAL_CATS), other='Other')
        .cat
        .remove_unused_categories()
    )

    base_pal = sc.pl.palettes.default_20
    if len(EPITHELIAL_CATS) > len(base_pal):
        palette_vals = plt.cm.tab20.colors * ((len(EPITHELIAL_CATS) // len(base_pal)) + 1)
    else:
        palette_vals = base_pal

    palette = {cat: palette_vals[i] for i, cat in enumerate(EPITHELIAL_CATS)}
    palette['Other'] = '#d3d3d3'

    # Plot UMAP
    sc.pl.umap(
        adata,
        color='highlight',
        palette=palette,
        size=0.5,
        legend_loc='right margin',
        title='Epithelial cell types (colored) vs others (gray)',
        show=False
    )
    plt.tight_layout()

    # Ensure the output directory exists, then save
    if save_path is None:
        figures_dir = Path.cwd() / 'figures'
        figures_dir.mkdir(parents=True, exist_ok=True)
        save_path = figures_dir / 'figure_2A.png'
    else:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')

def __figure_two_B(adata, save_path=None):
    """
    Subset to epithelial cells and plot UMAPs colored by six marker genes
    arranged in a 2×3 grid with a custom gray-to-emerald colormap and a single colorbar,
    then save as a high-quality image.
    """
    # Subset to epithelial cells
    epi = adata[adata.obs['leiden_res_20.00_celltype'].isin(EPITHELIAL_CATS)].copy()

    # Custom colormap
    cmap_custom = LinearSegmentedColormap.from_list('gray_emerald', ['darkgray', '#50C878'])

    # Marker genes to plot
    genes = ['AGER', 'HOPX', 'ABCA3', 'SFTPC', 'MUC1', 'SCGB1A1']

    # Create 2×3 subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    plt.subplots_adjust(wspace=0.1, hspace=0.3)

    # Plot each gene UMAP
    for i, gene in enumerate(genes):
        ax = axes[i]
        sc.pl.umap(
            epi,
            color=gene,
            size=0.5,
            sort_order=True,
            cmap=cmap_custom,
            ax=ax,
            show=False,
            colorbar_loc=None
        )
        # Tidy axes
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if i % 3 == 0:
            ax.set_ylabel('X2')
        else:
            ax.spines['left'].set_visible(False)
            ax.set_yticks([])
            ax.set_ylabel('')
        ax.set_xlabel('X1')
        ax.set_title(f'{gene} Expression')
        for spine in ax.spines.values():
            if spine.get_visible():
                spine.set_linewidth(1.5)
        ax.tick_params(width=1.5)

    # Shared colorbar
    expr = epi[:, genes[0]].X
    expr_arr = expr.toarray().flatten() if hasattr(expr, 'toarray') else np.array(expr).flatten()
    sm = plt.cm.ScalarMappable(cmap=cmap_custom)
    sm.set_array(expr_arr)
    sm.set_clim(expr_arr.min(), expr_arr.max())
    cbar = fig.colorbar(sm, ax=axes, fraction=0.015, pad=0.04)
    cbar.set_label('Expression')

    # --- Symmetric zoom around the data‐center ---
    zoom_frac = 0.8  # keep 80% of the span; tweak between 0 and 1

    # grab the UMAP coords
    umap = epi.obsm['X_umap']
    x, y = umap[:, 0], umap[:, 1]

    # compute center and half‐ranges
    x_c, y_c = x.mean(), y.mean()
    half_x = (x.max() - x.min()) * zoom_frac / 2
    half_y = (y.max() - y.min()) * zoom_frac / 2

    # apply identical limits to all axes
    for ax in axes:
        ax.set_xlim(x_c - half_x, x_c + half_x)
        ax.set_ylim(y_c - half_y, y_c + half_y)

    # Save figure
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def __figure_two_B_2(adata, save_path=None):
    epi = adata[adata.obs['leiden_res_20.00_celltype'].isin(EPITHELIAL_CATS)].copy()

    # Your six pathway‐score columns
    scores = [
        'REACTOME_SIGNALING_BY_EGFR_IN_CANCER',
        'HALLMARK_G2M_CHECKPOINT',
        'HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION',
        'HALLMARK_APOPTOSIS',
        'HALLMARK_P53_PATHWAY',
        'HALLMARK_KRAS_SIGNALING_UP'
    ]

    # Custom colormap
    cmap_custom = LinearSegmentedColormap.from_list('gray_emerald', ['darkgray', '#50C878'])

    # Compute global vmin/vmax across all six score columns for a shared color‐scale
    all_vals = epi.obs[scores].values.flatten()
    vmin, vmax = np.nanmin(all_vals), np.nanmax(all_vals)

    # Set up 2×3 axes
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    plt.subplots_adjust(wspace=0.1, hspace=0.3)

    for i, score in enumerate(scores):
        ax = axes[i]
        sc.pl.umap(
            epi,
            color=score,
            size=0.5,
            sort_order=True,
            cmap=cmap_custom,
            vmin=vmin,
            vmax=vmax,
            ax=ax,
            show=False,
            colorbar_loc=None
        )

        # tidy up the axes
        ax.set_title(score.replace('_', ' '), fontsize=14)
        ax.set_xlabel('X1')
        if i % 3 == 0:
            ax.set_ylabel('X2')
        else:
            ax.spines['left'].set_visible(False)
            ax.set_yticks([])
            ax.set_ylabel('')
        for spine in ('top','right'):
            ax.spines[spine].set_visible(False)
        for spine in ax.spines.values():
            if spine.get_visible():
                spine.set_linewidth(1.5)
        ax.tick_params(width=1.5)

    # Single shared colorbar at the bottom
    sm = plt.cm.ScalarMappable(cmap=cmap_custom)
    sm.set_clim(vmin, vmax)
    sm.set_array([])  # dummy
    cbar = fig.colorbar(sm, ax=axes, fraction=0.015, pad=0.04)
    cbar.set_label('Pathway score', fontsize=12)

    # --- Symmetric zoom around the data‐center ---
    zoom_frac = 0.8  # keep 80% of the span; tweak between 0 and 1

    # grab the UMAP coords
    umap = epi.obsm['X_umap']
    x, y = umap[:, 0], umap[:, 1]

    # compute center and half‐ranges
    x_c, y_c = x.mean(), y.mean()
    half_x = (x.max() - x.min()) * zoom_frac / 2
    half_y = (y.max() - y.min()) * zoom_frac / 2

    # apply identical limits to all axes
    for ax in axes:
        ax.set_xlim(x_c - half_x, x_c + half_x)
        ax.set_ylim(y_c - half_y, y_c + half_y)

    # Save
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def __figure_two_C(adata, save_path=None):
    epi = adata[adata.obs['leiden_res_20.00_celltype'].isin(EPITHELIAL_CATS)].copy()

    epi.obs['tumor_stage'] = epi.obs['tumor_stage'].replace(['nan', 'NaN', 'NAN'], np.nan)
    dist_pre = epi.obs['tumor_stage'].value_counts(dropna=False)

    # 1) define your obs‐columns and nicer labels
    cols = [
        'HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION',
        'HALLMARK_APOPTOSIS',
        'HALLMARK_KRAS_SIGNALING_UP',
        'REACTOME_SIGNALING_BY_EGFR_IN_CANCER'
    ]
    labels = ['EMT', 'Apoptosis', 'KRAS', 'EGFR']

    # 2) the three tumor stages (in the order you want them plotted)
    stages = ['non-cancer', 'early', 'advanced']
    stage_colors = {
        'non-cancer': '#84A970',
        'early':      '#E4C282',
        'advanced':   '#FF8C00'
    }

    # 3) some layout parameters
    n_groups, n_stages = len(cols), len(stages)

    width = 0.1                     # <- slimmer than before
    group_spacing = width * n_stages # <- exactly enough to hold n_stages boxes
    # offsets run from -group_spacing/2 + width/2  to  +group_spacing/2 - width/2
    offsets = np.linspace(
        -group_spacing/2 + width/2,
        group_spacing/2 - width/2,
        n_stages
    )

    # 4) prepare figure
    fig, ax = plt.subplots(figsize=(5, 5))

    # 5) for each stage, make a list of arrays (one per column) and call boxplot
    for i, stage in enumerate(stages):
        data = [ 
            epi.obs.loc[epi.obs['tumor_stage']==stage, col].dropna().values
            for col in cols
        ]
        positions = np.arange(n_groups) + offsets[i]
        bp = ax.boxplot(
            data,
            positions=positions,
            widths=width,
            patch_artist=True,
            showfliers=False
        )
        # color them
        for patch in bp['boxes']:
            patch.set_facecolor(stage_colors[stage])
            patch.set_edgecolor('black')
        for whisker in bp['whiskers']:
            whisker.set_color('black')
        for cap in bp['caps']:
            cap.set_color('black')
        for median in bp['medians']:
            median.set_color('black')

    # 6) set x‐ticks & labels
    ax.set_xticks(np.arange(n_groups))
    ax.set_xticklabels(labels, rotation=0, fontsize=12)

    # 7) legend
    handles = [
        plt.Line2D([0], [0], color=stage_colors[s], lw=6)
        for s in stages
    ]
    ax.legend(
        handles, stages,
        title='Tumor stage',
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        borderaxespad=0,
        frameon=False,          # <- no legend frame
        handlelength=1.5,       # <- shorter lines
        handleheight=0.8,       # <- a bit taller
        fontsize=10,            # <- slightly smaller font
        title_fontsize=11       # <- title font
    )

    # 8) tighten up the layout and make room for legend
    plt.tight_layout()
    plt.subplots_adjust(right=0.75)  # leave 25% of width on the right for the legend

    plt.savefig(save_path, dpi=300, bbox_inches='tight')

def __figure_two_G(adaa, save_path = None):
    # compute overall means
    kras_mean = adata.obs["HALLMARK_KRAS_SIGNALING_UP"].mean()
    egfr_mean = adata.obs["REACTOME_SIGNALING_BY_EGFR_IN_CANCER"].mean()

    # annotate boolean flags
    adata.obs["KRAS_high"] = adata.obs["HALLMARK_KRAS_SIGNALING_UP"] > kras_mean
    adata.obs["EGFR_high"] = adata.obs["REACTOME_SIGNALING_BY_EGFR_IN_CANCER"] > egfr_mean

    # sanity‐check
    print(f"HALLMARK_KRAS_SIGNALING_UP mean = {kras_mean:.3f}")
    print(adata.obs["KRAS_high"].value_counts())
    print(f"REACTOME_SIGNALING_BY_EGFR_IN_CANCER mean = {egfr_mean:.3f}")
    print(adata.obs["EGFR_high"].value_counts())

    cols   = [
        'REACTOME_CELL_CYCLE',
        'HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION',
        'HALLMARK_APOPTOSIS'
    ]
    labels = ['Cell cycle', 'EMT', 'Apoptosis']

    # 2) the four annotation categories + colors
    categories = [
        'Both',        # KRAS_high & EGFR_high
        'EGFR only',   # ~KRAS_high & EGFR_high
        'KRAS only',   # KRAS_high & ~EGFR_high
        'None'         # neither
    ]
    colors = {
        'Both':      '#84A970',
        'EGFR only': '#E4C282',
        'KRAS only': '#FF8C00',
        'None':      '#999999'
    }

    # 3) layout parameters
    n_groups  = len(cols)
    n_cats    = len(categories)
    width     = 0.1
    grp_space = width * n_cats
    offsets   = np.linspace(
        -grp_space/2 + width/2,
        grp_space/2 - width/2,
        n_cats
    )

    # 4) prep figure
    fig, ax = plt.subplots(figsize=(5, 5))

    # 5) for each annotation category, collect data & plot
    for i, cat in enumerate(categories):
        if cat == 'Both':
            mask = adata.obs['KRAS_high'] & adata.obs['EGFR_high']
        elif cat == 'EGFR only':
            mask = ~adata.obs['KRAS_high'] & adata.obs['EGFR_high']
        elif cat == 'KRAS only':
            mask = adata.obs['KRAS_high'] & ~adata.obs['EGFR_high']
        else:  # None
            mask = ~adata.obs['KRAS_high'] & ~adata.obs['EGFR_high']

        # gather arrays for each pathway
        data = [
            adata.obs.loc[mask, col].dropna().values
            for col in cols
        ]
        positions = np.arange(n_groups) + offsets[i]

        bp = ax.boxplot(
            data,
            positions=positions,
            widths=width,
            patch_artist=True,
            showfliers=False
        )
        # color boxes
        for box in bp['boxes']:
            box.set_facecolor(colors[cat])
            box.set_edgecolor('black')
        for elt in bp['whiskers'] + bp['caps'] + bp['medians']:
            elt.set_color('black')

    # 6) x‐axis ticks & labels
    ax.set_xticks(np.arange(n_groups))
    ax.set_xticklabels(labels, rotation=0, fontsize=12)
    ax.set_ylabel('Pathway score')

    # 7) legend
    handles = [
        plt.Line2D([0], [0], color=colors[c], lw=6)
        for c in categories
    ]
    ax.legend(
        handles, categories,
        title='Annotation',
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        borderaxespad=0,
        frameon=False,          # <- no legend frame
        handlelength=1.5,       # <- shorter lines
        handleheight=0.8,       # <- a bit taller
        fontsize=10,            # <- slightly smaller font
        title_fontsize=11       # <- title font
    )

    # 8) finalize
    plt.tight_layout()
    plt.subplots_adjust(right=0.75)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    BASE_DIR = Path.cwd()
    adata = sc.read_h5ad(filename=str(BASE_DIR / 'data/processed_data.h5ad'))
    __figure_two_A(adata, save_path=str(BASE_DIR / 'figures/figure_2A.png'))
    __figure_two_B(adata, save_path=str(BASE_DIR / 'figures/figure_2B.png'))
    __figure_two_B_2(adata, save_path=str(BASE_DIR / 'figures/figure_2B_2.png'))
    __figure_two_C(adata, save_path=str(BASE_DIR / 'figures/figure_2C.png'))
    __figure_two_G(adata, save_path=str(BASE_DIR / 'figures/figure_2G.png'))