import scanpy as sc
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

EPITHELIAL_CATS = [
    'AT2', 'AT1', 'Suprabasal', 'Basal resting',
    'Multiciliated (non-nasal)', 'Goblet (nasal)',
    'Club (nasal)', 'Ciliated (nasal)',
    'Club (non-nasal)', 'Multiciliated (nasal)',
    'Goblet (bronchial)', 'Transitional Club AT2',
    'AT2 proliferating', 'Goblet (subsegmental)'
]

def __figure_two_A(adata, save_path=None):
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

    if save_path is None:
        figures_dir = Path.cwd() / 'figures'
        figures_dir.mkdir(parents=True, exist_ok=True)
        save_path = figures_dir / 'figure_2A.png'
    else:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')

def __figure_two_B(adata, save_path=None):
    epi = adata[adata.obs['leiden_res_20.00_celltype'].isin(EPITHELIAL_CATS)].copy()

    cmap_custom = LinearSegmentedColormap.from_list('gray_emerald', ['darkgray', '#50C878'])

    genes = ['AGER', 'HOPX', 'ABCA3', 'SFTPC', 'MUC1', 'SCGB1A1']

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    plt.subplots_adjust(wspace=0.1, hspace=0.3)

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

    expr = epi[:, genes[0]].X
    expr_arr = expr.toarray().flatten() if hasattr(expr, 'toarray') else np.array(expr).flatten()
    sm = plt.cm.ScalarMappable(cmap=cmap_custom)
    sm.set_array(expr_arr)
    sm.set_clim(expr_arr.min(), expr_arr.max())
    cbar = fig.colorbar(sm, ax=axes, fraction=0.015, pad=0.04)
    cbar.set_label('Expression')

    zoom_frac = 0.8

    umap = epi.obsm['X_umap']
    x, y = umap[:, 0], umap[:, 1]

    x_c, y_c = x.mean(), y.mean()
    half_x = (x.max() - x.min()) * zoom_frac / 2
    half_y = (y.max() - y.min()) * zoom_frac / 2

    for ax in axes:
        ax.set_xlim(x_c - half_x, x_c + half_x)
        ax.set_ylim(y_c - half_y, y_c + half_y)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def __figure_two_B_2(adata, save_path=None):
    epi = adata[adata.obs['leiden_res_20.00_celltype'].isin(EPITHELIAL_CATS)].copy()

    scores = [
        'REACTOME_SIGNALING_BY_EGFR_IN_CANCER',
        'HALLMARK_G2M_CHECKPOINT',
        'HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION',
        'HALLMARK_APOPTOSIS',
        'HALLMARK_P53_PATHWAY',
        'HALLMARK_KRAS_SIGNALING_UP'
    ]

    cmap_custom = LinearSegmentedColormap.from_list('gray_emerald', ['darkgray', '#50C878'])

    all_vals = epi.obs[scores].values.flatten()
    vmin, vmax = np.nanmin(all_vals), np.nanmax(all_vals)

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

    sm = plt.cm.ScalarMappable(cmap=cmap_custom)
    sm.set_clim(vmin, vmax)
    sm.set_array([])  # dummy
    cbar = fig.colorbar(sm, ax=axes, fraction=0.015, pad=0.04)
    cbar.set_label('Pathway score', fontsize=12)

    zoom_frac = 0.8 

    umap = epi.obsm['X_umap']
    x, y = umap[:, 0], umap[:, 1]

    x_c, y_c = x.mean(), y.mean()
    half_x = (x.max() - x.min()) * zoom_frac / 2
    half_y = (y.max() - y.min()) * zoom_frac / 2

    for ax in axes:
        ax.set_xlim(x_c - half_x, x_c + half_x)
        ax.set_ylim(y_c - half_y, y_c + half_y)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def __figure_two_C(adata, save_path=None):
    epi = adata[adata.obs['leiden_res_20.00_celltype'].isin(EPITHELIAL_CATS)].copy()

    epi.obs['tumor_stage'] = epi.obs['tumor_stage'].replace(['nan', 'NaN', 'NAN'], np.nan)
    dist_pre = epi.obs['tumor_stage'].value_counts(dropna=False)

    cols = [
        'HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION',
        'HALLMARK_APOPTOSIS',
        'HALLMARK_KRAS_SIGNALING_UP',
        'REACTOME_SIGNALING_BY_EGFR_IN_CANCER'
    ]
    labels = ['EMT', 'Apoptosis', 'KRAS', 'EGFR']

    stages = ['non-cancer', 'early', 'advanced']
    stage_colors = {
        'non-cancer': '#84A970',
        'early':      '#E4C282',
        'advanced':   '#FF8C00'
    }

    n_groups, n_stages = len(cols), len(stages)

    width = 0.1     
    group_spacing = width * n_stages 
    offsets = np.linspace(
        -group_spacing/2 + width/2,
        group_spacing/2 - width/2,
        n_stages
    )

    fig, ax = plt.subplots(figsize=(5, 5))

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

    ax.set_xticks(np.arange(n_groups))
    ax.set_xticklabels(labels, rotation=0, fontsize=12)

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
        frameon=False,      
        handlelength=1.5,  
        handleheight=0.8,       
        fontsize=10,            
        title_fontsize=11      
    )

    plt.tight_layout()
    plt.subplots_adjust(right=0.75) 

    plt.savefig(save_path, dpi=300, bbox_inches='tight')

def __figure_two_G(adata, save_path=None):
    kras_mean = adata.obs["HALLMARK_KRAS_SIGNALING_UP"].mean()
    egfr_mean = adata.obs["REACTOME_SIGNALING_BY_EGFR_IN_CANCER"].mean()

    adata.obs["KRAS_high"] = adata.obs["HALLMARK_KRAS_SIGNALING_UP"] > kras_mean
    adata.obs["EGFR_high"] = adata.obs["REACTOME_SIGNALING_BY_EGFR_IN_CANCER"] > egfr_mean

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

    categories = [
        'Both',        
        'EGFR only',   
        'KRAS only',   
        'None'       
    ]
    colors = {
        'Both':      '#84A970',
        'EGFR only': '#E4C282',
        'KRAS only': '#FF8C00',
        'None':      '#999999'
    }

    n_groups  = len(cols)
    n_cats    = len(categories)
    width     = 0.1
    grp_space = width * n_cats
    offsets   = np.linspace(
        -grp_space/2 + width/2,
        grp_space/2 - width/2,
        n_cats
    )

    fig, ax = plt.subplots(figsize=(5, 5))

    for i, cat in enumerate(categories):
        if cat == 'Both':
            mask = adata.obs['KRAS_high'] & adata.obs['EGFR_high']
        elif cat == 'EGFR only':
            mask = ~adata.obs['KRAS_high'] & adata.obs['EGFR_high']
        elif cat == 'KRAS only':
            mask = adata.obs['KRAS_high'] & ~adata.obs['EGFR_high']
        else:  
            mask = ~adata.obs['KRAS_high'] & ~adata.obs['EGFR_high']

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
        for box in bp['boxes']:
            box.set_facecolor(colors[cat])
            box.set_edgecolor('black')
        for elt in bp['whiskers'] + bp['caps'] + bp['medians']:
            elt.set_color('black')

    ax.set_xticks(np.arange(n_groups))
    ax.set_xticklabels(labels, rotation=0, fontsize=12)
    ax.set_ylabel('Pathway score')

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
        frameon=False,        
        handlelength=1.5,     
        handleheight=0.8,      
        fontsize=10,          
        title_fontsize=11    
    )

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