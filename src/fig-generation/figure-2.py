import scanpy as sc
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd

import itertools
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

EPITHELIAL_CATS = [
    'AT2', 'AT1', 'Suprabasal', 'Basal resting',
    'Multiciliated (non-nasal)', 'Goblet (nasal)',
    'Club (nasal)', 'Ciliated (nasal)',
    'Club (non-nasal)', 'Multiciliated (nasal)',
    'Goblet (bronchial)', 'Transitional Club AT2',
    'AT2 proliferating', 'Goblet (subsegmental)'
]

TARGET_PROJECTS = ['SD2_raw_feature_bc_matrix','SD3_raw_feature_bc_matrix','SD6_raw_feature_bc_matrix','SD7_raw_feature_bc_matrix',
                   'SD8_raw_featuresbc_matrix','SD10_raw_feature_bc_matrix','SD12_raw_feature_bc_matrix', 'SD14_raw_feature_bc_matrix', 'SD15_raw_feature_bc_matrix',
                   'SD16_raw_feature_bc_matrix']

def _finite_minmax(arr):
    arr = np.asarray(arr).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        # fallback to a harmless span so colorbar works
        return (0.0, 1.0)
    vmin, vmax = float(np.nanmin(arr)), float(np.nanmax(arr))
    # avoid vmin == vmax
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        eps = 1e-6 if np.isfinite(vmin) else 0.0
        return (float(vmin - eps), float(vmax + eps if np.isfinite(vmax) else 1.0))
    return (vmin, vmax)

def _warn_if_no_targets(mask, where=""):
    if not np.any(mask):
        print(f"[WARN] No target (SD*) cells found{f' in {where}' if where else ''}. "
              "Rendering gray background only.")

def _umap_gray_bg_then_color(adata_subset, values, mask_targets, ax, cmap, size=1.0, vmin=None, vmax=None):
    """Gray non-target cells, then color target cells by `values` on the same axes."""
    coords = adata_subset.obsm['X_umap']
    ax.scatter(coords[~mask_targets, 0], coords[~mask_targets, 1],
               s=size, c='#A9A9A9', alpha=0.25, linewidths=0)
    # colored overlay (targets)
    sc = ax.scatter(coords[mask_targets, 0], coords[mask_targets, 1],
                    s=size, c=values[mask_targets], cmap=cmap,
                    vmin=vmin, vmax=vmax, linewidths=0)
    return sc


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
    palette['Other'] = '#A9A9A9'

    sc.pl.umap(
        adata,
        color='highlight',
        palette=palette,
        size=1,
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
    mask_targets = epi.obs['project'].isin(TARGET_PROJECTS).values

    cmap_custom = LinearSegmentedColormap.from_list('gray_emerald', ['darkgray', '#50C878'])
    genes = ['AGER', 'HOPX', 'ABCA3', 'SFTPC', 'MUC1', 'SCGB1A1']

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    plt.subplots_adjust(wspace=0.1, hspace=0.3)

    last_map = None
    def _to_array(x):
        return x.toarray().ravel() if hasattr(x, "toarray") else np.asarray(x).ravel()

    if not np.any(mask_targets):
        _warn_if_no_targets(mask_targets, "epithelial subset")

    for i, gene in enumerate(genes):
        ax = axes[i]
        expr = _to_array(epi[:, gene].X)

        # prefer scaling on target cells; if none/NaN-only, fall back to all finite
        target_vals = expr[mask_targets]
        if np.any(mask_targets) and np.isfinite(target_vals).any():
            vmin, vmax = _finite_minmax(target_vals)
        else:
            vmin, vmax = _finite_minmax(expr)

        last_map = _umap_gray_bg_then_color(
            adata_subset=epi,
            values=expr,
            mask_targets=mask_targets,
            ax=ax,
            cmap=cmap_custom,
            size=1.0,
            vmin=vmin,
            vmax=vmax
        )

        ax.set_title(f'{gene} Expression')
        ax.set_xlabel('X1')
        if i % 3 == 0:
            ax.set_ylabel('X2')
        else:
            ax.spines['left'].set_visible(False)
            ax.set_yticks([])
            ax.set_ylabel('')
        for s in ('top','right'):
            ax.spines[s].set_visible(False)
        for sp in ax.spines.values():
            if sp.get_visible():
                sp.set_linewidth(1.5)
        ax.tick_params(width=1.5)

    # if nothing colored, fabricate a valid mappable for the colorbar
    if last_map is None:
        sm = plt.cm.ScalarMappable(cmap=cmap_custom)
        sm.set_clim(0.0, 1.0)
        last_map = sm

    cbar = fig.colorbar(last_map, ax=axes, fraction=0.015, pad=0.04)
    cbar.set_label('Expression')

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

    mask_targets = epi.obs['project'].isin(TARGET_PROJECTS).values
    if not np.any(mask_targets):
        _warn_if_no_targets(mask_targets, "epithelial subset (scores)")

    # shared scaling: prefer target cells; else all cells
    if np.any(mask_targets):
        all_vals_target = epi.obs.loc[mask_targets, scores].values.astype(float).ravel()
        vmin, vmax = _finite_minmax(all_vals_target)
    else:
        all_vals = epi.obs[scores].values.astype(float).ravel()
        vmin, vmax = _finite_minmax(all_vals)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    plt.subplots_adjust(wspace=0.1, hspace=0.3)

    last_map = None
    for i, score in enumerate(scores):
        ax = axes[i]
        vals = epi.obs[score].values.astype(float)

        last_map = _umap_gray_bg_then_color(
            adata_subset=epi,
            values=vals,
            mask_targets=mask_targets,
            ax=ax,
            cmap=cmap_custom,
            size=1.0,
            vmin=vmin,
            vmax=vmax
        )

        ax.set_title(score.replace('_',' '), fontsize=14)
        ax.set_xlabel('X1')
        if i % 3 == 0:
            ax.set_ylabel('X2')
        else:
            ax.spines['left'].set_visible(False)
            ax.set_yticks([])
            ax.set_ylabel('')
        for s in ('top','right'):
            ax.spines[s].set_visible(False)
        for sp in ax.spines.values():
            if sp.get_visible():
                sp.set_linewidth(1.5)
        ax.tick_params(width=1.5)

    if last_map is None:
        sm = plt.cm.ScalarMappable(cmap=cmap_custom)
        sm.set_clim(vmin, vmax)
        last_map = sm

    cbar = fig.colorbar(last_map, ax=axes, fraction=0.015, pad=0.04)
    cbar.set_label('Pathway score', fontsize=12)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def __figure_two_C(adata, save_path=None):
    epi = adata[
        (adata.obs['leiden_res_0.10_celltype'].isin(EPITHELIAL_CATS))  
        & (adata.obs['project'].isin(TARGET_PROJECTS))
    ].copy()

    epi.obs['tumor_stage'] = epi.obs['project'].map({
        'SD2_raw_feature_bc_matrix': 'early',
        'SD3_raw_feature_bc_matrix': 'non-cancer',
        'SD4_raw_feature_bc_matrix': 'non-cancer',
        'SD6_raw_feature_bc_matrix': 'non-cancer',
        'SD7_raw_feature_bc_matrix': 'advanced',
        'SD8_raw_feature_bc_matrix': 'early',
        'SD9_raw_feature_bc_matrix': 'early',
        'SD10_raw_feature_bc_matrix': 'early',
        'SD12_raw_feature_bc_matrix': 'early',
        'SD14_raw_feature_bc_matrix': 'advanced',
        'SD15_raw_feature_bc_matrix': 'early',
        'SD16_raw_feature_bc_matrix': 'advanced',
    })

    epi.obs['tumor_stage'] = epi.obs['tumor_stage'].replace(['nan', 'NaN', 'NAN'], np.nan)

    cols  = ['HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION', 'HALLMARK_APOPTOSIS', 'REACTOME_CELL_CYCLE']
    labels = ['Cell cycle', 'EMT', 'Apoptosis']

    stages = ['non-cancer', 'early', 'advanced']
    stage_colors = {
        'non-cancer': '#84A970',
        'early':      '#E4C282',
        'advanced':   '#FF8C00'
    }

    n_groups, n_stages = len(cols), len(stages)

    width = 0.1
    group_spacing = width * n_stages
    offsets = np.linspace(-group_spacing/2 + width/2, group_spacing/2 - width/2, n_stages)

    all_results = []
    pvals_for_fdr = []
    fdr_index_map = [] 

    for col, lab in zip(cols, labels):
        data_by_stage = {st: epi.obs.loc[epi.obs['tumor_stage'] == st, col].dropna().values for st in stages}
        for g1, g2 in itertools.combinations(stages, 2):
            x, y = data_by_stage[g1], data_by_stage[g2]
            if len(x) == 0 or len(y) == 0:
                all_results.append({
                    "pathway": lab,
                    "feature": col,
                    "group1": g1, "n1": len(x),
                    "group2": g2, "n2": len(y),
                    "U_statistic": np.nan,
                    "p_value_raw": np.nan,
                    "p_value_adj_fdr_bh": np.nan,
                    "effect_size_rbc": np.nan
                })
                continue

            U, p = mannwhitneyu(x, y, alternative='two-sided')
            n1, n2 = len(x), len(y)
            rbc = 1.0 - (2.0 * U) / (n1 * n2)

            all_results.append({
                "pathway": lab,
                "feature": col,
                "group1": g1, "n1": n1,
                "group2": g2, "n2": n2,
                "U_statistic": float(U),
                "p_value_raw": float(p),
                "p_value_adj_fdr_bh": None,
                "effect_size_rbc": float(rbc)
            })
            pvals_for_fdr.append(p)
            fdr_index_map.append(len(all_results) - 1)

    if pvals_for_fdr:
        _, pvals_fdr, _, _ = multipletests(pvals_for_fdr, method='fdr_bh')
        for idx_in_list, adj_p in zip(fdr_index_map, pvals_fdr):
            all_results[idx_in_list]["p_value_adj_fdr_bh"] = float(adj_p)

    results_df = pd.DataFrame(all_results)
    figures_dir = Path("figures")
    figures_dir.mkdir(parents=True, exist_ok=True)
    csv_path = figures_dir / "figure_two_C_significance.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"Saved stage-wise significance to: {csv_path}")

    fig, ax = plt.subplots(figsize=(5, 5))
    for i, stage in enumerate(stages):
        data = [epi.obs.loc[epi.obs['tumor_stage'] == stage, col].dropna().values for col in cols]
        positions = np.arange(n_groups) + offsets[i]
        bp = ax.boxplot(data, positions=positions, widths=width, patch_artist=True, showfliers=False)
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

    handles = [plt.Line2D([0], [0], color=stage_colors[s], lw=6) for s in stages]
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
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

def __figure_two_G(adata, save_path=None):
    adata = adata[adata.obs['project'].isin(TARGET_PROJECTS)].copy()

    kras_mean = adata.obs["HALLMARK_KRAS_SIGNALING_UP"].mean()
    egfr_mean = adata.obs["REACTOME_SIGNALING_BY_EGFR_IN_CANCER"].mean()

    adata.obs["KRAS_high"] = adata.obs["HALLMARK_KRAS_SIGNALING_UP"] > kras_mean
    adata.obs["EGFR_high"] = adata.obs["REACTOME_SIGNALING_BY_EGFR_IN_CANCER"] > egfr_mean

    print(f"HALLMARK_KRAS_SIGNALING_UP mean = {kras_mean:.3f}")
    print(adata.obs["KRAS_high"].value_counts())
    print(f"REACTOME_SIGNALING_BY_EGFR_IN_CANCER mean = {egfr_mean:.3f}")
    print(adata.obs["EGFR_high"].value_counts())

    cols   = ['REACTOME_CELL_CYCLE', 'HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION', 'HALLMARK_APOPTOSIS']
    labels = ['Cell cycle', 'EMT', 'Apoptosis']

    categories = ['Both', 'EGFR only', 'KRAS only', 'None']
    colors = {'Both': '#7F00FF', 'EGFR only': '#00008B', 'KRAS only': '#8B0000', 'None': '#A9A9A9'}

    n_groups  = len(cols)
    n_cats    = len(categories)
    width     = 0.1
    grp_space = width * n_cats
    offsets   = np.linspace(-grp_space/2 + width/2, grp_space/2 - width/2, n_cats)

    # ---- Pairwise statistics across categories (Mann–Whitney U + FDR BH across all tests in this figure) ----
    def cat_mask(cat):
        if cat == 'Both':
            return adata.obs['KRAS_high'] & adata.obs['EGFR_high']
        elif cat == 'EGFR only':
            return ~adata.obs['KRAS_high'] & adata.obs['EGFR_high']
        elif cat == 'KRAS only':
            return adata.obs['KRAS_high'] & ~adata.obs['EGFR_high']
        else:  # 'None'
            return ~adata.obs['KRAS_high'] & ~adata.obs['EGFR_high']

    all_results = []
    pvals_for_fdr = []
    fdr_index_map = []

    for col, lab in zip(cols, labels):
        data_by_cat = {cat: adata.obs.loc[cat_mask(cat), col].dropna().values for cat in categories}
        for c1, c2 in itertools.combinations(categories, 2):
            x, y = data_by_cat[c1], data_by_cat[c2]
            if len(x) == 0 or len(y) == 0:
                all_results.append({
                    "pathway": lab,
                    "feature": col,
                    "category1": c1, "n1": len(x),
                    "category2": c2, "n2": len(y),
                    "U_statistic": np.nan,
                    "p_value_raw": np.nan,
                    "p_value_adj_fdr_bh": np.nan,
                    "effect_size_rbc": np.nan
                })
                continue

            U, p = mannwhitneyu(x, y, alternative='two-sided')
            n1, n2 = len(x), len(y)
            rbc = 1.0 - (2.0 * U) / (n1 * n2)

            all_results.append({
                "pathway": lab,
                "feature": col,
                "category1": c1, "n1": n1,
                "category2": c2, "n2": n2,
                "U_statistic": float(U),
                "p_value_raw": float(p),
                "p_value_adj_fdr_bh": None,  # fill later
                "effect_size_rbc": float(rbc)
            })
            pvals_for_fdr.append(p)
            fdr_index_map.append(len(all_results) - 1)

    if pvals_for_fdr:
        _, pvals_fdr, _, _ = multipletests(pvals_for_fdr, method='fdr_bh')
        for idx_in_list, adj_p in zip(fdr_index_map, pvals_fdr):
            all_results[idx_in_list]["p_value_adj_fdr_bh"] = float(adj_p)

    results_df = pd.DataFrame(all_results)
    figures_dir = Path("figures")
    figures_dir.mkdir(parents=True, exist_ok=True)
    csv_path = figures_dir / "figure_two_G_significance.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"Saved category-wise significance to: {csv_path}")

    # ---- Plot (unchanged, no annotations) ----
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

        data = [adata.obs.loc[mask, col].dropna().values for col in cols]
        positions = np.arange(n_groups) + offsets[i]
        bp = ax.boxplot(data, positions=positions, widths=width, patch_artist=True, showfliers=False)
        for box in bp['boxes']:
            box.set_facecolor(colors[cat])
            box.set_edgecolor('black')
        for elt in bp['whiskers'] + bp['caps'] + bp['medians']:
            elt.set_color('black')

    ax.set_xticks(np.arange(n_groups))
    ax.set_xticklabels(labels, rotation=0, fontsize=12)
    ax.set_ylabel('Pathway score')

    handles = [plt.Line2D([0], [0], color=colors[c], lw=6) for c in categories]
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
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

if __name__ == "__main__":
    BASE_DIR = Path.cwd()
    adata = sc.read_h5ad(filename=str(BASE_DIR / 'data/processed_data.h5ad'))
    __figure_two_A(adata, save_path=str(BASE_DIR / 'figures/figure_2A.png'))
    __figure_two_B(adata, save_path=str(BASE_DIR / 'figures/figure_2B.png'))
    __figure_two_B_2(adata, save_path=str(BASE_DIR / 'figures/figure_2B_2.png'))
    __figure_two_G(adata, save_path=str(BASE_DIR / 'figures/figure_2G.png'))
    __figure_two_C(adata, save_path=str(BASE_DIR / 'figures/figure_2C.png'))