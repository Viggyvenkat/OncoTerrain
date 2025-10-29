import scanpy as sc
import matplotlib.pyplot as plt 
import numpy as np
import sys
import random
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path

BASE_DIR = Path.cwd()

def __figure_one_D(adata):
    sc.pl.umap( adata, color='project', size = 0.2, show=False )

    ax = plt.gca()

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
        
    ax.tick_params(width=1.5)

    title_font = ax.title.get_fontproperties()
    ax.set_title('Project ID Annotation of HCA and LUCA', fontproperties=title_font)

    plt.savefig(str(BASE_DIR / 'figures/umap_project_annotation.png'), dpi=300, bbox_inches='tight')

    sc.pl.umap( adata, color='disease', size = 0.2, show=False )

    ax = plt.gca()

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
        
    ax.tick_params(width=1.5)

    title_font = ax.title.get_fontproperties()
    ax.set_title('Disease Annotation of HCA and LUCA', fontproperties=title_font)

    plt.savefig(str(BASE_DIR / 'figures/umap_disease_annotation.png'), dpi=300, bbox_inches='tight')

    stage_colors = {
        'non-cancer': '#84A970',
        'early':      '#E4C282',
        'advanced':   '#FF8C00'
    }

    sc.pl.umap( adata, color='tumor_stage', size = 0.2, show=False, palette=stage_colors)

    ax = plt.gca()

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
        
    ax.tick_params(width=1.5)

    title_font = ax.title.get_fontproperties()
    ax.set_title('Tumor Stage Annotation of HCA and LUCA', fontproperties=title_font)

    plt.savefig(str(BASE_DIR / 'figures/umap_tumor_stage_annotation.png'), dpi=300, bbox_inches='tight')

def __figure_one_E(adata):
    cmap_custom = LinearSegmentedColormap.from_list('gray_emerald', ['gray', '#50C878'])

    # Fixed set of genes instead of random sampling
    genes = ['EPCAM', 'CD163', 'COL1A2', 'CD3D']
    print(f"Using selected genes: {genes}")

    fig, axes = plt.subplots(1, len(genes), figsize=(25, 5))
    plt.subplots_adjust(wspace=0.15)

    for i, gene in enumerate(genes):
        sc.pl.umap(
            adata,
            color=gene,
            size = 0.2,
            sort_order=True,
            cmap=cmap_custom,
            ax=axes[i],
            show=False,
            colorbar_loc=None 
        )
        axes[i].spines['top'].set_visible(False)
        axes[i].spines['right'].set_visible(False)

        if i == 0:
            axes[i].set_ylabel('X2')
        else:
            axes[i].spines['left'].set_visible(False)
            axes[i].set_ylabel('')
            axes[i].set_yticks([])

        axes[i].set_xlabel('X1')
        axes[i].set_title(f'{gene} Expression')

        for spine in axes[i].spines.values():
            if spine.get_visible():
                spine.set_linewidth(1.5) 
        
        axes[i].tick_params(width=1.5)

    # Colorbar based on first gene
    expr = adata[:, genes[0]].X
    expr_arr = expr.toarray().flatten() if hasattr(expr, "toarray") else np.array(expr).flatten()

    sm = plt.cm.ScalarMappable(cmap=cmap_custom)
    sm.set_array(expr_arr)
    sm.set_clim(expr_arr.min(), expr_arr.max())

    cbar = plt.colorbar(sm, ax=axes, fraction=0.02, pad=0.04)
    cbar.set_label('Expression')

    plt.savefig(str(BASE_DIR / 'figures/figure-1-E.png'), dpi=300, bbox_inches='tight')
    plt.close()

def __figure_one_F(adata):
    sc.pl.umap(
        adata,
        color='leiden_res_20.00_celltype',
        size = 0.2,
        show=False 
    )

    ax = plt.gca()

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
        
    ax.tick_params(width=1.5)

    title_font = ax.title.get_fontproperties()
    ax.set_title('Cell Type Annotation of HCA and LUCA', fontproperties=title_font)

    plt.savefig(str(BASE_DIR/ 'figures/umap_celltype_annotation.png'), dpi=300, bbox_inches='tight')

def __figure_one_G(adata):
    disease_col = 'tumor_stage'
    celltype_col = 'leiden_res_20.00_celltype'

    pt = adata.obs.groupby([disease_col, celltype_col]).size().unstack(fill_value=0)
    pt_frac = pt.div(pt.sum(axis=1), axis=0)

    supplementary_table_path = BASE_DIR / 'figures/supplementary_table_one.csv'
    pt_frac.to_csv(supplementary_table_path)

    celltype_colors = adata.uns[f"{celltype_col}_colors"] 
    celltype_categories = adata.obs[celltype_col].cat.categories
    pt_frac = pt_frac.reindex(columns=celltype_categories)

    ax = pt_frac.plot(
        kind='bar',
        stacked=True,
        color=celltype_colors,
        edgecolor='black',
        linewidth=1,
        figsize=(10, 6)
    )

    ax.set_xlabel('tumor_stage')
    ax.set_ylabel('Fraction of Cells')
    ax.set_ylim(0, 1)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.title('Cell Type Fractions by tumor_stage')

    legend = ax.get_legend()
    if legend:
        legend.remove()

    plt.savefig(str(BASE_DIR / 'figures/barplot_celltype_fractions.png'), dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":

    adata = sc.read_h5ad(filename=str(BASE_DIR /'data/processed_data.h5ad'))

    rename_dict = {
        'chronic obstructive pulmonary disease': 'COPD',
        'lung adenocarcinoma': 'LUAD',
        'non-small cell lung carcinoma': 'NSCLC',
        'normal': 'normal',
        'squamous cell lung carcinoma': 'LUSC'
    }

    adata.obs['disease'] = adata.obs['disease'].replace(rename_dict)

    __figure_one_D(adata)
    __figure_one_E(adata)
    __figure_one_F(adata)
    __figure_one_G(adata)

