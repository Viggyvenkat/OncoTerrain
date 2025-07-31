# Core Python modules
import sys
import os
from pathlib import Path

# Numerical and Data Handling
import numpy as np
import pandas as pd
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# PyTorch and related tools
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Scikit-learn
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_curve,
    auc, classification_report, confusion_matrix, RocCurveDisplay
)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from sklearn.metrics.pairwise import cosine_similarity

# Dimensionality Reduction and Clustering
import umap
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN

# Statistics and Effect Size
import pingouin as pg
from pingouin import compute_effsize

# Logging
import logging

# TabNet
from pytorch_tabnet.tab_model import TabNetClassifier

# Single-cell analysis
import scanpy as sc
from py_monocle import (
    learn_graph, order_cells, compute_cell_states, regression_analysis, 
    differential_expression_genes
)

import joblib
import logging
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from TMEGPT.OncoTerrain import OncoTerrain

np.random.seed(42)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import scprep
from scipy import sparse

CONDITIONS = {
    'ADIPOGENESIS': ['EMT and Metastasis', 'Inflammation'],
    'ALLOGRAFT_REJECTION': ['Immune', 'Inflammation'],
    'ANDROGEN_RESPONSE': ['Sensitivity to growth'],
    'ANGIOGENESIS':['Angiogenesis', 'Immune', 'Inflammation'],
    'APICAL_JUNCTION':['EMT and Metastasis', 'Angiogenesis'],
    'APICAL_SURFACE':['EMT and Metastasis', 'Angiogenesis'],
    'APOPTOSIS':['Apoptosis', 'Replication'],
    'BILE_ACID_METABOLISM':['Sensitivity to growth', 'Immune', 'Energetics'],
    'CHOLESTEROL_HOMEOSTASIS':['Sensitivity to growth', 'Immune', 'Energetics'],
    'COAGULATION':['Angiogenesis', 'Immune'],
    'COMPLEMENT':['Immune','Apoptosis'],
    'DNA_REPAIR':['Proliferative signal', 'Genome instability'],
    'E2F_TARGETS':['Proliferative signal'],
    'EPITHELIAL_MESENCHYMAL_TRANSITION':['EMT and Metastasis'],
    'ESTROGEN_RESPONSE_EARLY':['Sensitivity to growth'],
    'FATTY_ACID_METABOLISM':['Sensitivity to growth', 'Immune', 'Energetics'],
    'G2M_CHECKPOINT':['Proliferative signal', 'Genome instability'],
    'GLYCOLYSIS':['Sensitivity to growth'],
    'HEDGEHOG_SIGNALING':['Insensitivity to antigrowth','Immune', 'Sensitivity to growth'],
    'HEME_METABOLISM':['Angiogenesis'],
    'HYPOXIA':['Sensitivity to growth','EMT and Metastasis', 'Energetics'],
    'IL2_STAT5_SIGNALING':['Immune'],
    'IL6_JAK_STAT3_SIGNALING':['Insensitivity to antigrowth'],
    'INFLAMMATORY_RESPONSE':['Inflammation', 'Sensitivity to growth'],
    'INTERFERON_ALPHA_RESPONSE': ['Immune', 'Inflammation'],
    'INTERFERON_GAMMA_RESPONSE':['Immune','Insensitivity to antigrowth'],
    'KRAS_SIGNALING_DN': ['Insensitivity to antigrowth'],
    'KRAS_SIGNALING_UP':['Sensitivity to growth'],
    'MITOTIC_SPINDLE':['Replication', 'Genome instability'],
    'MTORC1_SIGNALING':['Apoptosis','Insensitivity to antigrowth', 'Energetics'],  
    'MYC_TARGETS_V1':['Sensitivity to growth', 'Energetics'],
    'MYC_TARGETS_V2':['Sensitivity to growth', 'Energetics'],
    'MYOGENESIS':['EMT and Metastasis'],
    'NOTCH_SIGNALING':['EMT and Metastasis','Insensitivity to antigrowth', 'Sensitivity to growth' ],
    'OXIDATIVE_PHOSPHORYLATION':['Energetics'],
    'P53_PATHWAY':['Apoptosis', 'Replication'],
    'PANCREAS_BETA_CELLS':['Energetics'],
    'PEROXISOME':['Energetics'],
    'PI3K_AKT_MTOR_SIGNALING':['Apoptosis', 'Insensitivity to antigrowth', 'Energetics'],
    'PROTEIN_SECRETION':['EMT and Metastasis', 'Angiogenesis'],
    'REACTIVE_OXYGEN_SPECIES_PATHWAY':['Genome instability', 'Apoptosis'],
    'SPERMATOGENESIS':['Proliferative signal'],
    'TGF_BETA_SIGNALING':['Insensitivity to antigrowth', 'Immune'],
    'TNFA_SIGNALING_VIA_NFKB':['Immune'],
    'UNFOLDED_PROTEIN_RESPONSE':['EMT and Metastasis', 'Insensitivity to antigrowth', 'sensitivity to growth', 'Genome instability'],
    'UV_RESPONSE_DN':['Genome instability', 'Proliferative signal'],
    'UV_RESPONSE_UP':['Genome instability', 'Apoptosis'],
    'WNT_BETA_CATENIN_SIGNALING':['Replication', 'Energetics', 'Immune', 'sensitivity to growth'],
    'XENOBIOTIC_METABOLISM':['Immune']
}

def __feature_analysis_w_preprocessing(data):
    # Cell type and project name mappings
    celltype_mapping = {name: i for i, name in enumerate(data['leiden_res_20.00_celltype'].unique())}
    project_mapping = {name: i for i, name in enumerate(data['project'].unique())}

    logger.info(f"Cell type mapping:\n{celltype_mapping}")
    logger.info(f"Project name mapping:\n{project_mapping}")

    # Filter epithelial and balance by tumor_stage
    epithelial_data = data[data['leiden_res_20.00_celltype'] == 'Epithelial cell']
    counts = epithelial_data['tumor_stage'].value_counts()
    min_count = counts.min()
    balanced_epithelial_data = epithelial_data.groupby('tumor_stage').apply(lambda x: x.sample(min_count)).reset_index(drop=True)
    non_epithelial_data = data[data['leiden_res_20.00_celltype'] != 'Epithelial cell']
    data = pd.concat([non_epithelial_data, balanced_epithelial_data])

    # Map labels to integers
    data.loc[:, 'tumor_stage'] = data['tumor_stage'].map({'non-cancer': 0, 'early': 1, 'advanced': 2})
    data.loc[:, 'project'] = data['project'].map(project_mapping)
    data.loc[:, 'leiden_res_20.00_celltype'] = data['leiden_res_20.00_celltype'].map(celltype_mapping)

    data = data.dropna(subset=['tumor_stage', 'project', 'leiden_res_20.00_celltype'])

    logging.info("Preprocessing complete. Data shape: %s", data.shape)
    logging.info("Tumor stage mapping: %s", data['tumor_stage'].unique())
    logging.info("Project mapping: %s", data['project'].unique())
    logging.info("Cell type mapping: %s", data['leiden_res_20.00_celltype'].unique())

    return data, celltype_mapping, project_mapping

def __figure_five_B(adata, save_path, embedding):
    logging.info("Starting __figure_five_B plotting function.")

    # Define all color palettes for each stage-specific highlight
    palettes = [
        {0: "#84A970", 1: "#D3D3D3", 2: "#D3D3D3"},  # Highlight stage 0 (benign)
        {0: "#D3D3D3", 1: "#E4C282", 2: "#D3D3D3"},  # Highlight stage 1 (in situ)
        {0: "#D3D3D3", 1: "#D3D3D3", 2: "#FF8C00"},  # Highlight stage 2 (invasive)
    ]

    titles = ['Non-Cancer Highlighted', 'Early Highlighted', 'Advanced Highlighted']

    fig, axes = plt.subplots(1, 3, figsize=(24, 8), sharex=True, sharey=True)
    logging.info("Created figure and axes for subplots.")

    for i, (palette, title) in enumerate(zip(palettes, titles)):
        ax = axes[i]
        logging.info(f"Plotting subplot {i+1}: {title}")

        sns.scatterplot(
            x=embedding[:, 0],
            y=embedding[:, 1],
            hue=adata.obs['tumor_stage'] if hasattr(adata, 'obs') else adata['tumor_stage'],
            palette=palette,
            s=1,
            ax=ax,
            legend=False  # Suppress individual legends
        )
        ax.set_title(title, fontsize=16)
        ax.set_xlabel('UMAP1', fontsize=14)
        ax.set_ylabel('UMAP2' if i == 0 else '', fontsize=14)

        # Style axes
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(2)
        ax.grid(False)

    # Add a single legend outside the plot
    handles, labels = axes[0].get_legend_handles_labels()
    logging.info(f"Adding legend with labels: {labels}")
    fig.legend(handles, labels, title='Cancer Stage', fontsize=12, title_fontsize=14,
               loc='center left', bbox_to_anchor=(1.02, 0.5))

    plt.tight_layout(rect=[0, 0, 0.95, 1])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logging.info(f"Saved figure to {save_path}")
    plt.show()
    logging.info("Completed __figure_five_B plotting function.")


def __figure_five_C(adata, save_path, embedding, hallmark_list=None):
    logging.info("Starting __figure_five_C plotting function.")
    colors = ["gray", "#00a4ef"]
    cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", colors)
    logging.info(f"Created custom colormap with colors: {colors}")
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    for hallmark in hallmark_list:
        logging.info(f"Processing hallmark: {hallmark}")

        min_value = adata[hallmark].min()
        max_value = adata[hallmark].max()
        logging.info(f"Min and max expression values for {hallmark}: {min_value}, {max_value}")

        normed_expression = (adata[hallmark] - min_value) / (max_value - min_value)

        fig, ax = plt.subplots(figsize=(15, 10))
        scatter = ax.scatter(x=embedding[:, 0], y=embedding[:, 1], c=normed_expression, cmap=cmap, s=1)

        plt.title(f'UMAP projection colored by {hallmark}', fontsize=16)
        plt.xlabel('UMAP1', fontsize=14)
        plt.ylabel('UMAP2', fontsize=14)

        # Add a colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label(hallmark, rotation=270, labelpad=15)
        cbar.ax.yaxis.set_label_position('left')

        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(2)

        plt.grid(False)
        plt.tight_layout()
        plt.savefig(f"{save_path}/{hallmark}.png", dpi=300, bbox_inches='tight')
        logging.info(f"Saved figure to {save_path}/{hallmark}.png")

    logging.info("Completed __figure_five_C plotting function.")


def __figure_five_D(adata, save_path):
    logging.info("Starting __figure_five_D plotting function.")

    heatmap_colors = sns.color_palette("Purples", n_colors=5)
    logging.info(f"Using heatmap colors: {heatmap_colors}")

    pathway_1 = 'INTERFERON_ALPHA_RESPONSE'
    pathway_2 = 'INTERFERON_GAMMA_RESPONSE'

    benign_data = adata[adata['tumor_stage'] == 0][[pathway_1, pathway_2]].dropna()
    insitu_data = adata[adata['tumor_stage'] == 1][[pathway_1, pathway_2]].dropna()
    invasive_data = adata[adata['tumor_stage'] == 2][[pathway_1, pathway_2]].dropna()

    logging.info(f"Data counts - Benign: {len(benign_data)}, Insitu: {len(insitu_data)}, Invasive: {len(invasive_data)}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    sns.histplot(benign_data, x=pathway_1, y=pathway_2, bins=12, pmax=0.8, cmap=ListedColormap(heatmap_colors), ax=axes[0])
    axes[0].set_title('Benign')
    axes[0].set_xlabel('INTERFERON_ALPHA_RESPONSE')
    axes[0].set_ylabel('INTERFERON_GAMMA_RESPONSE')

    sns.histplot(insitu_data, x=pathway_1, y=pathway_2, bins=12, pmax=0.8, cmap=ListedColormap(heatmap_colors), ax=axes[1])
    axes[1].set_title('Insitu')
    axes[1].set_xlabel('INTERFERON_ALPHA_RESPONSE')
    axes[1].set_ylabel('INTERFERON_GAMMA_RESPONSE')

    sns.histplot(invasive_data, x=pathway_1, y=pathway_2, bins=12, pmax=0.8, cmap=ListedColormap(heatmap_colors), ax=axes[2])
    axes[2].set_title('Invasive')
    axes[2].set_xlabel('INTERFERON_ALPHA_RESPONSE')
    axes[2].set_ylabel('INTERFERON_GAMMA_RESPONSE')

    for ax in axes:
        ax.grid(False)
        ax.spines['top'].set_linewidth(2)
        ax.spines['right'].set_linewidth(2)
        ax.spines['bottom'].set_linewidth(2)
        ax.spines['left'].set_linewidth(2)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logging.info(f"Saved heatmap figure to {save_path}")

    plt.tight_layout()
    plt.show()
    logging.info("Completed __figure_five_D plotting function.")

def figure_five_E(meta_data, save_path, stage=2, stage_column='tumor_stage',
                                   groupby_column='project', scale_range=(0, 100),
                                   stage_label=None
                                   ):

    # 3. Compute pseudobulk matrix (average per sample)
    meta_data[groupby_column] = meta_data[groupby_column].values
    logging.info(f"meta data data types: {meta_data.dtypes}")

    pseudobulk_expr = meta_data.select_dtypes(include=[np.number]).copy()
    pseudobulk_expr[groupby_column] = meta_data[groupby_column].values
    pseudobulk_expr = pseudobulk_expr.groupby(groupby_column).mean()

    pseudobulk_expr.columns = [col.upper().replace("HALLMARK_", "") for col in pseudobulk_expr.columns]

    # 4. Join stage information
    sample_metadata = meta_data[[groupby_column, stage_column]].drop_duplicates().set_index(groupby_column)
    pseudobulk_expr = pseudobulk_expr.join(sample_metadata)

    # 5. Filter for selected stage
    selected_data = pseudobulk_expr[pseudobulk_expr[stage_column] == stage].drop(columns=stage_column).T

    # 7. Annotate with condition and average
    selected_data = selected_data.reset_index().rename(columns={'index': 'hallmark'})
    selected_data['Condition'] = selected_data['hallmark'].map(lambda x: ', '.join(CONDITIONS.get(x, []))).str.lower()
    selected_data = selected_data.dropna(subset=['Condition'])
    selected_data['mean_score'] = selected_data.drop(columns=['hallmark', 'Condition']).mean(axis=1)

    # 8. Explode multi-condition rows
    expanded_data = selected_data[['hallmark', 'mean_score', 'Condition']].copy()
    expanded_data = expanded_data.assign(Condition=expanded_data['Condition'].str.split(', '))
    expanded_data = expanded_data.explode('Condition').reset_index(drop=True)
    expanded_data = expanded_data.sort_values(by="Condition")

    VALUES = expanded_data["mean_score"].values
    LABELS = expanded_data['hallmark'].values
    GROUP = expanded_data["Condition"].values

    PAD = 3
    GROUPS_SIZE = [len(i[1]) for i in expanded_data.groupby("Condition")]
    ANGLES_N = len(VALUES) + PAD * len(GROUPS_SIZE)
    ANGLES = np.linspace(0, 2 * np.pi, num=ANGLES_N, endpoint=False)
    WIDTH = (2 * np.pi) / len(ANGLES)

    OFFSET = 0
    IDXS = []
    offset = 0
    for size in GROUPS_SIZE:
        IDXS += list(range(offset + PAD, offset + size + PAD))
        offset += size + PAD

    # 9. Normalize values
    scaler = MinMaxScaler(scale_range)
    VALUES = scaler.fit_transform(VALUES.reshape(-1, 1)).flatten()

    # 10. Assign colors
    cmap = plt.get_cmap("tab20", 50)
    unique_labels = np.unique(LABELS)
    color_map = {label: cmap(i % 20) for i, label in enumerate(unique_labels)}
    COLORS = [color_map[label] for label in LABELS]

    # 11. Plot
    fig, ax = plt.subplots(figsize=(20, 10), subplot_kw={"projection": "polar"})
    ax.set_theta_offset(OFFSET)
    ax.set_ylim(-scale_range[1], scale_range[1])
    ax.set_frame_on(False)
    ax.xaxis.grid(False)
    ax.yaxis.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

    ax.bar(ANGLES[IDXS], VALUES, width=WIDTH, color=COLORS, edgecolor="white", linewidth=2)

    # 12. Group separators + labels
    offset = 0
    for group, size in zip(np.unique(GROUP), GROUPS_SIZE):
        x1 = np.linspace(ANGLES[offset + PAD], ANGLES[offset + size + PAD - 1], num=50)
        ax.plot(x1, [-5] * 50, color="#333333")

        rotation_angle = np.rad2deg(np.mean(x1))
        if 90 < rotation_angle < 270:
            rotation_angle += 180

        ax.text(np.mean(x1), scale_range[1] * 0.85, group, color="#333", fontsize=14, fontweight="bold",
                ha="center", va="center", rotation=rotation_angle)
        offset += size + PAD

    # 13. Legend
    legend_handles = [mpatches.Patch(color=color_map[label], label=label) for label in unique_labels]
    plt.legend(handles=legend_handles, bbox_to_anchor=(1.1, 1), loc="upper left",
               fontsize=12, title="Pathways")

    # 14. Save
    suffix = stage_label or f"stage_{stage}"
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def __train_model(adata):
    adata = adata.dropna()

    y = adata['tumor_stage']
    X = adata.drop(['tumor_stage', 'project'], axis=1)

    columns_to_exclude = ['project', 'leiden_res_20.00_celltype']
    columns_to_scale = [col for col in X.columns if col not in columns_to_exclude]

    scaler = MinMaxScaler()
    X[columns_to_scale] = scaler.fit_transform(X[columns_to_scale])

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

    train_data = X_train.copy()
    train_data['tumor_stage'] = y_train

    X_train = train_data.drop('tumor_stage', axis=1)
    y_train = train_data['tumor_stage']

    y_train = y_train.astype(int)
    y_val = y_val.astype(int)

    X_train = X_train.astype(float)
    X_val = X_val.astype(float)

    param_grid = {
        'n_d': [8, 16, 24],
        'n_a': [8, 16, 24],
        'n_steps': [3, 5, 7],
        'gamma': [1.0, 1.5, 2.0],
        'lambda_sparse': [1e-3, 1e-4, 1e-5],
        'mask_type': ['sparsemax', 'entmax'],
        'n_independent': [2, 4],
        'n_shared': [2, 4],
    }

    clf = TabNetClassifier(verbose=0)

    random_search = RandomizedSearchCV(estimator=clf, param_distributions=param_grid, 
                                    scoring='accuracy', cv=3, verbose=0, n_jobs=-1, error_score='raise')

    random_search.fit(X_train.values, y_train.values, 
                    eval_set=[(X_val.values, y_val.values)], 
                    eval_metric=['accuracy', 'balanced_accuracy', 'logloss'])

    print("Best parameters found: ", random_search.best_params_)
    print("Best accuracy score: ", random_search.best_score_)

    y_val_probs = random_search.predict_proba(X_val.values)

    n_classes = len(np.unique(y_train))
    y_val_bin = label_binarize(y_val, classes=np.arange(n_classes))

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_val_bin[:, i], y_val_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(10, 8))
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label=f"Class {i} (AUC = {roc_auc[i]:.2f})")

    plt.plot([0, 1], [0, 1], 'k--') 
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('One-vs-Rest ROC-AUC')
    plt.legend(loc="lower right")

    BASE_DIR = Path.cwd()

    plt.savefig(BASE_DIR / "figures/ROC-AUC-PLOT.png", dpi=300, bbox_inches='tight')

    # Predict on validation set
    y_val_pred = random_search.predict(X_val.values)

    # Generate confusion matrix
    cm = confusion_matrix(y_val, y_val_pred)

    # Plot confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix - Type I vs Type II Errors')

    plt.savefig(BASE_DIR / "figures/CONFUSION_MATRIX.png", dpi=300, bbox_inches='tight')

    joblib.dump({
        'model': random_search,
        'features': X_train.columns.tolist()
    }, "OncoTerrain.joblib")

    return random_search

def __figure_five_H():
    BASE_DIR = Path.cwd()

    for dir in (BASE_DIR / 'data/scRNAseq-data').iterdir():
        if dir.is_dir() and dir.name.startswith('SD'):
            logging.info(f"Processing {dir.name} with OncoTerrain.")
            sd10_adata = sc.read_10x_mtx(dir, var_names='gene_symbols')
            onco_terrain_10 = OncoTerrain(sd10_adata)
            onco_terrain_10.inferencing(
                save_path=BASE_DIR / f"figures/{dir.name}_oncoterrain",
                save_adata=True
            )
            logging.info(f"OncoTerrain {dir.name} processing completed successfully.")


    for dir in (BASE_DIR / 'data/scRNAseq-data/PCLAcohort').iterdir():
        if dir.is_dir() and dir.name.startswith('SMP-'):
            logging.info(f"Processing {dir.name} with OncoTerrain.")
            sd10_adata = sc.read_10x_mtx(dir, var_names='gene_symbols')
            onco_terrain_10 = OncoTerrain(sd10_adata)
            onco_terrain_10.inferencing(
                save_path=BASE_DIR / f"figures/{dir.name}_oncoterrain",
                save_adata=True
            )
            logging.info(f"OncoTerrain {dir.name} processing completed successfully.")

def __get_top_de_genes(mice_data: pd.DataFrame, reference_col: str = 'Normal Lung', top_n: int = 1000):
    expression_cols = [col for col in mice_data.columns if col not in ['Gene', 'Human_Ortholog']]
    
    if reference_col not in expression_cols:
        raise ValueError(f"Reference column '{reference_col}' not found in mice_data.")
    
    pseudocount = 1e-6
    ref_expr = mice_data[reference_col] + pseudocount

    log2fc_df = mice_data[expression_cols].apply(lambda col: np.log2((col + pseudocount) / ref_expr))
    
    max_abs_log2fc = log2fc_df.drop(columns=[reference_col]).abs().max(axis=1)

    mice_data = mice_data.copy()
    mice_data['max_abs_log2FC'] = max_abs_log2fc
    top_genes = mice_data.nlargest(top_n, 'max_abs_log2FC').drop(columns='max_abs_log2FC')

    return top_genes

def __figure_five_F(save_path, mice_data: pd.DataFrame):
    BASE_DIR = Path.cwd()
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    logging.info("Starting __figure_five_F plotting function.")
    
    oncoterrain_files = list((BASE_DIR / 'figures').glob('*_oncoterrain/OncoTerrain_annotated.h5ad'))

    # mice_data = __get_top_de_genes(mice_data, reference_col='Normal Lung', top_n=1000)
    
    mice_columns = [col for col in mice_data.columns if col not in ['Gene', 'Human_Ortholog']]
    logging.info(f"Mice data columns for analysis: {mice_columns}")
    
    if 'Human_Ortholog' in mice_data.columns:
        gene_col = 'Human_Ortholog'
    elif 'Gene' in mice_data.columns:
        gene_col = 'Gene'
    else:
        raise ValueError("No gene identifier column found in mice_data")
    
    duplicate_genes = mice_data[mice_data[gene_col].duplicated(keep=False)]
    if not duplicate_genes.empty:
        logging.warning(f"Found {len(duplicate_genes)} duplicate gene entries in mice data")
        logging.info("Duplicate genes will be averaged across replicates")
        numeric_cols = mice_data.select_dtypes(include=[np.number]).columns.tolist()
        mice_data = mice_data.groupby(gene_col)[numeric_cols].mean().reset_index()
        logging.info(f"After averaging duplicates: {len(mice_data)} unique genes")
    
    all_results = []
    
    for i, file_path in enumerate(oncoterrain_files):
        full_path = BASE_DIR / file_path
        if not full_path.exists():
            logging.warning(f"File not found: {full_path}, skipping...")
            continue
            
        logging.info(f"Processing file {i+1}/{len(oncoterrain_files)}: {file_path}")
        oncoterrain_obj = sc.read_h5ad(full_path)
        sample_name = Path(file_path).parent.name.split('_')[0]

        logging.info(f"Normalizing {sample_name} with TPM via scprep")

        try:
            if sparse.issparse(oncoterrain_obj.X):
                oncoterrain_obj.X = oncoterrain_obj.X.toarray()

            tpm_normalized = scprep.normalize.library_size_normalize(oncoterrain_obj.X)
            tpm_normalized *= 1e6

            oncoterrain_obj.X = tpm_normalized
            logging.info(f"TPM normalization successful for {sample_name}")

        except Exception as e:
            logging.error(f"TPM normalization failed for {sample_name}: {e}")
            continue
        
        if 'oncoterrain_class' not in oncoterrain_obj.obs.columns:
            logging.warning(f"No 'oncoterrain_class' column found in {sample_name}, skipping...")
            continue
        
        oncoterrain_classes = oncoterrain_obj.obs['oncoterrain_class'].unique()
        logging.info(f"Found oncoterrain classes in {sample_name}: {oncoterrain_classes}")
        
        sc_genes = set(gene.upper() for gene in oncoterrain_obj.var_names)
        mice_genes = set(gene.upper() for gene in mice_data[gene_col].dropna())
        common_genes = sc_genes.intersection(mice_genes)
        
        if len(common_genes) == 0:
            logging.warning(f"No common genes found for {sample_name}, skipping...")
            continue
            
        logging.info(f"Found {len(common_genes)} common genes for {sample_name}")
        
        sc_gene_mapping = {gene.upper(): gene for gene in oncoterrain_obj.var_names}
        mice_gene_mapping = {gene.upper(): gene for gene in mice_data[gene_col].dropna()}
        
        common_genes_list = sorted(list(common_genes))
        common_sc_genes = [sc_gene_mapping[gene] for gene in common_genes_list]
        common_mice_genes = [mice_gene_mapping[gene] for gene in common_genes_list]
        
        assert len(common_sc_genes) == len(common_mice_genes)
        
        mice_gene_dict = {}
        for _, row in mice_data.iterrows():
            gene_name = row[gene_col]
            if gene_name in common_mice_genes:
                mice_gene_dict[gene_name] = row[mice_columns].values
        
        mice_expression_matrix = np.array([mice_gene_dict[gene] for gene in common_mice_genes], dtype=float)
        oncoterrain_filtered = oncoterrain_obj[:, common_sc_genes].copy()
        
        logging.info(f"After filtering: SC genes={oncoterrain_filtered.n_vars}, Mice genes={len(mice_expression_matrix)}")
        
        for onco_class in oncoterrain_classes:
            class_mask = oncoterrain_obj.obs['oncoterrain_class'] == onco_class
            class_cells = oncoterrain_filtered[class_mask, :]
            if class_cells.n_obs == 0:
                logging.warning(f"No cells found for class {onco_class} in {sample_name}")
                continue
            
            class_mean_expression = np.mean(class_cells.X, axis=0)
            if hasattr(class_mean_expression, 'A1'):
                class_mean_expression = class_mean_expression.A1
            class_mean_expression = np.log1p(class_mean_expression).flatten()
            
            for col_idx, mice_col in enumerate(mice_columns):
                mice_expression = np.log1p(mice_expression_matrix[:, col_idx])

                cosine_sim = cosine_similarity(
                    class_mean_expression.reshape(1, -1),
                    mice_expression.reshape(1, -1)
                )[0, 0]

                try:
                    pearson_corr = np.corrcoef(class_mean_expression, mice_expression)[0, 1]
                except Exception as e:
                    logging.warning(f"Pearson correlation failed for {sample_name}, {onco_class}, {mice_col}: {e}")
                    pearson_corr = np.nan

                all_results.append({
                    'Sample': sample_name,
                    'OncoTerrain_Class': onco_class,
                    'Mice_Condition': mice_col,
                    'Cosine_Similarity': cosine_sim,
                    'Pearson_Correlation': pearson_corr,
                    'Sample_Class': f"{sample_name}_{onco_class}"
                })
    
    results_df = pd.DataFrame(all_results)
    
    if results_df.empty:
        logging.error("No results generated. Check file paths and data compatibility.")
        return None
    
    logging.info(f"Generated {len(results_df)} similarity measurements")
    
    output_path = Path(save_path) / "figure_5F_detailed_similarities.csv"
    results_df.to_csv(output_path, index=False)
    logging.info(f"Detailed results saved to: {output_path}")
    
    summary_df = results_df.pivot_table(
        index='Sample_Class', 
        columns='Mice_Condition', 
        values='Cosine_Similarity', 
        aggfunc='mean'
    )
    
    summary_path = Path(save_path) / "figure_5F_summary_similarities.csv"
    summary_df.to_csv(summary_path)
    logging.info(f"Summary results saved to: {summary_path}")
    
    selected_conditions = ['Normal Lung', 'TMet', 'TnonMet', 'Pleural DTCs']
    selected_classes = ['Normal-like', 'Pre-malignant', 'Tumor-like']

    filtered_df = results_df[results_df['Mice_Condition'].isin(selected_conditions)]

    if filtered_df.empty:
        logging.error("No data available for the selected Mice Conditions. Aborting boxplot.")
    else:
        fig, ax = plt.subplots(figsize=(8, 8))

        box_data = []
        box_positions = []
        box_labels = []
        class_colors = {
            'Normal-like': "#84A970",
            'Pre-malignant': '#E4C282', 
            'Tumor-like': '#FF8C00'
        }

        xtick_positions = []
        xtick_labels = []
        box_width = 0.9  # Wider to make boxes touch
        group_spacing = 5  # Increase this value for more space between groups

        for idx, condition in enumerate(selected_conditions):
            base_position = idx * group_spacing  # increased spacing between groups
            condition_data = filtered_df[filtered_df['Mice_Condition'] == condition]

            for j, class_name in enumerate(selected_classes):
                class_data = condition_data[condition_data['OncoTerrain_Class'] == class_name]['Cosine_Similarity'].values
                if len(class_data) > 0:
                    box_data.append(class_data)
                    box_positions.append(base_position + j)  # boxes within group remain touching
                    box_labels.append(class_name)

            xtick_positions.append(base_position + 1)  # center label between 3 boxes
            xtick_labels.append(condition)

        bp = ax.boxplot(
            box_data,
            positions=box_positions,
            patch_artist=True,
            widths=box_width
        )

        # Color boxes + set black median lines
        for patch, label in zip(bp['boxes'], box_labels):
            patch.set_facecolor(class_colors.get(label, 'gray'))
            patch.set_alpha(0.85)

        for median in bp['medians']:
            median.set_color('black')
            median.set_linewidth(2)

        ax.set_xticks(xtick_positions)
        ax.set_xticklabels(xtick_labels, rotation=45, ha='right', fontsize=11)
        ax.set_ylabel("Cosine Similarity", fontsize=12)
        ax.set_title("OncoTerrain Class vs Mice Conditions", fontsize=14, fontweight='bold')

        # Remove grid and top/right spines
        ax.grid(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Add legend to the right, no frame
        handles = [plt.Line2D([0], [0], color=class_colors[c], lw=8) for c in selected_classes]
        ax.legend(handles, selected_classes, title="OncoTerrain Class", frameon=False,
                loc='center left', bbox_to_anchor=(1.02, 0.5))

        grouped_boxplot_path = Path(save_path) / "figure_5F_grouped_boxplot.png"
        plt.tight_layout()
        plt.savefig(grouped_boxplot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logging.info(f"Grouped box plot saved to: {grouped_boxplot_path}")

        # ---- Pearson Correlation Boxplot ----
        if filtered_df.empty:
            logging.error("No data available for the selected Mice Conditions. Aborting Pearson correlation boxplot.")
        else:
            fig, ax = plt.subplots(figsize=(8, 8))

            box_data = []
            box_positions = []
            box_labels = []

            xtick_positions = []
            xtick_labels = []

            for idx, condition in enumerate(selected_conditions):
                base_position = idx * group_spacing  # same spacing as cosine similarity
                condition_data = filtered_df[filtered_df['Mice_Condition'] == condition]

                for j, class_name in enumerate(selected_classes):
                    class_data = condition_data[condition_data['OncoTerrain_Class'] == class_name]['Pearson_Correlation'].values
                    if len(class_data) > 0:
                        box_data.append(class_data)
                        box_positions.append(base_position + j)
                        box_labels.append(class_name)

                xtick_positions.append(base_position + 1)
                xtick_labels.append(condition)

            bp = ax.boxplot(
                box_data,
                positions=box_positions,
                patch_artist=True,
                widths=box_width
            )

            for patch, label in zip(bp['boxes'], box_labels):
                patch.set_facecolor(class_colors.get(label, 'gray'))
                patch.set_alpha(0.85)

            for median in bp['medians']:
                median.set_color('black')
                median.set_linewidth(2)

            ax.set_xticks(xtick_positions)
            ax.set_xticklabels(xtick_labels, rotation=45, ha='right', fontsize=11)
            ax.set_ylabel("Pearson Correlation", fontsize=12)
            ax.set_title("OncoTerrain Class vs Mice Conditions (Pearson Correlation)", fontsize=14, fontweight='bold')

            ax.grid(False)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            handles = [plt.Line2D([0], [0], color=class_colors[c], lw=8) for c in selected_classes]
            ax.legend(handles, selected_classes, title="OncoTerrain Class", frameon=False,
                    loc='center left', bbox_to_anchor=(1.02, 0.5))

            pearson_boxplot_path = Path(save_path) / "figure_5F_pearson_boxplot.png"
            plt.tight_layout()
            plt.savefig(pearson_boxplot_path, dpi=300, bbox_inches='tight')
            plt.close()
            logging.info(f"Pearson correlation box plot saved to: {pearson_boxplot_path}")

    # ---- Heatmap Plot ----
    plt.figure(figsize=(12, 8))
    sns.heatmap(summary_df, annot=True, cmap='coolwarm', center=0, 
               fmt='.3f', cbar_kws={'label': 'Mean Cosine Similarity'})
    plt.title('Mean Cosine Similarity: OncoTerrain Classes vs Mice Conditions')
    plt.xlabel('Mice Data Conditions')
    plt.ylabel('Sample_OncoTerrain Class')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    heatmap_path = Path(save_path) / "figure_5F_heatmap.png"
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Heatmap saved to: {heatmap_path}")
    
    return results_df, summary_df

if __name__ == '__main__':
    logging.info("Script started.")
    
    BASE_DIR = Path.cwd()
    # data_path = BASE_DIR / 'data/processed_data.h5ad'
    # logging.info(f"Reading data from {data_path}")
    # adata = sc.read_h5ad(filename=str(data_path))
    
    # logging.info("Copying and preprocessing metadata.")
    # meta_data = adata.obs.copy()
    # meta_data.columns = meta_data.columns.str.replace('^HALLMARK_', '', regex=True)
    
    # # Drop object dtype columns except specified exceptions
    # meta_data = meta_data.drop(columns= ['disease', 'sample', 'source', 'tissue', 'n_genes', 'batch', 
    #                            'n_genes_by_counts', 'total_counts', 'total_counts_mt', 'pct_counts_mt', 'leiden_res_0.10', 
    #                            'leiden_res_1.00', 'leiden_res_5.00', 'leiden_res_10.00', 'leiden_res_20.00', 'leiden_res_0.10_celltype',
    #                            'leiden_res_1.00_celltype', 'leiden_res_5.00_celltype', 'leiden_res_10.00_celltype'])

    # logging.info("Running feature analysis preprocessing.")
    # updated_meta_data, _, _ = __feature_analysis_w_preprocessing(meta_data)
    
    # hallmark_list = updated_meta_data.columns[~updated_meta_data.columns.isin(['tumor_stage', 'project', 'leiden_res_20.00_celltype'])]
    # logging.info(f"Identified hallmark features: {list(hallmark_list)}")
    
    # features = updated_meta_data.drop(columns=['tumor_stage', 'leiden_res_20.00_celltype', 'project'])
    # tumor_stage = updated_meta_data['tumor_stage']
    
    # logging.info("Initializing UMAP reducer with parameters: n_neighbors=50, min_dist=0.05, metric='euclidean'")
    # reducer = umap.UMAP(n_neighbors=50, min_dist=0.05, metric='euclidean', random_state=42)
    
    # logging.info("Fitting UMAP to features and transforming.")
    # embedding = reducer.fit_transform(features)
    # logging.info("UMAP embedding shape: %s", embedding.shape)
    
    # fig5B_path = BASE_DIR / 'figures/fig-5B.png'
    # fig5C_path = BASE_DIR / 'figures/fig-5C'
    # fig5D_path = BASE_DIR / 'figures/fig-5D.png'
    # fig5E_path = BASE_DIR / 'figures/fig-5E.png'
    
    # logging.info("Generating figure 5B.")
    # __figure_five_B(updated_meta_data, save_path=str(fig5B_path), embedding=embedding)
    
    # logging.info("Generating figure 5C.")
    # __figure_five_C(updated_meta_data, save_path=str(fig5C_path), embedding=embedding, hallmark_list=hallmark_list)
    
    # logging.info("Generating figure 5D.")
    # __figure_five_D(updated_meta_data, save_path=str(fig5D_path))
    
    # logging.info("Generating figure 5E.")
    # figure_five_E(updated_meta_data, save_path=str(fig5E_path))
    
    # logging.info("Training model.")
    # model = __train_model(updated_meta_data)
    
    # __figure_five_H()
    __figure_five_F(save_path=BASE_DIR / 'figures/fig-5F', mice_data = pd.read_csv(BASE_DIR / 'data/averaged_gene_expression_nature_mice_supp_1.csv'))

    
    logging.info("Script finished successfully.")
