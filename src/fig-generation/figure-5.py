import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
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
import umap
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import pingouin as pg
from pingouin import compute_effsize
import logging
from pytorch_tabnet.tab_model import TabNetClassifier
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
from oncocli.OncoTerrain import OncoTerrain

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
    celltype_mapping = {name: i for i, name in enumerate(data['leiden_res_20.00_celltype'].unique())}
    project_mapping = {name: i for i, name in enumerate(data['project'].unique())}

    logger.info(f"Cell type mapping:\n{celltype_mapping}")
    logger.info(f"Project name mapping:\n{project_mapping}")

    epithelial_data = data[data['leiden_res_20.00_celltype'] == 'Epithelial cell']
    counts = epithelial_data['tumor_stage'].value_counts()
    min_count = counts.min()
    balanced_epithelial_data = epithelial_data.groupby('tumor_stage').apply(lambda x: x.sample(min_count)).reset_index(drop=True)
    non_epithelial_data = data[data['leiden_res_20.00_celltype'] != 'Epithelial cell']
    data = pd.concat([non_epithelial_data, balanced_epithelial_data])

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

    palettes = [
        {0: "#84A970", 1: "#D3D3D3", 2: "#D3D3D3"}, 
        {0: "#D3D3D3", 1: "#E4C282", 2: "#D3D3D3"},  
        {0: "#D3D3D3", 1: "#D3D3D3", 2: "#FF8C00"}, 
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
            legend=False 
        )
        ax.set_title(title, fontsize=16)
        ax.set_xlabel('UMAP1', fontsize=14)
        ax.set_ylabel('UMAP2' if i == 0 else '', fontsize=14)

        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(2)
        ax.grid(False)

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

    meta_data[groupby_column] = meta_data[groupby_column].values
    logging.info(f"meta data data types: {meta_data.dtypes}")

    pseudobulk_expr = meta_data.select_dtypes(include=[np.number]).copy()
    pseudobulk_expr[groupby_column] = meta_data[groupby_column].values
    pseudobulk_expr = pseudobulk_expr.groupby(groupby_column).mean()

    pseudobulk_expr.columns = [col.upper().replace("HALLMARK_", "") for col in pseudobulk_expr.columns]

    sample_metadata = meta_data[[groupby_column, stage_column]].drop_duplicates().set_index(groupby_column)
    pseudobulk_expr = pseudobulk_expr.join(sample_metadata)

    selected_data = pseudobulk_expr[pseudobulk_expr[stage_column] == stage].drop(columns=stage_column).T

    selected_data = selected_data.reset_index().rename(columns={'index': 'hallmark'})
    selected_data['Condition'] = selected_data['hallmark'].map(lambda x: ', '.join(CONDITIONS.get(x, []))).str.lower()
    selected_data = selected_data.dropna(subset=['Condition'])
    selected_data['mean_score'] = selected_data.drop(columns=['hallmark', 'Condition']).mean(axis=1)

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

    scaler = MinMaxScaler(scale_range)
    VALUES = scaler.fit_transform(VALUES.reshape(-1, 1)).flatten()

    cmap = plt.get_cmap("tab20", 50)
    unique_labels = np.unique(LABELS)
    color_map = {label: cmap(i % 20) for i, label in enumerate(unique_labels)}
    COLORS = [color_map[label] for label in LABELS]

    fig, ax = plt.subplots(figsize=(20, 10), subplot_kw={"projection": "polar"})
    ax.set_theta_offset(OFFSET)
    ax.set_ylim(-scale_range[1], scale_range[1])
    ax.set_frame_on(False)
    ax.xaxis.grid(False)
    ax.yaxis.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

    ax.bar(ANGLES[IDXS], VALUES, width=WIDTH, color=COLORS, edgecolor="white", linewidth=2)

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

    legend_handles = [mpatches.Patch(color=color_map[label], label=label) for label in unique_labels]
    plt.legend(handles=legend_handles, bbox_to_anchor=(1.1, 1), loc="upper left",
               fontsize=12, title="Pathways")

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

    y_val_pred = random_search.predict(X_val.values)

    cm = confusion_matrix(y_val, y_val_pred)

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

def __figure_five_F(save_path, mice_data: pd.DataFrame):
    from pathlib import Path
    import logging
    import numpy as np
    import pandas as pd
    import scanpy as sc
    import scprep
    from scipy import sparse
    from sklearn.metrics.pairwise import cosine_similarity
    import matplotlib.pyplot as plt
    import seaborn as sns

    BASE_DIR = Path.cwd()
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    logging.info("Starting __figure_five_F plotting function.")
    
    # find all annotated OncoTerrain files
    oncoterrain_files = list((BASE_DIR / 'figures').glob('*_oncoterrain/OncoTerrain_annotated.h5ad'))
    
    # determine which columns in mice_data are expression columns
    mice_columns = [col for col in mice_data.columns if col not in ['Gene', 'Human_Ortholog']]
    logging.info(f"Mice data columns for analysis: {mice_columns}")
    
    # pick gene identifier column
    if 'Human_Ortholog' in mice_data.columns:
        gene_col = 'Human_Ortholog'
    elif 'Gene' in mice_data.columns:
        gene_col = 'Gene'
    else:
        raise ValueError("No gene identifier column found in mice_data")
    
    # average duplicates
    dup = mice_data[mice_data[gene_col].duplicated(keep=False)]
    if not dup.empty:
        logging.warning(f"Found {len(dup)} duplicate gene entries; averaging")
        numeric_cols = mice_data.select_dtypes(include=[np.number]).columns.tolist()
        mice_data = mice_data.groupby(gene_col)[numeric_cols].mean().reset_index()
        logging.info(f"After averaging duplicates: {len(mice_data)} unique genes")
    
    all_results = []
    # loop through each sample
    for i, file_path in enumerate(oncoterrain_files):
        full_path = BASE_DIR / file_path
        if not full_path.exists():
            logging.warning(f"File not found: {full_path}, skipping...")
            continue
            
        logging.info(f"Processing file {i+1}/{len(oncoterrain_files)}: {file_path}")
        adata = sc.read_h5ad(full_path)
        sample_name = Path(file_path).parent.name.split('_')[0]

        # TPM normalization
        logging.info(f"Normalizing {sample_name} with TPM via scprep")
        try:
            if sparse.issparse(adata.X):
                adata.X = adata.X.toarray()
            tpm = scprep.normalize.library_size_normalize(adata.X) * 1e6
            adata.X = tpm
            logging.info(f"TPM normalization successful for {sample_name}")
        except Exception as e:
            logging.error(f"TPM normalization failed for {sample_name}: {e}")
            continue
        
        if 'oncoterrain_class' not in adata.obs.columns:
            logging.warning(f"No 'oncoterrain_class' in {sample_name}, skipping...")
            continue
        
        classes = adata.obs['oncoterrain_class'].unique()
        logging.info(f"Found classes in {sample_name}: {classes}")
        
        sc_genes = {g.upper() for g in adata.var_names}
        mice_genes = {g.upper() for g in mice_data[gene_col].dropna()}
        common = sc_genes & mice_genes
        if not common:
            logging.warning(f"No common genes for {sample_name}, skipping...")
            continue
        logging.info(f"{len(common)} common genes for {sample_name}")
        
        # mapping back to original names
        sc_map   = {g.upper():g for g in adata.var_names}
        mice_map = {g.upper():g for g in mice_data[gene_col].dropna()}
        common_list = sorted(common)
        sc_genes_sel   = [sc_map[g]   for g in common_list]
        mice_genes_sel = [mice_map[g] for g in common_list]
        
        # build mice expression matrix
        mice_dict = {
            row[gene_col]: row[mice_columns].values
            for _, row in mice_data.iterrows()
            if row[gene_col] in mice_genes_sel
        }
        mice_mat = np.vstack([mice_dict[g] for g in mice_genes_sel])
        
        # filter adata to common genes
        adata_sub = adata[:, sc_genes_sel].copy()
        
        # compute similarities per class and condition
        for cls in classes:
            mask = adata.obs['oncoterrain_class'] == cls
            sub_adata = adata_sub[mask, :]
            if sub_adata.n_obs == 0:
                logging.warning(f"No cells in class {cls} for {sample_name}")
                continue
            
            # mean expression, log1p
            mexpr = np.log1p(np.mean(sub_adata.X, axis=0).A1 if hasattr(sub_adata.X, 'A1') else np.mean(sub_adata.X, axis=0))
            
            for j, cond in enumerate(mice_columns):
                m_mouse = np.log1p(mice_mat[:, j])
                cos_sim = cosine_similarity(mexpr.reshape(1,-1), m_mouse.reshape(1,-1))[0,0]
                try:
                    pcorr = np.corrcoef(mexpr, m_mouse)[0,1]
                except:
                    logging.warning(f"Pearson failed for {sample_name}, {cls}, {cond}")
                    pcorr = np.nan
                
                all_results.append({
                    'Sample': sample_name,
                    'OncoTerrain_Class': cls,
                    'Mice_Condition': cond,
                    'Cosine_Similarity': cos_sim,
                    'Pearson_Correlation': pcorr,
                    'Sample_Class': f"{sample_name}_{cls}"
                })
    
    results_df = pd.DataFrame(all_results)
    if results_df.empty:
        logging.error("No results generated.")
        return None
    
    # save detailed and summary CSVs
    csv1 = save_path / "figure_5F_detailed_similarities.csv"
    results_df.to_csv(csv1, index=False)
    logging.info(f"Detailed saved to {csv1}")
    
    summary_df = results_df.pivot_table(
        index='Sample_Class',
        columns='Mice_Condition',
        values='Cosine_Similarity',
        aggfunc='mean'
    )
    csv2 = save_path / "figure_5F_summary_similarities.csv"
    summary_df.to_csv(csv2)
    logging.info(f"Summary saved to {csv2}")
    
    # select only the conditions we care about
    selected_conditions = ['Normal Lung', 'TMet', 'TnonMet', 'Pleural DTCs']
    filtered = results_df[results_df['Mice_Condition'].isin(selected_conditions)]
    
    # --- GROUPED BAR PLOTS ---
    class_colors = {
        'Pre-malignant': '#E4C282',
        'Tumor-like':    '#FF8C00'
    }

    # Cosine differences vs Normal-like
    diffs_cos = {'Pre-malignant': [], 'Tumor-like': []}
    for cond in selected_conditions:
        grp = filtered[filtered['Mice_Condition']==cond]
        m_norm = grp[grp['OncoTerrain_Class']=='Normal-like']['Cosine_Similarity'].mean()
        m_pre  = grp[grp['OncoTerrain_Class']=='Pre-malignant']['Cosine_Similarity'].mean()
        m_tum  = grp[grp['OncoTerrain_Class']=='Tumor-like']['Cosine_Similarity'].mean()
        diffs_cos['Pre-malignant'].append(m_pre - m_norm)
        diffs_cos['Tumor-like'].append(m_tum - m_norm)

    x = np.arange(len(selected_conditions))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8,6))
    ax.bar(x - width/2, diffs_cos['Pre-malignant'], width,
           label='Pre-malignant vs Normal-like',
           color=class_colors['Pre-malignant'], alpha=0.85)
    ax.bar(x + width/2, diffs_cos['Tumor-like'], width,
           label='Tumor-like vs Normal-like',
           color=class_colors['Tumor-like'], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(selected_conditions, rotation=45, ha='right', fontsize=11)
    ax.set_ylabel('Δ Mean Cosine Similarity', fontsize=12)
    ax.set_title('Cosine Similarity Δ vs Normal-like', fontsize=14, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(title='Comparison', frameon=False, loc='center left', bbox_to_anchor=(1.02, 0.5))

    plt.tight_layout()
    path_cos = save_path / "figure_5F_barplot_cosine.png"
    plt.savefig(path_cos, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Cosine bar plot saved to: {path_cos}")

    # Pearson differences vs Normal-like
    diffs_pear = {'Pre-malignant': [], 'Tumor-like': []}
    for cond in selected_conditions:
        grp = filtered[filtered['Mice_Condition']==cond]
        p_norm = grp[grp['OncoTerrain_Class']=='Normal-like']['Pearson_Correlation'].mean()
        p_pre  = grp[grp['OncoTerrain_Class']=='Pre-malignant']['Pearson_Correlation'].mean()
        p_tum  = grp[grp['OncoTerrain_Class']=='Tumor-like']['Pearson_Correlation'].mean()
        diffs_pear['Pre-malignant'].append(p_pre - p_norm)
        diffs_pear['Tumor-like'].append(p_tum - p_norm)

    fig, ax = plt.subplots(figsize=(8,6))
    ax.bar(x - width/2, diffs_pear['Pre-malignant'], width,
           label='Pre-malignant vs Normal-like',
           color=class_colors['Pre-malignant'], alpha=0.85)
    ax.bar(x + width/2, diffs_pear['Tumor-like'], width,
           label='Tumor-like vs Normal-like',
           color=class_colors['Tumor-like'], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(selected_conditions, rotation=45, ha='right', fontsize=11)
    ax.set_ylabel('Δ Mean Pearson Correlation', fontsize=12)
    ax.set_title('Pearson Correlation Δ vs Normal-like', fontsize=14, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(title='Comparison', frameon=False, loc='center left', bbox_to_anchor=(1.02, 0.5))

    plt.tight_layout()
    path_pear = save_path / "figure_5F_barplot_pearson.png"
    plt.savefig(path_pear, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Pearson bar plot saved to: {path_pear}")

    # --- HEATMAP ---
    plt.figure(figsize=(12, 8))
    sns.heatmap(summary_df, annot=True, cmap='coolwarm', center=0,
                fmt='.3f', cbar_kws={'label': 'Mean Cosine Similarity'})
    plt.title('Mean Cosine Similarity: OncoTerrain Classes vs Mice Conditions')
    plt.xlabel('Mice Data Conditions')
    plt.ylabel('Sample_OncoTerrain Class')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    heatmap_path = save_path / "figure_5F_heatmap.png"
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
