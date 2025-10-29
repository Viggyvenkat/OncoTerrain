import os
import sys
from pathlib import Path

import json
import joblib
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# allow local package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from OncoTerrain.OncoTerrain import OncoTerrain

# ----------------------------- #
# Constants
# ----------------------------- #
EPITHELIAL_CATS = [
    'AT2', 'AT1', 'Suprabasal', 'Basal resting', 'Multiciliated (non-nasal)',
    'Goblet (nasal)', 'Club (nasal)', 'Ciliated (nasal)', 'Club (non-nasal)',
    'Multiciliated (nasal)', 'Goblet (bronchial)', 'Transitional Club AT2',
    'AT2 proliferating', 'Goblet (subsegmental)'
]
FIBRO_CATS = [
    "Peribronchial fibroblasts", "Adventitial fibroblasts",
    "Alveolar fibroblasts", "Subpleural fibroblasts",
    "Myofibroblasts", "Fibromyocytes"
]
MYELOID_CATS = [
    'Alveolar macrophages', 'Alveolar Mφ MT-positive', 'Alveolar Mφ proliferating',
    'Interstitial Mφ perivascular', 'Alveolar Mφ CCL3+', 'Monocyte derived Mφ',
    'Mast cells', 'Plasmacytoid DCs', 'DC2', 'Migratory DCs',
    'Classical Monocytes', 'Non classical monocytes'
]
LYMPHOID_CATS = [
    'CD8 T cells', 'CD4 T cells', 'T cells proliferating',
    'NK cells', 'B cells', 'Plasma cells'
]
ENDOTHELIAL_CATS = [
    "EC venous pulmonary", "EC arterial", "EC venous systemic",
    "EC general capillary", "EC aerocyte capillary",
    "Lymphatic EC mature", "Lymphatic EC differentiating", "Lymphatic EC proliferating",
]
TARGET_PROJECTS = [
    'SD2_raw_feature_bc_matrix','SD3_raw_feature_bc_matrix','SD6_raw_feature_bc_matrix',
    'SD7_raw_feature_bc_matrix','SD8_raw_featuresbc_matrix','SD10_raw_feature_bc_matrix',
    'SD12_raw_feature_bc_matrix','SD14_raw_feature_bc_matrix','SD15_raw_feature_bc_matrix',
    'SD16_raw_feature_bc_matrix'
]

# CAT groups and order for plotting
CAT_GROUPS = {
    "EPITHELIAL": EPITHELIAL_CATS,
    "FIBRO": FIBRO_CATS,
    "MYELOID": MYELOID_CATS,
    "LYMPHOID": LYMPHOID_CATS,
    "ENDOTHELIAL": ENDOTHELIAL_CATS,
}
CAT_ORDER = ["EPITHELIAL", "FIBRO", "MYELOID", "LYMPHOID", "ENDOTHELIAL"]

# color palettes
CLASS_PALETTE = {"Normal-Like": "#84A970", "Pre-malignant": "#E4C282", "Tumor-like": "#FF8C00"}
INHOUSE_COLORS = sns.color_palette("Set2", n_colors=10)  # popping colors for strip points

# ----------------------------- #
# Helpers
# ----------------------------- #
def _detect_celltype_col(obs: pd.DataFrame, candidate_lists) -> str:
    """Return the obs column that contains the cell-type labels used in *_CATS."""
    return "leiden_res_20.00_celltype"  # project-specific; adjust if needed

def _mode_safe(x: pd.Series):
    """Return the mode value if available; else np.nan. Works with strings/numerics."""
    if x is None or x.empty:
        return np.nan
    x = x.dropna()
    if x.empty:
        return np.nan
    m = x.mode(dropna=True)
    return m.iloc[0] if not m.empty else np.nan

# ----------------------------- #
# Build tidy DF for plotting
# ----------------------------- #
def _make_violin_data(
    adata: sc.AnnData,
    onco_adata: sc.AnnData,
    cat_groups: dict,
    celltype_col: str = None
) -> pd.DataFrame:
    """
    Build a tidy DataFrame with columns:
      ['CAT', 'donor_id', 'value_raw', 'value_norm11', 'project', 'is_inhouse', 'cat_size']

    value_raw = fraction of cells within (CAT × donor) whose oncoterrain_class is in {Pre-malignant, Tumor-like}
    value_norm11 = per-CAT min–max scaling of value_raw to [-1, 1]
    """
    # start from a copy
    obs = adata.obs.copy()

    # bring in oncoterrain_class from onco_adata (aligned by cell index)
    common_idx = obs.index.intersection(onco_adata.obs_names)
    onco_obs = onco_adata.obs.loc[common_idx, ['oncoterrain_class']]
    if 'oncoterrain_class' not in obs.columns:
        obs = obs.join(onco_obs, how='left')
    else:
        obs.loc[common_idx, 'oncoterrain_class'] = (
            obs.loc[common_idx, 'oncoterrain_class']
            .where(obs.loc[common_idx, 'oncoterrain_class'].notna(), onco_obs['oncoterrain_class'])
        )

    # sanity checks
    required = ['donor_id', 'project', 'oncoterrain_class']
    for col in required:
        if col not in obs.columns:
            raise KeyError(f"Missing required column in obs: '{col}'")

    # detect celltype column if not provided
    if celltype_col is None:
        celltype_col = _detect_celltype_col(obs, list(cat_groups.values()))
    if celltype_col not in obs.columns:
        raise KeyError(f"Cell-type column '{celltype_col}' not found in adata.obs")

    # compute malignant fraction within CAT for each donor
    records = []
    malignant_labels = {'Tumor-like', 'Pre-malignant'}

    for cat_name, members in cat_groups.items():
        for donor_id, df in obs.groupby('donor_id', dropna=False):
            if df.empty:
                continue

            is_cat = df[celltype_col].isin(members)
            df_cat = df.loc[is_cat]

            cat_n = int(len(df_cat))
            if cat_n > 0:
                malig_mask = df_cat['oncoterrain_class'].isin(malignant_labels)
                value_raw = float(malig_mask.sum()) / cat_n
            else:
                value_raw = np.nan

            proj = _mode_safe(df['project'])
            is_inhouse = bool(proj in TARGET_PROJECTS) if pd.notna(proj) else False

            records.append({
                'CAT': cat_name,
                'donor_id': donor_id,
                'value_raw': value_raw,
                'cat_size': cat_n,
                'project': proj,
                'is_inhouse': is_inhouse,
            })

    df = pd.DataFrame.from_records(records)

    # vectorized per-CAT min–max normalization to [-1, 1]
    v = df['value_raw'].astype(float)
    g = df.groupby('CAT')['value_raw']
    vmin = g.transform(np.nanmin)
    vmax = g.transform(np.nanmax)
    good = (np.isfinite(vmin)) & (np.isfinite(vmax)) & (vmax != vmin)

    df['value_norm11'] = np.nan
    df.loc[good, 'value_norm11'] = 2 * (v[good] - vmin[good]) / (vmax[good] - vmin[good]) - 1

    return df

# ----------------------------- #
# Plotting
# ----------------------------- #
def _plot_violin_with_overlay(df: pd.DataFrame, outdir: Path, filename: str = "violin_all_CATS.png"):
    """
    Make ONE figure with a violin per CAT (continuous malignant fraction),
    overlay in-house points colored by project, and save to a single PNG.

    - Violin plots use soft pastel colors (distinct per CAT).
    - Strip plots (in-house projects) use popping colors from INHOUSE_COLORS.
    - Figure size: 4.85" (W) x 2.7" (H)
    """
    outdir.mkdir(parents=True, exist_ok=True)

    # color mapping for in-house points by project
    inhouse_projects = sorted(set(p for p in df.loc[df['is_inhouse'], 'project'].dropna()))
    proj2color = {p: INHOUSE_COLORS[i % len(INHOUSE_COLORS)] for i, p in enumerate(inhouse_projects)}

    # use the new [-1,1] scaling (or switch to 'value_raw' if you prefer)
    ycol = 'value_norm11'

    # enforce a specific CAT order if present
    cats_present = [c for c in CAT_ORDER if c in df['CAT'].unique().tolist()]
    if not cats_present:
        cats_present = list(df['CAT'].dropna().unique())

    # pastel palette for violins
    cat_palette = sns.color_palette("pastel", n_colors=len(cats_present))
    cat2color = {c: cat_palette[i] for i, c in enumerate(cats_present)}

    plt.figure(figsize=(4.85, 2.7), dpi=300)

    # violins across CATs with pastel fills
    ax = sns.violinplot(
        data=df,
        x='CAT',
        y=ycol,
        order=cats_present,
        cut=0,
        bw='scott',
        inner=None,
        palette=cat2color
    )

    ax.set_title("Malignant fraction within CATs", fontsize=10)
    ax.set_xlabel("")
    ax.set_ylabel("Normalized malignant fraction (−1 to 1)" if ycol == 'value_norm11'
                  else "Malignant fraction (Pre-malignant or Tumor-like)")

    # overlay in-house strip points (bright colors)
    sub_inhouse = df[df['is_inhouse']].copy()
    if not sub_inhouse.empty:
        sns.stripplot(
            data=sub_inhouse,
            x='CAT',
            y=ycol,
            order=cats_present,
            hue='project',
            dodge=False,
            jitter=True,
            size=3.5,
            edgecolor='k',
            linewidth=0.2,
            palette=proj2color
        )
        ax.legend(
            title="In-house projects",
            bbox_to_anchor=(1.02, 1),
            loc='upper left',
            borderaxespad=0.,
            fontsize=7,
            title_fontsize=8
        )
    else:
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()

    plt.xticks(rotation=20, ha='right')
    plt.tight_layout()
    plt.savefig(outdir / filename, dpi=300, bbox_inches='tight')
    plt.close()

# ----------------------------- #
# Orchestration
# ----------------------------- #
def __violin_plot_5D(adata: sc.AnnData = None, onco_adata: sc.AnnData = None, BASE_DIR: Path = Path(".")):
    """
    - If onco_adata is provided, skip inferencing.
    - If adata is None, fall back to onco_adata as the working object.
    - Ensures donor_id, builds tidy DF with malignant fractions, and plots.
    """
    if adata is None and onco_adata is None:
        raise ValueError("Provide at least one of adata or onco_adata.")

    # If adata not given, use onco_adata as the base
    if adata is None:
        adata = onco_adata.copy()

    # Load donor maps and fill donor_id (your original logic)
    luca = sc.read_h5ad('data/scRNAseq-data/c624f243-f3c9-4a43-8d8e-f8a366f73cca.h5ad')
    normal = sc.read_h5ad('data/scRNAseq-data/b351804c-293e-4aeb-9c4c-043db67f4540.h5ad')
    donor_map = pd.concat([
        pd.DataFrame(luca.obs['donor_id']),
        pd.DataFrame(normal.obs['donor_id'])
    ])

    if 'project' not in adata.obs.columns:
        raise KeyError("adata.obs is missing the required 'project' column.")

    adata.obs['donor_id'] = adata.obs_names.map(donor_map['donor_id'])
    adata.obs['donor_id'] = adata.obs['donor_id'].fillna(adata.obs['project'])

    # If no onco_adata given (or it lacks the class), run inferencing; else use provided
    if onco_adata is None or 'oncoterrain_class' not in onco_adata.obs.columns:
        runner = OncoTerrain(adata)
        runner.inferencing(save_path=BASE_DIR / "figures/all_oncoterrain", save_adata=True)
        onco_adata = sc.read_h5ad(BASE_DIR / "figures/all_oncoterrain/OncoTerrain_annotated.h5ad")

    # Build tidy DF (malignant fraction) + plots
    df = _make_violin_data(adata, onco_adata, CAT_GROUPS, celltype_col=None)
    outdir = BASE_DIR / "figures/violin_5D"
    _plot_violin_with_overlay(df, outdir)
    return df

# ----------------------------- #
# Pseudobulk + model training
# ----------------------------- #

from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler, FunctionTransformer

from pathlib import Path
import json, joblib, numpy as np, pandas as pd, scanpy as sc
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

def _make_sample_level_pseudobulk(adata: sc.AnnData, groupby: str = "donor_id") -> pd.DataFrame:
    """Return a gene × sample DataFrame of raw counts (sum across cells per sample)."""
    if groupby not in adata.obs.columns:
        raise KeyError(f"'{groupby}' not found in adata.obs.")

    groups = adata.obs[groupby].astype("category")
    sample_names = list(groups.cat.categories)

    counts = pd.DataFrame(
        0, index=adata.var_names.astype(str),
        columns=sample_names, dtype=np.float64
    )

    is_sparse = hasattr(adata.X, "tocsr")
    for sample in sample_names:
        cell_mask = (groups == sample).values
        if not np.any(cell_mask):
            continue
        X = adata.X[cell_mask]
        if is_sparse:
            s = np.asarray(X.sum(axis=0)).ravel()
        else:
            s = X.sum(axis=0)
            if hasattr(s, "A1"):
                s = s.A1
        counts[sample] = s

    return counts


def _malignant_fraction_per_sample(
    adata: sc.AnnData,
    label_col: str = "oncoterrain_class",
    sample_col: str = "donor_id",
) -> pd.Series:
    """
    malignant_fraction = (# cells with class in {Pre-malignant, Tumor-like}) / (total cells)
    Returns float Series indexed by sample.
    """
    if label_col not in adata.obs.columns:
        raise KeyError(f"'{label_col}' not found in adata.obs.")
    if sample_col not in adata.obs.columns:
        raise KeyError(f"'{sample_col}' not found in adata.obs.")

    malignant = {"Pre-malignant", "Tumor-like"}

    grp = adata.obs.groupby(sample_col, dropna=False, observed=False)[label_col]
    tot = grp.size().astype(float)
    mal = grp.apply(lambda s: float(s.isin(malignant).sum()))
    frac = mal / tot
    return frac.astype(float)

def _cpm_log1p(X: np.ndarray) -> np.ndarray:
    # X: (n_samples, n_genes) raw counts
    X = np.asarray(X, dtype=float)
    lib = X.sum(axis=1, keepdims=True)
    lib[lib == 0.0] = 1.0
    X = X / lib * 1e6
    return np.log1p(X)

def train_oncoterrain_counts_regressor(
    adata: sc.AnnData,
    outdir: Path,
    test_size: float = 0.2,
    random_state: int = 42,
    alphas=(10**np.linspace(-3, 3, 13)).tolist(),  # CV over 1e-3..1e3
):
    """
    Train a ridge regression model to predict malignant fraction per sample
    using pseudobulk raw counts. Normalization = CPM->log1p; scaling = Std(with_mean=False).
    Pipeline:
      counts (genes x samples) -> transpose to samples x genes
      -> Imputer(0) -> FunctionTransformer(cpm_log1p) -> VarianceThreshold(1e-12)
      -> StandardScaler(with_mean=False) -> RidgeCV
    """
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) Pseudobulk raw counts (genes x samples)
    counts = _make_sample_level_pseudobulk(adata, groupby="donor_id").astype(float)

    # Basic matrix sanity (genes as index, samples as columns)
    counts.index = counts.index.astype(str)
    before_dups = counts.shape[0]
    counts = counts[~counts.index.duplicated(keep="first")]
    after_dups = counts.shape[0]

    # Drop genes that are zero across all samples (safe, no leakage)
    nonzero = counts.sum(axis=1) > 0
    dropped_allzero = int((~nonzero).sum())
    counts = counts.loc[nonzero]

    # 2) Target per sample (malignant fraction)
    y_series = _malignant_fraction_per_sample(
        adata, label_col="oncoterrain_class", sample_col="donor_id"
    )
    y_series = y_series.reindex(counts.columns)
    mask = y_series.notna()
    counts = counts.loc[:, mask]
    y = y_series[mask].astype(float).values

    if not np.isfinite(y).all():
        raise ValueError("Target contains non-finite values.")
    if np.nanstd(y) == 0:
        raise ValueError("Target (malignant fraction) is constant across samples.")

    # 3) Features as DataFrame (keep names)
    X_df = counts.T  # samples x genes
    feature_genes_all = X_df.columns.astype(str).tolist()

    # 4) Train/test split (keep DataFrames to preserve names)
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X_df, y, test_size=test_size, random_state=random_state
    )

    if np.std(y_train) == 0:
        raise ValueError("y_train is constant after the split; try a different random_state.")

    # 5) Pipeline: impute -> cpm_log1p -> var filter -> scale -> ridgeCV
    preprocess = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
        ("cpm_log1p", FunctionTransformer(_cpm_log1p, validate=False)),
        ("var", VarianceThreshold(threshold=1e-12)),
        ("scaler", StandardScaler(with_mean=False)),
    ])
    reg = RidgeCV(alphas=alphas)
    pipe = Pipeline([("prep", preprocess), ("reg", reg)])

    pipe.fit(X_train_df.values, y_train)

    y_pred = pipe.predict(X_test_df.values)
    r2  = float(r2_score(y_test, y_pred))
    mae = float(mean_absolute_error(y_test, y_pred))
    corr = float(np.corrcoef(y_test, y_pred)[0, 1]) if (
        len(y_test) >= 2 and np.std(y_test) > 0 and np.std(y_pred) > 0
    ) else np.nan

    var_support = pipe.named_steps["prep"].named_steps["var"].get_support()
    n_kept = int(var_support.sum())
    if n_kept == 0:
        raise ValueError(
            "No features remained after VarianceThreshold. "
            "Check pseudobulk, gene IDs, and cross-cohort ID consistency."
        )
    kept_feature_genes = X_train_df.columns[var_support].astype(str).tolist()

    intercept = float(pipe.named_steps["reg"].intercept_)
    mean_ytr  = float(np.mean(y_train))

    Z_test = pipe.named_steps["prep"].transform(X_test_df.values)
    frac_zero_var_features = float((Z_test.var(axis=0) == 0).mean())
    median_per_sample_var  = float(np.median(Z_test.var(axis=1)))

    joblib.dump(pipe, outdir / "regressor.joblib")

    with open(outdir / "feature_genes_all.json", "w") as f:
        json.dump(feature_genes_all, f)
    with open(outdir / "feature_genes_kept.json", "w") as f:
        json.dump(kept_feature_genes, f)

    with open(outdir / "report.txt", "w") as f:
        f.write(
            "Counts-based RidgeCV with CPM->log1p + StdScaler(with_mean=False)\n"
            f"alphas: {alphas}\n"
            f"R^2:  {r2:.4f}\n"
            f"MAE:  {mae:.6f}\n"
            f"Pearson r: {corr:.4f}\n"
            f"n_train: {len(y_train)}, n_test: {len(y_test)}\n"
            f"genes_before_dedup: {before_dups}, after_dedup: {after_dups}\n"
            f"genes_dropped_allzero: {dropped_allzero}\n"
            f"n_features_kept_after_var: {n_kept}\n"
            f"intercept: {intercept:.6f}, mean(y_train): {mean_ytr:.6f}\n"
            f"frac_zero_var_features_on_test: {frac_zero_var_features:.6f}\n"
            f"median_per_sample_feature_var_on_test: {median_per_sample_var:.6f}\n"
        )

    with open(outdir / "metadata.json", "w") as f:
        json.dump({
            "target": "malignant_fraction (Pre-malignant + Tumor-like over all cells)",
            "inputs": "pseudobulk raw counts (genes x samples)",
            "transform": "Imputer(0); CPM->log1p; VarianceThreshold(1e-12); StandardScaler(w/o mean)",
            "model": "RidgeCV",
            "alphas": alphas,
            "random_state": random_state,
            "n_features_all": len(feature_genes_all),
            "n_features_kept": n_kept,
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
            "metrics": {"r2": r2, "mae": mae, "pearson_r": corr},
        }, f, indent=2)

    print(f"R^2={r2:.4f} | MAE={mae:.6f} | Pearson r={corr:.4f}")
    print(f"Kept features after variance: {n_kept} / {len(feature_genes_all)}")
    print(f"Intercept={intercept:.6f} | mean(y_train)={mean_ytr:.6f}")
    print(f"Saved regressor to: {outdir.resolve()}")

    return {
        "regressor_path": str((outdir / "regressor.joblib").resolve()),
        "features_all_path": str((outdir / "feature_genes_all.json").resolve()),
        "features_kept_path": str((outdir / "feature_genes_kept.json").resolve()),
        "report_path": str((outdir / "report.txt").resolve()),
        "metadata_path": str((outdir / "metadata.json").resolve()),
        "n_features_all": len(feature_genes_all),
        "n_features_kept": n_kept,
        "r2": r2,
        "mae": mae,
        "pearson_r": corr,
    }

def save_processed_by_stage(adata: sc.AnnData, outdir: Path) -> dict:
    """
    Save per-stage subsets to .h5ad in `outdir`.

    Mapping:
      Pre-malignant -> processed_early.h5ad
      Normal-Like   -> processed_nc.h5ad
      Tumor-like    -> processed_advanced.h5ad
    """
    if 'oncoterrain_class' not in adata.obs.columns:
        raise KeyError("`oncoterrain_class` not found in adata.obs; cannot split by stage.")

    outdir.mkdir(parents=True, exist_ok=True)

    class_to_file = {
        'Pre-malignant':  'processed_early.h5ad',
        'Normal-like':    'processed_nc.h5ad',
        'Tumor-like':     'processed_advanced.h5ad',
    }

    written = {}
    for cls, fname in class_to_file.items():
        mask = adata.obs['oncoterrain_class'] == cls
        n = int(mask.sum())
        if n == 0:
            # nothing to write for this class; skip politely
            continue

        sub = adata[mask].copy()

        # (optional) tidy categories to shrink file size a bit
        for col in sub.obs.select_dtypes(include='category').columns:
            sub.obs[col] = sub.obs[col].cat.remove_unused_categories()

        # write
        path = outdir / fname
        sub.write(path)
        written[cls] = str(path.resolve())

    return written


if __name__ == '__main__':
    BASE_DIR = Path.cwd()
    onco_adata = sc.read_h5ad(BASE_DIR / "figures/all_oncoterrain/OncoTerrain_annotated.h5ad")

    # __violin_plot_5D(adata=onco_adata, onco_adata=onco_adata, BASE_DIR=BASE_DIR)

    # Train model directly from the same annotated object
    out = train_oncoterrain_counts_regressor(
        adata=onco_adata,
        outdir=BASE_DIR / "models/oncoterrain_counts_reg",
        test_size=0.2,
        random_state=42
    )
    print(out)

    # written = save_processed_by_stage(onco_adata, BASE_DIR / "data")
    # print("Processed subsets written:", written)