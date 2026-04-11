from __future__ import annotations

import functools
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from scipy import sparse

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from OncoTerrain.OncoTerrain import OncoTerrain

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = REPO_ROOT / "data" / "processed_data.h5ad"
ONCOTERRAIN_JOBLIB = REPO_ROOT / "src" / "OncoTerrain" / "OncoTerrain.joblib"

DROP_META_COLUMNS = [
    "disease",
    "sample",
    "source",
    "tissue",
    "n_genes",
    "batch",
    "n_genes_by_counts",
    "total_counts",
    "total_counts_mt",
    "pct_counts_mt",
    "leiden_res_0.10",
    "leiden_res_1.00",
    "leiden_res_5.00",
    "leiden_res_10.00",
    "leiden_res_20.00",
    "tumor_stage",
    "project",
]


def densify(X):
    return X.toarray() if sparse.issparse(X) else np.asarray(X)


def load_processed_adata(*, refresh_hallmarks: bool = True):
    adata = sc.read_h5ad(DATA_PATH)
    if refresh_hallmarks:
        adata = OncoTerrain(adata).hp_calculation()
    return adata


def load_model_bundle(bundle_path: Path, *, device_name: str = "cpu"):
    device = torch.device(device_name)
    orig_torch_load = torch.load
    torch.load = functools.partial(orig_torch_load, map_location=device)
    try:
        bundle = joblib.load(bundle_path)
    finally:
        torch.load = orig_torch_load

    model = bundle["model"]
    if hasattr(model, "best_estimator_"):
        model = model.best_estimator_
    if hasattr(model, "device_name"):
        model.device_name = device.type
    if hasattr(model, "device"):
        model.device = device
    if hasattr(model, "network") and model.network is not None:
        model.network = model.network.to(device)
    return bundle, model


def align_feature_frame(adata, feature_list, *, logger: logging.Logger | None = None):
    meta = adata.obs.copy()
    cols_to_drop = [col for col in DROP_META_COLUMNS if col in meta.columns]
    if cols_to_drop:
        meta = meta.drop(columns=cols_to_drop)

    celltype_cols = [col for col in meta.columns if str(col).endswith("_celltype")]
    if celltype_cols:
        meta = meta.drop(columns=celltype_cols)

    meta.columns = meta.columns.astype(str).str.replace("^HALLMARK_", "", regex=True)
    meta = meta.apply(pd.to_numeric, errors="coerce")

    bool_cols = meta.select_dtypes(include=["boolean", "bool"]).columns
    if len(bool_cols):
        meta[bool_cols] = meta[bool_cols].astype(np.float32)

    meta.replace([np.inf, -np.inf], np.nan, inplace=True)
    meta.fillna(0.0, inplace=True)

    feature_set = set(feature_list)
    matched = feature_set & set(meta.columns)
    missing = feature_set - set(meta.columns)
    extra = set(meta.columns) - feature_set
    if logger is not None:
        logger.info(
            "Feature alignment: %d matched, %d missing (filled 0), %d extra (dropped)",
            len(matched),
            len(missing),
            len(extra),
        )
        if missing:
            logger.warning("Missing features (first 20): %s", sorted(missing)[:20])

    for col in missing:
        meta[col] = 0.0
    if extra:
        meta = meta.drop(columns=sorted(extra))
    return meta[list(feature_list)].astype(np.float32)


def extract_gene_expression_matrix(adata, gene_names, *, logger: logging.Logger | None = None):
    gene_names = list(gene_names)
    gene_index = {str(name): idx for idx, name in enumerate(adata.var_names)}
    present = [(out_idx, gene_index[name]) for out_idx, name in enumerate(gene_names) if name in gene_index]
    missing = [name for name in gene_names if name not in gene_index]
    if logger is not None and missing:
        logger.warning("Missing genes in AnnData (first 20): %s", missing[:20])

    X = np.zeros((adata.n_obs, len(gene_names)), dtype=np.float32)
    if present:
        out_idx, adata_idx = zip(*present)
        X[:, np.asarray(out_idx)] = densify(adata.X[:, np.asarray(adata_idx)]).astype(np.float32, copy=False)
    return X


def transform_features(scaler, X):
    X_scaled = scaler.transform(X)
    X_scaled = np.asarray(X_scaled, dtype=np.float32)
    if not np.isfinite(X_scaled).all():
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    return X_scaled


def align_probas(model, probas, target_classes):
    model_classes = getattr(model, "classes_", target_classes)
    if list(model_classes) != list(target_classes):
        order = [list(model_classes).index(c) for c in target_classes]
        return probas[:, order]
    return probas


def predict_proba(model, X, target_classes):
    proba = model.predict_proba(np.asarray(X, dtype=np.float32))
    proba = np.asarray(proba, dtype=np.float32)
    if not np.isfinite(proba).all():
        proba = np.nan_to_num(proba, nan=0.0, posinf=0.0, neginf=0.0)
    return align_probas(model, proba, target_classes)
