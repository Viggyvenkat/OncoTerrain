from __future__ import annotations

import functools
import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from scipy import sparse
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = REPO_ROOT / "data" / "processed_data.h5ad"
ONCOTERRAIN_JOBLIB = REPO_ROOT / "src" / "OncoTerrain" / "OncoTerrain.joblib"
TUMOR_STAGE_MAP = {"non-cancer": 0, "early": 1, "advanced": 2}
DEFAULT_LABEL_NAMES = {0: "Non-Cancer", 1: "Early", 2: "Advanced"}
RANDOM_STATE = 42
EVAL_COHORT_NAME = "figure5_manuscript_matched"


@dataclass(frozen=True)
class CohortArtifacts:
    adata: sc.AnnData
    obs: pd.DataFrame
    y: np.ndarray
    idx_train: np.ndarray
    idx_test: np.ndarray
    metadata: dict


def densify(X):
    return X.toarray() if sparse.issparse(X) else np.asarray(X)


def load_tabnet_bundle(bundle_path: Path):
    _orig_torch_load = torch.load
    torch.load = functools.partial(_orig_torch_load, map_location=torch.device("cpu"))
    try:
        bundle = joblib.load(bundle_path)
    finally:
        torch.load = _orig_torch_load

    model = bundle["model"]
    if hasattr(model, "best_estimator_"):
        model = model.best_estimator_
    if hasattr(model, "device_name"):
        model.device_name = "cpu"
    if hasattr(model, "device"):
        model.device = torch.device("cpu")
    if hasattr(model, "network") and model.network is not None:
        model.network = model.network.to("cpu")
    return bundle, model


def label_names_from_bundle(bundle) -> dict[int, str]:
    raw = bundle.get("label_map") or DEFAULT_LABEL_NAMES
    try:
        return {int(k): str(v) for k, v in sorted(raw.items(), key=lambda kv: int(kv[0]))}
    except Exception:
        return DEFAULT_LABEL_NAMES.copy()


def prepare_manuscript_matched_cohort(
    adata: sc.AnnData,
    *,
    random_state: int = RANDOM_STATE,
    logger: logging.Logger | None = None,
) -> CohortArtifacts:
    log = logger or logging.getLogger(__name__)

    data = adata.obs.copy()
    data["__row_id"] = np.arange(adata.n_obs)

    celltype_mapping = {name: i for i, name in enumerate(data["leiden_res_20.00_celltype"].unique())}
    project_mapping = {name: i for i, name in enumerate(data["project"].unique())}

    epithelial = data[data["leiden_res_20.00_celltype"] == "Epithelial cell"]
    counts = epithelial["tumor_stage"].value_counts()
    if counts.empty:
        raise ValueError("No epithelial cells found for manuscript-matched cohort balancing.")

    min_count = int(counts.min())
    balanced_epithelial = (
        epithelial.groupby("tumor_stage", observed=True, sort=False)
        .apply(lambda group: group.sample(min_count, random_state=random_state), include_groups=False)
        .reset_index(drop=True)
    )
    non_epithelial = data[data["leiden_res_20.00_celltype"] != "Epithelial cell"]
    data = pd.concat([non_epithelial, balanced_epithelial], axis=0)

    data.loc[:, "tumor_stage"] = data["tumor_stage"].astype(str).str.lower().map(TUMOR_STAGE_MAP)
    data.loc[:, "project"] = data["project"].astype(object).map(project_mapping)
    data.loc[:, "leiden_res_20.00_celltype"] = data["leiden_res_20.00_celltype"].astype(object).map(celltype_mapping)
    data = data.dropna(subset=["tumor_stage", "project", "leiden_res_20.00_celltype"]).copy()
    data.loc[:, "tumor_stage"] = data["tumor_stage"].astype(int)
    data.loc[:, "project"] = data["project"].astype(int)
    data.loc[:, "leiden_res_20.00_celltype"] = data["leiden_res_20.00_celltype"].astype(int)

    row_ids = data.pop("__row_id").to_numpy(dtype=int, copy=False)
    cohort_adata = adata[row_ids].copy()
    cohort_obs = data.reset_index(drop=True)
    y = cohort_obs["tumor_stage"].astype(int).to_numpy(copy=False)

    idx = np.arange(len(y))
    idx_train, idx_test, _, _ = train_test_split(
        idx,
        y,
        test_size=0.3,
        random_state=random_state,
        stratify=y,
    )

    stage_counts = pd.Series(y).value_counts().sort_index().to_dict()
    metadata = {
        "evaluation_cohort": EVAL_COHORT_NAME,
        "random_state": int(random_state),
        "n_cells": int(len(y)),
        "n_train": int(len(idx_train)),
        "n_test": int(len(idx_test)),
        "stage_counts": {int(k): int(v) for k, v in stage_counts.items()},
        "epithelial_stage_counts_before_balancing": {str(k): int(v) for k, v in counts.to_dict().items()},
        "epithelial_stage_min_count": int(min_count),
    }

    log.info("Prepared %s cohort with %d cells", EVAL_COHORT_NAME, metadata["n_cells"])
    log.info("Stage counts after preprocessing: %s", metadata["stage_counts"])
    log.info("Shared split sizes: train=%d test=%d", metadata["n_train"], metadata["n_test"])

    return CohortArtifacts(
        adata=cohort_adata,
        obs=cohort_obs,
        y=y,
        idx_train=idx_train,
        idx_test=idx_test,
        metadata=metadata,
    )


def oncoterrain_feature_matrix(obs_df: pd.DataFrame, feature_list):
    meta = obs_df.copy()
    meta.columns = meta.columns.str.replace("^HALLMARK_", "", regex=True)
    meta = meta.apply(pd.to_numeric, errors="coerce")

    bool_cols = meta.select_dtypes(include=["boolean", "bool"]).columns
    if len(bool_cols):
        meta[bool_cols] = meta[bool_cols].astype(np.float32)

    meta.replace([np.inf, -np.inf], np.nan, inplace=True)
    meta.fillna(0.0, inplace=True)

    feature_list = list(feature_list)
    for col in feature_list:
        if col not in meta.columns:
            meta[col] = 0.0
    extra = set(meta.columns) - set(feature_list)
    if extra:
        meta = meta.drop(columns=list(extra))

    return meta[feature_list].astype(np.float32)


def scale_feature_frame(X_df: pd.DataFrame, scaler, bundle) -> np.ndarray:
    X_df = X_df.copy()
    excluded = [col for col in bundle.get("columns_to_exclude", []) if col in X_df.columns]
    scaled_cols = [col for col in X_df.columns if col not in excluded]

    expected = getattr(scaler, "n_features_in_", None)
    if expected is not None and expected == len(scaled_cols):
        if scaled_cols:
            X_df.loc[:, scaled_cols] = scaler.transform(X_df[scaled_cols])
        X_scaled = X_df.to_numpy(dtype=np.float32, copy=False)
    else:
        X_scaled = scaler.transform(X_df).astype(np.float32, copy=False)

    if not np.isfinite(X_scaled).all():
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    return X_scaled
