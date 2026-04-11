from __future__ import annotations

import functools
import json
import logging
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from scipy import sparse
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_curve,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import MinMaxScaler, label_binarize

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("baseline_geneexp")

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = REPO_ROOT / "data" / "processed_data.h5ad"
ONCOTERRAIN_JOBLIB = REPO_ROOT / "src" / "OncoTerrain" / "OncoTerrain.joblib"
OUT_DIR = REPO_ROOT / "src" / "revision-analyses" / "outputs" / "baseline_geneexp"

TUMOR_STAGE_MAP = {"non-cancer": 0, "early": 1, "advanced": 2}
LABEL_NAMES = {0: "Normal-like", 1: "Pre-malignant", 2: "Tumor-like"}
N_HVG = 2000
RANDOM_STATE = 42
N_ITER_SEARCH = 10


def _densify(X):
    return X.toarray() if sparse.issparse(X) else np.asarray(X)


def _oncoterrain_feature_matrix(adata, feature_list):
    meta = adata.obs.copy()
    drop_cols = [
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
        "leiden_res_0.10_celltype",
        "leiden_res_1.00_celltype",
        "leiden_res_5.00_celltype",
        "leiden_res_10.00_celltype",
        "tumor_stage",
        "project",
    ]
    for c in drop_cols:
        if c in meta.columns:
            meta = meta.drop(columns=[c])
    meta.columns = meta.columns.str.replace("^HALLMARK_", "", regex=True)

    ctkey = "leiden_res_20.00_celltype"
    if ctkey in meta.columns:
        meta = meta.drop(columns=[ctkey])

    meta = meta.apply(pd.to_numeric, errors="coerce")
    bool_cols = meta.select_dtypes(include=["boolean", "bool"]).columns
    if len(bool_cols):
        meta[bool_cols] = meta[bool_cols].astype(np.float32)
    meta.replace([np.inf, -np.inf], np.nan, inplace=True)
    meta.fillna(0.0, inplace=True)

    for col in feature_list:
        if col not in meta.columns:
            meta[col] = 0.0
    extra = set(meta.columns) - set(feature_list)
    if extra:
        meta = meta.drop(columns=list(extra))
    return meta[feature_list].astype(np.float32)


def _metrics_row(model_name, y_true, y_pred, y_proba, class_ids):
    rows = []
    rows.append({"model": model_name, "metric": "accuracy", "class": "overall", "value": accuracy_score(y_true, y_pred)})
    rows.append({"model": model_name, "metric": "balanced_accuracy", "class": "overall", "value": balanced_accuracy_score(y_true, y_pred)})
    rows.append({"model": model_name, "metric": "f1_macro", "class": "overall", "value": f1_score(y_true, y_pred, average="macro")})

    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=class_ids, zero_division=0)
    for i, cid in enumerate(class_ids):
        name = LABEL_NAMES.get(int(cid), str(cid))
        rows.append({"model": model_name, "metric": "precision", "class": name, "value": p[i]})
        rows.append({"model": model_name, "metric": "recall", "class": name, "value": r[i]})
        rows.append({"model": model_name, "metric": "f1", "class": name, "value": f[i]})

    y_bin = label_binarize(y_true, classes=class_ids)
    aucs = []
    for i, cid in enumerate(class_ids):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        a = auc(fpr, tpr)
        aucs.append(a)
        rows.append({"model": model_name, "metric": "roc_auc", "class": LABEL_NAMES.get(int(cid), str(cid)), "value": a})
    rows.append({"model": model_name, "metric": "roc_auc_macro", "class": "overall", "value": float(np.mean(aucs))})
    return rows


def _roc_curves(y_true, y_proba, class_ids):
    y_bin = label_binarize(y_true, classes=class_ids)
    out = {}
    for i, cid in enumerate(class_ids):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        out[int(cid)] = (fpr, tpr, auc(fpr, tpr))
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Loading %s", DATA_PATH)
    adata = sc.read_h5ad(DATA_PATH)

    stage = adata.obs["tumor_stage"].astype(str)
    if stage.isin(TUMOR_STAGE_MAP).all():
        y_full = stage.map(TUMOR_STAGE_MAP).astype(int).values
    else:
        y_full = adata.obs["tumor_stage"].astype(int).values

    log.info("Selecting top %d highly variable genes", N_HVG)
    if "highly_variable" not in adata.var.columns:
        try:
            sc.pp.highly_variable_genes(adata, n_top_genes=N_HVG, flavor="seurat_v3")
        except Exception:
            sc.pp.highly_variable_genes(adata, n_top_genes=N_HVG, flavor="seurat")
    hvg_mask = adata.var["highly_variable"].values
    gene_names = adata.var_names[hvg_mask].tolist()
    X_gene = _densify(adata[:, hvg_mask].X).astype(np.float32)
    log.info("Gene-expression feature matrix: %s", X_gene.shape)

    log.info("Loading OncoTerrain bundle: %s", ONCOTERRAIN_JOBLIB)
    _orig_torch_load = torch.load
    torch.load = functools.partial(_orig_torch_load, map_location=torch.device("cpu"))
    try:
        bundle = joblib.load(ONCOTERRAIN_JOBLIB)
    finally:
        torch.load = _orig_torch_load
    ot_model = bundle["model"]
    if hasattr(ot_model, "best_estimator_"):
        ot_model = ot_model.best_estimator_
    ot_model.device = torch.device("cpu")
    if hasattr(ot_model, "network") and ot_model.network is not None:
        ot_model.network = ot_model.network.to("cpu")
    ot_features = list(bundle["features"])
    ot_scaler = bundle["scaler"]

    X_ot_df = _oncoterrain_feature_matrix(adata, ot_features)

    idx = np.arange(len(y_full))
    idx_train, idx_test, y_train, y_test = train_test_split(
        idx, y_full, test_size=0.3, random_state=RANDOM_STATE, stratify=y_full
    )
    log.info("Train %d / Test %d", len(idx_train), len(idx_test))

    X_gene_train = X_gene[idx_train]
    X_gene_test = X_gene[idx_test]

    base_scaler = MinMaxScaler()
    X_gene_train_s = base_scaler.fit_transform(X_gene_train)
    X_gene_test_s = base_scaler.transform(X_gene_test)

    param_grid = {
        "n_d": [8, 16, 24],
        "n_a": [8, 16, 24],
        "n_steps": [3, 5, 7],
        "gamma": [1.0, 1.5, 2.0],
        "lambda_sparse": [1e-3, 1e-4, 1e-5],
        "mask_type": ["sparsemax", "entmax"],
        "n_independent": [2, 4],
        "n_shared": [2, 4],
    }
    log.info("Running RandomizedSearchCV over TabNet (n_iter=%d)", N_ITER_SEARCH)
    base_clf = TabNetClassifier(verbose=0)
    base_search = RandomizedSearchCV(
        estimator=base_clf,
        param_distributions=param_grid,
        n_iter=N_ITER_SEARCH,
        scoring="accuracy",
        cv=3,
        verbose=0,
        n_jobs=1,
        random_state=RANDOM_STATE,
        error_score="raise",
    )
    base_search.fit(
        X_gene_train_s.astype(np.float32),
        y_train.astype(int),
        eval_set=[(X_gene_test_s.astype(np.float32), y_test.astype(int))],
        eval_metric=["accuracy", "balanced_accuracy", "logloss"],
    )
    log.info("Baseline best params: %s", base_search.best_params_)

    base_proba = base_search.predict_proba(X_gene_test_s.astype(np.float32))
    base_pred = np.argmax(base_proba, axis=1)

    X_ot_test = X_ot_df.iloc[idx_test]
    X_ot_test_s = ot_scaler.transform(X_ot_test)
    if not np.isfinite(X_ot_test_s).all():
        X_ot_test_s = np.nan_to_num(X_ot_test_s, nan=0.0, posinf=0.0, neginf=0.0)

    ot_proba = ot_model.predict_proba(X_ot_test_s.astype(np.float32))
    ot_pred = np.argmax(ot_proba, axis=1)

    class_ids = np.array([0, 1, 2])

    ot_classes = getattr(ot_model, "classes_", class_ids)
    if list(ot_classes) != list(class_ids):
        order = [list(ot_classes).index(c) for c in class_ids]
        ot_proba = ot_proba[:, order]

    rows = []
    rows.extend(_metrics_row("baseline_geneexp_tabnet", y_test, base_pred, base_proba, class_ids))
    rows.extend(_metrics_row("OncoTerrain", y_test, ot_pred, ot_proba, class_ids))
    metrics_df = pd.DataFrame(rows)
    metrics_csv = OUT_DIR / "metrics_comparison.csv"
    metrics_df.to_csv(metrics_csv, index=False)
    log.info("Wrote %s", metrics_csv)

    base_roc = _roc_curves(y_test, base_proba, class_ids)
    ot_roc = _roc_curves(y_test, ot_proba, class_ids)
    fig, ax = plt.subplots(figsize=(9, 7))
    colors = {0: "#1f77b4", 1: "#ff7f0e", 2: "#2ca02c"}
    for cid in class_ids:
        fpr, tpr, a = base_roc[int(cid)]
        ax.plot(fpr, tpr, linestyle="--", color=colors[int(cid)], label=f"baseline {LABEL_NAMES[int(cid)]} (AUC={a:.2f})")
        fpr, tpr, a = ot_roc[int(cid)]
        ax.plot(fpr, tpr, linestyle="-", color=colors[int(cid)], label=f"OncoTerrain {LABEL_NAMES[int(cid)]} (AUC={a:.2f})")
    ax.plot([0, 1], [0, 1], "k:", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("OncoTerrain vs gene-expression baseline (One-vs-Rest ROC)")
    ax.legend(loc="lower right", fontsize=9)
    roc_png = OUT_DIR / "roc_overlay.png"
    plt.tight_layout()
    plt.savefig(roc_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", roc_png)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (name, preds) in zip(axes, [("baseline (gene expr)", base_pred), ("OncoTerrain", ot_pred)]):
        cm = confusion_matrix(y_test, preds, labels=class_ids)
        disp = ConfusionMatrixDisplay(cm, display_labels=[LABEL_NAMES[int(c)] for c in class_ids])
        disp.plot(ax=ax, cmap=plt.cm.Blues, colorbar=False)
        ax.set_title(name)
    cm_png = OUT_DIR / "confusion_matrices.png"
    plt.tight_layout()
    plt.savefig(cm_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", cm_png)

    out_bundle = OUT_DIR / "baseline_geneexp.joblib"
    joblib.dump(
        {
            "model": base_search,
            "scaler": base_scaler,
            "features": gene_names,
            "label_map": LABEL_NAMES,
            "best_params": base_search.best_params_,
            "n_hvg": N_HVG,
            "random_state": RANDOM_STATE,
        },
        out_bundle,
    )
    log.info("Wrote %s", out_bundle)

    with (OUT_DIR / "summary.json").open("w") as f:
        json.dump(
            {
                "n_train": int(len(idx_train)),
                "n_test": int(len(idx_test)),
                "n_hvg_features": int(X_gene.shape[1]),
                "n_oncoterrain_features": int(len(ot_features)),
                "baseline_best_params": base_search.best_params_,
            },
            f,
            indent=2,
            default=str,
        )

    print(metrics_df.pivot_table(index=["metric", "class"], columns="model", values="value"))


if __name__ == "__main__":
    main()
