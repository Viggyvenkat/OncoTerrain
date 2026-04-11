from __future__ import annotations

import gc
import json
import logging
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_curve,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import MinMaxScaler, label_binarize

from analysis_common import (
    ONCOTERRAIN_JOBLIB,
    REPO_ROOT,
    densify,
    extract_gene_expression_matrix,
    load_model_bundle,
    load_processed_adata,
    align_feature_frame,
    predict_proba,
    transform_features,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("baseline_geneexp")

OUT_DIR = REPO_ROOT / "src" / "revision-analyses" / "outputs" / "baseline_geneexp"

TUMOR_STAGE_MAP = {"non-cancer": 0, "early": 1, "advanced": 2}
LABEL_NAMES = {0: "Normal-like", 1: "Pre-malignant", 2: "Tumor-like"}
N_HVG = 2000
RANDOM_STATE = 42
N_ITER_SEARCH = 10

OT_COLORS = {0: "#84A970", 1: "#E4C282", 2: "#FF8C00"}
BASE_COLORS = {0: "#5B8FA8", 1: "#8B7EB8", 2: "#A0A0A0"}


def _encode_tumor_stage(adata):
    stage = adata.obs["tumor_stage"].astype(str)
    if stage.isin(TUMOR_STAGE_MAP).all():
        return stage.map(TUMOR_STAGE_MAP).astype(int).to_numpy()
    return adata.obs["tumor_stage"].astype(int).to_numpy()


def _metrics_row(model_name, y_true, y_pred, y_proba, class_ids):
    rows = [
        {"model": model_name, "metric": "accuracy", "class": "overall", "value": accuracy_score(y_true, y_pred)},
        {
            "model": model_name,
            "metric": "balanced_accuracy",
            "class": "overall",
            "value": balanced_accuracy_score(y_true, y_pred),
        },
        {"model": model_name, "metric": "f1_macro", "class": "overall", "value": f1_score(y_true, y_pred, average="macro")},
    ]

    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=class_ids, zero_division=0)
    for i, cid in enumerate(class_ids):
        label = LABEL_NAMES.get(int(cid), str(cid))
        rows.extend(
            [
                {"model": model_name, "metric": "precision", "class": label, "value": precision[i]},
                {"model": model_name, "metric": "recall", "class": label, "value": recall[i]},
                {"model": model_name, "metric": "f1", "class": label, "value": f1[i]},
            ]
        )

    y_bin = label_binarize(y_true, classes=class_ids)
    roc_aucs = []
    ap_scores = []
    for i, cid in enumerate(class_ids):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        avg_precision = average_precision_score(y_bin[:, i], y_proba[:, i])
        roc_aucs.append(roc_auc)
        ap_scores.append(avg_precision)
        rows.extend(
            [
                {"model": model_name, "metric": "roc_auc", "class": LABEL_NAMES[int(cid)], "value": roc_auc},
                {"model": model_name, "metric": "avg_precision", "class": LABEL_NAMES[int(cid)], "value": avg_precision},
            ]
        )

    rows.extend(
        [
            {"model": model_name, "metric": "roc_auc_macro", "class": "overall", "value": float(np.mean(roc_aucs))},
            {"model": model_name, "metric": "avg_precision_macro", "class": "overall", "value": float(np.mean(ap_scores))},
        ]
    )
    return rows


def _roc_curves(y_true, y_proba, class_ids):
    y_bin = label_binarize(y_true, classes=class_ids)
    curves = {}
    for i, cid in enumerate(class_ids):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        curves[int(cid)] = (fpr, tpr, auc(fpr, tpr))
    return curves


def _pr_curves(y_true, y_proba, class_ids):
    y_bin = label_binarize(y_true, classes=class_ids)
    curves = {}
    for i, cid in enumerate(class_ids):
        precision, recall, _ = precision_recall_curve(y_bin[:, i], y_proba[:, i])
        curves[int(cid)] = (precision, recall, average_precision_score(y_bin[:, i], y_proba[:, i]))
    return curves


def _load_or_train_baseline(adata, idx_train, idx_test, y_train, y_test, out_bundle: Path, *, device_name: str):
    if out_bundle.exists():
        log.info("Found existing baseline bundle at %s. Skipping training.", out_bundle)
        bundle, model = load_model_bundle(out_bundle, device_name=device_name)
        scaler = bundle["scaler"]
        gene_names = list(bundle["features"])
        best_params = bundle.get("best_params", {})
        X_gene = extract_gene_expression_matrix(adata, gene_names, logger=log)
        X_gene_test_s = transform_features(scaler, X_gene[idx_test])
        del X_gene
        gc.collect()
        return model, scaler, gene_names, best_params, X_gene_test_s

    log.info("No baseline bundle found. Proceeding with training.")
    log.info("Selecting top %d highly variable genes", N_HVG)
    if "highly_variable" not in adata.var.columns:
        try:
            sc.pp.highly_variable_genes(adata, n_top_genes=N_HVG, flavor="seurat_v3")
        except Exception:
            sc.pp.highly_variable_genes(adata, n_top_genes=N_HVG, flavor="seurat")

    hvg_mask = adata.var["highly_variable"].to_numpy()
    gene_names = adata.var_names[hvg_mask].tolist()
    X_gene = densify(adata[:, hvg_mask].X).astype(np.float32)
    log.info("Gene-expression feature matrix: %s", X_gene.shape)

    X_gene_train = X_gene[idx_train].copy()
    X_gene_test = X_gene[idx_test].copy()
    del X_gene
    gc.collect()

    scaler = MinMaxScaler()
    X_gene_train_s = scaler.fit_transform(X_gene_train).astype(np.float32, copy=False)
    X_gene_test_s = transform_features(scaler, X_gene_test)
    del X_gene_train, X_gene_test
    gc.collect()

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

    rng = np.random.default_rng(RANDOM_STATE)
    sub_idx = rng.choice(len(y_train), size=min(100_000, len(y_train)), replace=False)
    X_search = X_gene_train_s[sub_idx].astype(np.float32)
    y_search = y_train[sub_idx].astype(int)
    log.info("Subsampled %d cells for RandomizedSearchCV", len(sub_idx))

    base_search = RandomizedSearchCV(
        estimator=TabNetClassifier(verbose=0, device_name=device_name),
        param_distributions=param_grid,
        n_iter=N_ITER_SEARCH,
        scoring="accuracy",
        cv=3,
        verbose=0,
        n_jobs=1,
        random_state=RANDOM_STATE,
        error_score="raise",
    )
    log.info("Running RandomizedSearchCV over TabNet (n_iter=%d)", N_ITER_SEARCH)
    base_search.fit(X_search, y_search)
    best_params = base_search.best_params_
    log.info("Baseline best params: %s", best_params)
    del X_search, y_search, base_search
    gc.collect()

    model = TabNetClassifier(**best_params, verbose=1, device_name=device_name)
    log.info("Refitting best TabNet on full training set (%d cells)", len(y_train))
    model.fit(
        X_gene_train_s,
        y_train.astype(int),
        eval_set=[(X_gene_test_s, y_test.astype(int))],
        eval_metric=["accuracy", "balanced_accuracy", "logloss"],
        batch_size=16384,
        virtual_batch_size=512,
        num_workers=0,
        drop_last=False,
    )

    joblib.dump(
        {
            "model": model,
            "scaler": scaler,
            "features": gene_names,
            "label_map": LABEL_NAMES,
            "best_params": best_params,
            "n_hvg": N_HVG,
            "random_state": RANDOM_STATE,
        },
        out_bundle,
    )
    log.info("Wrote %s", out_bundle)
    return model, scaler, gene_names, best_params, X_gene_test_s


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_bundle = OUT_DIR / "baseline_geneexp.joblib"
    device_name = "cuda" if torch.cuda.is_available() else "cpu"

    log.info("Loading processed AnnData and refreshing hallmark scores")
    adata = load_processed_adata(refresh_hallmarks=True)
    y_full = _encode_tumor_stage(adata)

    idx = np.arange(len(y_full))
    idx_train, idx_test, y_train, y_test = train_test_split(
        idx,
        y_full,
        test_size=0.3,
        random_state=RANDOM_STATE,
        stratify=y_full,
    )
    log.info("Train %d / Test %d", len(idx_train), len(idx_test))

    base_model, _, gene_names, best_params, X_gene_test_s = _load_or_train_baseline(
        adata,
        idx_train,
        idx_test,
        y_train,
        y_test,
        out_bundle,
        device_name=device_name,
    )

    log.info("Loading OncoTerrain bundle: %s", ONCOTERRAIN_JOBLIB)
    ot_bundle, ot_model = load_model_bundle(ONCOTERRAIN_JOBLIB, device_name=device_name)
    ot_features = list(ot_bundle["features"])
    ot_scaler = ot_bundle["scaler"]
    X_ot_df = align_feature_frame(adata, ot_features, logger=log)
    X_ot_test_s = transform_features(ot_scaler, X_ot_df.iloc[idx_test])

    del adata, X_ot_df
    gc.collect()

    class_ids = np.array([0, 1, 2])
    base_proba = predict_proba(base_model, X_gene_test_s, class_ids)
    ot_proba = predict_proba(ot_model, X_ot_test_s, class_ids)
    base_pred = class_ids[np.argmax(base_proba, axis=1)]
    ot_pred = class_ids[np.argmax(ot_proba, axis=1)]

    rows = []
    rows.extend(_metrics_row("baseline_geneexp_tabnet", y_test, base_pred, base_proba, class_ids))
    rows.extend(_metrics_row("OncoTerrain", y_test, ot_pred, ot_proba, class_ids))
    metrics_df = pd.DataFrame(rows)
    metrics_csv = OUT_DIR / "metrics_comparison.csv"
    metrics_df.to_csv(metrics_csv, index=False)
    log.info("Wrote %s", metrics_csv)

    sns.set_theme(style="white", context="talk")

    base_roc = _roc_curves(y_test, base_proba, class_ids)
    ot_roc = _roc_curves(y_test, ot_proba, class_ids)
    fig, ax = plt.subplots(figsize=(9, 7))
    for cid in class_ids:
        fpr, tpr, roc_auc = base_roc[int(cid)]
        ax.plot(
            fpr,
            tpr,
            linestyle="--",
            linewidth=2,
            color=BASE_COLORS[int(cid)],
            label=f"Baseline {LABEL_NAMES[int(cid)]} (AUC={roc_auc:.2f})",
        )
        fpr, tpr, roc_auc = ot_roc[int(cid)]
        ax.plot(
            fpr,
            tpr,
            linestyle="-",
            linewidth=2,
            color=OT_COLORS[int(cid)],
            label=f"OncoTerrain {LABEL_NAMES[int(cid)]} (AUC={roc_auc:.2f})",
        )
    ax.plot([0, 1], [0, 1], "k:", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("OncoTerrain vs Gene-Expression Baseline (One-vs-Rest ROC)")
    ax.legend(loc="lower right", fontsize=9)
    sns.despine(ax=ax)
    plt.tight_layout()
    roc_png = OUT_DIR / "roc_overlay.png"
    plt.savefig(roc_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", roc_png)

    base_pr = _pr_curves(y_test, base_proba, class_ids)
    ot_pr = _pr_curves(y_test, ot_proba, class_ids)
    fig, ax = plt.subplots(figsize=(9, 7))
    for cid in class_ids:
        precision, recall, avg_precision = base_pr[int(cid)]
        ax.plot(
            recall,
            precision,
            linestyle="--",
            linewidth=2,
            color=BASE_COLORS[int(cid)],
            label=f"Baseline {LABEL_NAMES[int(cid)]} (AP={avg_precision:.2f})",
        )
        precision, recall, avg_precision = ot_pr[int(cid)]
        ax.plot(
            recall,
            precision,
            linestyle="-",
            linewidth=2,
            color=OT_COLORS[int(cid)],
            label=f"OncoTerrain {LABEL_NAMES[int(cid)]} (AP={avg_precision:.2f})",
        )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_title("OncoTerrain vs Gene-Expression Baseline (One-vs-Rest PR)")
    ax.legend(loc="lower left", fontsize=9)
    sns.despine(ax=ax)
    plt.tight_layout()
    pr_png = OUT_DIR / "pr_overlay.png"
    plt.savefig(pr_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", pr_png)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (title, pred) in zip(axes, [("baseline (gene expr)", base_pred), ("OncoTerrain", ot_pred)]):
        cm = confusion_matrix(y_test, pred, labels=class_ids)
        disp = ConfusionMatrixDisplay(cm, display_labels=[LABEL_NAMES[int(c)] for c in class_ids])
        disp.plot(ax=ax, cmap=plt.cm.Blues, colorbar=False)
        ax.set_title(title)
    plt.tight_layout()
    cm_png = OUT_DIR / "confusion_matrices.png"
    plt.savefig(cm_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", cm_png)

    with (OUT_DIR / "summary.json").open("w") as handle:
        json.dump(
            {
                "n_train": int(len(idx_train)),
                "n_test": int(len(idx_test)),
                "n_hvg_features": int(len(gene_names)),
                "n_oncoterrain_features": int(len(ot_features)),
                "baseline_best_params": best_params,
            },
            handle,
            indent=2,
            default=str,
        )

    print(metrics_df.pivot_table(index=["metric", "class"], columns="model", values="value"))


if __name__ == "__main__":
    main()
