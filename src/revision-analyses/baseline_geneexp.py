from __future__ import annotations

import json
import logging

import matplotlib.pyplot as plt
import joblib
import numpy as np
import pandas as pd
import scanpy as sc
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_curve,
)
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler, label_binarize

from revision_common import (
    DATA_PATH,
    ONCOTERRAIN_JOBLIB,
    RANDOM_STATE,
    REPO_ROOT,
    densify,
    label_names_from_bundle,
    load_tabnet_bundle,
    oncoterrain_feature_matrix,
    prepare_manuscript_matched_cohort,
    scale_feature_frame,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("baseline_geneexp")

OUT_DIR = REPO_ROOT / "src" / "revision-analyses" / "outputs" / "baseline_geneexp"
N_HVG = 2000
N_ITER_SEARCH = 10


def _metrics_rows(model_name, y_true, y_pred, y_proba, class_ids, label_names, metadata):
    rows = []
    base = {
        "model": model_name,
        "evaluation_cohort": metadata["evaluation_cohort"],
        "split_seed": metadata["random_state"],
        "n_cells": metadata["n_cells"],
        "n_test": metadata["n_test"],
    }

    rows.append({**base, "metric": "accuracy", "class": "overall", "value": accuracy_score(y_true, y_pred)})
    rows.append(
        {
            **base,
            "metric": "balanced_accuracy",
            "class": "overall",
            "value": balanced_accuracy_score(y_true, y_pred),
        }
    )
    rows.append({**base, "metric": "f1_macro", "class": "overall", "value": f1_score(y_true, y_pred, average="macro")})

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=class_ids,
        zero_division=0,
    )
    y_bin = label_binarize(y_true, classes=class_ids)
    roc_aucs = []
    avg_precs = []
    for i, cid in enumerate(class_ids):
        class_name = label_names[int(cid)]
        rows.append({**base, "metric": "precision", "class": class_name, "value": precision[i]})
        rows.append({**base, "metric": "recall", "class": class_name, "value": recall[i]})
        rows.append({**base, "metric": "f1", "class": class_name, "value": f1[i]})

        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        avg_precision = average_precision_score(y_bin[:, i], y_proba[:, i])
        roc_aucs.append(roc_auc)
        avg_precs.append(avg_precision)
        rows.append({**base, "metric": "roc_auc", "class": class_name, "value": roc_auc})
        rows.append({**base, "metric": "avg_precision", "class": class_name, "value": avg_precision})

    rows.append({**base, "metric": "roc_auc_macro", "class": "overall", "value": float(np.mean(roc_aucs))})
    rows.append({**base, "metric": "avg_precision_macro", "class": "overall", "value": float(np.mean(avg_precs))})
    return rows


def _roc_curves(y_true, y_proba, class_ids):
    y_bin = label_binarize(y_true, classes=class_ids)
    curves = {}
    for i, cid in enumerate(class_ids):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        curves[int(cid)] = (fpr, tpr, auc(fpr, tpr))
    return curves


def _gene_expression_matrix(adata, gene_names):
    gene_names = list(gene_names)
    X = np.zeros((adata.n_obs, len(gene_names)), dtype=np.float32)
    present = [gene for gene in gene_names if gene in adata.var_names]
    if present:
        present_idx = [gene_names.index(gene) for gene in present]
        X[:, present_idx] = densify(adata[:, present].X).astype(np.float32, copy=False)
    missing = [gene for gene in gene_names if gene not in adata.var_names]
    if missing:
        log.warning("Missing genes in AnnData (first 20): %s", missing[:20])
    return X


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    baseline_bundle_path = OUT_DIR / "baseline_geneexp.joblib"

    log.info("Loading %s", DATA_PATH)
    adata = sc.read_h5ad(DATA_PATH)

    cohort = prepare_manuscript_matched_cohort(adata, random_state=RANDOM_STATE, logger=log)
    y_full = cohort.y
    idx_train = cohort.idx_train
    idx_test = cohort.idx_test

    log.info("Selecting top %d highly variable genes on manuscript-matched cohort", N_HVG)
    if "highly_variable" not in cohort.adata.var.columns:
        try:
            sc.pp.highly_variable_genes(cohort.adata, n_top_genes=N_HVG, flavor="seurat_v3")
        except Exception:
            sc.pp.highly_variable_genes(cohort.adata, n_top_genes=N_HVG, flavor="seurat")
    hvg_mask = cohort.adata.var["highly_variable"].to_numpy()
    gene_names = cohort.adata.var_names[hvg_mask].tolist()
    X_gene = densify(cohort.adata[:, hvg_mask].X).astype(np.float32)
    log.info("Gene-expression feature matrix: %s", X_gene.shape)

    log.info("Loading OncoTerrain bundle: %s", ONCOTERRAIN_JOBLIB)
    ot_bundle, ot_model = load_tabnet_bundle(ONCOTERRAIN_JOBLIB)
    label_names = label_names_from_bundle(ot_bundle)
    class_ids = np.array(sorted(label_names))
    X_ot = oncoterrain_feature_matrix(cohort.obs, ot_bundle["features"])

    X_gene_train = X_gene[idx_train]
    X_gene_test = X_gene[idx_test]
    y_train = y_full[idx_train].astype(int)
    y_test = y_full[idx_test].astype(int)
    if baseline_bundle_path.exists():
        log.info("Reusing existing baseline bundle: %s", baseline_bundle_path)
        baseline_bundle, baseline_model = load_tabnet_bundle(baseline_bundle_path)
        baseline_features = list(baseline_bundle["features"])
        baseline_scaler = baseline_bundle["scaler"]
        X_gene = _gene_expression_matrix(cohort.adata, baseline_features)
        X_gene_test = X_gene[idx_test]
        X_gene_test_s = baseline_scaler.transform(X_gene_test).astype(np.float32, copy=False)
        if not np.isfinite(X_gene_test_s).all():
            X_gene_test_s = np.nan_to_num(X_gene_test_s, nan=0.0, posinf=0.0, neginf=0.0)
        base_proba = baseline_model.predict_proba(X_gene_test_s)
        base_pred = np.argmax(base_proba, axis=1)
    else:
        base_scaler = MinMaxScaler()
        X_gene_train_s = base_scaler.fit_transform(X_gene_train).astype(np.float32, copy=False)
        X_gene_test_s = base_scaler.transform(X_gene_test).astype(np.float32, copy=False)

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
        base_search = RandomizedSearchCV(
            estimator=TabNetClassifier(verbose=0),
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
            X_gene_train_s,
            y_train,
            eval_set=[(X_gene_test_s, y_test)],
            eval_metric=["accuracy", "balanced_accuracy", "logloss"],
        )
        log.info("Baseline best params: %s", base_search.best_params_)

        base_proba = base_search.predict_proba(X_gene_test_s)
        base_pred = np.argmax(base_proba, axis=1)

        baseline_bundle = {
            "model": base_search,
            "features": gene_names,
            "scaler": base_scaler,
            "label_map": label_names,
        }
        joblib.dump(baseline_bundle, baseline_bundle_path)
        log.info("Wrote %s", baseline_bundle_path)

    X_ot_test = scale_feature_frame(X_ot.iloc[idx_test].reset_index(drop=True), ot_bundle["scaler"], ot_bundle)
    ot_proba = ot_model.predict_proba(X_ot_test)
    ot_classes = np.asarray(getattr(ot_model, "classes_", class_ids))
    if list(ot_classes) != list(class_ids):
        order = [list(ot_classes).index(cid) for cid in class_ids]
        ot_proba = ot_proba[:, order]
    ot_pred = np.argmax(ot_proba, axis=1)

    metrics_rows = []
    metrics_rows.extend(_metrics_rows("baseline_geneexp_tabnet", y_test, base_pred, base_proba, class_ids, label_names, cohort.metadata))
    metrics_rows.extend(_metrics_rows("OncoTerrain", y_test, ot_pred, ot_proba, class_ids, label_names, cohort.metadata))
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_csv = OUT_DIR / "metrics_comparison.csv"
    metrics_df.to_csv(metrics_csv, index=False)
    log.info("Wrote %s", metrics_csv)

    base_roc = _roc_curves(y_test, base_proba, class_ids)
    ot_roc = _roc_curves(y_test, ot_proba, class_ids)
    fig, ax = plt.subplots(figsize=(9, 7))
    colors = {int(cid): color for cid, color in zip(class_ids, ["#1f77b4", "#ff7f0e", "#2ca02c"], strict=False)}
    for cid in class_ids:
        fpr, tpr, score = base_roc[int(cid)]
        ax.plot(fpr, tpr, linestyle="--", color=colors[int(cid)], label=f"baseline {label_names[int(cid)]} (AUC={score:.2f})")
        fpr, tpr, score = ot_roc[int(cid)]
        ax.plot(fpr, tpr, linestyle="-", color=colors[int(cid)], label=f"OncoTerrain {label_names[int(cid)]} (AUC={score:.2f})")
    ax.plot([0, 1], [0, 1], "k:", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("OncoTerrain vs gene-expression baseline (Figure-5-matched cohort)")
    ax.legend(loc="lower right", fontsize=9)
    roc_png = OUT_DIR / "roc_overlay.png"
    plt.tight_layout()
    plt.savefig(roc_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", roc_png)

    cm = confusion_matrix(y_test, ot_pred, labels=class_ids)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[label_names[int(cid)] for cid in class_ids])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("OncoTerrain confusion matrix")
    cm_png = OUT_DIR / "oncoterrain_confusion_matrix.png"
    plt.savefig(cm_png, dpi=300, bbox_inches="tight")
    plt.close()
    log.info("Wrote %s", cm_png)

    summary = {
        **cohort.metadata,
        "label_names": {str(k): v for k, v in label_names.items()},
        "n_hvg": int(len(gene_names)),
        "baseline_gene_expression_accuracy": float(accuracy_score(y_test, base_pred)),
        "oncoterrain_accuracy": float(accuracy_score(y_test, ot_pred)),
    }
    summary_path = OUT_DIR / "evaluation_summary.json"
    with summary_path.open("w") as handle:
        json.dump(summary, handle, indent=2)
    log.info("Wrote %s", summary_path)


if __name__ == "__main__":
    main()
