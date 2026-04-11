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
from scipy import sparse
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("sensitivity_analysis")

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = REPO_ROOT / "data" / "processed_data.h5ad"
ONCOTERRAIN_JOBLIB = REPO_ROOT / "src" / "OncoTerrain" / "OncoTerrain.joblib"
BASELINE_BUNDLE = REPO_ROOT / "src" / "revision-analyses" / "outputs" / "baseline_geneexp" / "baseline_geneexp.joblib"
OUT_DIR = REPO_ROOT / "src" / "revision-analyses" / "outputs" / "sensitivity"

TUMOR_STAGE_MAP = {"non-cancer": 0, "early": 1, "advanced": 2}
LABEL_NAMES = {0: "Normal-like", 1: "Pre-malignant", 2: "Tumor-like"}
RANDOM_STATE = 42
CLASS_IDS = np.array([0, 1, 2])
DROPOUT_FRACTIONS = (0.00, 0.01, 0.05, 0.10, 0.20, 0.30)
DROPOUT_REPEATS = 20
MODEL_COLORS = {"OncoTerrain": "#FF8C00", "baseline_geneexp_tabnet": "#5B8FA8"}


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


def _gene_expression_matrix(adata, gene_names):
    gene_names = list(gene_names)
    missing = [gene for gene in gene_names if gene not in adata.var_names]
    if missing:
        log.warning("Missing genes in AnnData (first 20): %s", missing[:20])

    present = [gene for gene in gene_names if gene in adata.var_names]
    X = np.zeros((adata.n_obs, len(gene_names)), dtype=np.float32)
    if present:
        present_idx = [gene_names.index(gene) for gene in present]
        X[:, present_idx] = _densify(adata[:, present].X).astype(np.float32, copy=False)
    return X


def _predict(model, scaler, X_df):
    X = scaler.transform(X_df)
    if not np.isfinite(X).all():
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return model.predict_proba(X.astype(np.float32))


def _predict_scaled(model, X_scaled):
    proba = model.predict_proba(np.asarray(X_scaled, dtype=np.float32))
    if not np.isfinite(proba).all():
        proba = np.nan_to_num(proba, nan=0.0, posinf=0.0, neginf=0.0)
    model_classes = getattr(model, "classes_", CLASS_IDS)
    if list(model_classes) != list(CLASS_IDS):
        order = [list(model_classes).index(c) for c in CLASS_IDS]
        proba = proba[:, order]
    return proba


def _load_tabnet_bundle(bundle_path: Path):
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


def _dropout_sensitivity(model_name, model, X_scaled, y_true):
    base_proba = _predict_scaled(model, X_scaled)
    base_pred = np.argmax(base_proba, axis=1)
    base_acc = accuracy_score(y_true, base_pred)
    n_features = X_scaled.shape[1]

    records = []
    for frac in DROPOUT_FRACTIONS:
        n_drop = int(round(frac * n_features))
        if frac > 0.0 and n_drop == 0 and n_features > 0:
            n_drop = 1
        log.info("%s dropout sensitivity at %.2f (%d features)", model_name, frac, n_drop)
        for repeat in range(DROPOUT_REPEATS):
            rng = np.random.default_rng(RANDOM_STATE + repeat + int(frac * 1000))
            X_corrupt = X_scaled.copy()
            if n_drop > 0:
                drop_idx = rng.choice(n_features, size=n_drop, replace=False)
                X_corrupt[:, drop_idx] = 0.0

            proba = _predict_scaled(model, X_corrupt)
            pred = np.argmax(proba, axis=1)
            abs_delta = np.abs(proba - base_proba)
            acc = accuracy_score(y_true, pred)
            records.append(
                {
                    "model": model_name,
                    "dropout_fraction": float(frac),
                    "repeat": int(repeat),
                    "n_features_total": int(n_features),
                    "n_features_dropped": int(n_drop),
                    "accuracy": float(acc),
                    "delta_accuracy": float(acc - base_acc),
                    "mean_L1_proba_shift": float(abs_delta.sum(axis=1).mean()),
                    "pred_flip_fraction": float((pred != base_pred).mean()),
                }
            )

    repeat_df = pd.DataFrame(records)
    summary_df = (
        repeat_df.groupby(["model", "dropout_fraction", "n_features_total", "n_features_dropped"], as_index=False)
        .agg(
            mean_accuracy=("accuracy", "mean"),
            std_accuracy=("accuracy", "std"),
            mean_delta_accuracy=("delta_accuracy", "mean"),
            std_delta_accuracy=("delta_accuracy", "std"),
            mean_L1_proba_shift=("mean_L1_proba_shift", "mean"),
            std_L1_proba_shift=("mean_L1_proba_shift", "std"),
            mean_pred_flip_fraction=("pred_flip_fraction", "mean"),
            std_pred_flip_fraction=("pred_flip_fraction", "std"),
        )
        .fillna(0.0)
    )
    summary_df["base_accuracy"] = float(base_acc)
    return repeat_df, summary_df, float(base_acc)


def _plot_robustness_curve(curve_df):
    fig, ax = plt.subplots(figsize=(8, 6))
    for model_name, model_df in curve_df.groupby("model"):
        model_df = model_df.sort_values("dropout_fraction")
        x = model_df["dropout_fraction"].to_numpy()
        y = model_df["mean_accuracy"].to_numpy()
        yerr = model_df["std_accuracy"].to_numpy()
        color = MODEL_COLORS.get(model_name, "#333333")
        ax.plot(x, y, marker="o", linewidth=2, color=color, label=model_name)
        ax.fill_between(x, np.clip(y - yerr, 0.0, 1.0), np.clip(y + yerr, 0.0, 1.0), color=color, alpha=0.18)

    ax.set_xlabel("Feature dropout fraction")
    ax.set_ylabel("Accuracy on held-out test cells")
    ax.set_title("Robustness to random feature dropout")
    ax.set_xlim(min(DROPOUT_FRACTIONS), max(DROPOUT_FRACTIONS))
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="best")
    ax.grid(alpha=0.25, linewidth=0.5)
    plt.tight_layout()
    out_path = OUT_DIR / "robustness_curve.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", out_path)


def main():
    if not BASELINE_BUNDLE.exists():
        raise FileNotFoundError(f"Missing baseline bundle at {BASELINE_BUNDLE}. Run baseline_geneexp.py first.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Loading OncoTerrain bundle: %s", ONCOTERRAIN_JOBLIB)
    bundle, model = _load_tabnet_bundle(ONCOTERRAIN_JOBLIB)
    features = list(bundle["features"])
    scaler = bundle["scaler"]

    log.info("Loading %s", DATA_PATH)
    adata = sc.read_h5ad(DATA_PATH)

    stage = adata.obs["tumor_stage"].astype(str)
    if stage.isin(TUMOR_STAGE_MAP).all():
        y_full = stage.map(TUMOR_STAGE_MAP).astype(int).values
    else:
        y_full = adata.obs["tumor_stage"].astype(int).values

    X_full = _oncoterrain_feature_matrix(adata, features)

    idx = np.arange(len(y_full))
    _, idx_test, _, y_test = train_test_split(
        idx, y_full, test_size=0.3, random_state=RANDOM_STATE, stratify=y_full
    )
    log.info("Held-out test cells: %d", len(idx_test))

    X_test = X_full.iloc[idx_test].reset_index(drop=True)
    y_test = np.asarray(y_test)

    proba_base = _predict(model, scaler, X_test)

    model_classes = getattr(model, "classes_", CLASS_IDS)
    if list(model_classes) != list(CLASS_IDS):
        order = [list(model_classes).index(c) for c in CLASS_IDS]
        proba_base = proba_base[:, order]

    pred_base = np.argmax(proba_base, axis=1)
    base_acc = accuracy_score(y_test, pred_base)
    log.info("Baseline accuracy on test fold: %.4f", base_acc)

    records = []
    for i, feat in enumerate(features, start=1):
        X_ab = X_test.copy()
        X_ab[feat] = 0.0
        proba_ab = _predict(model, scaler, X_ab)
        if list(model_classes) != list(CLASS_IDS):
            proba_ab = proba_ab[:, order]
        pred_ab = np.argmax(proba_ab, axis=1)

        delta = proba_ab - proba_base
        abs_delta = np.abs(delta)
        l1 = abs_delta.sum(axis=1).mean()
        flip_frac = float((pred_ab != pred_base).mean())
        ab_acc = accuracy_score(y_test, pred_ab)
        delta_acc = ab_acc - base_acc

        row = {
            "feature": feat,
            "mean_abs_delta_Normal-like": float(abs_delta[:, 0].mean()),
            "mean_abs_delta_Pre-malignant": float(abs_delta[:, 1].mean()),
            "mean_abs_delta_Tumor-like": float(abs_delta[:, 2].mean()),
            "mean_L1_proba_shift": float(l1),
            "pred_flip_fraction": flip_frac,
            "ablated_accuracy": float(ab_acc),
            "delta_accuracy": float(delta_acc),
        }
        for cid in CLASS_IDS:
            mask = y_test == cid
            if mask.any():
                row[f"mean_L1_shift_{LABEL_NAMES[int(cid)]}"] = float(abs_delta[mask].sum(axis=1).mean())
                row[f"flip_fraction_{LABEL_NAMES[int(cid)]}"] = float((pred_ab[mask] != pred_base[mask]).mean())
            else:
                row[f"mean_L1_shift_{LABEL_NAMES[int(cid)]}"] = float("nan")
                row[f"flip_fraction_{LABEL_NAMES[int(cid)]}"] = float("nan")
        records.append(row)

        if i % 5 == 0 or i == len(features):
            log.info("Ablated %d/%d features", i, len(features))

    df = pd.DataFrame(records).sort_values("mean_L1_proba_shift", ascending=False).reset_index(drop=True)
    csv_path = OUT_DIR / "feature_sensitivity.csv"
    df.to_csv(csv_path, index=False)
    log.info("Wrote %s", csv_path)

    top = df.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.barh(top["feature"], top["mean_L1_proba_shift"], color="#4472C4")
    ax.set_xlabel("Mean L1 probability shift after ablation")
    ax.set_title("Top-20 OncoTerrain features by ablation sensitivity")
    plt.tight_layout()
    bar_png = OUT_DIR / "top_features_bar.png"
    plt.savefig(bar_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", bar_png)

    heat_df = df.set_index("feature")[[
        "mean_abs_delta_Normal-like",
        "mean_abs_delta_Pre-malignant",
        "mean_abs_delta_Tumor-like",
    ]].head(30)
    fig, ax = plt.subplots(figsize=(7, 10))
    im = ax.imshow(heat_df.values, aspect="auto", cmap="viridis")
    ax.set_yticks(range(len(heat_df.index)))
    ax.set_yticklabels(heat_df.index, fontsize=8)
    ax.set_xticks(range(heat_df.shape[1]))
    ax.set_xticklabels(["Normal-like", "Pre-malignant", "Tumor-like"], rotation=30, ha="right")
    ax.set_title("Per-class mean |Δ probability| (top-30 features)")
    fig.colorbar(im, ax=ax, shrink=0.6)
    plt.tight_layout()
    heat_png = OUT_DIR / "sensitivity_heatmap.png"
    plt.savefig(heat_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", heat_png)

    baseline_bundle, baseline_model = _load_tabnet_bundle(BASELINE_BUNDLE)
    baseline_features = list(baseline_bundle["features"])
    baseline_scaler = baseline_bundle["scaler"]
    X_gene = _gene_expression_matrix(adata, baseline_features)
    X_gene_test = X_gene[idx_test]
    X_gene_test_s = baseline_scaler.transform(X_gene_test)
    X_gene_test_s = np.asarray(X_gene_test_s, dtype=np.float32)
    if not np.isfinite(X_gene_test_s).all():
        X_gene_test_s = np.nan_to_num(X_gene_test_s, nan=0.0, posinf=0.0, neginf=0.0)

    X_ot_test_s = scaler.transform(X_test)
    X_ot_test_s = np.asarray(X_ot_test_s, dtype=np.float32)
    if not np.isfinite(X_ot_test_s).all():
        X_ot_test_s = np.nan_to_num(X_ot_test_s, nan=0.0, posinf=0.0, neginf=0.0)

    _, ot_curve, ot_dropout_base_acc = _dropout_sensitivity("OncoTerrain", model, X_ot_test_s, y_test)
    _, baseline_curve, baseline_dropout_base_acc = _dropout_sensitivity(
        "baseline_geneexp_tabnet",
        baseline_model,
        X_gene_test_s,
        y_test,
    )

    baseline_csv = OUT_DIR / "gene_expression_dropout_sensitivity.csv"
    baseline_curve.to_csv(baseline_csv, index=False)
    log.info("Wrote %s", baseline_csv)

    robustness_curve = pd.concat([ot_curve, baseline_curve], ignore_index=True)
    robustness_curve = robustness_curve.sort_values(["model", "dropout_fraction"]).reset_index(drop=True)
    robustness_csv = OUT_DIR / "robustness_curve.csv"
    robustness_curve.to_csv(robustness_csv, index=False)
    log.info("Wrote %s", robustness_csv)
    _plot_robustness_curve(robustness_curve)

    ot_curve_sorted = ot_curve.sort_values("dropout_fraction")
    baseline_curve_sorted = baseline_curve.sort_values("dropout_fraction")
    ot_auc = float(np.trapz(ot_curve_sorted["mean_accuracy"], x=ot_curve_sorted["dropout_fraction"]))
    baseline_auc = float(np.trapz(baseline_curve_sorted["mean_accuracy"], x=baseline_curve_sorted["dropout_fraction"]))
    summary = {
        "dropout_fractions": list(DROPOUT_FRACTIONS),
        "dropout_repeats": DROPOUT_REPEATS,
        "heldout_test_cells": int(len(idx_test)),
        "oncoterrain_base_accuracy": float(ot_dropout_base_acc),
        "oncoterrain_feature_ablation_base_accuracy": float(base_acc),
        "baseline_gene_expression_base_accuracy": float(baseline_dropout_base_acc),
        "oncoterrain_robustness_auc": ot_auc,
        "baseline_gene_expression_robustness_auc": baseline_auc,
        "delta_robustness_auc": float(baseline_auc - ot_auc),
    }
    summary_path = OUT_DIR / "robustness_summary.json"
    with summary_path.open("w") as handle:
        json.dump(summary, handle, indent=2)
    log.info("Wrote %s", summary_path)

    print(df[["feature", "mean_L1_proba_shift", "pred_flip_fraction", "delta_accuracy"]].head(15).to_string(index=False))
    print(robustness_curve[["model", "dropout_fraction", "mean_accuracy", "mean_L1_proba_shift"]].to_string(index=False))


if __name__ == "__main__":
    main()
