from __future__ import annotations

import json
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import accuracy_score

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
log = logging.getLogger("sensitivity_analysis")

BASELINE_BUNDLE = REPO_ROOT / "src" / "revision-analyses" / "outputs" / "baseline_geneexp" / "baseline_geneexp.joblib"
OUT_DIR = REPO_ROOT / "src" / "revision-analyses" / "outputs" / "sensitivity"

CLASS_IDS = np.array([0, 1, 2])
DROPOUT_FRACTIONS = (0.00, 0.01, 0.05, 0.10, 0.20, 0.30)
DROPOUT_REPEATS = 20
MODEL_COLORS = {"OncoTerrain": "#FF8C00", "baseline_geneexp_tabnet": "#5B8FA8"}


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


def _predict_scaled(model, X_scaled, class_ids):
    proba = model.predict_proba(np.asarray(X_scaled, dtype=np.float32))
    if not np.isfinite(proba).all():
        proba = np.nan_to_num(proba, nan=0.0, posinf=0.0, neginf=0.0)
    model_classes = np.asarray(getattr(model, "classes_", class_ids))
    if list(model_classes) != list(class_ids):
        order = [list(model_classes).index(cid) for cid in class_ids]
        proba = proba[:, order]
    return proba


def _dropout_sensitivity(model_name, model, X_scaled, y_true, class_ids, metadata):
    base_proba = _predict_scaled(model, X_scaled, class_ids)
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

            proba = _predict_scaled(model, X_corrupt, class_ids)
            pred = np.argmax(proba, axis=1)
            abs_delta = np.abs(proba - base_proba)
            acc = accuracy_score(y_true, pred)
            records.append(
                {
                    "model": model_name,
                    "evaluation_cohort": metadata["evaluation_cohort"],
                    "split_seed": metadata["random_state"],
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
        repeat_df.groupby(
            ["model", "evaluation_cohort", "split_seed", "dropout_fraction", "n_features_total", "n_features_dropped"],
            as_index=False,
        )
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
    ot_bundle, model = load_tabnet_bundle(ONCOTERRAIN_JOBLIB)
    label_names = label_names_from_bundle(ot_bundle)
    class_ids = np.array(sorted(label_names))

    log.info("Loading %s", DATA_PATH)
    adata = sc.read_h5ad(DATA_PATH)
    cohort = prepare_manuscript_matched_cohort(adata, random_state=RANDOM_STATE, logger=log)

    X_full = oncoterrain_feature_matrix(cohort.obs, ot_bundle["features"])
    y_test = cohort.y[cohort.idx_test].astype(int)
    X_test_df = X_full.iloc[cohort.idx_test].reset_index(drop=True)
    X_test_scaled = scale_feature_frame(X_test_df, ot_bundle["scaler"], ot_bundle)

    proba_base = _predict_scaled(model, X_test_scaled, class_ids)
    pred_base = np.argmax(proba_base, axis=1)
    base_acc = accuracy_score(y_test, pred_base)
    log.info("Figure-5-matched OncoTerrain accuracy on shared split: %.4f", base_acc)

    records = []
    for i, feat in enumerate(X_test_df.columns, start=1):
        X_ab = X_test_df.copy()
        X_ab.loc[:, feat] = 0.0
        proba_ab = _predict_scaled(model, scale_feature_frame(X_ab, ot_bundle["scaler"], ot_bundle), class_ids)
        pred_ab = np.argmax(proba_ab, axis=1)

        abs_delta = np.abs(proba_ab - proba_base)
        row = {
            "feature": feat,
            "evaluation_cohort": cohort.metadata["evaluation_cohort"],
            "split_seed": cohort.metadata["random_state"],
            "mean_L1_proba_shift": float(abs_delta.sum(axis=1).mean()),
            "pred_flip_fraction": float((pred_ab != pred_base).mean()),
            "ablated_accuracy": float(accuracy_score(y_test, pred_ab)),
            "delta_accuracy": float(accuracy_score(y_test, pred_ab) - base_acc),
        }
        for col_idx, cid in enumerate(class_ids):
            class_name = label_names[int(cid)]
            row[f"mean_abs_delta_{class_name}"] = float(abs_delta[:, col_idx].mean())
            mask = y_test == cid
            if mask.any():
                row[f"mean_L1_shift_{class_name}"] = float(abs_delta[mask].sum(axis=1).mean())
                row[f"flip_fraction_{class_name}"] = float((pred_ab[mask] != pred_base[mask]).mean())
            else:
                row[f"mean_L1_shift_{class_name}"] = float("nan")
                row[f"flip_fraction_{class_name}"] = float("nan")
        records.append(row)

        if i % 5 == 0 or i == X_test_df.shape[1]:
            log.info("Ablated %d/%d features", i, X_test_df.shape[1])

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

    heat_cols = [f"mean_abs_delta_{label_names[int(cid)]}" for cid in class_ids]
    heat_labels = [label_names[int(cid)] for cid in class_ids]
    heat_df = df.set_index("feature")[heat_cols].head(30)
    fig, ax = plt.subplots(figsize=(7, 10))
    im = ax.imshow(heat_df.values, aspect="auto", cmap="viridis")
    ax.set_yticks(range(len(heat_df.index)))
    ax.set_yticklabels(heat_df.index, fontsize=8)
    ax.set_xticks(range(len(heat_labels)))
    ax.set_xticklabels(heat_labels, rotation=30, ha="right")
    ax.set_title("Per-class mean |Δ probability| (top-30 features)")
    fig.colorbar(im, ax=ax, shrink=0.6)
    plt.tight_layout()
    heat_png = OUT_DIR / "sensitivity_heatmap.png"
    plt.savefig(heat_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", heat_png)

    baseline_bundle, baseline_model = load_tabnet_bundle(BASELINE_BUNDLE)
    baseline_features = list(baseline_bundle["features"])
    baseline_scaler = baseline_bundle["scaler"]
    X_gene = _gene_expression_matrix(cohort.adata, baseline_features)
    X_gene_test = X_gene[cohort.idx_test]
    X_gene_test_s = baseline_scaler.transform(X_gene_test).astype(np.float32, copy=False)
    if not np.isfinite(X_gene_test_s).all():
        X_gene_test_s = np.nan_to_num(X_gene_test_s, nan=0.0, posinf=0.0, neginf=0.0)

    _, ot_curve, ot_dropout_base_acc = _dropout_sensitivity("OncoTerrain", model, X_test_scaled, y_test, class_ids, cohort.metadata)
    _, baseline_curve, baseline_dropout_base_acc = _dropout_sensitivity(
        "baseline_geneexp_tabnet",
        baseline_model,
        X_gene_test_s,
        y_test,
        class_ids,
        cohort.metadata,
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
        **cohort.metadata,
        "label_names": {str(k): v for k, v in label_names.items()},
        "dropout_fractions": list(DROPOUT_FRACTIONS),
        "dropout_repeats": DROPOUT_REPEATS,
        "heldout_test_cells": int(len(cohort.idx_test)),
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
