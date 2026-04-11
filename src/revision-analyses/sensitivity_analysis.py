from __future__ import annotations

import json
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from analysis_common import (
    ONCOTERRAIN_JOBLIB,
    REPO_ROOT,
    align_feature_frame,
    extract_gene_expression_matrix,
    load_model_bundle,
    load_processed_adata,
    predict_proba,
    transform_features,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("sensitivity_analysis")

OUT_DIR = REPO_ROOT / "src" / "revision-analyses" / "outputs" / "sensitivity"
BASELINE_BUNDLE = REPO_ROOT / "src" / "revision-analyses" / "outputs" / "baseline_geneexp" / "baseline_geneexp.joblib"

TUMOR_STAGE_MAP = {"non-cancer": 0, "early": 1, "advanced": 2}
LABEL_NAMES = {0: "Normal-like", 1: "Pre-malignant", 2: "Tumor-like"}
RANDOM_STATE = 42
CLASS_IDS = np.array([0, 1, 2])
DROPOUT_FRACTIONS = (0.00, 0.01, 0.05, 0.10, 0.20, 0.30)
DROPOUT_REPEATS = 20
MODEL_COLORS = {"OncoTerrain": "#FF8C00", "baseline_geneexp_tabnet": "#5B8FA8"}


def _encode_tumor_stage(adata):
    stage = adata.obs["tumor_stage"].astype(str)
    if stage.isin(TUMOR_STAGE_MAP).all():
        return stage.map(TUMOR_STAGE_MAP).astype(int).to_numpy()
    return adata.obs["tumor_stage"].astype(int).to_numpy()


def _feature_ablation(model, X_base, y_true, feature_names):
    proba_base = predict_proba(model, X_base, CLASS_IDS)
    pred_base = CLASS_IDS[np.argmax(proba_base, axis=1)]
    base_acc = accuracy_score(y_true, pred_base)
    log.info("OncoTerrain baseline accuracy on test fold: %.4f", base_acc)

    records = []
    for i, feature in enumerate(feature_names, start=1):
        X_ab = X_base.copy()
        X_ab[:, i - 1] = 0.0
        proba_ab = predict_proba(model, X_ab, CLASS_IDS)
        pred_ab = CLASS_IDS[np.argmax(proba_ab, axis=1)]

        delta = proba_ab - proba_base
        abs_delta = np.abs(delta)
        row = {
            "feature": feature,
            "mean_abs_delta_Normal-like": float(abs_delta[:, 0].mean()),
            "mean_abs_delta_Pre-malignant": float(abs_delta[:, 1].mean()),
            "mean_abs_delta_Tumor-like": float(abs_delta[:, 2].mean()),
            "mean_L1_proba_shift": float(abs_delta.sum(axis=1).mean()),
            "pred_flip_fraction": float((pred_ab != pred_base).mean()),
            "ablated_accuracy": float(accuracy_score(y_true, pred_ab)),
        }
        row["delta_accuracy"] = row["ablated_accuracy"] - float(base_acc)
        for cid in CLASS_IDS:
            mask = y_true == cid
            label = LABEL_NAMES[int(cid)]
            if mask.any():
                row[f"mean_L1_shift_{label}"] = float(abs_delta[mask].sum(axis=1).mean())
                row[f"flip_fraction_{label}"] = float((pred_ab[mask] != pred_base[mask]).mean())
            else:
                row[f"mean_L1_shift_{label}"] = float("nan")
                row[f"flip_fraction_{label}"] = float("nan")
        records.append(row)

        if i % 5 == 0 or i == len(feature_names):
            log.info("Ablated %d/%d OncoTerrain features", i, len(feature_names))

    df = pd.DataFrame(records).sort_values("mean_L1_proba_shift", ascending=False).reset_index(drop=True)
    return df, float(base_acc)


def _run_dropout_sensitivity(model_name, model, X_base, y_true):
    base_proba = predict_proba(model, X_base, CLASS_IDS)
    base_pred = CLASS_IDS[np.argmax(base_proba, axis=1)]
    base_acc = accuracy_score(y_true, base_pred)
    n_features = X_base.shape[1]
    rng = np.random.default_rng(RANDOM_STATE)

    records = []
    for frac in DROPOUT_FRACTIONS:
        n_drop = int(round(frac * n_features))
        if frac > 0.0 and n_drop == 0 and n_features > 0:
            n_drop = 1
        log.info("%s dropout sensitivity at %.2f (%d features)", model_name, frac, n_drop)

        for repeat in range(DROPOUT_REPEATS):
            X_corrupt = X_base.copy()
            if n_drop > 0:
                drop_idx = rng.choice(n_features, size=n_drop, replace=False)
                X_corrupt[:, drop_idx] = 0.0

            proba = predict_proba(model, X_corrupt, CLASS_IDS)
            pred = CLASS_IDS[np.argmax(proba, axis=1)]
            abs_delta = np.abs(proba - base_proba)
            acc = accuracy_score(y_true, pred)
            records.append(
                {
                    "model": model_name,
                    "dropout_fraction": float(frac),
                    "repeat": repeat,
                    "n_features_total": int(n_features),
                    "n_features_dropped": int(n_drop),
                    "accuracy": float(acc),
                    "delta_accuracy": float(acc - base_acc),
                    "mean_L1_proba_shift": float(abs_delta.sum(axis=1).mean()),
                    "pred_flip_fraction": float((pred != base_pred).mean()),
                }
            )

    per_repeat = pd.DataFrame(records)
    summary = (
        per_repeat.groupby(["model", "dropout_fraction", "n_features_total", "n_features_dropped"], as_index=False)
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
    summary["base_accuracy"] = float(base_acc)
    return per_repeat, summary, float(base_acc)


def _plot_top_feature_bar(df):
    top = df.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.barh(top["feature"], top["mean_L1_proba_shift"], color="#4472C4")
    ax.set_xlabel("Mean L1 probability shift after ablation")
    ax.set_title("Top-20 OncoTerrain features by ablation sensitivity")
    plt.tight_layout()
    path = OUT_DIR / "top_features_bar.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", path)


def _plot_heatmap(df):
    heat_df = df.set_index("feature")[
        [
            "mean_abs_delta_Normal-like",
            "mean_abs_delta_Pre-malignant",
            "mean_abs_delta_Tumor-like",
        ]
    ].head(30)
    fig, ax = plt.subplots(figsize=(7, 10))
    image = ax.imshow(heat_df.values, aspect="auto", cmap="viridis")
    ax.set_yticks(range(len(heat_df.index)))
    ax.set_yticklabels(heat_df.index, fontsize=8)
    ax.set_xticks(range(heat_df.shape[1]))
    ax.set_xticklabels(["Normal-like", "Pre-malignant", "Tumor-like"], rotation=30, ha="right")
    ax.set_title("Per-class mean |Δ probability| (top-30 features)")
    fig.colorbar(image, ax=ax, shrink=0.6)
    plt.tight_layout()
    path = OUT_DIR / "sensitivity_heatmap.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", path)


def _plot_robustness_curve(df):
    fig, ax = plt.subplots(figsize=(8, 6))
    for model_name, model_df in df.groupby("model"):
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
    path = OUT_DIR / "robustness_curve.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", path)


def main():
    if not BASELINE_BUNDLE.exists():
        raise FileNotFoundError(f"Missing baseline bundle at {BASELINE_BUNDLE}. Run baseline_geneexp.py first.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log.info("Loading processed AnnData and refreshing hallmark scores")
    adata = load_processed_adata(refresh_hallmarks=True)
    y_full = _encode_tumor_stage(adata)

    idx = np.arange(len(y_full))
    _, idx_test, _, y_test = train_test_split(
        idx,
        y_full,
        test_size=0.3,
        random_state=RANDOM_STATE,
        stratify=y_full,
    )
    y_test = np.asarray(y_test)
    log.info("Held-out test cells: %d", len(idx_test))

    ot_bundle, ot_model = load_model_bundle(ONCOTERRAIN_JOBLIB, device_name="cpu")
    ot_features = list(ot_bundle["features"])
    ot_scaler = ot_bundle["scaler"]
    X_ot_df = align_feature_frame(adata, ot_features, logger=log)
    X_ot_test_s = transform_features(ot_scaler, X_ot_df.iloc[idx_test])

    base_bundle, base_model = load_model_bundle(BASELINE_BUNDLE, device_name="cpu")
    base_features = list(base_bundle["features"])
    base_scaler = base_bundle["scaler"]
    X_gene = extract_gene_expression_matrix(adata, base_features, logger=log)
    X_gene_test_s = transform_features(base_scaler, X_gene[idx_test])
    del adata, X_ot_df, X_gene

    feature_df, ot_base_acc = _feature_ablation(ot_model, X_ot_test_s, y_test, ot_features)
    feature_csv = OUT_DIR / "feature_sensitivity.csv"
    feature_df.to_csv(feature_csv, index=False)
    log.info("Wrote %s", feature_csv)
    _plot_top_feature_bar(feature_df)
    _plot_heatmap(feature_df)

    _, ot_curve, ot_dropout_base_acc = _run_dropout_sensitivity("OncoTerrain", ot_model, X_ot_test_s, y_test)
    _, base_curve, baseline_base_acc = _run_dropout_sensitivity(
        "baseline_geneexp_tabnet",
        base_model,
        X_gene_test_s,
        y_test,
    )

    baseline_csv = OUT_DIR / "gene_expression_dropout_sensitivity.csv"
    base_curve.to_csv(baseline_csv, index=False)
    log.info("Wrote %s", baseline_csv)

    robustness_curve = pd.concat([ot_curve, base_curve], ignore_index=True)
    robustness_curve = robustness_curve.sort_values(["model", "dropout_fraction"]).reset_index(drop=True)
    robustness_csv = OUT_DIR / "robustness_curve.csv"
    robustness_curve.to_csv(robustness_csv, index=False)
    log.info("Wrote %s", robustness_csv)
    _plot_robustness_curve(robustness_curve)

    ot_curve_sorted = ot_curve.sort_values("dropout_fraction")
    base_curve_sorted = base_curve.sort_values("dropout_fraction")
    ot_auc = float(np.trapz(ot_curve_sorted["mean_accuracy"], x=ot_curve_sorted["dropout_fraction"]))
    baseline_auc = float(np.trapz(base_curve_sorted["mean_accuracy"], x=base_curve_sorted["dropout_fraction"]))
    summary = {
        "dropout_fractions": list(DROPOUT_FRACTIONS),
        "dropout_repeats": DROPOUT_REPEATS,
        "heldout_test_cells": int(len(idx_test)),
        "oncoterrain_base_accuracy": float(ot_dropout_base_acc),
        "oncoterrain_feature_ablation_base_accuracy": float(ot_base_acc),
        "baseline_gene_expression_base_accuracy": float(baseline_base_acc),
        "oncoterrain_robustness_auc": ot_auc,
        "baseline_gene_expression_robustness_auc": baseline_auc,
        "delta_robustness_auc": float(baseline_auc - ot_auc),
    }
    summary_path = OUT_DIR / "robustness_summary.json"
    with summary_path.open("w") as handle:
        json.dump(summary, handle, indent=2)
    log.info("Wrote %s", summary_path)

    print(feature_df[["feature", "mean_L1_proba_shift", "pred_flip_fraction", "delta_accuracy"]].head(15).to_string(index=False))
    print(robustness_curve[["model", "dropout_fraction", "mean_accuracy", "mean_L1_proba_shift"]].to_string(index=False))


if __name__ == "__main__":
    main()
