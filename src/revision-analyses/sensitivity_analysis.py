from __future__ import annotations

import logging
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("sensitivity_analysis")

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = REPO_ROOT / "data" / "processed_data.h5ad"
ONCOTERRAIN_JOBLIB = REPO_ROOT / "src" / "OncoTerrain" / "OncoTerrain.joblib"
OUT_DIR = REPO_ROOT / "src" / "revision-analyses" / "outputs" / "sensitivity"

TUMOR_STAGE_MAP = {"non-cancer": 0, "early": 1, "advanced": 2}
LABEL_NAMES = {0: "Normal-like", 1: "Pre-malignant", 2: "Tumor-like"}
RANDOM_STATE = 42


def _oncoterrain_feature_matrix(adata, feature_list):
    meta = adata.obs.copy()
    drop_cols = [
        "disease", "sample", "source", "tissue", "n_genes", "batch",
        "n_genes_by_counts", "total_counts", "total_counts_mt", "pct_counts_mt",
        "leiden_res_0.10", "leiden_res_1.00", "leiden_res_5.00", "leiden_res_10.00", "leiden_res_20.00",
        "leiden_res_0.10_celltype", "leiden_res_1.00_celltype", "leiden_res_5.00_celltype", "leiden_res_10.00_celltype",
        "tumor_stage", "project",
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


def _predict(model, scaler, X_df):
    X = scaler.transform(X_df)
    if not np.isfinite(X).all():
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return model.predict_proba(X.astype(np.float32))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Loading OncoTerrain bundle: %s", ONCOTERRAIN_JOBLIB)
    bundle = joblib.load(ONCOTERRAIN_JOBLIB)
    model = bundle["model"]
    if hasattr(model, "best_estimator_"):
        model = model.best_estimator_
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

    class_ids = np.array([0, 1, 2])
    proba_base = _predict(model, scaler, X_test)

    model_classes = getattr(model, "classes_", class_ids)
    if list(model_classes) != list(class_ids):
        order = [list(model_classes).index(c) for c in class_ids]
        proba_base = proba_base[:, order]

    pred_base = np.argmax(proba_base, axis=1)
    base_acc = accuracy_score(y_test, pred_base)
    log.info("Baseline accuracy on test fold: %.4f", base_acc)

    records = []
    for i, feat in enumerate(features, start=1):
        X_ab = X_test.copy()
        X_ab[feat] = 0.0
        proba_ab = _predict(model, scaler, X_ab)
        if list(model_classes) != list(class_ids):
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
        for cid in class_ids:
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

    print(df[["feature", "mean_L1_proba_shift", "pred_flip_fraction", "delta_accuracy"]].head(15).to_string(index=False))


if __name__ == "__main__":
    main()
