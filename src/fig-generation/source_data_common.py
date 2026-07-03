"""Shared helpers for exporting Communications Biology source data + exact p-values.

Every quantitative figure panel (box / bar / violin / polar / correlation / similarity)
routes its plotted values through :func:`panel_csv` and its statistical tests through
:func:`write_pvalues`. One tidy CSV is written per panel plus one unified
``figure_<N>_pvalues.csv`` per figure, and :func:`aggregate_to_excel` bundles them into a
single ``Figure_<N>_source_data.xlsx`` workbook (one sheet per panel + a ``pvalues`` sheet)
in the format the journal prefers to receive.

Modeled on ``src/revision-analyses/revision_common.py`` (REPO_ROOT anchor, tidy CSVs,
``log.info("Wrote %s", path)`` after every write).
"""
from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DATA_DIR = REPO_ROOT / "src" / "fig-generation" / "source-data"

# Canonical column order for every figure_<N>_pvalues.csv so all figures share one schema.
PVALUE_COLUMNS = [
    "figure", "panel", "feature",
    "group1", "group2", "n1", "n2",
    "median1", "median2", "higher_median_group",
    "U_statistic", "p_value_raw", "p_value_adj", "correction_method",
    "effect_size_rbc",
]

# In-memory accumulator so multiple panels can each call write_pvalues(fig, rows) and the
# full figure_<N>_pvalues.csv is rewritten from everything collected so far this process.
_PVALUE_ROWS: dict[int, list[dict]] = defaultdict(list)


def figure_dir(fig: int) -> Path:
    """source-data/figure_<N>/, created on demand."""
    d = SOURCE_DATA_DIR / f"figure_{fig}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def panel_csv(fig: int, panel: str, df: pd.DataFrame, *, index: bool = False) -> Path:
    """Write one panel's plotted values to figure_<N>_<panel>_source_data.csv."""
    path = figure_dir(fig) / f"figure_{fig}_{panel}_source_data.csv"
    df.to_csv(path, index=index)
    log.info("Wrote %s", path)
    return path


def long_from_group_dict(data_by_group: dict, *, feature: str,
                         group_col: str = "group", value_col: str = "value") -> pd.DataFrame:
    """Tidy {group -> 1-D array} into long form [feature, group, value] for source data."""
    frames = []
    for group, values in data_by_group.items():
        values = np.asarray(values, dtype=float).ravel()
        frames.append(pd.DataFrame({
            "feature": feature,
            group_col: group,
            value_col: values,
        }))
    if not frames:
        return pd.DataFrame(columns=["feature", group_col, value_col])
    return pd.concat(frames, ignore_index=True)


def mwu_with_direction(x, y, group1: str, group2: str, *, alternative: str = "two-sided") -> dict:
    """Two-sided Mann-Whitney U with medians + direction (which group is higher).

    Returns raw p only; multiple-comparison correction is applied by the caller across the
    panel's full set of tests, matching the existing figure scripts.
    """
    x = np.asarray(x, dtype=float); x = x[np.isfinite(x)]
    y = np.asarray(y, dtype=float); y = y[np.isfinite(y)]
    n1, n2 = int(x.size), int(y.size)
    if n1 == 0 or n2 == 0:
        return {
            "group1": group1, "group2": group2, "n1": n1, "n2": n2,
            "median1": np.nan, "median2": np.nan, "higher_median_group": None,
            "U_statistic": np.nan, "p_value_raw": np.nan, "effect_size_rbc": np.nan,
        }
    U, p = mannwhitneyu(x, y, alternative=alternative)
    rbc = 1.0 - (2.0 * U) / (n1 * n2)          # rank-biserial effect size
    m1, m2 = float(np.median(x)), float(np.median(y))
    higher = group1 if m1 > m2 else (group2 if m2 > m1 else "tie")
    return {
        "group1": group1, "group2": group2, "n1": n1, "n2": n2,
        "median1": m1, "median2": m2, "higher_median_group": higher,
        "U_statistic": float(U), "p_value_raw": float(p), "effect_size_rbc": float(rbc),
    }


def write_pvalues(fig: int, rows: list[dict]) -> Path:
    """Accumulate p-value rows for a figure and (re)write figure_<N>_pvalues.csv.

    Each row may set any subset of PVALUE_COLUMNS; ``figure`` is filled automatically.
    Safe to call once per panel — the full file is rebuilt from all rows collected so far.
    """
    for r in rows:
        r.setdefault("figure", fig)
        _PVALUE_ROWS[fig].append(r)

    df = pd.DataFrame(_PVALUE_ROWS[fig])
    for col in PVALUE_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    df = df[PVALUE_COLUMNS]

    path = figure_dir(fig) / f"figure_{fig}_pvalues.csv"
    df.to_csv(path, index=False)
    log.info("Wrote %s (%d comparisons)", path, len(df))
    return path


def _sheet_name(stem: str, used: set[str]) -> str:
    """Excel sheet names must be <=31 chars and unique."""
    name = stem[:31]
    i = 1
    while name in used:
        suffix = f"_{i}"
        name = stem[:31 - len(suffix)] + suffix
        i += 1
    used.add(name)
    return name


def aggregate_to_excel(fig: int) -> Path | None:
    """Bundle every figure_<N>_*_source_data.csv + the pvalues CSV into one workbook."""
    d = figure_dir(fig)
    panel_csvs = sorted(d.glob(f"figure_{fig}_*_source_data.csv"))
    pval_csv = d / f"figure_{fig}_pvalues.csv"
    if not panel_csvs and not pval_csv.exists():
        log.warning("No source-data CSVs found for figure %d; skipping Excel aggregation.", fig)
        return None

    # keep_default_na=False + na_values=[""] preserves legitimate string labels (e.g. a category
    # literally named "None"/"NA") while still treating empty numeric cells as missing.
    read = lambda p: pd.read_csv(p, keep_default_na=False, na_values=[""])

    xlsx = d / f"Figure_{fig}_source_data.xlsx"
    used: set[str] = set()
    with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
        for csv in panel_csvs:
            stem = csv.name.replace(f"figure_{fig}_", "").replace("_source_data.csv", "")
            read(csv).to_excel(writer, sheet_name=_sheet_name(stem, used), index=False)
        if pval_csv.exists():
            read(pval_csv).to_excel(writer, sheet_name=_sheet_name("pvalues", used), index=False)
    log.info("Wrote %s", xlsx)
    return xlsx
