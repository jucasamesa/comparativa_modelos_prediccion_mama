# -*- coding: utf-8 -*-
"""Utilidades QQ para decidir transformaciones log1p (Fase 3.5).

Provee:
  - plot_qq_raw_vs_log1p: QQ raw vs log1p, transformada bajo su versión cruda.
  - find_log1p_candidates: recomienda log1p según curvatura/cola del QQ y mejora real.
"""
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


def plot_qq_raw_vs_log1p(
    df,
    variables,
    max_cols=6,
    figsize_per_col=3.2,
    figsize_per_row=3.0,
    marker_size=2,
    clip_lower=0,
    title_suffix_raw="raw",
    title_suffix_log="log1p",
):
    """QQ plots de versiones raw y log1p, con la transformada bajo su raw."""
    variables = list(variables)
    if len(variables) == 0:
        raise ValueError("No variables provided.")

    ncols = min(max_cols, len(variables))
    n_blocks = math.ceil(len(variables) / ncols)

    fig, axes = plt.subplots(
        2 * n_blocks,
        ncols,
        figsize=(figsize_per_col * ncols, figsize_per_row * 2 * n_blocks),
        squeeze=False,
    )
    for ax in axes.flat:
        ax.set_visible(False)

    for i, c in enumerate(variables):
        block = i // ncols
        col = i % ncols
        ax_raw = axes[2 * block, col]
        ax_log = axes[2 * block + 1, col]
        ax_raw.set_visible(True)
        ax_log.set_visible(True)

        x = df[c].dropna()
        stats.probplot(x, dist="norm", plot=ax_raw)
        ax_raw.set_title(f"{c} {title_suffix_raw}")

        x_log = np.log1p(x.clip(lower=clip_lower))
        stats.probplot(x_log, dist="norm", plot=ax_log)
        ax_log.set_title(f"{c} {title_suffix_log}")

        for ax in (ax_raw, ax_log):
            lines = ax.get_lines()
            if lines:
                lines[0].set_markersize(marker_size)
            ax.set_xlabel("")
            ax.set_ylabel("")

    plt.tight_layout()
    return fig, axes


def _qq_metrics(x, tail_frac=0.15):
    """Métricas de forma a partir de un QQ plot normal."""
    x = pd.Series(x).dropna().astype(float)

    if len(x) < 8:
        return {
            "n": len(x), "skew": np.nan, "qq_r": np.nan, "qq_rmse": np.nan,
            "tail_lift": np.nan, "tail_lift_z": np.nan,
            "right_tail_mean_resid": np.nan, "center_mean_resid": np.nan,
            "right_tail_monotone": np.nan, "valid": False, "reason": "too_few_points",
        }

    if x.nunique() < 4:
        return {
            "n": len(x), "skew": stats.skew(x, bias=False), "qq_r": np.nan,
            "qq_rmse": np.nan, "tail_lift": np.nan, "tail_lift_z": np.nan,
            "right_tail_mean_resid": np.nan, "center_mean_resid": np.nan,
            "right_tail_monotone": np.nan, "valid": False,
            "reason": "too_few_unique_values",
        }

    (osm, osr), (slope, intercept, r) = stats.probplot(x, dist="norm", fit=True)
    fitted = slope * osm + intercept
    resid = osr - fitted

    n = len(resid)
    k = max(2, int(np.ceil(n * tail_frac)))
    right_tail = resid[-k:]
    center = resid[k:n - k] if n > 2 * k else resid[k // 2:n - k // 2]

    mad = np.median(np.abs(resid - np.median(resid)))
    robust_scale = 1.4826 * mad if mad > 0 else (np.std(resid, ddof=1) + 1e-12)

    right_tail_mean = np.mean(right_tail)
    center_mean = np.mean(center)
    tail_lift = right_tail_mean - center_mean
    tail_lift_z = tail_lift / (robust_scale + 1e-12)

    diffs = np.diff(right_tail)
    right_tail_monotone = np.mean(diffs > 0) if len(diffs) > 0 else np.nan
    qq_rmse = np.sqrt(np.mean(resid ** 2))

    return {
        "n": n, "skew": stats.skew(x, bias=False), "qq_r": r, "qq_rmse": qq_rmse,
        "tail_lift": tail_lift, "tail_lift_z": tail_lift_z,
        "right_tail_mean_resid": right_tail_mean, "center_mean_resid": center_mean,
        "right_tail_monotone": right_tail_monotone, "valid": True, "reason": "",
    }


def find_log1p_candidates(
    df,
    cols,
    tail_frac=0.15,
    skew_thr=1.0,
    bend_z_thr=2.5,
    monotone_thr=0.60,
    rmse_improve_thr=0.12,
    tail_improve_thr=0.25,
    clip_lower=0,
):
    """Identifica variables cuyo QQ raw se curva en cola derecha y se endereza con log1p."""
    rows = []
    for c in cols:
        x = pd.Series(df[c]).dropna().astype(float)
        raw = _qq_metrics(x, tail_frac=tail_frac)
        x_log = np.log1p(x.clip(lower=clip_lower))
        log = _qq_metrics(x_log, tail_frac=tail_frac)

        if raw["valid"] and log["valid"]:
            rmse_improve = (raw["qq_rmse"] - log["qq_rmse"]) / (raw["qq_rmse"] + 1e-12)
            tail_improve = (abs(raw["tail_lift_z"]) - abs(log["tail_lift_z"])) / (
                abs(raw["tail_lift_z"]) + 1e-12
            )
            raw_right_bend = (
                (raw["skew"] > skew_thr)
                and (raw["tail_lift_z"] > bend_z_thr)
                and (raw["right_tail_monotone"] >= monotone_thr)
            )
            log_straighter = (
                (rmse_improve >= rmse_improve_thr)
                and (tail_improve >= tail_improve_thr)
                and (abs(log["tail_lift_z"]) < abs(raw["tail_lift_z"]))
            )
            recommend_log1p = raw_right_bend and log_straighter
        else:
            rmse_improve = np.nan
            tail_improve = np.nan
            raw_right_bend = False
            log_straighter = False
            recommend_log1p = False

        rows.append({
            "variable": c,
            "raw_n": raw["n"], "raw_skew": raw["skew"], "raw_qq_r": raw["qq_r"],
            "raw_qq_rmse": raw["qq_rmse"], "raw_tail_lift_z": raw["tail_lift_z"],
            "raw_right_tail_monotone": raw["right_tail_monotone"],
            "log_qq_r": log["qq_r"], "log_qq_rmse": log["qq_rmse"],
            "log_tail_lift_z": log["tail_lift_z"], "log_skew": log["skew"],
            "rmse_improve_pct": 100 * rmse_improve if pd.notna(rmse_improve) else np.nan,
            "tail_improve_pct": 100 * tail_improve if pd.notna(tail_improve) else np.nan,
            "raw_right_bend": raw_right_bend, "log_straighter": log_straighter,
            "recommend_log1p": recommend_log1p,
            "raw_valid": raw["valid"], "log_valid": log["valid"],
            "raw_reason": raw["reason"], "log_reason": log["reason"],
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(
            ["recommend_log1p", "rmse_improve_pct", "tail_improve_pct"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
    return out
