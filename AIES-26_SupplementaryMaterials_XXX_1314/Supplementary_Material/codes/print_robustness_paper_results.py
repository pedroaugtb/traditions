#!/usr/bin/env python3
"""
Print paper-ready robustness results from robustness_framing_diagnostics.py.

This script reads the CSV outputs produced by robustness_framing_diagnostics.py
and prints the tables/statements most useful for a Results subsection. It also
writes compact CSV/LaTeX/Markdown artifacts into <outdir>/paper_results/.

Typical use:

  python print_robustness_paper_results.py \
    --outdir robustness_diagnostics

Optional:

  python print_robustness_paper_results.py \
    --outdir robustness_diagnostics \
    --axis-order assertive_commitment doctrinal_attribution hedging mythologized

Expected input files, if available:
  response_level_summary.csv
  aggregate_stability_summary.csv
  lexical_baseline_summary.csv
  residualization_summary.csv
  residualized_vs_original_summary.csv
  residualized_aggregate_cells.csv
  placebo_summary.csv

The script is defensive: if a file is missing, it prints what is missing and
continues with the other tables.
"""

from __future__ import annotations

import argparse
import math
import textwrap
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


AXIS_LABELS = {
    "assertive_commitment": "Assertive commitment",
    "doctrinal_attribution": "Doctrinal attribution",
    "hedging": "Hedging",
    "mythologized": "Mythologization",
    "mythologization": "Mythologization",
}

BASELINE_LABELS = {
    "cue_length_only": "Cue/length only",
    "tfidf_only": "TF-IDF only",
    "cue_length_prompt_model": "Cue/length + prompt/model",
}

LEVEL_LABELS = {
    "religion": "Religion",
    "prompt_group": "Prompt group",
    "religion_prompt_group": "Religion × prompt group",
    "model_religion": "Model × religion",
    "model_religion_prompt_group": "Model × religion × prompt group",
}

DEFAULT_AXIS_ORDER = [
    "assertive_commitment",
    "doctrinal_attribution",
    "hedging",
    "mythologized",
]


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------

def read_csv_optional(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        print(f"[missing] {path}")
        return None
    try:
        df = pd.read_csv(path)
        print(f"[loaded] {path}  shape={df.shape}")
        return df
    except Exception as e:
        print(f"[error] could not read {path}: {e}")
        return None


def finite_values(series: pd.Series) -> np.ndarray:
    vals = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    return vals[np.isfinite(vals)]


def mean_or_nan(series: pd.Series) -> float:
    vals = finite_values(series)
    return float(np.mean(vals)) if len(vals) else float("nan")


def min_or_nan(series: pd.Series) -> float:
    vals = finite_values(series)
    return float(np.min(vals)) if len(vals) else float("nan")


def max_or_nan(series: pd.Series) -> float:
    vals = finite_values(series)
    return float(np.max(vals)) if len(vals) else float("nan")


def range_text(series: pd.Series, ndigits: int = 3) -> str:
    vals = finite_values(series)
    if len(vals) == 0:
        return "NA"
    return f"{np.min(vals):.{ndigits}f}–{np.max(vals):.{ndigits}f}"


def fmt(x: Any, ndigits: int = 3) -> str:
    try:
        y = float(x)
        if not math.isfinite(y):
            return "NA"
        return f"{y:.{ndigits}f}"
    except Exception:
        if x is None:
            return "NA"
        return str(x)


def pct(x: Any, ndigits: int = 1) -> str:
    try:
        y = float(x)
        if not math.isfinite(y):
            return "NA"
        return f"{100*y:.{ndigits}f}%"
    except Exception:
        return "NA"


def axis_label(axis: Any) -> str:
    return AXIS_LABELS.get(str(axis), str(axis))


def baseline_label(baseline: Any) -> str:
    return BASELINE_LABELS.get(str(baseline), str(baseline))


def level_label(level: Any) -> str:
    return LEVEL_LABELS.get(str(level), str(level))


def sort_axes(df: pd.DataFrame, axis_order: Sequence[str]) -> pd.DataFrame:
    if "axis" not in df.columns:
        return df
    order = {a: i for i, a in enumerate(axis_order)}
    tmp = df.copy()
    tmp["_axis_order"] = tmp["axis"].map(order).fillna(999).astype(int)
    tmp = tmp.sort_values(["_axis_order", "axis"]).drop(columns=["_axis_order"])
    return tmp


def ensure_cols(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            out[c] = np.nan
    return out


def md_table(df: pd.DataFrame, float_digits: int = 3) -> str:
    """Small markdown table writer without requiring tabulate."""
    if df is None or df.empty:
        return "_No rows._"
    display = df.copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]) or pd.api.types.is_integer_dtype(display[col]):
            display[col] = display[col].map(lambda x: fmt(x, float_digits))
        else:
            display[col] = display[col].astype(str)
    headers = list(display.columns)
    rows = display.values.tolist()
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    def line(vals: Sequence[Any]) -> str:
        return "| " + " | ".join(str(v).ljust(widths[i]) for i, v in enumerate(vals)) + " |"
    out = [line(headers), "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |"]
    out.extend(line(row) for row in rows)
    return "\n".join(out)


def write_table_artifacts(df: pd.DataFrame, outdir: Path, name: str, caption: str = "") -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(outdir / f"{name}.csv", index=False)
    (outdir / f"{name}.md").write_text(md_table(df), encoding="utf-8")
    try:
        latex = df.to_latex(index=False, escape=True, float_format=lambda x: f"{x:.3f}")
        if caption:
            latex = latex.replace("\\begin{tabular}", f"% {caption}\n\\begin{{tabular}}")
        (outdir / f"{name}.tex").write_text(latex, encoding="utf-8")
    except Exception as e:
        (outdir / f"{name}.tex.error.txt").write_text(str(e), encoding="utf-8")


def print_block(title: str, body: str) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)
    print(body)


# -----------------------------------------------------------------------------
# Table builders
# -----------------------------------------------------------------------------

def table_full_audit_response(response: Optional[pd.DataFrame], axis_order: Sequence[str]) -> pd.DataFrame:
    if response is None or response.empty:
        return pd.DataFrame()
    df = ensure_cols(response, [
        "axis", "subset", "n_used", "cue_coverage_all", "mean_delta", "flip_rate",
        "spearman_response", "mean_abs_delta", "abs_cohen_dz",
    ])
    df = df[df["subset"].astype(str).eq("full_audit")].copy()
    if df.empty:
        return df
    g = df.groupby("axis", as_index=False).agg(
        n_used=("n_used", "sum"),
        cue_coverage=("cue_coverage_all", "mean"),
        mean_delta=("mean_delta", "mean"),
        flip_rate=("flip_rate", "mean"),
        spearman_response=("spearman_response", "mean"),
        mean_abs_delta=("mean_abs_delta", "mean"),
        abs_dz=("abs_cohen_dz", "mean"),
    )
    g = sort_axes(g, axis_order)
    g.insert(0, "Axis", g["axis"].map(axis_label))
    g = g.drop(columns=["axis"])
    return g.rename(columns={
        "n_used": "N",
        "cue_coverage": "Cue coverage",
        "mean_delta": "Mean Δ",
        "flip_rate": "Flip rate",
        "spearman_response": "ρ response",
        "mean_abs_delta": "Mean |Δ|",
        "abs_dz": "|dz|",
    })


def table_cue_present_response(response: Optional[pd.DataFrame], axis_order: Sequence[str]) -> pd.DataFrame:
    if response is None or response.empty:
        return pd.DataFrame()
    df = ensure_cols(response, [
        "axis", "subset", "n_used", "cue_coverage_all", "mean_delta", "flip_rate",
        "spearman_response", "mean_abs_delta", "abs_cohen_dz",
    ])
    df = df[df["subset"].astype(str).eq("cue_present")].copy()
    if df.empty:
        return df
    g = df.groupby("axis", as_index=False).agg(
        n_used=("n_used", "sum"),
        cue_coverage=("cue_coverage_all", "mean"),
        mean_delta=("mean_delta", "mean"),
        flip_rate=("flip_rate", "mean"),
        spearman_response=("spearman_response", "mean"),
        mean_abs_delta=("mean_abs_delta", "mean"),
        abs_dz=("abs_cohen_dz", "mean"),
    )
    g = sort_axes(g, axis_order)
    g.insert(0, "Axis", g["axis"].map(axis_label))
    g = g.drop(columns=["axis"])
    return g.rename(columns={
        "n_used": "N cue-present",
        "cue_coverage": "Cue coverage",
        "mean_delta": "Mean Δ",
        "flip_rate": "Flip rate",
        "spearman_response": "ρ response",
        "mean_abs_delta": "Mean |Δ|",
        "abs_dz": "|dz|",
    })


def table_aggregate_stability(agg: Optional[pd.DataFrame], axis_order: Sequence[str], subset: str = "full_audit") -> pd.DataFrame:
    if agg is None or agg.empty:
        return pd.DataFrame()
    df = ensure_cols(agg, [
        "axis", "subset", "level", "spearman_cells", "same_sign_cell_rate",
        "avg_rank_movement", "same_top_cell", "same_bottom_cell", "mean_abs_cell_shift",
    ])
    df = df[df["subset"].astype(str).eq(subset)].copy()
    df = df[df["level"].astype(str).isin(["religion", "religion_prompt_group"])]
    if df.empty:
        return df
    g = df.groupby(["axis", "level"], as_index=False).agg(
        spearman_cells=("spearman_cells", "mean"),
        same_sign_cell_rate=("same_sign_cell_rate", "mean"),
        avg_rank_movement=("avg_rank_movement", "mean"),
        same_top_cell=("same_top_cell", "mean"),
        same_bottom_cell=("same_bottom_cell", "mean"),
        mean_abs_cell_shift=("mean_abs_cell_shift", "mean"),
    )
    g = sort_axes(g, axis_order)
    g.insert(0, "Axis", g["axis"].map(axis_label))
    g.insert(1, "Level", g["level"].map(level_label))
    g = g.drop(columns=["axis", "level"])
    return g.rename(columns={
        "spearman_cells": "ρ cells",
        "same_sign_cell_rate": "Same-sign rate",
        "avg_rank_movement": "Avg. rank movement",
        "same_top_cell": "Same top",
        "same_bottom_cell": "Same bottom",
        "mean_abs_cell_shift": "Mean |cell shift|",
    })


def table_lexical_baselines(lex: Optional[pd.DataFrame], axis_order: Sequence[str]) -> pd.DataFrame:
    if lex is None or lex.empty:
        return pd.DataFrame()
    df = ensure_cols(lex, ["axis", "baseline", "n", "spearman_pred_score", "pearson_pred_score", "r2_pred_score"])
    df = df[df.get("error", pd.Series([np.nan] * len(df))).isna()].copy()
    if df.empty:
        return df
    df["Axis"] = df["axis"].map(axis_label)
    df["Baseline"] = df["baseline"].map(baseline_label)
    df = sort_axes(df, axis_order)
    out = df[["Axis", "Baseline", "n", "spearman_pred_score", "pearson_pred_score", "r2_pred_score"]]
    return out.rename(columns={
        "n": "N",
        "spearman_pred_score": "ρ lexical→score",
        "pearson_pred_score": "r lexical→score",
        "r2_pred_score": "CV R²",
    })


def table_residualization_summary(resid: Optional[pd.DataFrame], axis_order: Sequence[str]) -> pd.DataFrame:
    if resid is None or resid.empty:
        return pd.DataFrame()
    df = ensure_cols(resid, [
        "axis", "n", "spearman_score_control_pred", "pearson_score_control_pred",
        "r2_in_sample_control_pred", "score_sd", "residual_sd", "religion_in_controls",
    ])
    df = sort_axes(df, axis_order)
    df.insert(0, "Axis", df["axis"].map(axis_label))
    out = df[[
        "Axis", "n", "spearman_score_control_pred", "r2_in_sample_control_pred",
        "score_sd", "residual_sd", "religion_in_controls",
    ]]
    return out.rename(columns={
        "n": "N",
        "spearman_score_control_pred": "ρ controls→score",
        "r2_in_sample_control_pred": "Control R²",
        "score_sd": "Score SD",
        "residual_sd": "Residual SD",
        "religion_in_controls": "Religion controlled?",
    })


def table_residualized_vs_original(resid_vs: Optional[pd.DataFrame], axis_order: Sequence[str]) -> pd.DataFrame:
    if resid_vs is None or resid_vs.empty:
        return pd.DataFrame()
    df = ensure_cols(resid_vs, [
        "axis", "level", "n_cells", "spearman_orig_residualized_cells", "avg_rank_movement",
        "same_top_cell", "same_bottom_cell", "orig_range", "residualized_range",
    ])
    df = df[df["level"].astype(str).isin(["religion", "religion_prompt_group", "model_religion"])]
    if df.empty:
        return df
    df = sort_axes(df, axis_order)
    df.insert(0, "Axis", df["axis"].map(axis_label))
    df.insert(1, "Level", df["level"].map(level_label))
    out = df[[
        "Axis", "Level", "n_cells", "spearman_orig_residualized_cells", "avg_rank_movement",
        "same_top_cell", "same_bottom_cell", "orig_range", "residualized_range",
    ]]
    return out.rename(columns={
        "n_cells": "Cells",
        "spearman_orig_residualized_cells": "ρ orig,resid",
        "avg_rank_movement": "Avg. rank movement",
        "same_top_cell": "Same top",
        "same_bottom_cell": "Same bottom",
        "orig_range": "Orig. range",
        "residualized_range": "Residual range",
    })


def table_residualized_rankings(cells: Optional[pd.DataFrame], axis_order: Sequence[str]) -> pd.DataFrame:
    if cells is None or cells.empty:
        return pd.DataFrame()
    df = ensure_cols(cells, ["axis", "level", "religion", "mean_orig", "mean_residualized"])
    df = df[df["level"].astype(str).eq("religion")].copy()
    if df.empty:
        return df
    rows: List[Dict[str, Any]] = []
    for axis in axis_order + [a for a in sorted(df["axis"].dropna().unique()) if a not in axis_order]:
        sub = df[df["axis"].eq(axis)].copy()
        if sub.empty:
            continue
        sub["mean_orig"] = pd.to_numeric(sub["mean_orig"], errors="coerce")
        sub["mean_residualized"] = pd.to_numeric(sub["mean_residualized"], errors="coerce")
        high_orig = sub.loc[sub["mean_orig"].idxmax()]
        low_orig = sub.loc[sub["mean_orig"].idxmin()]
        high_res = sub.loc[sub["mean_residualized"].idxmax()]
        low_res = sub.loc[sub["mean_residualized"].idxmin()]
        rows.append({
            "Axis": axis_label(axis),
            "Highest original": f"{high_orig['religion']} ({fmt(high_orig['mean_orig'])})",
            "Lowest original": f"{low_orig['religion']} ({fmt(low_orig['mean_orig'])})",
            "Highest residualized": f"{high_res['religion']} ({fmt(high_res['mean_residualized'])})",
            "Lowest residualized": f"{low_res['religion']} ({fmt(low_res['mean_residualized'])})",
        })
    return pd.DataFrame(rows)


def table_placebo(placebo: Optional[pd.DataFrame], axis_order: Sequence[str]) -> pd.DataFrame:
    if placebo is None or placebo.empty:
        return pd.DataFrame()
    df = ensure_cols(placebo, [
        "axis", "model", "n_placebo", "real_religion_range", "placebo_religion_range_mean",
        "p_placebo_religion_range_ge_real", "real_rpg_range", "placebo_rpg_range_mean",
        "p_placebo_rpg_range_ge_real", "mean_spearman_real_placebo_religion",
        "mean_spearman_real_placebo_rpg", "mean_placebo_val_sign_accuracy_positive",
    ])
    g = df.groupby("axis", as_index=False).agg(
        n_model_axes=("model", "count"),
        n_placebo=("n_placebo", "mean"),
        real_religion_range=("real_religion_range", "mean"),
        placebo_religion_range_mean=("placebo_religion_range_mean", "mean"),
        p_placebo_religion_range_ge_real=("p_placebo_religion_range_ge_real", "mean"),
        real_rpg_range=("real_rpg_range", "mean"),
        placebo_rpg_range_mean=("placebo_rpg_range_mean", "mean"),
        p_placebo_rpg_range_ge_real=("p_placebo_rpg_range_ge_real", "mean"),
        mean_spearman_real_placebo_religion=("mean_spearman_real_placebo_religion", "mean"),
        mean_spearman_real_placebo_rpg=("mean_spearman_real_placebo_rpg", "mean"),
        mean_placebo_val_sign_accuracy_positive=("mean_placebo_val_sign_accuracy_positive", "mean"),
    )
    g = sort_axes(g, axis_order)
    g.insert(0, "Axis", g["axis"].map(axis_label))
    out = g[[
        "Axis", "n_model_axes", "n_placebo", "real_religion_range", "placebo_religion_range_mean",
        "p_placebo_religion_range_ge_real", "real_rpg_range", "placebo_rpg_range_mean",
        "p_placebo_rpg_range_ge_real", "mean_spearman_real_placebo_religion",
        "mean_spearman_real_placebo_rpg", "mean_placebo_val_sign_accuracy_positive",
    ]]
    return out.rename(columns={
        "n_model_axes": "Model-axes",
        "n_placebo": "Placebos/model-axis",
        "real_religion_range": "Real religion range",
        "placebo_religion_range_mean": "Placebo religion range",
        "p_placebo_religion_range_ge_real": "P(placebo ≥ real), religion",
        "real_rpg_range": "Real rel×group range",
        "placebo_rpg_range_mean": "Placebo rel×group range",
        "p_placebo_rpg_range_ge_real": "P(placebo ≥ real), rel×group",
        "mean_spearman_real_placebo_religion": "Mean ρ real/placebo religion",
        "mean_spearman_real_placebo_rpg": "Mean ρ real/placebo rel×group",
        "mean_placebo_val_sign_accuracy_positive": "Placebo validation sign+",
    })


# -----------------------------------------------------------------------------
# Natural-language summary builder
# -----------------------------------------------------------------------------

def get_col(df: pd.DataFrame, col: str) -> pd.Series:
    if df is None or df.empty or col not in df.columns:
        return pd.Series(dtype=float)
    return df[col]


def bullet_summary(
    full_resp: pd.DataFrame,
    agg_full: pd.DataFrame,
    lex_tbl: pd.DataFrame,
    resid_vs_tbl: pd.DataFrame,
    placebo_tbl: pd.DataFrame,
) -> str:
    bullets: List[str] = []
    if full_resp is not None and not full_resp.empty:
        bullets.append(
            "Full-audit ablation: cue coverage ranged from "
            f"{range_text(full_resp['Cue coverage'])}; mean sign-flip rates ranged from "
            f"{range_text(full_resp['Flip rate'])}; response-level original-vs.-ablated "
            f"Spearman correlations ranged from {range_text(full_resp['ρ response'])}."
        )
    if agg_full is not None and not agg_full.empty:
        rel = agg_full[agg_full["Level"].eq("Religion")]
        rpg = agg_full[agg_full["Level"].eq("Religion × prompt group")]
        if not rel.empty:
            bullets.append(
                "Aggregate ablation stability at the religion level: cell-level Spearman "
                f"correlations ranged from {range_text(rel['ρ cells'])}, with same-sign rates "
                f"from {range_text(rel['Same-sign rate'])}."
            )
        if not rpg.empty:
            bullets.append(
                "Aggregate ablation stability at the religion × prompt-group level: "
                f"cell-level Spearman correlations ranged from {range_text(rpg['ρ cells'])}, "
                f"with same-sign rates from {range_text(rpg['Same-sign rate'])}."
            )
    if lex_tbl is not None and not lex_tbl.empty:
        bullets.append(
            "Lexical-only baselines: cross-validated lexical predictors achieved Spearman "
            f"correlations with contextual scores in the range {range_text(lex_tbl['ρ lexical→score'])} "
            f"and CV R² in the range {range_text(lex_tbl['CV R²'])}."
        )
    if resid_vs_tbl is not None and not resid_vs_tbl.empty:
        rel = resid_vs_tbl[resid_vs_tbl["Level"].eq("Religion")]
        rpg = resid_vs_tbl[resid_vs_tbl["Level"].eq("Religion × prompt group")]
        if not rel.empty:
            bullets.append(
                "Residualized audit scores: after regressing out lexical/length features, model, "
                "and prompt group, original and residualized religion-level cell means had "
                f"Spearman correlations in the range {range_text(rel['ρ orig,resid'])}."
            )
        if not rpg.empty:
            bullets.append(
                "For religion × prompt-group cells, original and residualized scores had "
                f"Spearman correlations in the range {range_text(rpg['ρ orig,resid'])}."
            )
    if placebo_tbl is not None and not placebo_tbl.empty:
        bullets.append(
            "Placebo axes: the fraction of shuffled/placebo axes with religion-level range at "
            f"least as large as the real axis ranged from {range_text(placebo_tbl['P(placebo ≥ real), religion'])}; "
            "for religion × prompt-group ranges, the corresponding fraction was "
            f"{range_text(placebo_tbl['P(placebo ≥ real), rel×group'])}."
        )
    return "\n".join(f"- {b}" for b in bullets) if bullets else "- No summary available; required CSV files were not found."


def results_section_draft(
    full_resp: pd.DataFrame,
    cue_resp: pd.DataFrame,
    agg_full: pd.DataFrame,
    lex_tbl: pd.DataFrame,
    resid_tbl: pd.DataFrame,
    resid_vs_tbl: pd.DataFrame,
    rankings_tbl: pd.DataFrame,
    placebo_tbl: pd.DataFrame,
) -> str:
    """Draft text with numbers inserted. Keep cautious wording."""
    paragraphs: List[str] = []

    paragraphs.append(
        "### Robustness to lexical and placebo controls\n\n"
        "We next evaluate whether the contextual framing axes are reducible to surface lexical markers or arbitrary directions in representation space. "
        "These analyses are diagnostic rather than causal: they test whether the main comparisons remain stable after removing or statistically controlling for obvious lexical shortcuts."
    )

    if full_resp is not None and not full_resp.empty:
        paragraphs.append(
            "In the full-audit lexical ablation, cue-free responses were retained unchanged and cue-containing responses were rescored after removing axis-specific cue expressions. "
            f"Across axes, cue coverage ranged from {range_text(full_resp['Cue coverage'])}. "
            f"The ablation produced mean sign-flip rates between {range_text(full_resp['Flip rate'])}, while response-level original-vs.-ablated Spearman correlations ranged from {range_text(full_resp['ρ response'])}. "
            "Thus, removing explicit cues changed scores nontrivially but did not generally collapse the response-level ordering."
        )

    if agg_full is not None and not agg_full.empty:
        rel = agg_full[agg_full["Level"].eq("Religion")]
        rpg = agg_full[agg_full["Level"].eq("Religion × prompt group")]
        pieces = []
        if not rel.empty:
            pieces.append(
                f"religion-level cell correlations ranged from {range_text(rel['ρ cells'])} "
                f"with same-sign rates of {range_text(rel['Same-sign rate'])}"
            )
        if not rpg.empty:
            pieces.append(
                f"religion × prompt-group correlations ranged from {range_text(rpg['ρ cells'])} "
                f"with same-sign rates of {range_text(rpg['Same-sign rate'])}"
            )
        if pieces:
            paragraphs.append(
                "The aggregate comparisons used in the main audit were also largely stable under full-audit ablation: "
                + "; ".join(pieces)
                + ". This suggests that the reported religion and prompt-group patterns are not explained solely by the presence of a compact cue list."
            )

    if lex_tbl is not None and not lex_tbl.empty:
        paragraphs.append(
            "We also fit lexical-only baselines to estimate how much of the contextual score is predictable from surface form alone. "
            f"Across cue/length, TF-IDF, and cue/length-plus-prompt/model baselines, cross-validated Spearman correlations with contextual scores ranged from {range_text(lex_tbl['ρ lexical→score'])}, and CV R² ranged from {range_text(lex_tbl['CV R²'])}. "
            "These baselines should be interpreted as an upper-bound diagnostic for lexical reproducibility, not as a replacement for the contextual scoring procedure."
        )

    if resid_tbl is not None and not resid_tbl.empty:
        paragraphs.append(
            "To test whether the audit survives statistical control for surface features, we residualized contextual scores against cue counts, response length, model identity, and prompt group, while intentionally leaving religion out of the controls. "
            f"The control model's in-sample R² ranged from {range_text(resid_tbl['Control R²'])}, and the residual standard deviation ranged from {range_text(resid_tbl['Residual SD'])}."
        )

    if resid_vs_tbl is not None and not resid_vs_tbl.empty:
        rel = resid_vs_tbl[resid_vs_tbl["Level"].eq("Religion")]
        rpg = resid_vs_tbl[resid_vs_tbl["Level"].eq("Religion × prompt group")]
        pieces = []
        if not rel.empty:
            pieces.append(f"religion-level original/residualized correlations of {range_text(rel['ρ orig,resid'])}")
        if not rpg.empty:
            pieces.append(f"religion × prompt-group correlations of {range_text(rpg['ρ orig,resid'])}")
        if pieces:
            paragraphs.append(
                "After residualization, the aggregate structure remained comparable to the original audit, with "
                + " and ".join(pieces)
                + ". We therefore treat lexical and length features as contributing to the scores, but not as fully explaining the comparative framing patterns."
            )

    if placebo_tbl is not None and not placebo_tbl.empty:
        paragraphs.append(
            "Finally, we compared the real axes against label-shuffled placebo axes built from the same pair representations and split structure. "
            f"The mean placebo validation sign-positive rate ranged from {range_text(placebo_tbl['Placebo validation sign+'])}. "
            f"The probability that a placebo axis produced a religion-level range at least as large as the real axis ranged from {range_text(placebo_tbl['P(placebo ≥ real), religion'])}; for religion × prompt-group ranges, it ranged from {range_text(placebo_tbl['P(placebo ≥ real), rel×group'])}. "
            "These controls test whether arbitrary contextual directions reproduce the same aggregate structure."
        )

    paragraphs.append(
        "Overall, these diagnostics support a cautious robustness claim: the contextual axes are lexically sensitive, as expected for discourse-framing phenomena, but the main response-level and aggregate patterns are not reducible to a small inventory of explicit lexical triggers. The results should still be interpreted as measurements of operationalized framing, not as judgments of theological correctness or religious adequacy."
    )

    return "\n\n".join(paragraphs)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=Path, default=Path("resultados_robustness_diagnostics"),
                    help="Directory containing CSV outputs from robustness_framing_diagnostics.py")
    ap.add_argument("--axis-order", nargs="*", default=DEFAULT_AXIS_ORDER)
    ap.add_argument("--float-digits", type=int, default=3)
    args = ap.parse_args()

    outdir: Path = args.outdir
    paper_dir = outdir / "paper_results"
    paper_dir.mkdir(parents=True, exist_ok=True)

    response = read_csv_optional(outdir / "response_level_summary.csv")
    agg = read_csv_optional(outdir / "aggregate_stability_summary.csv")
    lex = read_csv_optional(outdir / "lexical_baseline_summary.csv")
    resid = read_csv_optional(outdir / "residualization_summary.csv")
    resid_vs = read_csv_optional(outdir / "residualized_vs_original_summary.csv")
    resid_cells = read_csv_optional(outdir / "residualized_aggregate_cells.csv")
    placebo = read_csv_optional(outdir / "placebo_summary.csv")

    full_resp_tbl = table_full_audit_response(response, args.axis_order)
    cue_resp_tbl = table_cue_present_response(response, args.axis_order)
    agg_full_tbl = table_aggregate_stability(agg, args.axis_order, subset="full_audit")
    lex_tbl = table_lexical_baselines(lex, args.axis_order)
    resid_tbl = table_residualization_summary(resid, args.axis_order)
    resid_vs_tbl = table_residualized_vs_original(resid_vs, args.axis_order)
    rankings_tbl = table_residualized_rankings(resid_cells, args.axis_order)
    placebo_tbl = table_placebo(placebo, args.axis_order)

    tables = [
        ("table_robust_full_audit_ablation", full_resp_tbl, "Full-audit lexical ablation, averaged across models."),
        ("table_robust_cue_present_ablation", cue_resp_tbl, "Cue-present lexical ablation subset, averaged across models."),
        ("table_robust_aggregate_stability", agg_full_tbl, "Aggregate stability under full-audit ablation."),
        ("table_robust_lexical_baselines", lex_tbl, "Lexical-only baseline prediction of contextual scores."),
        ("table_robust_residualization_controls", resid_tbl, "Residualization control-model summary."),
        ("table_robust_residualized_vs_original", resid_vs_tbl, "Original vs residualized aggregate-cell stability."),
        ("table_robust_residualized_rankings", rankings_tbl, "Highest and lowest religion-level original/residualized cells."),
        ("table_robust_placebo", placebo_tbl, "Placebo-axis summary."),
    ]

    for name, df, caption in tables:
        if df is None or df.empty:
            print_block(name, "No table produced; input file or required rows were missing.")
            continue
        write_table_artifacts(df, paper_dir, name, caption)
        print_block(name, md_table(df, float_digits=args.float_digits))

    bullets = bullet_summary(full_resp_tbl, agg_full_tbl, lex_tbl, resid_vs_tbl, placebo_tbl)
    section = results_section_draft(
        full_resp=full_resp_tbl,
        cue_resp=cue_resp_tbl,
        agg_full=agg_full_tbl,
        lex_tbl=lex_tbl,
        resid_tbl=resid_tbl,
        resid_vs_tbl=resid_vs_tbl,
        rankings_tbl=rankings_tbl,
        placebo_tbl=placebo_tbl,
    )

    (paper_dir / "robustness_key_findings.md").write_text(bullets + "\n", encoding="utf-8")
    (paper_dir / "robustness_results_section_draft.md").write_text(section + "\n", encoding="utf-8")

    print_block("KEY FINDINGS TO SEND BACK", bullets)
    print_block("DRAFT RESULTS SUBSECTION", section)
    print(f"\nSaved paper-ready outputs in: {paper_dir.resolve()}")


if __name__ == "__main__":
    main()
