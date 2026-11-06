#!/usr/bin/env python3
"""
annotation_aggregate_gwet_ac2.py

Aggregate multiple annotator JSON files and compute:
- Gwet's AC2 (ordinal) for the 4 main questions
- Exact agreement and adjacent agreement
- Descriptive scores per axis
- Core / usable / flagged subset counts
- Per-pair summaries useful for paper

Requires:
    pip install irrCAC pandas

Usage:
    python annotation_aggregate_gwet_ac2.py \
        --inputs annotator_a.json annotator_b.json annotator_c.json \
        --output-dir annotation_results
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from irrCAC.raw import CAC


QUESTION_KEYS = [
    "positive_pole_correct",
    "negative_pole_correct",
    "proposition_preserved",
    "contrast_strength",
]

QUESTION_LEVELS = {
    "positive_pole_correct": [0, 1, 2],
    "negative_pole_correct": [0, 1, 2],
    "proposition_preserved": [0, 1, 2],
    "contrast_strength": [1, 2, 3, 4, 5],
}


# -----------------------------
# IO
# -----------------------------

def load_annotation_file(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must be a JSON object.")
    if "annotator_id" not in data or "responses" not in data:
        raise ValueError(f"{path} must contain 'annotator_id' and 'responses'.")
    return data


def collect_pair_metadata(files: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    meta = {}
    for f in files:
        for pair_id, resp in f["responses"].items():
            meta[pair_id] = {
                "axis_key": resp.get("axis_key"),
                "pair_id": resp.get("pair_id", pair_id),
                "religion": resp.get("religion"),
                "topic": resp.get("topic"),
            }
    return meta


# -----------------------------
# Generic helpers
# -----------------------------

def mean_or_none(vals: List[float]) -> Optional[float]:
    if not vals:
        return None
    return float(statistics.mean(vals))


def median_or_none(vals: List[float]) -> Optional[float]:
    if not vals:
        return None
    return float(statistics.median(vals))


def classify_pair(pair_summary: Dict[str, Any]) -> str:
    pos = pair_summary["positive_pole_correct_mean"]
    neg = pair_summary["negative_pole_correct_mean"]
    prop = pair_summary["proposition_preserved_mean"]
    contrast = pair_summary["contrast_strength_mean"]

    if (
        pos is not None and neg is not None and prop is not None and contrast is not None
        and pos >= 1.67 and neg >= 1.67 and prop >= 1.67 and contrast >= 4.0
    ):
        return "core"

    if (
        pos is not None and neg is not None and prop is not None and contrast is not None
        and pos >= 1.0 and neg >= 1.0 and prop >= 1.0 and contrast >= 3.0
    ):
        return "usable"

    return "flagged"


# -----------------------------
# Agreement metrics
# -----------------------------

def build_rating_df(
    files: List[Dict[str, Any]],
    pair_ids: List[str],
    question_key: str,
) -> pd.DataFrame:
    rows = []
    annotator_ids = [f["annotator_id"] for f in files]

    for pair_id in pair_ids:
        row = {"pair_id": pair_id}
        n_present = 0
        for f in files:
            aid = f["annotator_id"]
            resp = f["responses"].get(pair_id)
            if resp is None or question_key not in resp:
                row[aid] = pd.NA
            else:
                row[aid] = resp[question_key]
                n_present += 1
        if n_present >= 2:
            rows.append(row)

    df = pd.DataFrame(rows)
    if len(df) == 0:
        return pd.DataFrame(columns=annotator_ids)

    df = df.set_index("pair_id")
    return df[annotator_ids]


def _valid_rows_matrix(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    return df.loc[df.notna().sum(axis=1) >= 2].copy()


def exact_agreement(df: pd.DataFrame) -> Optional[float]:
    df = _valid_rows_matrix(df)
    if df.empty:
        return None

    agrees = []
    for _, row in df.iterrows():
        vals = [float(v) for v in row.tolist() if pd.notna(v)]
        agrees.append(1.0 if len(set(vals)) == 1 else 0.0)
    return float(sum(agrees) / len(agrees)) if agrees else None


def adjacent_agreement(df: pd.DataFrame, tolerance: float = 1.0) -> Optional[float]:
    df = _valid_rows_matrix(df)
    if df.empty:
        return None

    agrees = []
    for _, row in df.iterrows():
        vals = [float(v) for v in row.tolist() if pd.notna(v)]
        agrees.append(1.0 if (max(vals) - min(vals)) <= tolerance else 0.0)
    return float(sum(agrees) / len(agrees)) if agrees else None


def gwet_ac2(df: pd.DataFrame, categories: List[int]) -> Dict[str, Any]:
    """
    Compute Gwet's AC2 using irrCAC on raw ratings.

    We use ordinal weights because all four questions are ordinal.
    The library can infer categories from the data; to keep the reported
    category set stable, we also return the intended category list.
    """
    df = _valid_rows_matrix(df)
    if df.empty:
        return {
            "level": "ordinal",
            "coefficient_name": "Gwet's AC2",
            "coefficient_value": None,
            "confidence_interval": None,
            "p_value": None,
            "z": None,
            "se": None,
            "pa": None,
            "pe": None,
            "categories_intended": categories,
        }

    # irrCAC expects numeric columns and NaN for missing values
    ratings = df.astype(float)
    cac = CAC(ratings, weights="ordinal")
    result = cac.gwet()
    est = result["est"]

    ci = est.get("confidence_interval")
    if ci is not None:
        ci = [float(ci[0]), float(ci[1])]

    return {
        "level": "ordinal",
        "coefficient_name": est.get("coefficient_name", "Gwet's AC2"),
        "coefficient_value": float(est["coefficient_value"]),
        "confidence_interval": ci,
        "p_value": float(est["p_value"]) if est.get("p_value") is not None else None,
        "z": float(est["z"]) if est.get("z") is not None else None,
        "se": float(est["se"]) if est.get("se") is not None else None,
        "pa": float(est["pa"]) if est.get("pa") is not None else None,
        "pe": float(est["pe"]) if est.get("pe") is not None else None,
        "categories_intended": categories,
        "categories_used_by_irrcac": list(result.get("categories", [])),
    }


def compute_question_metrics(
    files: List[Dict[str, Any]],
    pair_ids: List[str],
    question_key: str,
) -> Dict[str, Any]:
    df = build_rating_df(files, pair_ids, question_key)
    ac2 = gwet_ac2(df, QUESTION_LEVELS[question_key])
    return {
        "gwet_ac2": ac2,
        "exact_agreement": exact_agreement(df),
        "adjacent_agreement": adjacent_agreement(df, tolerance=1.0),
        "n_units_used": int(len(_valid_rows_matrix(df))),
    }


def compute_global_metrics(files: List[Dict[str, Any]], pair_ids: List[str]) -> Dict[str, Any]:
    return {
        q: compute_question_metrics(files, pair_ids, q)
        for q in QUESTION_KEYS
    }


def compute_axis_metrics(files: List[Dict[str, Any]], pair_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    grouped: Dict[str, List[str]] = defaultdict(list)
    for row in pair_rows:
        grouped[row["axis_key"]].append(row["pair_id"])

    out = {}
    for axis_key, ids in grouped.items():
        out[axis_key] = compute_global_metrics(files, sorted(ids))
    return out


# -----------------------------
# Descriptive summaries
# -----------------------------

def aggregate_pair_level(files: List[Dict[str, Any]], pair_meta: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    pair_ids = sorted(pair_meta.keys())
    out = []

    for pair_id in pair_ids:
        responses = []
        for f in files:
            resp = f["responses"].get(pair_id)
            if resp is not None:
                responses.append(resp)

        pos_vals = [float(r["positive_pole_correct"]) for r in responses if "positive_pole_correct" in r]
        neg_vals = [float(r["negative_pole_correct"]) for r in responses if "negative_pole_correct" in r]
        prop_vals = [float(r["proposition_preserved"]) for r in responses if "proposition_preserved" in r]
        contrast_vals = [float(r["contrast_strength"]) for r in responses if "contrast_strength" in r]

        row = {
            **pair_meta[pair_id],
            "n_annotators_present": len(responses),
            "positive_pole_correct_mean": mean_or_none(pos_vals),
            "negative_pole_correct_mean": mean_or_none(neg_vals),
            "proposition_preserved_mean": mean_or_none(prop_vals),
            "contrast_strength_mean": mean_or_none(contrast_vals),
            "positive_pole_correct_median": median_or_none(pos_vals),
            "negative_pole_correct_median": median_or_none(neg_vals),
            "proposition_preserved_median": median_or_none(prop_vals),
            "contrast_strength_median": median_or_none(contrast_vals),
            "annotator_responses": responses,
        }
        row["status"] = classify_pair(row)
        out.append(row)

    return out


def aggregate_axis_level(pair_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in pair_rows:
        grouped[row["axis_key"]].append(row)

    out = {}
    for axis_key, rows in grouped.items():
        pos = [r["positive_pole_correct_mean"] for r in rows if r["positive_pole_correct_mean"] is not None]
        neg = [r["negative_pole_correct_mean"] for r in rows if r["negative_pole_correct_mean"] is not None]
        prop = [r["proposition_preserved_mean"] for r in rows if r["proposition_preserved_mean"] is not None]
        contrast = [r["contrast_strength_mean"] for r in rows if r["contrast_strength_mean"] is not None]
        statuses = Counter(r["status"] for r in rows)

        out[axis_key] = {
            "n_pairs": len(rows),
            "positive_pole_correct_mean": mean_or_none(pos),
            "negative_pole_correct_mean": mean_or_none(neg),
            "proposition_preserved_mean": mean_or_none(prop),
            "contrast_strength_mean": mean_or_none(contrast),
            "status_counts": dict(statuses),
        }
    return out


# -----------------------------
# Main
# -----------------------------

def main(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)

    files = [load_annotation_file(p) for p in args.inputs]
    annotator_ids = [f["annotator_id"] for f in files]
    if len(set(annotator_ids)) != len(annotator_ids):
        raise ValueError("Annotator IDs must be unique across input files.")

    pair_meta = collect_pair_metadata(files)
    pair_ids = sorted(pair_meta.keys())

    pair_rows = aggregate_pair_level(files, pair_meta)
    axis_summary = aggregate_axis_level(pair_rows)

    global_metrics = compute_global_metrics(files, pair_ids)
    axis_metrics = compute_axis_metrics(files, pair_rows)

    overall_status_counts = Counter(r["status"] for r in pair_rows)

    paper_summary = {
        "n_annotators": len(files),
        "annotator_ids": annotator_ids,
        "n_pairs_total": len(pair_rows),
        "global_agreement": global_metrics,
        "axis_level_agreement": axis_metrics,
        "axis_summary": axis_summary,
        "overall_status_counts": dict(overall_status_counts),
        "recommended_reporting": {
            "main_pool": "all pairs",
            "core_subset": "pairs with strong validation",
            "sensitivity_subset": "core + usable",
            "primary_metric": "Gwet's AC2 (ordinal)",
            "secondary_metrics": ["exact agreement", "adjacent agreement (<=1 point)"],
        },
    }

    full = {
        "paper_summary": paper_summary,
        "pair_rows": pair_rows,
    }

    (args.output_dir / "paper_summary_gwet_ac2.json").write_text(
        json.dumps(paper_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "pair_rows_gwet_ac2.json").write_text(
        json.dumps(pair_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "full_results_gwet_ac2.json").write_text(
        json.dumps(full, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Done.")
    print(f"Saved summary to: {args.output_dir / 'paper_summary_gwet_ac2.json'}")
    print(f"Saved pair rows to: {args.output_dir / 'pair_rows_gwet_ac2.json'}")
    print(f"Saved full results to: {args.output_dir / 'full_results_gwet_ac2.json'}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Aggregate annotator JSON files and compute Gwet's AC2 paper metrics.")
    p.add_argument("--inputs", type=Path, nargs="+", required=True, help="Annotator JSON files.")
    p.add_argument("--output-dir", type=Path, required=True, help="Directory for outputs.")
    return p


if __name__ == "__main__":
    parser = build_argparser()
    args = parser.parse_args()
    main(args)
