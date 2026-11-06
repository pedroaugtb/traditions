#!/usr/bin/env python3
"""
select_human_eval_examples_v2.py

Updated selector for the user's actual stage-2 layout:

resultados_religioes_reps/
  <model_key>/
    <representation_key>/
      <axis_key>/
        full.json
        summary.json
        prompt_catalog.json

The main source is full.json, which stores generated outputs nested as:
  religions -> <religion> -> items -> generations
with per-generation scores in either:
  - axis_seed_summary.mean
  - mean_score / score
  - or reducible from axis_seed_scores

What this script does
---------------------
1) Reads every full.json under resultados_religioes_reps.
2) Extracts one response-level row per generation.
3) Standardizes scores within (model, representation, axis).
4) Selects, for each axis:
   - high examples
   - low examples
   - middle examples
   - disagreement examples (contextual > output, output > contextual),
     only when the same response can be paired across representations.
5) Uses diversity-aware proportional sampling so the final pack is spread
   across models, religions, and prompt groups instead of being dominated
   by a few cells.
6) Saves:
   - annotation_items_full.json
   - annotation_items_blind.json
   - selection_summary.json

Recommended usage
-----------------
python select_human_eval_examples_v2.py \
    --input-root resultados_religioes_reps \
    --output-dir human_eval_pack

If you want a smaller pack:
python select_human_eval_examples_v2.py \
    --input-root resultados_religioes_reps \
    --output-dir human_eval_pack_small \
    --high-per-axis 16 --low-per-axis 16 --middle-per-axis 12 \
    --disagree-contextual-per-axis 8 --disagree-output-per-axis 8
"""

import argparse
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd


KNOWN_REPS = {"contextual_last_mean", "output_projection_mean"}
KNOWN_AXES = {
    "assertive_commitment",
    "doctrinal_attribution",
    "hedging",
    "mythologized",
}


def normalize_text(x: Optional[str]) -> str:
    if x is None:
        return ""
    return " ".join(str(x).split())


def stable_hash(parts: Iterable[Any]) -> str:
    blob = "||".join("" if p is None else str(p) for p in parts)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()


def safe_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, bool):
            return None
        return float(x)
    except Exception:
        return None


def infer_metadata_from_path(path: Path) -> Dict[str, Optional[str]]:
    """
    Expected path:
    resultados_religioes_reps/<model_key>/<representation_key>/<axis_key>/full.json
    """
    parts = list(path.parts)
    model_key = None
    rep_key = None
    axis_key = None

    for i, p in enumerate(parts):
        if p in KNOWN_REPS:
            rep_key = p
            if i - 1 >= 0:
                model_key = parts[i - 1]
        if p in KNOWN_AXES:
            axis_key = p

    return {
        "model_key": model_key,
        "representation_key": rep_key,
        "axis_key": axis_key,
    }


def extract_generation_score(gen: Dict[str, Any]) -> Optional[float]:
    """
    Priority:
      1) axis_seed_summary.mean
      2) mean_score / score / value
      3) mean over axis_seed_scores[*].score
    """
    ass = gen.get("axis_seed_summary")
    if isinstance(ass, dict):
        m = safe_float(ass.get("mean"))
        if m is not None:
            return m

    for key in ["mean_score", "score", "value", "axis_score"]:
        v = safe_float(gen.get(key))
        if v is not None:
            return v

    seed_scores = gen.get("axis_seed_scores")
    if isinstance(seed_scores, list):
        vals = []
        for s in seed_scores:
            if isinstance(s, dict):
                v = safe_float(s.get("score"))
                if v is not None:
                    vals.append(v)
        if vals:
            return float(sum(vals) / len(vals))

    return None


def parse_full_json(path: Path) -> List[Dict[str, Any]]:
    meta = infer_metadata_from_path(path)
    if meta["representation_key"] not in KNOWN_REPS:
        return []
    if meta["axis_key"] not in KNOWN_AXES:
        return []

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []

    religions = payload.get("religions")
    if not isinstance(religions, dict):
        return []

    rows = []
    for religion, rinfo in religions.items():
        if not isinstance(rinfo, dict):
            continue
        items = rinfo.get("items", [])
        if not isinstance(items, list):
            continue

        for item in items:
            if not isinstance(item, dict):
                continue

            prompt_group = item.get("group", "")
            group_prompt_index = item.get("group_prompt_index", None)
            prompt_text = normalize_text(item.get("prompt", ""))

            gens = item.get("generations", [])
            if not isinstance(gens, list):
                continue

            for gen in gens:
                if not isinstance(gen, dict):
                    continue
                if gen.get("is_empty_failure", False):
                    continue

                response_text = normalize_text(gen.get("response", ""))
                if not response_text:
                    continue

                score = extract_generation_score(gen)
                if score is None:
                    continue

                generation_index = gen.get("generation_index", None)
                if generation_index in [None, ""]:
                    generation_index = None
                else:
                    try:
                        generation_index = int(generation_index)
                    except Exception:
                        generation_index = None

                row = {
                    "source_file": str(path),
                    "model_key": meta["model_key"],
                    "representation_key": meta["representation_key"],
                    "axis_key": meta["axis_key"],
                    "religion": religion,
                    "prompt_group": prompt_group,
                    "group_prompt_index": group_prompt_index,
                    "prompt_text": prompt_text,
                    "generation_index": generation_index,
                    "response_text": response_text,
                    "score_raw": float(score),
                }

                # strict response id: same metadata + same exact response text
                row["response_uid_strict"] = stable_hash([
                    row["model_key"],
                    row["axis_key"],
                    row["religion"],
                    row["prompt_group"],
                    row["group_prompt_index"],
                    row["prompt_text"],
                    row["generation_index"],
                    row["response_text"],
                ])

                # meta id: same prompt cell / generation slot, regardless of text
                row["response_uid_meta"] = stable_hash([
                    row["model_key"],
                    row["axis_key"],
                    row["religion"],
                    row["prompt_group"],
                    row["group_prompt_index"],
                    row["prompt_text"],
                    row["generation_index"],
                ])
                rows.append(row)

    return rows


def load_all_rows(input_root: Path) -> pd.DataFrame:
    full_files = list(input_root.rglob("full.json"))
    rows = []
    for path in full_files:
        rows.extend(parse_full_json(path))

    if not rows:
        raise RuntimeError(
            "No response rows found in full.json files. "
            "Check the input root and the expected folder layout."
        )

    df = pd.DataFrame(rows).drop_duplicates()
    return df


def zscore_within_groups(df: pd.DataFrame, group_cols: List[str], value_col: str, out_col: str):
    def _z(x: pd.Series) -> pd.Series:
        mu = x.mean()
        sd = x.std(ddof=0)
        if sd is None or math.isclose(sd, 0.0):
            return pd.Series([0.0] * len(x), index=x.index)
        return (x - mu) / sd
    df[out_col] = df.groupby(group_cols, observed=False)[value_col].transform(_z)
    return df


def build_paired_table(rows_df: pd.DataFrame) -> pd.DataFrame:
    """
    Pairs contextual and output rows for the SAME model/axis/religion/prompt/generation slot.

    Strong preference:
      - exact same response text across representations (strict pair)
    Fallback:
      - same metadata slot but different text (meta pair), kept but flagged
    """
    ctx = rows_df[rows_df["representation_key"] == "contextual_last_mean"].copy()
    out = rows_df[rows_df["representation_key"] == "output_projection_mean"].copy()

    ctx = ctx.rename(columns={
        "score_raw": "score_raw_contextual",
        "score_z": "score_z_contextual",
        "response_text": "response_text_contextual",
        "source_file": "source_file_contextual",
    })
    out = out.rename(columns={
        "score_raw": "score_raw_output",
        "score_z": "score_z_output",
        "response_text": "response_text_output",
        "source_file": "source_file_output",
    })

    strict_cols = [
        "response_uid_strict", "model_key", "axis_key", "religion", "prompt_group",
        "group_prompt_index", "prompt_text", "generation_index"
    ]
    strict = pd.merge(
        ctx[strict_cols + ["score_raw_contextual", "score_z_contextual", "response_text_contextual", "source_file_contextual"]],
        out[strict_cols + ["score_raw_output", "score_z_output", "response_text_output", "source_file_output"]],
        on=strict_cols,
        how="inner",
    )
    if not strict.empty:
        strict["pair_match_type"] = "strict_text_match"

    # Fallback meta-pair only for those not already strict-paired
    meta_cols = [
        "response_uid_meta", "model_key", "axis_key", "religion", "prompt_group",
        "group_prompt_index", "prompt_text", "generation_index"
    ]
    loose = pd.merge(
        ctx[meta_cols + ["response_uid_strict", "score_raw_contextual", "score_z_contextual", "response_text_contextual", "source_file_contextual"]],
        out[meta_cols + ["response_uid_strict", "score_raw_output", "score_z_output", "response_text_output", "source_file_output"]],
        on=meta_cols,
        how="inner",
        suffixes=("_ctx_strictid", "_out_strictid"),
    )
    if not loose.empty:
        loose["pair_match_type"] = "meta_match"

    if strict.empty and loose.empty:
        return pd.DataFrame()

    paired = pd.concat([strict, loose], ignore_index=True, sort=False)
    if paired.empty:
        return paired

    paired["gap_contextual_minus_output"] = paired["score_z_contextual"] - paired["score_z_output"]
    paired["texts_identical"] = paired["response_text_contextual"] == paired["response_text_output"]
    return paired


def _group_key_for_diversity(row: pd.Series) -> Tuple[str, str, str]:
    return (str(row["model_key"]), str(row["religion"]), str(row["prompt_group"]))


def diverse_take(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """
    Diversity-aware selection:
    - input df should already be sorted by desirability
    - we then choose rows while balancing across (model, religion, prompt_group)
    """
    if df.empty or n <= 0:
        return df.head(0).copy()

    counts = Counter()
    chosen_indices = []
    remaining = df.reset_index(drop=False)

    while len(chosen_indices) < n and not remaining.empty:
        # score candidates by how underrepresented their cell currently is,
        # with original ranking used as a tie-break through current order.
        best_pos = None
        best_count = None
        for pos, row in remaining.iterrows():
            key = _group_key_for_diversity(row)
            c = counts[key]
            if best_pos is None or c < best_count:
                best_pos = pos
                best_count = c
                if c == 0:
                    break

        row = remaining.loc[best_pos]
        chosen_indices.append(row["index"])
        counts[_group_key_for_diversity(row)] += 1
        remaining = remaining.drop(index=best_pos)

    return df.loc[chosen_indices].copy()


def select_axis_examples(
    ctx_df: pd.DataFrame,
    paired_df: pd.DataFrame,
    axis_key: str,
    n_high: int,
    n_low: int,
    n_mid: int,
    n_dis_ctx: int,
    n_dis_out: int,
) -> List[Dict[str, Any]]:
    out_items: List[Dict[str, Any]] = []

    sub = ctx_df[ctx_df["axis_key"] == axis_key].copy()
    if sub.empty:
        return out_items

    # high / low / middle from contextual, since contextual is the main analysis
    high = sub.sort_values(["score_z", "response_uid_strict"], ascending=[False, True])
    low = sub.sort_values(["score_z", "response_uid_strict"], ascending=[True, True])
    mid = sub.assign(absz=sub["score_z"].abs()).sort_values(["absz", "response_uid_strict"], ascending=[True, True])

    buckets = {
        "high_contextual": diverse_take(high, n_high),
        "low_contextual": diverse_take(low, n_low),
        "middle_contextual": diverse_take(mid, n_mid),
    }

    for bucket_name, bucket_df in buckets.items():
        for _, r in bucket_df.iterrows():
            out_items.append({
                "target_axis": axis_key,
                "selection_bucket": bucket_name,
                "model_key": r["model_key"],
                "religion": r["religion"],
                "prompt_group": r["prompt_group"],
                "group_prompt_index": r["group_prompt_index"],
                "prompt_text": r["prompt_text"],
                "generation_index": r["generation_index"],
                "response_text": r["response_text"],
                "contextual_score_raw": float(r["score_raw"]),
                "contextual_score_z": float(r["score_z"]),
                "output_score_raw": None,
                "output_score_z": None,
                "pair_match_type": None,
                "texts_identical_across_reps": None,
            })

    if not paired_df.empty:
        psub = paired_df[paired_df["axis_key"] == axis_key].copy()
        if not psub.empty:
            # Prefer strict/identical-text pairs for disagreement examples
            psub = psub.sort_values(
                ["texts_identical", "pair_match_type", "gap_contextual_minus_output"],
                ascending=[False, True, False]
            )

            more_ctx = psub.sort_values(["texts_identical", "gap_contextual_minus_output"], ascending=[False, False])
            more_out = psub.sort_values(["texts_identical", "gap_contextual_minus_output"], ascending=[False, True])

            more_ctx = diverse_take(more_ctx, n_dis_ctx)
            more_out = diverse_take(more_out, n_dis_out)

            for bucket_name, bucket_df in [
                ("disagreement_contextual_gt_output", more_ctx),
                ("disagreement_output_gt_contextual", more_out),
            ]:
                for _, r in bucket_df.iterrows():
                    # annotate the contextual text by default
                    response_text = r["response_text_contextual"]
                    out_items.append({
                        "target_axis": axis_key,
                        "selection_bucket": bucket_name,
                        "model_key": r["model_key"],
                        "religion": r["religion"],
                        "prompt_group": r["prompt_group"],
                        "group_prompt_index": r["group_prompt_index"],
                        "prompt_text": r["prompt_text"],
                        "generation_index": r["generation_index"],
                        "response_text": response_text,
                        "contextual_score_raw": float(r["score_raw_contextual"]),
                        "contextual_score_z": float(r["score_z_contextual"]),
                        "output_score_raw": float(r["score_raw_output"]),
                        "output_score_z": float(r["score_z_output"]),
                        "pair_match_type": r["pair_match_type"],
                        "texts_identical_across_reps": bool(r["texts_identical"]),
                    })

    return out_items


def deduplicate_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for item in items:
        key = (
            item["target_axis"],
            item["selection_bucket"],
            item["model_key"],
            item["religion"],
            item["prompt_group"],
            item["group_prompt_index"],
            item["prompt_text"],
            item["generation_index"],
            item["response_text"],
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def attach_item_ids(full_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    blind = []
    for i, item in enumerate(full_items):
        item_id = f"item_{i:05d}"
        item["item_id"] = item_id
        blind.append({
            "item_id": item_id,
            "target_axis": item["target_axis"],
            "selection_bucket": item["selection_bucket"],
            "religion": item["religion"],
            "prompt_group": item["prompt_group"],
            "prompt_text": item["prompt_text"],
            "response_text": item["response_text"],
        })
    return blind


def build_summary(items: List[Dict[str, Any]], rows_df: pd.DataFrame, paired_df: pd.DataFrame) -> Dict[str, Any]:
    by_axis_bucket = defaultdict(lambda: defaultdict(int))
    by_axis_model = defaultdict(lambda: defaultdict(int))
    by_axis_religion = defaultdict(lambda: defaultdict(int))
    by_axis_prompt_group = defaultdict(lambda: defaultdict(int))

    for item in items:
        ax = item["target_axis"]
        by_axis_bucket[ax][item["selection_bucket"]] += 1
        by_axis_model[ax][item["model_key"]] += 1
        by_axis_religion[ax][item["religion"]] += 1
        by_axis_prompt_group[ax][item["prompt_group"]] += 1

    return {
        "n_response_rows_found": int(len(rows_df)),
        "n_paired_rows_found": int(len(paired_df)),
        "n_items_selected": int(len(items)),
        "items_per_axis_bucket": {k: dict(v) for k, v in by_axis_bucket.items()},
        "items_per_axis_model": {k: dict(v) for k, v in by_axis_model.items()},
        "items_per_axis_religion": {k: dict(v) for k, v in by_axis_religion.items()},
        "items_per_axis_prompt_group": {k: dict(v) for k, v in by_axis_prompt_group.items()},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-root", type=str, default="resultados_religioes_reps")
    ap.add_argument("--output-dir", type=str, default="human_eval_pack")
    ap.add_argument("--high-per-axis", type=int, default=24)
    ap.add_argument("--low-per-axis", type=int, default=24)
    ap.add_argument("--middle-per-axis", type=int, default=16)
    ap.add_argument("--disagree-contextual-per-axis", type=int, default=12)
    ap.add_argument("--disagree-output-per-axis", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)

    input_root = Path(args.input_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows_df = load_all_rows(input_root)
    rows_df = zscore_within_groups(
        rows_df,
        ["model_key", "representation_key", "axis_key"],
        "score_raw",
        "score_z",
    )

    ctx_df = rows_df[rows_df["representation_key"] == "contextual_last_mean"].copy()
    paired_df = build_paired_table(rows_df)

    selected: List[Dict[str, Any]] = []
    for axis_key in sorted(KNOWN_AXES):
        selected.extend(select_axis_examples(
            ctx_df=ctx_df,
            paired_df=paired_df,
            axis_key=axis_key,
            n_high=args.high_per_axis,
            n_low=args.low_per_axis,
            n_mid=args.middle_per_axis,
            n_dis_ctx=args.disagree_contextual_per_axis,
            n_dis_out=args.disagree_output_per_axis,
        ))

    selected = deduplicate_items(selected)
    blind_items = attach_item_ids(selected)
    summary = build_summary(selected, rows_df, paired_df)

    (output_dir / "annotation_items_full.json").write_text(
        json.dumps(selected, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "annotation_items_blind.json").write_text(
        json.dumps(blind_items, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "selection_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Done.")
    print(f"Response rows found: {len(rows_df)}")
    print(f"Paired rows found:   {len(paired_df)}")
    print(f"Items selected:      {len(selected)}")
    print(f"Saved to:            {output_dir}")


if __name__ == "__main__":
    main()
