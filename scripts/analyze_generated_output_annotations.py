#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


def safe_float(x):
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def rankdata(x: np.ndarray) -> np.ndarray:
    s = pd.Series(x)
    return s.rank(method='average').to_numpy(dtype=float)


def spearman_corr(x: Iterable[float], y: Iterable[float]) -> Optional[float]:
    x = np.asarray(list(x), dtype=float)
    y = np.asarray(list(y), dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 2:
        return None
    rx = rankdata(x)
    ry = rankdata(y)
    sx = rx.std()
    sy = ry.std()
    if math.isclose(sx, 0.0) or math.isclose(sy, 0.0):
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def exact_agreement(ratings: List[float]) -> Optional[float]:
    if len(ratings) < 2:
        return None
    return float(all(r == ratings[0] for r in ratings))


def adjacent_agreement(ratings: List[float]) -> Optional[float]:
    if len(ratings) < 2:
        return None
    return float((max(ratings) - min(ratings)) <= 1.0)


def mean_or_none(vals: List[Optional[float]]) -> Optional[float]:
    vals = [v for v in vals if v is not None and np.isfinite(v)]
    if not vals:
        return None
    return float(np.mean(vals))


def stdev_or_none(vals: List[Optional[float]]) -> Optional[float]:
    vals = [v for v in vals if v is not None and np.isfinite(v)]
    if len(vals) < 2:
        return None
    return float(np.std(vals, ddof=0))


def first_existing(df: pd.DataFrame, candidates: List[str], required: bool = True):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"Could not find any of these columns: {candidates}")
    return None


def pairwise_spearman_by_annotator(df_long: pd.DataFrame) -> Dict[str, Any]:
    annotators = sorted(df_long['annotator_id'].unique())
    pairs = []
    for i in range(len(annotators)):
        for j in range(i + 1, len(annotators)):
            a = annotators[i]
            b = annotators[j]
            da = df_long[df_long['annotator_id'] == a][['item_id', 'rating']].rename(columns={'rating': 'rating_a'})
            db = df_long[df_long['annotator_id'] == b][['item_id', 'rating']].rename(columns={'rating': 'rating_b'})
            m = da.merge(db, on='item_id', how='inner')
            corr = spearman_corr(m['rating_a'], m['rating_b'])
            pairs.append({
                'annotator_a': a,
                'annotator_b': b,
                'n_common_items': int(len(m)),
                'spearman': corr,
            })
    return {
        'pairs': pairs,
        'mean_pairwise_spearman': mean_or_none([p['spearman'] for p in pairs]),
    }


def load_full_items(path: Path) -> pd.DataFrame:
    items = json.loads(path.read_text(encoding='utf-8'))
    return pd.DataFrame(items)


def load_annotations(annotations_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(annotations_dir.glob('*.json')):
        payload = json.loads(path.read_text(encoding='utf-8'))
        annotator_id = payload.get('annotator_id', path.stem)
        ann = payload.get('annotations', {})
        for item_id, item in ann.items():
            rows.append({
                'annotator_id': annotator_id,
                'item_id': item_id,
                'target_axis': item.get('target_axis'),
                'selection_bucket': item.get('selection_bucket'),
                'rating': safe_float(item.get('rating')),
                'confidence': safe_float(item.get('confidence')),
                'note': item.get('note', ''),
                'status': item.get('status', 'rated' if item.get('rating') is not None else 'unknown'),
            })
    if not rows:
        raise RuntimeError('No annotation files found.')
    return pd.DataFrame(rows)


def standardize_within_axis(df: pd.DataFrame, axis_col: str, value_col: str, out_col: str):
    def _z(x):
        mu = x.mean()
        sd = x.std(ddof=0)
        if sd is None or math.isclose(sd, 0.0):
            return pd.Series([0.0] * len(x), index=x.index)
        return (x - mu) / sd
    df[out_col] = df.groupby(axis_col, observed=False)[value_col].transform(_z)
    return df


def analyze_agreement(ann_df: pd.DataFrame) -> Dict[str, Any]:
    rated_df = ann_df[ann_df['status'] == 'rated'].copy()

    item_groups = rated_df.groupby('item_id', observed=False)
    item_rows = []
    for item_id, sub in item_groups:
        ratings = [float(x) for x in sub['rating'].tolist() if x is not None and np.isfinite(x)]
        item_rows.append({
            'item_id': item_id,
            'target_axis': sub['target_axis'].iloc[0],
            'selection_bucket': sub['selection_bucket'].iloc[0],
            'n_ratings': len(ratings),
            'exact_agreement': exact_agreement(ratings),
            'adjacent_agreement': adjacent_agreement(ratings),
            'mean_rating': mean_or_none(ratings),
            'std_rating': stdev_or_none(ratings),
        })
    item_df = pd.DataFrame(item_rows)

    overall = {
        'n_annotation_rows_total': int(len(ann_df)),
        'n_annotation_rows_rated': int(len(rated_df)),
        'n_bugged_skipped_rows': int((ann_df['status'] == 'bugged_skipped').sum()),
        'n_unique_items_rated': int(rated_df['item_id'].nunique()),
        'n_annotators': int(rated_df['annotator_id'].nunique()),
        'exact_agreement_mean': mean_or_none(item_df['exact_agreement'].tolist()) if not item_df.empty else None,
        'adjacent_agreement_mean': mean_or_none(item_df['adjacent_agreement'].tolist()) if not item_df.empty else None,
        'pairwise_annotator_spearman': pairwise_spearman_by_annotator(rated_df) if not rated_df.empty else {'pairs': [], 'mean_pairwise_spearman': None},
    }

    by_axis = {}
    if not item_df.empty:
        for axis, sub in item_df.groupby('target_axis', observed=False):
            axis_ann = rated_df[rated_df['target_axis'] == axis]
            by_axis[axis] = {
                'n_items': int(len(sub)),
                'exact_agreement_mean': mean_or_none(sub['exact_agreement'].tolist()),
                'adjacent_agreement_mean': mean_or_none(sub['adjacent_agreement'].tolist()),
                'mean_item_std': mean_or_none(sub['std_rating'].tolist()),
                'pairwise_annotator_spearman': pairwise_spearman_by_annotator(axis_ann),
            }

    return {'overall': overall, 'by_axis': by_axis, 'item_level': item_df}


def compare_representation_match(item_means: pd.DataFrame, axis_col: str) -> Dict[str, Any]:
    out = {'overall': {}, 'by_axis': {}}
    tmp = item_means.copy()
    tmp = standardize_within_axis(tmp, axis_col, 'mean_rating', 'human_z')

    if 'contextual_score_z' in tmp.columns:
        tmp['contextual_abs_err'] = (tmp['human_z'] - tmp['contextual_score_z']).abs()
    if 'output_score_z' in tmp.columns:
        tmp['output_abs_err'] = (tmp['human_z'] - tmp['output_score_z']).abs()

    if 'contextual_abs_err' in tmp.columns and 'output_abs_err' in tmp.columns:
        valid = tmp[tmp['output_score_z'].notna()].copy()
        if not valid.empty:
            out['overall'] = {
                'n_items_with_both_reps': int(len(valid)),
                'mean_abs_err_contextual': float(valid['contextual_abs_err'].mean()),
                'mean_abs_err_output': float(valid['output_abs_err'].mean()),
                'contextual_better_fraction': float((valid['contextual_abs_err'] < valid['output_abs_err']).mean()),
                'output_better_fraction': float((valid['output_abs_err'] < valid['contextual_abs_err']).mean()),
                'tie_fraction': float((valid['output_abs_err'] == valid['contextual_abs_err']).mean()),
            }
            for axis, sub in valid.groupby(axis_col, observed=False):
                out['by_axis'][axis] = {
                    'n_items': int(len(sub)),
                    'mean_abs_err_contextual': float(sub['contextual_abs_err'].mean()),
                    'mean_abs_err_output': float(sub['output_abs_err'].mean()),
                    'contextual_better_fraction': float((sub['contextual_abs_err'] < sub['output_abs_err']).mean()),
                    'output_better_fraction': float((sub['output_abs_err'] < sub['contextual_abs_err']).mean()),
                    'tie_fraction': float((sub['output_abs_err'] == sub['contextual_abs_err']).mean()),
                }
    return out


def analyze_construct_validity(full_df: pd.DataFrame, ann_df: pd.DataFrame) -> Dict[str, Any]:
    rated_df = ann_df[ann_df['status'] == 'rated'].copy()

    mean_df = rated_df.groupby(['item_id', 'target_axis', 'selection_bucket'], observed=False, as_index=False).agg(
        mean_rating=('rating', 'mean'),
        std_rating=('rating', lambda x: float(np.std(x, ddof=0)) if len(x) > 1 else 0.0),
        n_raters=('rating', 'count'),
        mean_confidence=('confidence', 'mean'),
    )

    merged = mean_df.merge(full_df, on='item_id', how='left', suffixes=('_ann', '_full'))

    axis_col = first_existing(merged, ['target_axis_ann', 'target_axis', 'target_axis_full'])
    bucket_col = first_existing(merged, ['selection_bucket_ann', 'selection_bucket', 'selection_bucket_full'])

    merged = standardize_within_axis(merged, axis_col, 'mean_rating', 'human_z')

    validity = {'overall': {}, 'by_axis': {}, 'bucket_checks': {}}

    if 'contextual_score_z' in merged.columns:
        validity['overall']['spearman_human_vs_contextual_z'] = spearman_corr(
            merged['mean_rating'], merged['contextual_score_z']
        )
    if 'output_score_z' in merged.columns:
        validity['overall']['spearman_human_vs_output_z'] = spearman_corr(
            merged['mean_rating'], merged['output_score_z']
        )

    for axis, sub in merged.groupby(axis_col, observed=False):
        axis_info = {
            'n_items': int(len(sub)),
            'spearman_human_vs_contextual_z': spearman_corr(sub['mean_rating'], sub['contextual_score_z']) if 'contextual_score_z' in sub.columns else None,
            'spearman_human_vs_output_z': spearman_corr(sub['mean_rating'], sub['output_score_z']) if 'output_score_z' in sub.columns else None,
            'spearman_human_vs_contextual_raw': spearman_corr(sub['mean_rating'], sub['contextual_score_raw']) if 'contextual_score_raw' in sub.columns else None,
            'spearman_human_vs_output_raw': spearman_corr(sub['mean_rating'], sub['output_score_raw']) if 'output_score_raw' in sub.columns else None,
        }
        validity['by_axis'][axis] = axis_info

        bucket_means = {}
        for bucket, bsub in sub.groupby(bucket_col, observed=False):
            bucket_means[bucket] = {
                'n_items': int(len(bsub)),
                'mean_human_rating': float(bsub['mean_rating'].mean()),
                'mean_contextual_z': float(bsub['contextual_score_z'].mean()) if 'contextual_score_z' in bsub.columns and bsub['contextual_score_z'].notna().any() else None,
                'mean_output_z': float(bsub['output_score_z'].mean()) if 'output_score_z' in bsub.columns and bsub['output_score_z'].notna().any() else None,
            }
        validity['bucket_checks'][axis] = bucket_means

    rep_match = compare_representation_match(merged, axis_col)
    return {'summary': validity, 'representation_match': rep_match, 'item_means': merged}


def write_text_report(agreement: Dict[str, Any], validity: Dict[str, Any], out_path: Path):
    lines = []
    lines.append('Generated-output construct-validity analysis')
    lines.append('=' * 72)
    lines.append('')
    lines.append('Agreement')
    lines.append('-' * 72)
    lines.append(f"Rated unique items: {agreement['overall']['n_unique_items_rated']}")
    lines.append(f"Annotators:         {agreement['overall']['n_annotators']}")
    lines.append(f"Bugged/skipped:     {agreement['overall']['n_bugged_skipped_rows']}")
    lines.append(f"Exact agreement mean:    {agreement['overall']['exact_agreement_mean']}")
    lines.append(f"Adjacent agreement mean: {agreement['overall']['adjacent_agreement_mean']}")
    lines.append(f"Mean pairwise Spearman:  {agreement['overall']['pairwise_annotator_spearman']['mean_pairwise_spearman']}")
    lines.append('')
    lines.append('Construct validity')
    lines.append('-' * 72)
    ov = validity['summary']['overall']
    lines.append(f"Spearman human vs contextual z: {ov.get('spearman_human_vs_contextual_z')}")
    lines.append(f"Spearman human vs output z:     {ov.get('spearman_human_vs_output_z')}")
    lines.append('')
    for axis, info in validity['summary']['by_axis'].items():
        lines.append(f'[{axis}]')
        lines.append(f"  n_items: {info['n_items']}")
        lines.append(f"  rho(human, contextual z): {info['spearman_human_vs_contextual_z']}")
        lines.append(f"  rho(human, output z):     {info['spearman_human_vs_output_z']}")
        lines.append('')
    lines.append('Representation match')
    lines.append('-' * 72)
    rm = validity['representation_match'].get('overall', {})
    if rm:
        lines.append(f"Items with both reps: {rm.get('n_items_with_both_reps')}")
        lines.append(f"Mean abs err contextual: {rm.get('mean_abs_err_contextual')}")
        lines.append(f"Mean abs err output:     {rm.get('mean_abs_err_output')}")
        lines.append(f"Contextual better frac:  {rm.get('contextual_better_fraction')}")
        lines.append(f"Output better frac:      {rm.get('output_better_fraction')}")
    else:
        lines.append('No matched contextual/output items available for representation comparison.')
    out_path.write_text('\n'.join(lines), encoding='utf-8')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--full-items', type=str, default='human_eval_pack/annotation_items_full.json')
    ap.add_argument('--annotations-dir', type=str, default='human_eval_annotations')
    ap.add_argument('--output-dir', type=str, default='human_eval_analysis')
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    full_df = load_full_items(Path(args.full_items))
    ann_df = load_annotations(Path(args.annotations_dir))

    agreement = analyze_agreement(ann_df)
    validity = analyze_construct_validity(full_df, ann_df)

    agreement['item_level'].to_csv(output_dir / 'agreement_item_level.csv', index=False)
    validity['item_means'].to_csv(output_dir / 'per_axis_item_means.csv', index=False)

    agreement_json = {'overall': agreement['overall'], 'by_axis': agreement['by_axis']}
    (output_dir / 'agreement_summary.json').write_text(json.dumps(agreement_json, ensure_ascii=False, indent=2), encoding='utf-8')

    construct_json = {'summary': validity['summary'], 'representation_match': validity['representation_match']}
    (output_dir / 'construct_validity_summary.json').write_text(json.dumps(construct_json, ensure_ascii=False, indent=2), encoding='utf-8')

    write_text_report(agreement, validity, output_dir / 'report.txt')

    print('Done.')
    print(f"Agreement summary:        {output_dir / 'agreement_summary.json'}")
    print(f"Construct validity:      {output_dir / 'construct_validity_summary.json'}")
    print(f"Per-item merged table:   {output_dir / 'per_axis_item_means.csv'}")
    print(f"Text report:             {output_dir / 'report.txt'}")


if __name__ == '__main__':
    main()
