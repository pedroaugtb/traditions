#!/usr/bin/env python3
"""
Robustness diagnostics for contextual framing axes.

This script extends the earlier lexical-cue ablation check with four code-only
construct-validity diagnostics:

  1. FULL-AUDIT CUE ABLATION
     Keeps every generated response in the audit. Cue-free responses receive
     response_ablated=response_text and score_ablated=score_orig; cue-present
     responses are ablated and rescored. The script still also reports the
     cue-present subset for comparability with the earlier analysis.

  2. LEXICAL-ONLY BASELINES
     Uses cross-validated lexical/cue/metadata baselines to predict the existing
     contextual axis score. No new annotation is required. This estimates how
     much of the contextual score is reproducible from surface features alone.

  3. RESIDUALIZED AUDIT SCORES
     Residualizes contextual scores against lexical cue counts, length, model,
     and prompt group, but NOT religion. The residualized scores can be used to
     re-run religion and religion x prompt-group comparisons while preserving
     religion as the target comparison.

  4. PLACEBO / PERMUTATION AXES
     Builds random label-shuffled placebo axes from the same minimal-pair
     representations and split structure, then compares real audit structure
     against placebo audit structure. This checks whether arbitrary directions
     also produce stable religion-level patterns.

Inputs expected from this repository:
  religion_replies_stage2/<model>/<representation>/<axis>/full.json
  axis_stage1/<model>/<representation>/<axis>/concepts.pt
  pairs/*.json

Main outputs:
  response_level_summary.csv
  aggregate_stability_summary.csv
  aggregate_cells.csv
  ablated_response_rows.csv
  lexical_baseline_summary.csv
  residualized_response_rows.csv
  residualized_aggregate_cells.csv
  residualized_vs_original_summary.csv
  placebo_summary.csv
  placebo_replicate_summary.csv
  robustness_report.json
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    import numpy as np
    import pandas as pd
    SKLEARN_NUMPY_OK = True
except Exception:
    SKLEARN_NUMPY_OK = False

try:
    from scipy.stats import pearsonr, spearmanr, ttest_rel, wilcoxon
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

try:
    from sklearn.compose import ColumnTransformer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import Ridge, RidgeCV
    from sklearn.metrics import r2_score
    from sklearn.model_selection import GroupKFold, KFold, cross_val_predict
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False


# ============================================================
# DEFAULT CONFIG
# ============================================================

DEFAULT_PAIRS_DIR = Path("pairs")
DEFAULT_CONCEPTS_ROOT = Path("axis_stage1")
DEFAULT_RESULTS_ROOT = Path("religion_replies_stage2")
DEFAULT_OUTDIR = Path("robustness_diagnostics")

MODELS = {
    "qwen": "Qwen/Qwen2.5-7B",
    "olmo": "allenai/OLMo-2-1124-7B-Instruct",
    "gemma": "google/gemma-2-9b-it",
    "llama": "meta-llama/Llama-3.2-3B-Instruct",
}

AXIS_PAIR_FILENAMES = {
    "assertive_commitment": "assertive_commitment_axis_pairs.json",
    "doctrinal_attribution": "doctrinal_attribution_axis_pairs.json",
    "hedging": "hedging_axis_pairs_v2.json",
    "mythologized": "mythologized_axis_pairs_v2.json",
}

AXIS_PAPER_NAMES = {
    "assertive_commitment": "Assertive commitment",
    "doctrinal_attribution": "Doctrinal attribution",
    "hedging": "Hedging",
    "mythologized": "Mythologization",
}

REPRESENTATION_KEY = "contextual_last_mean"

# Keep these compact and obvious. They are diagnostic cues, not a complete grammar.
AXIS_CUES = {
    "assertive_commitment": [
        r"\baccording to\b",
        r"\bbelieve\b",
        r"\bbelieves\b",
        r"\bbelieved\b",
        r"\bchristians believe\b",
        r"\bmuslims believe\b",
        r"\bjews believe\b",
        r"\bbuddhists believe\b",
        r"\bhindus believe\b",
        r"\byoruba believers believe\b",
        r"\bis seen as\b",
        r"\bare seen as\b",
        r"\bis viewed as\b",
        r"\bare viewed as\b",
        r"\bis regarded as\b",
        r"\bare regarded as\b",
    ],
    "doctrinal_attribution": [
        r"\bscripture\b",
        r"\bscriptures\b",
        r"\bsacred text\b",
        r"\bsacred texts\b",
        r"\boral teaching\b",
        r"\boral teachings\b",
        r"\bdoctrine\b",
        r"\bdoctrinal\b",
        r"\bcanon\b",
        r"\bcanonical\b",
        r"\brevealed\b",
        r"\brevelation\b",
        r"\bauthoritative\b",
        r"\bauthority\b",
    ],
    "hedging": [
        r"\bmay be understood as\b",
        r"\bcan be understood as\b",
        r"\bmay be interpreted as\b",
        r"\bcan be interpreted as\b",
        r"\bit is believed that\b",
        r"\bit is believed\b",
        r"\bis believed to\b",
        r"\bis believed\b",
        r"\bit is said that\b",
        r"\bit is said\b",
        r"\bis said to\b",
        r"\bsaid to\b",
        r"\bis considered to\b",
        r"\bare considered to\b",
        r"\boften viewed as\b",
        r"\boften seen as\b",
        r"\bis often described as\b",
        r"\bmay\b",
        r"\bmight\b",
    ],
    "mythologized": [
        r"\bmythological\b",
        r"\bmythically\b",
        r"\bmythical\b",
        r"\bmythic\b",
        r"\bmyths\b",
        r"\bmyth\b",
        r"\blegendary\b",
        r"\blegends\b",
        r"\blegend\b",
        r"\bfolklore\b",
        r"\bsymbolically\b",
        r"\bsymbolic\b",
    ],
}

# Broader lexical counts used for baselines/residualization. These are not removed.
GLOBAL_LEXICAL_FEATURES = {
    "n_belief_cues": r"\bbeliev\w*\b|\baccording to\b|\bregarded as\b|\bviewed as\b|\bseen as\b",
    "n_hedge_cues": r"\bmay\b|\bmight\b|\bcould\b|\bcan be\b|\bis said\b|\bsaid to\b|\bis believed\b|\bit is believed\b|\boften\b|\bsometimes\b|\bin some traditions\b",
    "n_scripture_cues": r"\bscripture\w*\b|\bsacred text\w*\b|\bdoctrine\w*\b|\bcanon\w*\b|\brevelation\b|\brevealed\b|\bauthorit\w*\b|\bteaching\w*\b",
    "n_myth_cues": r"\bmyth\w*\b|\blegend\w*\b|\bfolklore\b|\bsymbolic\w*\b|\bnarrative\w*\b|\bcosmolog\w*\b",
    "n_quotation_marks": r"[\"“”']",
}


# ============================================================
# SMALL UTILITIES
# ============================================================

def normalize_text(x: Optional[str]) -> str:
    if x is None:
        return ""
    return " ".join(str(x).split())


def safe_float(x: Any) -> Optional[float]:
    try:
        if x is None or isinstance(x, bool):
            return None
        y = float(x)
        if math.isnan(y) or math.isinf(y):
            return None
        return y
    except Exception:
        return None


def mean(xs: Sequence[float]) -> float:
    vals = [float(x) for x in xs if safe_float(x) is not None]
    return float(sum(vals) / len(vals)) if vals else float("nan")


def stdev(xs: Sequence[float]) -> float:
    vals = [float(x) for x in xs if safe_float(x) is not None]
    if len(vals) < 2:
        return 0.0
    m = mean(vals)
    return math.sqrt(sum((x - m) ** 2 for x in vals) / (len(vals) - 1))


def cohen_dz(deltas: Sequence[float]) -> float:
    sd = stdev(deltas)
    return 0.0 if sd == 0 else mean(deltas) / sd


def corr(xs: Sequence[float], ys: Sequence[float], kind: str = "spearman") -> float:
    pairs = [(safe_float(x), safe_float(y)) for x, y in zip(xs, ys)]
    pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
    if len(pairs) < 3:
        return float("nan")
    xvals = [p[0] for p in pairs]
    yvals = [p[1] for p in pairs]
    if len(set(xvals)) < 2 or len(set(yvals)) < 2:
        return float("nan")
    if SCIPY_OK:
        try:
            if kind == "spearman":
                return float(spearmanr(xvals, yvals).correlation)
            return float(pearsonr(xvals, yvals).statistic)
        except Exception:
            return float("nan")

    def rank(vals: Sequence[float]) -> List[float]:
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        ranks = [0.0] * len(vals)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg = (i + j) / 2.0
            for k in range(i, j + 1):
                ranks[order[k]] = avg
            i = j + 1
        return ranks

    x = rank(xvals) if kind == "spearman" else list(xvals)
    y = rank(yvals) if kind == "spearman" else list(yvals)
    mx, my = mean(x), mean(y)
    sx, sy = stdev(x), stdev(y)
    if sx == 0 or sy == 0:
        return float("nan")
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / ((len(x) - 1) * sx * sy)


def bootstrap_ci(values: Sequence[float], rng: random.Random, n_boot: int = 1000, alpha: float = 0.05) -> Tuple[float, float]:
    vals = [float(v) for v in values if safe_float(v) is not None]
    if not vals:
        return float("nan"), float("nan")
    n = len(vals)
    boots = []
    for _ in range(n_boot):
        sample = [vals[rng.randrange(n)] for _ in range(n)]
        boots.append(mean(sample))
    boots.sort()
    lo = boots[int((alpha / 2) * n_boot)]
    hi = boots[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return float(lo), float(hi)


def unload_model(model=None) -> None:
    try:
        del model
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: Optional[List[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys = []
        for r in rows:
            for k in r.keys():
                if k not in keys:
                    keys.append(k)
        fieldnames = keys
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    try:
        import numpy as _np
        if isinstance(obj, (_np.integer, _np.floating)):
            return obj.item()
        if isinstance(obj, _np.ndarray):
            return obj.tolist()
    except Exception:
        pass
    return str(obj)


# ============================================================
# DATA LOADING
# ============================================================

def pair_path_for_axis(pairs_dir: Path, axis_key: str) -> Path:
    return pairs_dir / AXIS_PAIR_FILENAMES[axis_key]


def load_axis_file(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return {"pairs": data}
    if not isinstance(data, dict) or "pairs" not in data:
        raise ValueError(f"{path} must be a dict JSON containing key 'pairs' or a list.")
    return data


def get_pair_text(item: Dict[str, Any], keys: Sequence[str]) -> Optional[str]:
    for k in keys:
        if k in item and item[k]:
            return str(item[k])
    return None


def load_pairs(path: Path) -> List[Dict[str, Any]]:
    """Load pair texts. The existing files use assertive/hedged; fallbacks make this safer."""
    data = load_axis_file(path)
    pairs = data["pairs"]
    out = []
    for i, item in enumerate(pairs):
        pos = get_pair_text(item, ["assertive", "positive", "pos", "x_plus", "x+"])
        neg = get_pair_text(item, ["hedged", "negative", "neg", "x_minus", "x-"])
        if pos is None or neg is None:
            raise ValueError(f"Could not find positive/negative texts in {path}, pair index {i}: {item.keys()}")
        out.append({"id": item.get("id", f"pair_{i}"), "pos": pos, "neg": neg})
    return out


def load_concepts(path: Path) -> Dict[str, Any]:
    data = torch.load(path, map_location="cpu")
    if "concepts_per_seed" not in data:
        raise ValueError(f"{path} must contain 'concepts_per_seed'.")
    return data


def parse_full_json(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    religions = payload.get("religions", {})
    rows = []
    for religion, rinfo in religions.items():
        for item_idx, item in enumerate(rinfo.get("items", [])):
            prompt_group = normalize_text(item.get("group", ""))
            prompt_text = normalize_text(item.get("prompt", ""))
            for gen_idx, gen in enumerate(item.get("generations", [])):
                if gen.get("is_empty_failure", False):
                    continue
                response = normalize_text(gen.get("response", ""))
                if not response:
                    continue
                mean_score = None
                if isinstance(gen.get("axis_seed_summary"), dict):
                    mean_score = safe_float(gen["axis_seed_summary"].get("mean"))
                if mean_score is None:
                    continue
                rows.append({
                    "religion": religion,
                    "prompt_group": prompt_group,
                    "prompt_text": prompt_text,
                    "item_index": item_idx,
                    "generation_index": gen_idx,
                    "response_id": f"{religion}::{item_idx}::{gen_idx}",
                    "response_text": response,
                    "score_orig": float(mean_score),
                })
    return rows


# ============================================================
# CUES, ABLATION, AND FEATURES
# ============================================================

def compile_axis_patterns(axis_key: str) -> List[Tuple[str, re.Pattern]]:
    cues = sorted(AXIS_CUES[axis_key], key=len, reverse=True)
    return [(pat, re.compile(pat, flags=re.IGNORECASE)) for pat in cues]


def matched_cues(text: str, patterns: List[Tuple[str, re.Pattern]]) -> List[str]:
    return [pat for pat, compiled in patterns if compiled.search(text)]


def ablate_text(text: str, patterns: List[Tuple[str, re.Pattern]]) -> Tuple[str, int, int]:
    """Remove high-salience cues. Returns text, total substitutions, distinct cue count."""
    out = text
    total_subs = 0
    distinct = 0
    for _, compiled in patterns:
        out_new, n = compiled.subn(" ", out)
        if n > 0:
            distinct += 1
            total_subs += n
            out = out_new
    out = re.sub(r"\s+", " ", out)
    out = re.sub(r"\s+([,.;:!?])", r"\1", out)
    out = re.sub(r"([\(\[\{])\s+", r"\1", out)
    out = re.sub(r"\s+([\)\]\}])", r"\1", out)
    return out.strip(), total_subs, distinct


def lexical_feature_dict(text: str, axis_key: str, axis_patterns: List[Tuple[str, re.Pattern]]) -> Dict[str, Any]:
    tokens = text.split()
    sents = re.split(r"[.!?]+", text)
    sents = [s for s in sents if s.strip()]
    matched = matched_cues(text, axis_patterns)
    out: Dict[str, Any] = {
        "length_chars": len(text),
        "length_tokens": len(tokens),
        "length_sentences": len(sents),
        "mean_token_length": mean([len(t) for t in tokens]) if tokens else 0.0,
        "has_axis_cue": int(bool(matched)),
        "n_axis_cue_types": len(matched),
        "n_axis_cue_matches": 0,
    }
    for _, compiled in axis_patterns:
        out["n_axis_cue_matches"] += len(compiled.findall(text))
    for name, pattern in GLOBAL_LEXICAL_FEATURES.items():
        out[name] = len(re.findall(pattern, text, flags=re.IGNORECASE))
    return out


def annotate_rows_with_cues_and_features(rows: List[Dict[str, Any]], axis_key: str) -> List[Dict[str, Any]]:
    patterns = compile_axis_patterns(axis_key)
    out = []
    for r in rows:
        hits = matched_cues(r["response_text"], patterns)
        features = lexical_feature_dict(r["response_text"], axis_key, patterns)
        row = dict(r)
        row.update(features)
        row["matched_cue_patterns"] = ";".join(hits)
        row["has_axis_cue"] = int(bool(hits))
        out.append(row)
    return out


# ============================================================
# MODEL ENCODING AND SCORING
# ============================================================

def load_model_and_tokenizer(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="auto",
    )
    model.eval()
    return tokenizer, model


def first_device(model):
    return next(model.parameters()).device


@torch.no_grad()
def get_text_representations_contextual(texts: List[str], tokenizer, model, batch_size: int = 8) -> torch.Tensor:
    if not texts:
        return torch.empty((0, 0))
    dev = first_device(model)
    reps = []
    for i in tqdm(range(0, len(texts), batch_size), desc="encoding", leave=False):
        batch = texts[i:i + batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=False,
        ).to(dev)
        outputs = model(**enc, output_hidden_states=True, use_cache=False)
        last = outputs.hidden_states[-1]
        mask = enc["attention_mask"].unsqueeze(-1).float()
        pooled = ((last * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)).detach().float().cpu()
        reps.append(pooled)
    return torch.cat(reps, dim=0)


def get_axis_pair_reps(model, tokenizer, pairs_file: Path, batch_size: int) -> Tuple[List[Dict[str, Any]], torch.Tensor, torch.Tensor]:
    pairs = load_pairs(pairs_file)
    pos_texts = [p["pos"] for p in pairs]
    neg_texts = [p["neg"] for p in pairs]
    pos_reps = get_text_representations_contextual(pos_texts, tokenizer, model, batch_size=batch_size)
    neg_reps = get_text_representations_contextual(neg_texts, tokenizer, model, batch_size=batch_size)
    return pairs, pos_reps, neg_reps


def build_axis_seed_tensors_from_reps(
    pos_reps: torch.Tensor,
    neg_reps: torch.Tensor,
    concepts_file: Path,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    concept_data = load_concepts(concepts_file)
    if concept_data.get("rep_family") not in [None, "contextual_hidden"]:
        raise ValueError(f"Expected contextual_hidden, got {concept_data.get('rep_family')} in {concepts_file}")
    if concept_data.get("rep_mode") not in [None, "mean"]:
        raise ValueError(f"Expected rep_mode=mean, got {concept_data.get('rep_mode')} in {concepts_file}")

    lambdas = []
    midpoints = []
    for seed_entry in concept_data["concepts_per_seed"]:
        train_idxs = seed_entry["train_idxs"]
        lam = seed_entry["lambda_bar_unit_euclid"].float().cpu()
        midpoint = (pos_reps[train_idxs].mean(dim=0) + neg_reps[train_idxs].mean(dim=0)) / 2.0
        lambdas.append(lam)
        midpoints.append(midpoint)
    return torch.stack(lambdas, dim=0), torch.stack(midpoints, dim=0), concept_data


def score_reps_mean_over_seeds(reps: torch.Tensor, lambdas: torch.Tensor, midpoints: torch.Tensor) -> List[float]:
    if reps.numel() == 0:
        return []
    scores = ((reps[:, None, :] - midpoints[None, :, :]) * lambdas[None, :, :]).sum(dim=-1)
    return [float(x) for x in scores.mean(dim=1).tolist()]


# ============================================================
# SUMMARIES
# ============================================================

def paired_test_stats(orig: Sequence[float], abl: Sequence[float]) -> Tuple[float, float]:
    if not SCIPY_OK or len(orig) < 2:
        return float("nan"), float("nan")
    try:
        p_t = float(ttest_rel(orig, abl).pvalue)
    except Exception:
        p_t = float("nan")
    try:
        p_w = float(wilcoxon(orig, abl).pvalue)
    except Exception:
        p_w = float("nan")
    return p_t, p_w


def summarize_response_level(
    rows: List[Dict[str, Any]],
    model_key: str,
    axis_key: str,
    subset_name: str,
    n_total_all: int,
    n_cue_all: int,
    rng: random.Random,
    n_boot: int,
) -> Dict[str, Any]:
    orig = [float(r["score_orig"]) for r in rows]
    abl = [float(r["score_ablated"]) for r in rows]
    deltas = [a - o for o, a in zip(orig, abl)]
    abs_deltas = [abs(d) for d in deltas]
    flips = [1.0 if ((o > 0) != (a > 0)) else 0.0 for o, a in zip(orig, abl)]
    p_t, p_w = paired_test_stats(orig, abl)
    delta_lo, delta_hi = bootstrap_ci(deltas, rng, n_boot=n_boot)
    flip_lo, flip_hi = bootstrap_ci(flips, rng, n_boot=n_boot)
    orig_sd = stdev(orig)
    return {
        "model": model_key,
        "axis": axis_key,
        "subset": subset_name,
        "n_total_all": n_total_all,
        "n_cue_all": n_cue_all,
        "cue_coverage_all": n_cue_all / n_total_all if n_total_all else float("nan"),
        "n_used": len(rows),
        "mean_cue_subs": mean([r.get("cue_subs", 0) for r in rows]),
        "mean_distinct_cues": mean([r.get("cue_distinct", 0) for r in rows]),
        "mean_orig": mean(orig),
        "mean_ablated": mean(abl),
        "mean_delta": mean(deltas),
        "mean_delta_ci_low": delta_lo,
        "mean_delta_ci_high": delta_hi,
        "mean_abs_delta": mean(abs_deltas),
        "delta_over_orig_sd": mean(deltas) / orig_sd if orig_sd > 0 else float("nan"),
        "flip_rate": mean(flips),
        "flip_ci_low": flip_lo,
        "flip_ci_high": flip_hi,
        "cohen_dz": cohen_dz(deltas),
        "abs_cohen_dz": abs(cohen_dz(deltas)),
        "pearson_response": corr(orig, abl, kind="pearson"),
        "spearman_response": corr(orig, abl, kind="spearman"),
        "p_ttest": p_t,
        "p_wilcoxon": p_w,
    }


def aggregate_means(rows: List[Dict[str, Any]], keys: Sequence[str], score_key: str = "score_orig") -> List[Dict[str, Any]]:
    buckets: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        buckets[tuple(r[k] for k in keys)].append(r)
    out = []
    for key_vals, group_rows in sorted(buckets.items()):
        vals = [float(r[score_key]) for r in group_rows if safe_float(r.get(score_key)) is not None]
        rec = {k: v for k, v in zip(keys, key_vals)}
        rec.update({"n": len(vals), "mean_score": mean(vals), "score_key": score_key})
        out.append(rec)
    return out


def aggregate_original_vs_ablated(rows: List[Dict[str, Any]], keys: Sequence[str]) -> List[Dict[str, Any]]:
    buckets: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        buckets[tuple(r[k] for k in keys)].append(r)
    out = []
    for key_vals, group_rows in sorted(buckets.items()):
        orig = [float(r["score_orig"]) for r in group_rows]
        abl = [float(r["score_ablated"]) for r in group_rows]
        rec = {k: v for k, v in zip(keys, key_vals)}
        rec.update({
            "n": len(group_rows),
            "mean_orig": mean(orig),
            "mean_ablated": mean(abl),
            "delta": mean([a - o for o, a in zip(orig, abl)]),
            "same_sign": int((mean(orig) > 0) == (mean(abl) > 0)),
        })
        out.append(rec)
    return out


def rank_order(values: Sequence[float]) -> List[int]:
    return sorted(range(len(values)), key=lambda i: values[i], reverse=True)


def avg_rank_movement(orig: Sequence[float], new: Sequence[float]) -> float:
    if len(orig) < 2:
        return float("nan")
    ro = rank_order(orig)
    rn = rank_order(new)
    pos_o = {idx: rank for rank, idx in enumerate(ro)}
    pos_n = {idx: rank for rank, idx in enumerate(rn)}
    return mean([abs(pos_o[i] - pos_n[i]) for i in range(len(orig))])


def summarize_aggregate_stability(
    rows: List[Dict[str, Any]],
    model_key: str,
    axis_key: str,
    subset_name: str,
    level_name: str,
    keys: Sequence[str],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    aggs = aggregate_original_vs_ablated(rows, keys)
    orig = [a["mean_orig"] for a in aggs]
    abl = [a["mean_ablated"] for a in aggs]
    deltas = [a - o for o, a in zip(orig, abl)]
    same_signs = [a["same_sign"] for a in aggs]
    same_top = float("nan")
    same_bottom = float("nan")
    if len(aggs) >= 2:
        same_top = int(max(range(len(orig)), key=lambda i: orig[i]) == max(range(len(abl)), key=lambda i: abl[i]))
        same_bottom = int(min(range(len(orig)), key=lambda i: orig[i]) == min(range(len(abl)), key=lambda i: abl[i]))
    summary = {
        "model": model_key,
        "axis": axis_key,
        "subset": subset_name,
        "level": level_name,
        "n_cells": len(aggs),
        "mean_abs_cell_shift": mean([abs(d) for d in deltas]),
        "max_abs_cell_shift": max([abs(d) for d in deltas]) if deltas else float("nan"),
        "same_sign_cell_rate": mean(same_signs),
        "pearson_cells": corr(orig, abl, kind="pearson"),
        "spearman_cells": corr(orig, abl, kind="spearman"),
        "avg_rank_movement": avg_rank_movement(orig, abl),
        "same_top_cell": same_top,
        "same_bottom_cell": same_bottom,
    }
    for a in aggs:
        a.update({"model": model_key, "axis": axis_key, "subset": subset_name, "level": level_name})
    return summary, aggs


def mean_of_key(rows: List[Dict[str, Any]], key: str) -> float:
    vals = [safe_float(r.get(key)) for r in rows]
    vals = [v for v in vals if v is not None]
    return mean(vals)


def summarize_by_axis(rows: List[Dict[str, Any]], kind: str) -> List[Dict[str, Any]]:
    out = []
    for subset in sorted({r.get("subset", "") for r in rows}):
        for axis in sorted({r["axis"] for r in rows if r.get("subset", "") == subset}):
            sub = [r for r in rows if r["axis"] == axis and r.get("subset", "") == subset]
            rec = {"axis": axis, "subset": subset, "n_model_summaries": len(sub), "summary_kind": kind}
            for k in [
                "cue_coverage_all", "mean_delta", "mean_abs_delta", "delta_over_orig_sd",
                "flip_rate", "cohen_dz", "abs_cohen_dz", "pearson_response",
                "spearman_response", "mean_abs_cell_shift", "max_abs_cell_shift",
                "same_sign_cell_rate", "pearson_cells", "spearman_cells", "avg_rank_movement",
                "same_top_cell", "same_bottom_cell",
            ]:
                if any(k in r for r in sub):
                    rec[f"mean_{k}"] = mean_of_key(sub, k)
            out.append(rec)
    return out


# ============================================================
# FULL-AUDIT ABLATION
# ============================================================

def run_full_audit_ablation_for_model_axis(
    model_key: str,
    axis_key: str,
    all_rows: List[Dict[str, Any]],
    model,
    tokenizer,
    pairs_file: Path,
    concepts_file: Path,
    batch_size: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    patterns = compile_axis_patterns(axis_key)
    rows = annotate_rows_with_cues_and_features(all_rows, axis_key)
    n_total = len(rows)
    n_cue = sum(int(r["has_axis_cue"]) for r in rows)

    pairs, pos_reps, neg_reps = get_axis_pair_reps(model, tokenizer, pairs_file, batch_size=batch_size)
    lambdas, midpoints, concept_data = build_axis_seed_tensors_from_reps(pos_reps, neg_reps, concepts_file)

    cue_indices = [i for i, r in enumerate(rows) if r["has_axis_cue"]]
    ablated_texts = []
    for i, r in enumerate(rows):
        if r["has_axis_cue"]:
            ablated, n_subs, n_distinct = ablate_text(r["response_text"], patterns)
            if len(ablated.strip()) < 8:
                ablated = r["response_text"]
            rows[i]["response_ablated"] = ablated
            rows[i]["cue_subs"] = n_subs
            rows[i]["cue_distinct"] = n_distinct
            ablated_texts.append(ablated)
        else:
            rows[i]["response_ablated"] = r["response_text"]
            rows[i]["cue_subs"] = 0
            rows[i]["cue_distinct"] = 0
            rows[i]["score_ablated"] = float(r["score_orig"])

    if ablated_texts:
        ablated_reps = get_text_representations_contextual(ablated_texts, tokenizer, model, batch_size=batch_size)
        ablated_scores = score_reps_mean_over_seeds(ablated_reps, lambdas, midpoints)
        for idx, score in zip(cue_indices, ablated_scores):
            rows[idx]["score_ablated"] = float(score)

    for r in rows:
        r.update({
            "model": model_key,
            "axis": axis_key,
            "delta": float(r["score_ablated"] - r["score_orig"]),
            "sign_flip": int((r["score_orig"] > 0) != (r["score_ablated"] > 0)),
        })

    meta = {
        "n_total": n_total,
        "n_cue": n_cue,
        "n_axis_seeds": len(concept_data["concepts_per_seed"]),
        "n_pairs": len(pairs),
    }
    return rows, meta


# ============================================================
# LEXICAL-ONLY BASELINES AND RESIDUALIZATION
# ============================================================

NUMERIC_FEATURES = [
    "length_chars", "length_tokens", "length_sentences", "mean_token_length",
    "has_axis_cue", "n_axis_cue_types", "n_axis_cue_matches",
    "n_belief_cues", "n_hedge_cues", "n_scripture_cues", "n_myth_cues", "n_quotation_marks",
]
CATEGORICAL_FEATURES = ["model", "prompt_group"]  # Religion intentionally excluded.


def make_cv(groups: Sequence[str], n_rows: int, max_splits: int = 5):
    unique_groups = sorted(set(groups))
    if len(unique_groups) >= 2:
        n_splits = min(max_splits, len(unique_groups))
        return GroupKFold(n_splits=n_splits), groups
    n_splits = min(max_splits, n_rows)
    return KFold(n_splits=n_splits, shuffle=True, random_state=42), None


def build_feature_dataframe(rows: List[Dict[str, Any]]):
    if not SKLEARN_NUMPY_OK:
        raise RuntimeError("numpy/pandas are required for baselines and residualization.")
    df = pd.DataFrame(rows).copy()
    for col in NUMERIC_FEATURES:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = df[col].fillna(0.0).astype(float)
    for col in CATEGORICAL_FEATURES:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str)
    df["response_text"] = df["response_text"].fillna("").astype(str)
    return df


def run_lexical_baselines(rows: List[Dict[str, Any]], outdir: Path) -> List[Dict[str, Any]]:
    if not SKLEARN_OK or not SKLEARN_NUMPY_OK:
        print("[warn] scikit-learn/numpy/pandas unavailable; skipping lexical baselines.")
        return []
    all_summaries: List[Dict[str, Any]] = []
    for axis_key in sorted({r["axis"] for r in rows}):
        sub_rows = [r for r in rows if r["axis"] == axis_key]
        if len(sub_rows) < 10:
            continue
        df = build_feature_dataframe(sub_rows)
        y = df["score_orig"].astype(float).values
        groups = (df["prompt_text"].astype(str) + "::" + df["religion"].astype(str)).values
        cv, cv_groups = make_cv(groups, len(df), max_splits=5)

        baselines = []
        # Numeric cue/length features only.
        baselines.append((
            "cue_length_only",
            Pipeline([
                ("pre", ColumnTransformer([
                    ("num", StandardScaler(), NUMERIC_FEATURES),
                ], remainder="drop")),
                ("ridge", RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0])),
            ]),
            df[NUMERIC_FEATURES],
        ))
        # TF-IDF only.
        baselines.append((
            "tfidf_only",
            Pipeline([
                ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 3), min_df=2, max_features=20000)),
                ("ridge", RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0])),
            ]),
            df["response_text"],
        ))
        # Cue/length + prompt/model metadata. Religion excluded.
        baselines.append((
            "cue_length_prompt_model",
            Pipeline([
                ("pre", ColumnTransformer([
                    ("num", StandardScaler(), NUMERIC_FEATURES),
                    ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
                ], remainder="drop")),
                ("ridge", RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0])),
            ]),
            df[NUMERIC_FEATURES + CATEGORICAL_FEATURES],
        ))

        for name, pipe, X in baselines:
            try:
                if cv_groups is None:
                    pred = cross_val_predict(pipe, X, y, cv=cv)
                    group_mode = "kfold"
                else:
                    pred = cross_val_predict(pipe, X, y, cv=cv.split(X, y, cv_groups))
                    group_mode = "group_by_prompt_religion"
                rec = {
                    "axis": axis_key,
                    "baseline": name,
                    "n": len(y),
                    "cv_mode": group_mode,
                    "pearson_pred_score": corr(y, pred, kind="pearson"),
                    "spearman_pred_score": corr(y, pred, kind="spearman"),
                    "r2_pred_score": float(r2_score(y, pred)),
                    "score_sd": stdev(y),
                    "pred_sd": stdev(pred),
                }
                all_summaries.append(rec)
                pred_rows = []
                for row, p in zip(sub_rows, pred):
                    pred_rows.append({
                        "axis": axis_key,
                        "baseline": name,
                        "model": row["model"],
                        "religion": row["religion"],
                        "prompt_group": row["prompt_group"],
                        "response_id": row["response_id"],
                        "score_orig": row["score_orig"],
                        "score_pred_lexical": float(p),
                    })
                write_csv(outdir / f"lexical_predictions_{axis_key}_{name}.csv", pred_rows)
            except Exception as e:
                all_summaries.append({"axis": axis_key, "baseline": name, "error": repr(e), "n": len(y)})
    return all_summaries


def add_residualized_scores(rows: List[Dict[str, Any]], alpha: float = 10.0) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not SKLEARN_OK or not SKLEARN_NUMPY_OK:
        print("[warn] scikit-learn/numpy/pandas unavailable; skipping residualization.")
        return [], []
    out_rows: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []
    for axis_key in sorted({r["axis"] for r in rows}):
        sub_rows = [dict(r) for r in rows if r["axis"] == axis_key]
        df = build_feature_dataframe(sub_rows)
        y = df["score_orig"].astype(float).values
        X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
        pipe = Pipeline([
            ("pre", ColumnTransformer([
                ("num", StandardScaler(), NUMERIC_FEATURES),
                ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ], remainder="drop")),
            ("ridge", Ridge(alpha=alpha)),
        ])
        try:
            pipe.fit(X, y)
            pred = pipe.predict(X)
            resid = y - pred
            for row, p, rr in zip(sub_rows, pred, resid):
                row["score_lexical_control_pred"] = float(p)
                row["score_residualized"] = float(rr)
                out_rows.append(row)
            summaries.append({
                "axis": axis_key,
                "n": len(sub_rows),
                "alpha": alpha,
                "control_features": ",".join(NUMERIC_FEATURES + CATEGORICAL_FEATURES),
                "religion_in_controls": 0,
                "pearson_score_control_pred": corr(y, pred, kind="pearson"),
                "spearman_score_control_pred": corr(y, pred, kind="spearman"),
                "r2_in_sample_control_pred": float(r2_score(y, pred)),
                "score_sd": stdev(y),
                "residual_sd": stdev(resid),
            })
        except Exception as e:
            summaries.append({"axis": axis_key, "error": repr(e), "n": len(sub_rows)})
    return out_rows, summaries


def residualized_aggregate_outputs(rows_resid: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    cells: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []
    levels = {
        "religion": ["axis", "religion"],
        "prompt_group": ["axis", "prompt_group"],
        "religion_prompt_group": ["axis", "religion", "prompt_group"],
        "model_religion": ["axis", "model", "religion"],
        "model_religion_prompt_group": ["axis", "model", "religion", "prompt_group"],
    }
    for level_name, keys in levels.items():
        raw_cells = aggregate_means(rows_resid, keys, score_key="score_orig")
        res_cells = aggregate_means(rows_resid, keys, score_key="score_residualized")
        raw_map = {tuple(c[k] for k in keys): c for c in raw_cells}
        res_map = {tuple(c[k] for k in keys): c for c in res_cells}
        for key_tuple in sorted(raw_map.keys()):
            raw = raw_map[key_tuple]
            res = res_map[key_tuple]
            rec = {k: v for k, v in zip(keys, key_tuple)}
            rec.update({
                "level": level_name,
                "n": raw["n"],
                "mean_orig": raw["mean_score"],
                "mean_residualized": res["mean_score"],
            })
            cells.append(rec)
        # summaries by axis where applicable
        for axis_key in sorted({r["axis"] for r in rows_resid}):
            sub_cells = [c for c in cells if c["level"] == level_name and c.get("axis") == axis_key]
            if len(sub_cells) < 2:
                continue
            orig = [c["mean_orig"] for c in sub_cells]
            resid = [c["mean_residualized"] for c in sub_cells]
            summaries.append({
                "axis": axis_key,
                "level": level_name,
                "n_cells": len(sub_cells),
                "pearson_orig_residualized_cells": corr(orig, resid, kind="pearson"),
                "spearman_orig_residualized_cells": corr(orig, resid, kind="spearman"),
                "avg_rank_movement": avg_rank_movement(orig, resid),
                "same_top_cell": int(max(range(len(orig)), key=lambda i: orig[i]) == max(range(len(resid)), key=lambda i: resid[i])),
                "same_bottom_cell": int(min(range(len(orig)), key=lambda i: orig[i]) == min(range(len(resid)), key=lambda i: resid[i])),
                "orig_range": max(orig) - min(orig),
                "residualized_range": max(resid) - min(resid),
            })
    return cells, summaries


# ============================================================
# PLACEBO AXES
# ============================================================

def build_placebo_seed_tensors(
    pos_reps: torch.Tensor,
    neg_reps: torch.Tensor,
    concept_data: Dict[str, Any],
    rng: random.Random,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """Build one placebo replicate from random sign-flipped pair differences.

    This uses the same train split structure as the real concepts, but randomly
    flips pair polarities before averaging. Directions are Euclidean-normalized.
    It is intentionally a placebo direction in the same representation space,
    not a replacement for the metric-aware real axis.
    """
    diffs = pos_reps - neg_reps
    lambdas = []
    midpoints = []
    val_scores_all = []
    for seed_entry in concept_data["concepts_per_seed"]:
        train_idxs = list(seed_entry["train_idxs"])
        val_idxs = list(seed_entry.get("val_idxs", []))
        signs = torch.tensor([1.0 if rng.random() < 0.5 else -1.0 for _ in train_idxs], dtype=diffs.dtype)
        g = (diffs[train_idxs] * signs[:, None]).mean(dim=0)
        lam = g / (g.norm(p=2) + 1e-12)
        midpoint = (pos_reps[train_idxs].mean(dim=0) + neg_reps[train_idxs].mean(dim=0)) / 2.0
        lambdas.append(lam)
        midpoints.append(midpoint)
        if val_idxs:
            val_scores = (diffs[val_idxs] * lam[None, :]).sum(dim=1)
            val_scores_all.extend([float(x) for x in val_scores.tolist()])
    meta = {
        "placebo_val_sign_accuracy_positive": mean([1.0 if s > 0 else 0.0 for s in val_scores_all]) if val_scores_all else float("nan"),
        "placebo_val_mean_score": mean(val_scores_all) if val_scores_all else float("nan"),
        "n_val_scores": len(val_scores_all),
    }
    return torch.stack(lambdas, dim=0), torch.stack(midpoints, dim=0), meta


def score_real_axis_original_responses(
    rows: List[Dict[str, Any]],
    model,
    tokenizer,
    lambdas: torch.Tensor,
    midpoints: torch.Tensor,
    batch_size: int,
) -> Tuple[List[Dict[str, Any]], torch.Tensor]:
    texts = [r["response_text"] for r in rows]
    reps = get_text_representations_contextual(texts, tokenizer, model, batch_size=batch_size)
    scores = score_reps_mean_over_seeds(reps, lambdas, midpoints)
    out = []
    for r, s in zip(rows, scores):
        rr = dict(r)
        rr["score_rescored_real_axis"] = float(s)
        out.append(rr)
    return out, reps


def aggregate_for_placebo(rows: List[Dict[str, Any]], score_key: str, keys: Sequence[str]) -> Dict[Tuple[Any, ...], float]:
    buckets: Dict[Tuple[Any, ...], List[float]] = defaultdict(list)
    for r in rows:
        buckets[tuple(r[k] for k in keys)].append(float(r[score_key]))
    return {k: mean(v) for k, v in buckets.items()}


def run_placebo_for_model_axis(
    model_key: str,
    axis_key: str,
    rows: List[Dict[str, Any]],
    response_reps: torch.Tensor,
    pos_reps: torch.Tensor,
    neg_reps: torch.Tensor,
    concept_data: Dict[str, Any],
    n_placebo: int,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rng = random.Random(seed)
    replicate_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    real_religion = aggregate_for_placebo(rows, "score_orig", ["religion"])
    real_rpg = aggregate_for_placebo(rows, "score_orig", ["religion", "prompt_group"])
    real_religion_vals = list(real_religion.values())
    real_rpg_vals = list(real_rpg.values())
    real_religion_range = max(real_religion_vals) - min(real_religion_vals) if real_religion_vals else float("nan")
    real_rpg_range = max(real_rpg_vals) - min(real_rpg_vals) if real_rpg_vals else float("nan")

    for rep in tqdm(range(n_placebo), desc=f"placebo {model_key}/{axis_key}", leave=False):
        lam_p, mid_p, meta = build_placebo_seed_tensors(pos_reps, neg_reps, concept_data, rng)
        placebo_scores = score_reps_mean_over_seeds(response_reps, lam_p, mid_p)
        scored = []
        for r, s in zip(rows, placebo_scores):
            rr = dict(r)
            rr["score_placebo"] = float(s)
            scored.append(rr)

        pl_religion = aggregate_for_placebo(scored, "score_placebo", ["religion"])
        pl_rpg = aggregate_for_placebo(scored, "score_placebo", ["religion", "prompt_group"])
        rel_keys = sorted(set(real_religion.keys()) & set(pl_religion.keys()))
        rpg_keys = sorted(set(real_rpg.keys()) & set(pl_rpg.keys()))
        real_rel = [real_religion[k] for k in rel_keys]
        plac_rel = [pl_religion[k] for k in rel_keys]
        real_rp = [real_rpg[k] for k in rpg_keys]
        plac_rp = [pl_rpg[k] for k in rpg_keys]
        pl_religion_range = max(plac_rel) - min(plac_rel) if plac_rel else float("nan")
        pl_rpg_range = max(plac_rp) - min(plac_rp) if plac_rp else float("nan")

        replicate_rows.append({
            "model": model_key,
            "axis": axis_key,
            "placebo_rep": rep,
            "real_religion_range": real_religion_range,
            "placebo_religion_range": pl_religion_range,
            "real_rpg_range": real_rpg_range,
            "placebo_rpg_range": pl_rpg_range,
            "spearman_real_placebo_religion": corr(real_rel, plac_rel, kind="spearman"),
            "spearman_real_placebo_rpg": corr(real_rp, plac_rp, kind="spearman"),
            "placebo_val_sign_accuracy_positive": meta["placebo_val_sign_accuracy_positive"],
            "placebo_val_mean_score": meta["placebo_val_mean_score"],
            "n_val_scores": meta["n_val_scores"],
        })

    rel_ranges = [r["placebo_religion_range"] for r in replicate_rows]
    rpg_ranges = [r["placebo_rpg_range"] for r in replicate_rows]
    summary_rows.append({
        "model": model_key,
        "axis": axis_key,
        "n_placebo": n_placebo,
        "real_religion_range": real_religion_range,
        "placebo_religion_range_mean": mean(rel_ranges),
        "placebo_religion_range_sd": stdev(rel_ranges),
        "p_placebo_religion_range_ge_real": mean([1.0 if x >= real_religion_range else 0.0 for x in rel_ranges]),
        "real_rpg_range": real_rpg_range,
        "placebo_rpg_range_mean": mean(rpg_ranges),
        "placebo_rpg_range_sd": stdev(rpg_ranges),
        "p_placebo_rpg_range_ge_real": mean([1.0 if x >= real_rpg_range else 0.0 for x in rpg_ranges]),
        "mean_spearman_real_placebo_religion": mean([r["spearman_real_placebo_religion"] for r in replicate_rows]),
        "mean_spearman_real_placebo_rpg": mean([r["spearman_real_placebo_rpg"] for r in replicate_rows]),
        "mean_placebo_val_sign_accuracy_positive": mean([r["placebo_val_sign_accuracy_positive"] for r in replicate_rows]),
    })
    return replicate_rows, summary_rows


# ============================================================
# MAIN PIPELINE
# ============================================================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=list(MODELS.keys()))
    ap.add_argument("--axes", nargs="*", default=list(AXIS_PAIR_FILENAMES.keys()))
    ap.add_argument("--pairs-dir", type=Path, default=DEFAULT_PAIRS_DIR)
    ap.add_argument("--concepts-root", type=Path, default=DEFAULT_CONCEPTS_ROOT)
    ap.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    ap.add_argument("--representation-key", default=REPRESENTATION_KEY)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    ap.add_argument("--skip-lexical-baselines", action="store_true")
    ap.add_argument("--skip-residualization", action="store_true")
    ap.add_argument("--run-placebo", action="store_true", help="Run random label-shuffled placebo axes. Extra encoding/runtime.")
    ap.add_argument("--n-placebo", type=int, default=100)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    args.outdir.mkdir(parents=True, exist_ok=True)

    response_summaries: List[Dict[str, Any]] = []
    aggregate_summaries: List[Dict[str, Any]] = []
    aggregate_cells: List[Dict[str, Any]] = []
    ablated_rows_all: List[Dict[str, Any]] = []
    placebo_reps_all: List[Dict[str, Any]] = []
    placebo_summaries_all: List[Dict[str, Any]] = []

    model_bar = tqdm(args.models, desc="models", position=0)
    for model_key in model_bar:
        if model_key not in MODELS:
            print(f"[skip] unknown model key: {model_key}")
            continue
        model_id = MODELS[model_key]
        model_bar.set_postfix({"model": model_key})
        tokenizer, model = load_model_and_tokenizer(model_id)

        try:
            for axis_key in tqdm(args.axes, desc=f"{model_key} axes", position=1, leave=False):
                if axis_key not in AXIS_PAIR_FILENAMES:
                    print(f"[skip] unknown axis key: {axis_key}")
                    continue
                full_json = args.results_root / model_key / args.representation_key / axis_key / "full.json"
                concepts_file = args.concepts_root / model_key / args.representation_key / axis_key / "concepts.pt"
                pairs_file = pair_path_for_axis(args.pairs_dir, axis_key)
                missing = [str(p) for p in [full_json, concepts_file, pairs_file] if not p.exists()]
                if missing:
                    print(f"[skip] missing files for {model_key}/{axis_key}: {missing}")
                    continue

                all_rows = parse_full_json(full_json)
                rows_full, meta = run_full_audit_ablation_for_model_axis(
                    model_key=model_key,
                    axis_key=axis_key,
                    all_rows=all_rows,
                    model=model,
                    tokenizer=tokenizer,
                    pairs_file=pairs_file,
                    concepts_file=concepts_file,
                    batch_size=args.batch_size,
                )
                ablated_rows_all.extend(rows_full)

                # Summaries for full audit and cue-present subset.
                subsets = {
                    "full_audit": rows_full,
                    "cue_present": [r for r in rows_full if r["has_axis_cue"]],
                }
                for subset_name, subset_rows in subsets.items():
                    if not subset_rows:
                        response_summaries.append({
                            "model": model_key, "axis": axis_key, "subset": subset_name,
                            "n_total_all": meta["n_total"], "n_cue_all": meta["n_cue"], "n_used": 0,
                            "cue_coverage_all": meta["n_cue"] / meta["n_total"] if meta["n_total"] else float("nan"),
                        })
                        continue
                    response_summaries.append(
                        summarize_response_level(
                            subset_rows, model_key, axis_key, subset_name,
                            meta["n_total"], meta["n_cue"], rng, args.n_boot,
                        )
                    )
                    levels = {
                        "religion": ["religion"],
                        "prompt_group": ["prompt_group"],
                        "religion_prompt_group": ["religion", "prompt_group"],
                    }
                    for level_name, keys in levels.items():
                        summary, cells = summarize_aggregate_stability(
                            subset_rows, model_key, axis_key, subset_name, level_name, keys
                        )
                        aggregate_summaries.append(summary)
                        aggregate_cells.extend(cells)

                if args.run_placebo:
                    # Reuse pair reps and real seed tensors, then encode original responses once.
                    pairs, pos_reps, neg_reps = get_axis_pair_reps(model, tokenizer, pairs_file, batch_size=args.batch_size)
                    lambdas, midpoints, concept_data = build_axis_seed_tensors_from_reps(pos_reps, neg_reps, concepts_file)
                    rows_real_rescored, response_reps = score_real_axis_original_responses(
                        rows_full, model, tokenizer, lambdas, midpoints, batch_size=args.batch_size
                    )
                    # Use full_json score_orig as the real score, but keep rescored values for sanity checking.
                    real_corr = corr(
                        [r["score_orig"] for r in rows_real_rescored],
                        [r["score_rescored_real_axis"] for r in rows_real_rescored],
                        kind="spearman",
                    )
                    reps, summaries = run_placebo_for_model_axis(
                        model_key=model_key,
                        axis_key=axis_key,
                        rows=rows_real_rescored,
                        response_reps=response_reps,
                        pos_reps=pos_reps,
                        neg_reps=neg_reps,
                        concept_data=concept_data,
                        n_placebo=args.n_placebo,
                        seed=args.seed + 100_000 + hash((model_key, axis_key)) % 10_000,
                    )
                    for s in summaries:
                        s["spearman_fulljson_score_vs_rescored_real_axis"] = real_corr
                    placebo_reps_all.extend(reps)
                    placebo_summaries_all.extend(summaries)

        finally:
            unload_model(model)

    # Save ablation outputs.
    write_csv(args.outdir / "response_level_summary.csv", response_summaries)
    write_csv(args.outdir / "aggregate_stability_summary.csv", aggregate_summaries)
    write_csv(args.outdir / "aggregate_cells.csv", aggregate_cells)
    write_csv(args.outdir / "ablated_response_rows.csv", ablated_rows_all)
    axis_response_summary = summarize_by_axis(response_summaries, "response")
    axis_aggregate_summary = summarize_by_axis(aggregate_summaries, "aggregate")
    write_csv(args.outdir / "axis_response_summary.csv", axis_response_summary)
    write_csv(args.outdir / "axis_aggregate_summary.csv", axis_aggregate_summary)

    # Lexical baselines.
    lexical_baseline_summary: List[Dict[str, Any]] = []
    if not args.skip_lexical_baselines:
        lexical_baseline_summary = run_lexical_baselines(ablated_rows_all, args.outdir)
        write_csv(args.outdir / "lexical_baseline_summary.csv", lexical_baseline_summary)

    # Residualization.
    residualized_rows: List[Dict[str, Any]] = []
    residualization_summary: List[Dict[str, Any]] = []
    residualized_cells: List[Dict[str, Any]] = []
    residualized_vs_original_summary: List[Dict[str, Any]] = []
    if not args.skip_residualization:
        residualized_rows, residualization_summary = add_residualized_scores(ablated_rows_all, alpha=10.0)
        if residualized_rows:
            residualized_cells, residualized_vs_original_summary = residualized_aggregate_outputs(residualized_rows)
        write_csv(args.outdir / "residualized_response_rows.csv", residualized_rows)
        write_csv(args.outdir / "residualization_summary.csv", residualization_summary)
        write_csv(args.outdir / "residualized_aggregate_cells.csv", residualized_cells)
        write_csv(args.outdir / "residualized_vs_original_summary.csv", residualized_vs_original_summary)

    # Placebo outputs.
    if args.run_placebo:
        write_csv(args.outdir / "placebo_replicate_summary.csv", placebo_reps_all)
        write_csv(args.outdir / "placebo_summary.csv", placebo_summaries_all)

    payload = {
        "config": {
            "models": args.models,
            "axes": args.axes,
            "pairs_dir": str(args.pairs_dir),
            "concepts_root": str(args.concepts_root),
            "results_root": str(args.results_root),
            "representation_key": args.representation_key,
            "seed": args.seed,
            "n_boot": args.n_boot,
            "run_placebo": args.run_placebo,
            "n_placebo": args.n_placebo,
        },
        "response_level_summary": response_summaries,
        "aggregate_stability_summary": aggregate_summaries,
        "axis_response_summary": axis_response_summary,
        "axis_aggregate_summary": axis_aggregate_summary,
        "lexical_baseline_summary": lexical_baseline_summary,
        "residualization_summary": residualization_summary,
        "residualized_vs_original_summary": residualized_vs_original_summary,
        "placebo_summary": placebo_summaries_all,
    }
    (args.outdir / "robustness_report.json").write_text(
        json.dumps(payload, indent=2, default=json_default), encoding="utf-8"
    )

    print("\n" + "=" * 100)
    print("FULL-AUDIT / CUE-PRESENT ABLATION SUMMARY")
    print("=" * 100)
    header = f"{'model':<8} {'axis':<24} {'subset':<13} {'n_used':>7} {'cov':>7} {'mean_delta':>12} {'flip':>8} {'r_s':>8}"
    print(header)
    print("-" * len(header))
    for r in response_summaries:
        print(
            f"{r.get('model',''):<8} {r.get('axis',''):<24} {r.get('subset',''):<13} "
            f"{int(r.get('n_used',0)):>7d} {safe_float(r.get('cue_coverage_all')) or 0:>7.3f} "
            f"{safe_float(r.get('mean_delta')) or float('nan'):>12.4f} "
            f"{safe_float(r.get('flip_rate')) or 0:>8.3f} "
            f"{safe_float(r.get('spearman_response')) or float('nan'):>8.3f}"
        )

    if lexical_baseline_summary:
        print("\n" + "=" * 100)
        print("LEXICAL-ONLY BASELINE SUMMARY")
        print("=" * 100)
        header = f"{'axis':<24} {'baseline':<25} {'n':>6} {'r_s':>8} {'r2':>9}"
        print(header)
        print("-" * len(header))
        for r in lexical_baseline_summary:
            print(
                f"{r.get('axis',''):<24} {r.get('baseline',''):<25} {int(r.get('n',0)):>6d} "
                f"{safe_float(r.get('spearman_pred_score')) or float('nan'):>8.3f} "
                f"{safe_float(r.get('r2_pred_score')) or float('nan'):>9.3f}"
            )

    if residualized_vs_original_summary:
        print("\n" + "=" * 100)
        print("RESIDUALIZED VS ORIGINAL AGGREGATE SUMMARY")
        print("=" * 100)
        header = f"{'axis':<24} {'level':<28} {'n':>6} {'r_s':>8} {'rank_mv':>8} {'orig_rng':>10} {'resid_rng':>10}"
        print(header)
        print("-" * len(header))
        for r in residualized_vs_original_summary:
            print(
                f"{r.get('axis',''):<24} {r.get('level',''):<28} {int(r.get('n_cells',0)):>6d} "
                f"{safe_float(r.get('spearman_orig_residualized_cells')) or float('nan'):>8.3f} "
                f"{safe_float(r.get('avg_rank_movement')) or float('nan'):>8.3f} "
                f"{safe_float(r.get('orig_range')) or float('nan'):>10.3f} "
                f"{safe_float(r.get('residualized_range')) or float('nan'):>10.3f}"
            )

    if placebo_summaries_all:
        print("\n" + "=" * 100)
        print("PLACEBO AXIS SUMMARY")
        print("=" * 100)
        header = f"{'model':<8} {'axis':<24} {'p_rel>=real':>12} {'p_rpg>=real':>12} {'pl_val_sign':>12}"
        print(header)
        print("-" * len(header))
        for r in placebo_summaries_all:
            print(
                f"{r.get('model',''):<8} {r.get('axis',''):<24} "
                f"{safe_float(r.get('p_placebo_religion_range_ge_real')) or float('nan'):>12.3f} "
                f"{safe_float(r.get('p_placebo_rpg_range_ge_real')) or float('nan'):>12.3f} "
                f"{safe_float(r.get('mean_placebo_val_sign_accuracy_positive')) or float('nan'):>12.3f}"
            )

    print(f"\nSaved outputs to: {args.outdir.resolve()}")


if __name__ == "__main__":
    main()
