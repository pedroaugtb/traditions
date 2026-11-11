#!/usr/bin/env python3
"""
Lexical cue ablation + stability checks for contextual framing axes.

This version keeps the original cue-removal test, but adds the diagnostics that
reviewers usually need for construct-validity claims:
  1. cue coverage by model/axis,
  2. response-level original-vs-ablated correlations,
  3. sign-flip rates,
  4. religion / prompt-group / religion×prompt-group aggregate stability,
  5. per-axis pooled stability summaries,
  6. CSV + JSON outputs for easy paper tables.

The script does NOT try to make the model outputs look better; it reports
whether the main comparisons survive removal of obvious lexical triggers.
"""

import argparse
import csv
import gc
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from scipy.stats import pearsonr, spearmanr, ttest_rel, wilcoxon
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False


# ============================================================
# CONFIG
# ============================================================

PAIRS_DIR = Path("pairs")
CONCEPTS_ROOT = Path("axis_stage1")
RESULTS_ROOT = Path("religion_replies_stage2")
DEFAULT_OUTDIR = Path("lexical_initial_ablation")

MODELS = {
    "qwen": "Qwen/Qwen2.5-7B",
    "olmo": "allenai/OLMo-2-1124-7B-Instruct",
    "gemma": "google/gemma-2-9b-it",
    "llama": "meta-llama/Llama-3.2-3B-Instruct",
}

AXES = {
    "assertive_commitment": {
        "pairs_file": PAIRS_DIR / "assertive_commitment_axis_pairs.json",
        "paper_name": "Assertive commitment",
    },
    "doctrinal_attribution": {
        "pairs_file": PAIRS_DIR / "doctrinal_attribution_axis_pairs.json",
        "paper_name": "Doctrinal attribution",
    },
    "hedging": {
        "pairs_file": PAIRS_DIR / "hedging_axis_pairs_v2.json",
        "paper_name": "Hedging",
    },
    "mythologized": {
        "pairs_file": PAIRS_DIR / "mythologized_axis_pairs_v2.json",
        "paper_name": "Mythologization",
    },
}

REPRESENTATION_KEY = "contextual_last_mean"

# High-salience lexical cues. Keep these intentionally small and obvious.
# Regexes are sorted longest-first before substitution to avoid deleting "may"
# before matching "may be interpreted as", etc.
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
    return float(sum(xs) / len(xs)) if xs else float("nan")


def stdev(xs: Sequence[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def cohen_dz(deltas: Sequence[float]) -> float:
    sd = stdev(deltas)
    return 0.0 if sd == 0 else mean(deltas) / sd


def abs_cohen_dz(deltas: Sequence[float]) -> float:
    return abs(cohen_dz(deltas))


def corr(xs: Sequence[float], ys: Sequence[float], kind: str = "spearman") -> float:
    if len(xs) < 3 or len(ys) < 3:
        return float("nan")
    if len(set(xs)) < 2 or len(set(ys)) < 2:
        return float("nan")
    if SCIPY_OK:
        try:
            return float(spearmanr(xs, ys).correlation if kind == "spearman" else pearsonr(xs, ys).statistic)
        except Exception:
            return float("nan")
    # Fallback: Pearson only. For Spearman, rank then Pearson.
    def rank(vals: Sequence[float]) -> List[float]:
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        ranks = [0.0] * len(vals)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg_rank = (i + j) / 2.0
            for k in range(i, j + 1):
                ranks[order[k]] = avg_rank
            i = j + 1
        return ranks
    x = rank(xs) if kind == "spearman" else list(xs)
    y = rank(ys) if kind == "spearman" else list(ys)
    mx, my = mean(x), mean(y)
    sx, sy = stdev(x), stdev(y)
    if sx == 0 or sy == 0:
        return float("nan")
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / ((len(x) - 1) * sx * sy)


def bootstrap_ci(
    values: Sequence[float],
    rng: random.Random,
    n_boot: int = 1000,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    vals = list(values)
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
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


# ============================================================
# DATA LOADING
# ============================================================

def load_axis_file(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "pairs" not in data:
        raise ValueError(f"{path} must be a dict JSON containing key 'pairs'.")
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
        for item in rinfo.get("items", []):
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
                    "generation_index": gen_idx,
                    "response_text": response,
                    "score_orig": float(mean_score),
                })
    return rows


# ============================================================
# CUE ABLATION
# ============================================================

def compile_axis_patterns(axis_key: str) -> List[Tuple[str, re.Pattern]]:
    # Longest regex first prevents partial deletion of multi-word cues.
    cues = sorted(AXIS_CUES[axis_key], key=len, reverse=True)
    return [(pat, re.compile(pat, flags=re.IGNORECASE)) for pat in cues]


def matched_cues(text: str, patterns: List[Tuple[str, re.Pattern]]) -> List[str]:
    hits = []
    for pat, compiled in patterns:
        if compiled.search(text):
            hits.append(pat)
    return hits


def cue_present(text: str, patterns: List[Tuple[str, re.Pattern]]) -> bool:
    return any(compiled.search(text) for _, compiled in patterns)


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
def get_text_representations_contextual(
    texts: List[str],
    tokenizer,
    model,
    batch_size: int = 8,
) -> torch.Tensor:
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

    return torch.cat(reps, dim=0) if reps else torch.empty((0, 0))


def build_axis_seed_tensors(
    model,
    tokenizer,
    pairs_file: Path,
    concepts_file: Path,
    batch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    pairs = load_pairs(pairs_file)
    concept_data = load_concepts(concepts_file)

    if concept_data.get("rep_family") != "contextual_hidden":
        raise ValueError(f"Expected contextual_hidden, got {concept_data.get('rep_family')} in {concepts_file}")
    if concept_data.get("rep_mode") != "mean":
        raise ValueError(f"Expected rep_mode=mean, got {concept_data.get('rep_mode')} in {concepts_file}")

    pos_texts = [p["pos"] for p in pairs]
    neg_texts = [p["neg"] for p in pairs]

    pos_reps = get_text_representations_contextual(pos_texts, tokenizer, model, batch_size=batch_size)
    neg_reps = get_text_representations_contextual(neg_texts, tokenizer, model, batch_size=batch_size)

    lambdas = []
    midpoints = []
    for seed_entry in concept_data["concepts_per_seed"]:
        train_idxs = seed_entry["train_idxs"]
        lam = seed_entry["lambda_bar_unit_euclid"].float().cpu()
        midpoint = (pos_reps[train_idxs].mean(dim=0) + neg_reps[train_idxs].mean(dim=0)) / 2.0
        lambdas.append(lam)
        midpoints.append(midpoint)

    return torch.stack(lambdas, dim=0), torch.stack(midpoints, dim=0)


def score_reps_mean_over_seeds(
    reps: torch.Tensor,
    lambdas: torch.Tensor,
    midpoints: torch.Tensor,
) -> List[float]:
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
    n_total: int,
    n_cue: int,
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
    delta_over_orig_sd = mean(deltas) / orig_sd if orig_sd > 0 else float("nan")

    return {
        "model": model_key,
        "axis": axis_key,
        "n_total": n_total,
        "n_cue": n_cue,
        "n_used": len(rows),
        "cue_coverage": n_cue / n_total if n_total else float("nan"),
        "mean_cue_subs": mean([r["cue_subs"] for r in rows]),
        "mean_distinct_cues": mean([r["cue_distinct"] for r in rows]),
        "mean_orig": mean(orig),
        "mean_ablated": mean(abl),
        "mean_delta": mean(deltas),
        "mean_delta_ci_low": delta_lo,
        "mean_delta_ci_high": delta_hi,
        "mean_abs_delta": mean(abs_deltas),
        "delta_over_orig_sd": delta_over_orig_sd,
        "flip_rate": mean(flips),
        "flip_ci_low": flip_lo,
        "flip_ci_high": flip_hi,
        "cohen_dz": cohen_dz(deltas),
        "abs_cohen_dz": abs_cohen_dz(deltas),
        "pearson_response": corr(orig, abl, kind="pearson"),
        "spearman_response": corr(orig, abl, kind="spearman"),
        "p_ttest": p_t,
        "p_wilcoxon": p_w,
    }


def aggregate_means(rows: List[Dict[str, Any]], keys: Sequence[str]) -> List[Dict[str, Any]]:
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
    # Descending rank order indices.
    return sorted(range(len(values)), key=lambda i: values[i], reverse=True)


def avg_rank_movement(orig: Sequence[float], abl: Sequence[float]) -> float:
    if len(orig) < 2:
        return float("nan")
    ro = rank_order(orig)
    ra = rank_order(abl)
    pos_o = {idx: rank for rank, idx in enumerate(ro)}
    pos_a = {idx: rank for rank, idx in enumerate(ra)}
    return mean([abs(pos_o[i] - pos_a[i]) for i in range(len(orig))])


def summarize_aggregate_stability(
    rows: List[Dict[str, Any]],
    model_key: str,
    axis_key: str,
    level_name: str,
    keys: Sequence[str],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    aggs = aggregate_means(rows, keys)
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
        a.update({"model": model_key, "axis": axis_key, "level": level_name})
    return summary, aggs


def mean_of_key(rows: List[Dict[str, Any]], key: str) -> float:
    vals = [safe_float(r.get(key)) for r in rows]
    vals = [v for v in vals if v is not None]
    return mean(vals)


def summarize_by_axis(rows: List[Dict[str, Any]], kind: str) -> List[Dict[str, Any]]:
    out = []
    for axis in sorted({r["axis"] for r in rows}):
        sub = [r for r in rows if r["axis"] == axis]
        rec = {"axis": axis, "n_model_summaries": len(sub), "summary_kind": kind}
        for k in [
            "cue_coverage", "mean_delta", "mean_abs_delta", "delta_over_orig_sd",
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
# MAIN PIPELINE
# ============================================================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=list(MODELS.keys()))
    ap.add_argument("--axes", nargs="*", default=list(AXES.keys()))
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-samples", type=int, default=0,
                    help="Max cue-present responses per model/axis. Use 0 for all; default uses all.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    args.outdir.mkdir(parents=True, exist_ok=True)

    response_summaries: List[Dict[str, Any]] = []
    aggregate_summaries: List[Dict[str, Any]] = []
    aggregate_cells: List[Dict[str, Any]] = []
    ablated_rows_all: List[Dict[str, Any]] = []

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
                if axis_key not in AXES:
                    print(f"[skip] unknown axis key: {axis_key}")
                    continue

                full_json = RESULTS_ROOT / model_key / REPRESENTATION_KEY / axis_key / "full.json"
                concepts_file = CONCEPTS_ROOT / model_key / REPRESENTATION_KEY / axis_key / "concepts.pt"
                pairs_file = AXES[axis_key]["pairs_file"]

                missing = [str(p) for p in [full_json, concepts_file, pairs_file] if not p.exists()]
                if missing:
                    print(f"[skip] missing files for {model_key}/{axis_key}: {missing}")
                    continue

                patterns = compile_axis_patterns(axis_key)
                all_rows = parse_full_json(full_json)
                n_total = len(all_rows)
                cue_rows = []
                for r in all_rows:
                    hits = matched_cues(r["response_text"], patterns)
                    if hits:
                        r = dict(r)
                        r["matched_cue_patterns"] = ";".join(hits)
                        cue_rows.append(r)
                n_cue = len(cue_rows)

                if n_cue == 0:
                    empty = {
                        "model": model_key, "axis": axis_key,
                        "n_total": n_total, "n_cue": 0, "n_used": 0,
                        "cue_coverage": 0.0 if n_total else float("nan"),
                    }
                    response_summaries.append(empty)
                    continue

                if args.max_samples and n_cue > args.max_samples:
                    cue_rows = rng.sample(cue_rows, args.max_samples)

                lambdas, midpoints = build_axis_seed_tensors(
                    model=model,
                    tokenizer=tokenizer,
                    pairs_file=pairs_file,
                    concepts_file=concepts_file,
                    batch_size=args.batch_size,
                )

                ablated_texts = []
                for i, r in enumerate(cue_rows):
                    ablated, n_subs, n_distinct = ablate_text(r["response_text"], patterns)
                    if len(ablated.strip()) < 8:
                        ablated = r["response_text"]
                    cue_rows[i]["response_ablated"] = ablated
                    cue_rows[i]["cue_subs"] = n_subs
                    cue_rows[i]["cue_distinct"] = n_distinct
                    ablated_texts.append(ablated)

                ablated_reps = get_text_representations_contextual(
                    ablated_texts, tokenizer, model, batch_size=args.batch_size
                )
                ablated_scores = score_reps_mean_over_seeds(ablated_reps, lambdas, midpoints)

                rows_scored = []
                for r, s_abl in zip(cue_rows, ablated_scores):
                    out = dict(r)
                    out.update({
                        "model": model_key,
                        "axis": axis_key,
                        "score_ablated": float(s_abl),
                        "delta": float(s_abl - r["score_orig"]),
                        "sign_flip": int((r["score_orig"] > 0) != (s_abl > 0)),
                    })
                    rows_scored.append(out)
                    ablated_rows_all.append(out)

                response_summaries.append(
                    summarize_response_level(rows_scored, model_key, axis_key, n_total, n_cue, rng, args.n_boot)
                )

                levels = {
                    "religion": ["religion"],
                    "prompt_group": ["prompt_group"],
                    "religion_prompt_group": ["religion", "prompt_group"],
                }
                for level_name, keys in levels.items():
                    summary, cells = summarize_aggregate_stability(
                        rows_scored, model_key, axis_key, level_name, keys
                    )
                    aggregate_summaries.append(summary)
                    aggregate_cells.extend(cells)

        finally:
            unload_model(model)

    # Save detailed outputs.
    write_csv(args.outdir / "response_level_summary.csv", response_summaries)
    write_csv(args.outdir / "aggregate_stability_summary.csv", aggregate_summaries)
    write_csv(args.outdir / "aggregate_cells.csv", aggregate_cells)
    write_csv(args.outdir / "ablated_response_rows.csv", ablated_rows_all)

    axis_response_summary = summarize_by_axis(response_summaries, "response")
    axis_aggregate_summary = summarize_by_axis(aggregate_summaries, "aggregate")
    write_csv(args.outdir / "axis_response_summary.csv", axis_response_summary)
    write_csv(args.outdir / "axis_aggregate_summary.csv", axis_aggregate_summary)

    payload = {
        "config": {
            "models": args.models,
            "axes": args.axes,
            "representation_key": REPRESENTATION_KEY,
            "max_samples": args.max_samples,
            "seed": args.seed,
            "n_boot": args.n_boot,
        },
        "response_level_summary": response_summaries,
        "aggregate_stability_summary": aggregate_summaries,
        "axis_response_summary": axis_response_summary,
        "axis_aggregate_summary": axis_aggregate_summary,
    }
    (args.outdir / "lexical_ablation_report.json").write_text(
        json.dumps(payload, indent=2, default=json_default), encoding="utf-8"
    )

    # Console tables.
    print("\n" + "=" * 130)
    print("RESPONSE-LEVEL LEXICAL ABLATION SUMMARY")
    print("=" * 130)
    header = (
        f"{'model':<8} {'axis':<24} {'n_total':>7} {'n_cue':>7} {'cov':>7} {'n_used':>7} "
        f"{'mean_delta':>11} {'flip':>8} {'dz':>8} {'r_s_resp':>9} {'mean_abs_d':>11}"
    )
    print(header)
    print("-" * len(header))
    for r in response_summaries:
        print(
            f"{r.get('model',''):<8} {r.get('axis',''):<24} "
            f"{int(r.get('n_total',0)):>7d} {int(r.get('n_cue',0)):>7d} {safe_float(r.get('cue_coverage')) or 0:>7.3f} "
            f"{int(r.get('n_used',0)):>7d} {safe_float(r.get('mean_delta')) or float('nan'):>11.4f} "
            f"{safe_float(r.get('flip_rate')) or 0:>8.3f} {safe_float(r.get('cohen_dz')) or float('nan'):>8.3f} "
            f"{safe_float(r.get('spearman_response')) or float('nan'):>9.3f} {safe_float(r.get('mean_abs_delta')) or float('nan'):>11.4f}"
        )

    print("\n" + "=" * 100)
    print("AXIS RESPONSE SUMMARY, AVERAGED ACROSS MODELS")
    print("=" * 100)
    header = f"{'axis':<24} {'coverage':>9} {'mean_delta':>12} {'flip':>8} {'abs_dz':>8} {'r_s_resp':>9}"
    print(header)
    print("-" * len(header))
    for r in axis_response_summary:
        print(
            f"{r['axis']:<24} {safe_float(r.get('mean_cue_coverage')) or float('nan'):>9.3f} "
            f"{safe_float(r.get('mean_mean_delta')) or float('nan'):>12.4f} "
            f"{safe_float(r.get('mean_flip_rate')) or float('nan'):>8.3f} "
            f"{safe_float(r.get('mean_abs_cohen_dz')) or float('nan'):>8.3f} "
            f"{safe_float(r.get('mean_spearman_response')) or float('nan'):>9.3f}"
        )

    print("\n" + "=" * 118)
    print("AGGREGATE STABILITY SUMMARY, AVERAGED ACROSS MODEL/AXIS WITHIN EACH LEVEL")
    print("=" * 118)
    for level in ["religion", "prompt_group", "religion_prompt_group"]:
        sub = [r for r in aggregate_summaries if r["level"] == level]
        print(f"\n[{level}]")
        header = f"{'axis':<24} {'r_s_cells':>10} {'same_sign':>10} {'rank_move':>10} {'same_top':>9} {'same_bottom':>12} {'mean_abs_shift':>15}"
        print(header)
        print("-" * len(header))
        for axis in args.axes:
            ss = [r for r in sub if r["axis"] == axis]
            if not ss:
                continue
            print(
                f"{axis:<24} {mean_of_key(ss, 'spearman_cells'):>10.3f} "
                f"{mean_of_key(ss, 'same_sign_cell_rate'):>10.3f} "
                f"{mean_of_key(ss, 'avg_rank_movement'):>10.3f} "
                f"{mean_of_key(ss, 'same_top_cell'):>9.3f} "
                f"{mean_of_key(ss, 'same_bottom_cell'):>12.3f} "
                f"{mean_of_key(ss, 'mean_abs_cell_shift'):>15.4f}"
            )

    print(f"\nSaved outputs to: {args.outdir.resolve()}")


if __name__ == "__main__":
    main()
