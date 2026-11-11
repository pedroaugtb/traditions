import gc
import json
import random
import statistics
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# ============================================================
# CONFIG
# ============================================================

PAIRS_DIR = Path("pairs")
RESULTS_ROOT = Path("axis_stage1")

AXES = {
    "assertive_commitment": PAIRS_DIR / "assertive_commitment_axis_pairs.json",
    "doctrinal_attribution": PAIRS_DIR / "doctrinal_attribution_axis_pairs.json",
    "hedging": PAIRS_DIR / "hedging_axis_pairs_v2.json",
    "mythologized": PAIRS_DIR / "mythologized_axis_pairs_v2.json",
}

MODELS = {
    "qwen": "Qwen/Qwen2.5-7B",
    "olmo": "allenai/OLMo-2-1124-7B-Instruct",
    "gemma": "google/gemma-2-9b-it",
    "llama": "meta-llama/Llama-3.2-3B-Instruct"
}

# Two representation families:
# 1) output_projection_mean  -> current lexical/output-space proxy
# 2) contextual_last_mean    -> contextual hidden states (last layer, mean pooling)
REPRESENTATIONS = {
    "output_projection_mean": {
        "rep_family": "output_projection",
        "rep_mode": "mean",
    },
    "contextual_last_mean": {
        "rep_family": "contextual_hidden",
        "rep_mode": "mean",
    },
}

TEXT_FIELD_ASSERTIVE = "assertive"
TEXT_FIELD_HEDGED = "hedged"

# Concept / metric settings
METRIC_MODE = "full"   # "full" or "diag"
VOCAB_SAMPLE_SIZE = 20000
COV_BATCH_SIZE = 256
REG_EPS = 1e-3
RANDOM_BASELINE_PAIRS = 512

# Multi-seed settings
BASE_SEED = 42
N_RANDOM_SEEDS = 30
TRAIN_FRACTION = 0.7

# Bootstrap over seeds
BOOTSTRAP_RESAMPLES = 500
CI_LEVEL = 0.95

SAVE_CONCEPTS_PT = True


# ============================================================
# IO
# ============================================================

def load_pair_items(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))

    if isinstance(data, dict) and "pairs" in data:
        pairs = data["pairs"]
    elif isinstance(data, list):
        pairs = data
    else:
        raise ValueError(f"{path} must be a list or a dict with key 'pairs'.")

    out = []
    for i, item in enumerate(pairs):
        if not isinstance(item, dict):
            raise ValueError(f"Pair {i} in {path} must be a dict.")
        if TEXT_FIELD_ASSERTIVE not in item or TEXT_FIELD_HEDGED not in item:
            raise ValueError(
                f"Pair {i} in {path} must contain '{TEXT_FIELD_ASSERTIVE}' and '{TEXT_FIELD_HEDGED}'."
            )
        out.append({
            "id": item.get("id", f"pair_{i}"),
            "assertive": item[TEXT_FIELD_ASSERTIVE],
            "hedged": item[TEXT_FIELD_HEDGED],
            "religion": item.get("religion", None),
            "topic": item.get("topic", None),
        })
    return out


def split_indices_train_val(n_items: int, seed: int) -> Tuple[List[int], List[int]]:
    idxs = list(range(n_items))
    rng = random.Random(seed)
    rng.shuffle(idxs)

    n_train = int(n_items * TRAIN_FRACTION)
    train_idxs = idxs[:n_train]
    val_idxs = idxs[n_train:]
    return train_idxs, val_idxs


def generate_random_seeds(base_seed: int, n: int) -> List[int]:
    rng = random.Random(base_seed)
    return rng.sample(range(1, 10_000_000), n)


# ============================================================
# MODEL
# ============================================================

def is_phi_model_name(name: str) -> bool:
    s = (name or "").lower()
    return "microsoft/phi" in s or "phi-3.5-mini-instruct" in s or "phi3" in s


def load_model_and_tokenizer(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="auto",
    )
    model.eval()

    if is_phi_model_name(model_id):
        try:
            model.generation_config.use_cache = False
        except Exception:
            pass

    return tokenizer, model


def unload_model(model=None):
    try:
        del model
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def first_device(model):
    return next(model.parameters()).device


def get_output_embedding_weight(model) -> torch.Tensor:
    head = model.get_output_embeddings()
    if head is None or not hasattr(head, "weight"):
        raise ValueError("Could not find output embedding weight via model.get_output_embeddings().")
    return head.weight


# ============================================================
# REPRESENTATIONS
# ============================================================

def encode_text_tokens(text: str, tokenizer) -> List[int]:
    ids = tokenizer(text, add_special_tokens=False).input_ids
    if len(ids) == 0:
        raise ValueError(f"Text {text!r} tokenized to 0 tokens.")
    return ids


@torch.no_grad()
def get_text_representation(
    text: str,
    tokenizer,
    model,
    rep_family: str,
    rep_mode: str,
) -> torch.Tensor:
    if rep_family == "output_projection":
        W = get_output_embedding_weight(model)
        token_ids = encode_text_tokens(text, tokenizer)
        ids = torch.tensor(token_ids, device=W.device, dtype=torch.long)

        token_reps = W.index_select(0, ids).detach().float().cpu()  # [T, H]

        if rep_mode == "mean":
            return token_reps.mean(dim=0)
        if rep_mode == "last":
            return token_reps[-1]
        if rep_mode == "mean_last2":
            return token_reps[-1] if token_reps.shape[0] == 1 else token_reps[-2:].mean(dim=0)

        raise ValueError(f"Unknown output_projection rep_mode: {rep_mode}")

    if rep_family == "contextual_hidden":
        dev = first_device(model)
        inputs = tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=False,
        ).to(dev)

        outputs = model(
            **inputs,
            output_hidden_states=True,
            use_cache=False,
        )
        hidden_states = outputs.hidden_states

        if hidden_states is None:
            raise ValueError("Model did not return hidden_states.")

        last = hidden_states[-1][0].detach().float().cpu()  # [T, H]

        if rep_mode == "mean":
            return last.mean(dim=0)
        if rep_mode == "last_token":
            return last[-1]
        if rep_mode == "mean_last4":
            if len(hidden_states) < 4:
                stacked = torch.stack([h[0].detach().float().cpu() for h in hidden_states[1:]], dim=0)
            else:
                stacked = torch.stack([h[0].detach().float().cpu() for h in hidden_states[-4:]], dim=0)
            return stacked.mean(dim=0).mean(dim=0)

        raise ValueError(f"Unknown contextual_hidden rep_mode: {rep_mode}")

    raise ValueError(f"Unknown rep_family: {rep_family}")


@torch.no_grad()
def get_text_representations(
    texts: List[str],
    tokenizer,
    model,
    rep_family: str,
    rep_mode: str,
    desc: str = "",
) -> torch.Tensor:
    reps = []
    iterator = tqdm(texts, desc=desc, leave=False) if desc else texts
    for text in iterator:
        reps.append(get_text_representation(text, tokenizer, model, rep_family, rep_mode))
    return torch.stack(reps, dim=0)


# ============================================================
# CAUSAL METRIC APPROXIMATION
# ============================================================

@torch.no_grad()
def estimate_covariance_from_vocab_sample(
    model,
    vocab_size: int,
    sample_size: int,
    batch_size: int,
    seed: int = 42,
) -> torch.Tensor:
    rng = random.Random(seed)
    sample_ids = list(range(vocab_size))
    rng.shuffle(sample_ids)
    sample_ids = sample_ids[:min(sample_size, vocab_size)]

    W = get_output_embedding_weight(model)
    hidden_size = W.shape[1]

    sum_x = torch.zeros(hidden_size, dtype=torch.float64)
    sum_xx = torch.zeros(hidden_size, hidden_size, dtype=torch.float64)
    n = 0

    batch_iter = range(0, len(sample_ids), batch_size)
    for start in tqdm(batch_iter, desc="Estimating covariance", leave=False):
        batch_ids = sample_ids[start:start + batch_size]
        ids = torch.tensor(batch_ids, device=W.device, dtype=torch.long)

        X = W.index_select(0, ids).detach().to(dtype=torch.float32)
        X_cpu = X.cpu().to(dtype=torch.float64)

        sum_x += X_cpu.sum(dim=0)
        sum_xx += X_cpu.T @ X_cpu
        n += X_cpu.shape[0]

    mean = sum_x / n
    cov = (sum_xx - n * torch.outer(mean, mean)) / max(n - 1, 1)
    return cov


def build_metric_solver(cov: torch.Tensor, mode: str = "full", reg_eps: float = 1e-3):
    d = cov.shape[0]
    scale = cov.diag().mean().item()
    reg = reg_eps * max(scale, 1e-8)

    if mode == "diag":
        diag = cov.diag().clone() + reg

        def apply_M(v: torch.Tensor) -> torch.Tensor:
            return v / diag.to(v.dtype)

        meta = {"mode": "diag", "reg": reg}
        return apply_M, meta

    cov_reg = cov + reg * torch.eye(d, dtype=cov.dtype)
    L = torch.linalg.cholesky(cov_reg)

    def apply_M(v: torch.Tensor) -> torch.Tensor:
        orig_dtype = v.dtype
        x = torch.cholesky_solve(v.to(torch.float64).unsqueeze(1), L).squeeze(1)
        return x.to(orig_dtype)

    meta = {"mode": "full", "reg": reg}
    return apply_M, meta


# ============================================================
# CONCEPT / EVAL
# ============================================================

def causal_normalize(gamma_vec: torch.Tensor, apply_M):
    Mg = apply_M(gamma_vec)
    norm_sq = torch.dot(gamma_vec, Mg).item()
    norm_sq = max(norm_sq, 1e-12)
    norm = norm_sq ** 0.5
    return gamma_vec / norm, norm


def build_concept_direction(
    pos_gamma: torch.Tensor,
    neg_gamma: torch.Tensor,
    apply_M,
):
    diffs = pos_gamma - neg_gamma
    gamma_raw = diffs.mean(dim=0)
    gamma_bar, gamma_causal_norm = causal_normalize(gamma_raw, apply_M)

    lambda_bar = apply_M(gamma_bar)
    lambda_bar_unit_euclid = lambda_bar / (lambda_bar.norm(p=2) + 1e-12)

    return {
        "diffs": diffs,
        "gamma_raw": gamma_raw,
        "gamma_bar": gamma_bar,
        "lambda_bar": lambda_bar,
        "lambda_bar_unit_euclid": lambda_bar_unit_euclid,
        "gamma_causal_norm_before_normalization": gamma_causal_norm,
    }


def leave_one_out_scores(diffs: torch.Tensor, apply_M) -> List[float]:
    n = diffs.shape[0]
    scores = []

    if n == 1:
        gamma_bar, _ = causal_normalize(diffs[0], apply_M)
        lambda_bar = apply_M(gamma_bar)
        scores.append(float(torch.dot(diffs[0], lambda_bar).item()))
        return scores

    full_sum = diffs.sum(dim=0)
    for i in range(n):
        loo_mean = (full_sum - diffs[i]) / (n - 1)
        gamma_bar_loo, _ = causal_normalize(loo_mean, apply_M)
        lambda_bar_loo = apply_M(gamma_bar_loo)
        score_i = torch.dot(diffs[i], lambda_bar_loo).item()
        scores.append(float(score_i))
    return scores


@torch.no_grad()
def random_baseline_scores(model, vocab_size: int, lambda_bar: torch.Tensor, n_pairs: int, seed: int):
    rng = random.Random(seed)
    W = get_output_embedding_weight(model)
    scores = []

    for _ in range(n_pairs):
        a = rng.randrange(vocab_size)
        b = rng.randrange(vocab_size)
        if a == b:
            continue
        ids = torch.tensor([a, b], device=W.device, dtype=torch.long)
        gamma = W.index_select(0, ids).detach().float().cpu()
        diff = gamma[0] - gamma[1]
        score = torch.dot(diff, lambda_bar).item()
        scores.append(float(score))

    return scores


def score_diffs_against_lambda(diffs: torch.Tensor, lambda_bar: torch.Tensor) -> List[float]:
    return [float(torch.dot(d, lambda_bar).item()) for d in diffs]


# ============================================================
# STATS
# ============================================================

def summarize_values(values: List[float]) -> Dict[str, Any]:
    if len(values) == 0:
        return {
            "n": 0,
            "mean": 0.0,
            "std": 0.0,
            "median": 0.0,
            "min": 0.0,
            "max": 0.0,
        }

    x = [float(v) for v in values]
    return {
        "n": len(x),
        "mean": float(statistics.mean(x)),
        "std": float(statistics.pstdev(x)) if len(x) > 1 else 0.0,
        "median": float(statistics.median(x)),
        "min": float(min(x)),
        "max": float(max(x)),
    }


def bootstrap_mean_ci(values: List[float], n_boot: int = 5000, ci_level: float = 0.95, seed: int = 42) -> Dict[str, float]:
    x = [float(v) for v in values]
    if len(x) == 0:
        return {"ci_low": 0.0, "ci_high": 0.0}
    if len(x) == 1:
        return {"ci_low": x[0], "ci_high": x[0]}

    rng = random.Random(seed)
    means = []
    n = len(x)

    for _ in range(n_boot):
        sample = [x[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)

    means.sort()
    alpha = 1.0 - ci_level
    lo_idx = int((alpha / 2.0) * n_boot)
    hi_idx = int((1.0 - alpha / 2.0) * n_boot) - 1
    lo_idx = max(0, min(lo_idx, n_boot - 1))
    hi_idx = max(0, min(hi_idx, n_boot - 1))

    return {
        "ci_low": float(means[lo_idx]),
        "ci_high": float(means[hi_idx]),
    }


def summarize_metric_across_seeds(values: List[float], seed: int) -> Dict[str, Any]:
    out = summarize_values(values)
    out.update(bootstrap_mean_ci(values, n_boot=BOOTSTRAP_RESAMPLES, ci_level=CI_LEVEL, seed=seed))
    return out


def sign_accuracy(scores: List[float]) -> float:
    if len(scores) == 0:
        return 0.0
    x = torch.tensor(scores, dtype=torch.float32)
    return float((x > 0).float().mean().item())


def effect_size_vs_baseline(scores: List[float], baseline_scores: List[float]) -> float:
    x = torch.tensor(scores, dtype=torch.float32)
    y = torch.tensor(baseline_scores, dtype=torch.float32)

    if len(x) < 2 or len(y) < 2:
        return 0.0

    mx = x.mean().item()
    my = y.mean().item()
    vx = x.var(unbiased=False).item()
    vy = y.var(unbiased=False).item()

    pooled = ((vx + vy) / 2.0) ** 0.5
    if pooled < 1e-12:
        return 0.0
    return float((mx - my) / pooled)


# ============================================================
# RUN ONE MODEL × ONE AXIS × ONE REPRESENTATION
# ============================================================

def run_one_job(
    model_key: str,
    model_id: str,
    representation_key: str,
    rep_family: str,
    rep_mode: str,
    axis_key: str,
    pairs_path: Path,
):
    out_dir = RESULTS_ROOT / model_key / representation_key / axis_key
    out_dir.mkdir(parents=True, exist_ok=True)

    output_summary = out_dir / "summary.json"
    output_per_seed = out_dir / "per_seed.json"
    output_per_pair_oos = out_dir / "per_pair_oos.json"
    output_concepts_pt = out_dir / "concepts.pt"

    torch.manual_seed(BASE_SEED)
    random.seed(BASE_SEED)

    all_pairs = load_pair_items(pairs_path)
    n_total = len(all_pairs)
    if n_total < 4:
        raise ValueError(f"{pairs_path} needs at least 4 pairs.")

    tokenizer, model = load_model_and_tokenizer(model_id)

    try:
        all_assertive_texts = [p["assertive"] for p in all_pairs]
        all_hedged_texts = [p["hedged"] for p in all_pairs]

        all_assertive_token_counts = [len(encode_text_tokens(t, tokenizer)) for t in all_assertive_texts]
        all_hedged_token_counts = [len(encode_text_tokens(t, tokenizer)) for t in all_hedged_texts]

        all_assertive_reps = get_text_representations(
            all_assertive_texts,
            tokenizer,
            model,
            rep_family=rep_family,
            rep_mode=rep_mode,
            desc=f"{model_key}/{representation_key}/{axis_key} assertive -> rep",
        )
        all_hedged_reps = get_text_representations(
            all_hedged_texts,
            tokenizer,
            model,
            rep_family=rep_family,
            rep_mode=rep_mode,
            desc=f"{model_key}/{representation_key}/{axis_key} hedged -> rep",
        )

        vocab_size = get_output_embedding_weight(model).shape[0]
        hidden_size = get_output_embedding_weight(model).shape[1]

        cov = estimate_covariance_from_vocab_sample(
            model=model,
            vocab_size=vocab_size,
            sample_size=VOCAB_SAMPLE_SIZE,
            batch_size=COV_BATCH_SIZE,
            seed=BASE_SEED,
        )
        apply_M, metric_meta = build_metric_solver(cov, mode=METRIC_MODE, reg_eps=REG_EPS)

        seeds = generate_random_seeds(BASE_SEED, N_RANDOM_SEEDS)

        per_seed_results = []
        per_pair_oos = {p["id"]: [] for p in all_pairs}
        concepts_to_save = []

        all_oos_scores_pooled = []
        all_train_scores_pooled = []
        all_train_loo_scores_pooled = []
        all_baseline_scores_pooled = []

        seed_iter = tqdm(seeds, desc=f"{model_key}/{representation_key}/{axis_key} seeds", leave=False)

        for seed in seed_iter:
            train_idxs, val_idxs = split_indices_train_val(n_total, seed)

            train_pos = all_assertive_reps[train_idxs]
            train_neg = all_hedged_reps[train_idxs]
            val_pos = all_assertive_reps[val_idxs]
            val_neg = all_hedged_reps[val_idxs]

            concept = build_concept_direction(train_pos, train_neg, apply_M)
            train_diffs = concept["diffs"]
            val_diffs = val_pos - val_neg

            train_scores = score_diffs_against_lambda(train_diffs, concept["lambda_bar"])
            val_scores = score_diffs_against_lambda(val_diffs, concept["lambda_bar"])
            train_loo = leave_one_out_scores(train_diffs, apply_M)

            baseline_scores = random_baseline_scores(
                model=model,
                vocab_size=vocab_size,
                lambda_bar=concept["lambda_bar"],
                n_pairs=RANDOM_BASELINE_PAIRS,
                seed=seed + 1000,
            )

            all_train_scores_pooled.extend(train_scores)
            all_oos_scores_pooled.extend(val_scores)
            all_train_loo_scores_pooled.extend(train_loo)
            all_baseline_scores_pooled.extend(baseline_scores)

            seed_result = {
                "seed": seed,
                "n_train": len(train_idxs),
                "n_val": len(val_idxs),
                "train_pair_score_mean": float(statistics.mean(train_scores)),
                "train_pair_score_std": float(statistics.pstdev(train_scores)) if len(train_scores) > 1 else 0.0,
                "train_sign_accuracy_positive": sign_accuracy(train_scores),
                "train_loo_mean": float(statistics.mean(train_loo)),
                "train_loo_std": float(statistics.pstdev(train_loo)) if len(train_loo) > 1 else 0.0,
                "val_pair_score_mean": float(statistics.mean(val_scores)),
                "val_pair_score_std": float(statistics.pstdev(val_scores)) if len(val_scores) > 1 else 0.0,
                "val_sign_accuracy_positive": sign_accuracy(val_scores),
                "baseline_mean": float(statistics.mean(baseline_scores)),
                "baseline_std": float(statistics.pstdev(baseline_scores)) if len(baseline_scores) > 1 else 0.0,
                "train_effect_size_vs_baseline": effect_size_vs_baseline(train_scores, baseline_scores),
                "val_effect_size_vs_baseline": effect_size_vs_baseline(val_scores, baseline_scores),
                "train_pair_ids": [all_pairs[i]["id"] for i in train_idxs],
                "val_pair_ids": [all_pairs[i]["id"] for i in val_idxs],
            }
            per_seed_results.append(seed_result)

            for local_j, global_idx in enumerate(val_idxs):
                pair_id = all_pairs[global_idx]["id"]
                per_pair_oos[pair_id].append({
                    "seed": seed,
                    "score": float(val_scores[local_j]),
                    "assertive": all_pairs[global_idx]["assertive"],
                    "hedged": all_pairs[global_idx]["hedged"],
                    "religion": all_pairs[global_idx]["religion"],
                    "topic": all_pairs[global_idx]["topic"],
                    "assertive_n_tokens": all_assertive_token_counts[global_idx],
                    "hedged_n_tokens": all_hedged_token_counts[global_idx],
                })

            if SAVE_CONCEPTS_PT:
                concepts_to_save.append({
                    "seed": seed,
                    "train_idxs": train_idxs,
                    "val_idxs": val_idxs,
                    "gamma_bar": concept["gamma_bar"],
                    "lambda_bar": concept["lambda_bar"],
                    "lambda_bar_unit_euclid": concept["lambda_bar_unit_euclid"],
                })

        train_mean_values = [x["train_pair_score_mean"] for x in per_seed_results]
        train_sign_values = [x["train_sign_accuracy_positive"] for x in per_seed_results]
        train_loo_values = [x["train_loo_mean"] for x in per_seed_results]
        val_mean_values = [x["val_pair_score_mean"] for x in per_seed_results]
        val_sign_values = [x["val_sign_accuracy_positive"] for x in per_seed_results]
        train_effect_values = [x["train_effect_size_vs_baseline"] for x in per_seed_results]
        val_effect_values = [x["val_effect_size_vs_baseline"] for x in per_seed_results]
        baseline_mean_values = [x["baseline_mean"] for x in per_seed_results]

        aggregated = {
            "train_pair_score_mean_across_seeds": summarize_metric_across_seeds(train_mean_values, BASE_SEED + 1),
            "train_sign_accuracy_across_seeds": summarize_metric_across_seeds(train_sign_values, BASE_SEED + 2),
            "train_loo_mean_across_seeds": summarize_metric_across_seeds(train_loo_values, BASE_SEED + 3),
            "validation_pair_score_mean_across_seeds": summarize_metric_across_seeds(val_mean_values, BASE_SEED + 4),
            "validation_sign_accuracy_across_seeds": summarize_metric_across_seeds(val_sign_values, BASE_SEED + 5),
            "train_effect_size_vs_baseline_across_seeds": summarize_metric_across_seeds(train_effect_values, BASE_SEED + 6),
            "validation_effect_size_vs_baseline_across_seeds": summarize_metric_across_seeds(val_effect_values, BASE_SEED + 7),
            "baseline_mean_across_seeds": summarize_metric_across_seeds(baseline_mean_values, BASE_SEED + 8),
        }

        pooled = {
            "train_scores_pooled": {
                **summarize_values(all_train_scores_pooled),
                "sign_accuracy_positive": sign_accuracy(all_train_scores_pooled),
            },
            "train_loo_scores_pooled": {
                **summarize_values(all_train_loo_scores_pooled),
                "sign_accuracy_positive": sign_accuracy(all_train_loo_scores_pooled),
            },
            "validation_scores_pooled_out_of_sample": {
                **summarize_values(all_oos_scores_pooled),
                "sign_accuracy_positive": sign_accuracy(all_oos_scores_pooled),
            },
            "baseline_scores_pooled": {
                **summarize_values(all_baseline_scores_pooled),
                "sign_accuracy_positive": sign_accuracy(all_baseline_scores_pooled),
            },
        }

        per_pair_oos_summary = []
        for pair in all_pairs:
            pair_id = pair["id"]
            rows = per_pair_oos[pair_id]
            scores = [r["score"] for r in rows]

            entry = {
                "id": pair_id,
                "religion": pair["religion"],
                "topic": pair["topic"],
                "assertive": pair["assertive"],
                "hedged": pair["hedged"],
                "n_out_of_sample_evals": len(scores),
                "out_of_sample_score_summary": summarize_values(scores),
                "out_of_sample_sign_accuracy_positive": sign_accuracy(scores),
            }

            if len(scores) > 0:
                entry["out_of_sample_score_summary"].update(
                    bootstrap_mean_ci(scores, n_boot=BOOTSTRAP_RESAMPLES, ci_level=CI_LEVEL, seed=BASE_SEED + 999)
                )

            per_pair_oos_summary.append(entry)

        summary = {
            "model_key": model_key,
            "model_id": model_id,
            "representation_key": representation_key,
            "rep_family": rep_family,
            "rep_mode": rep_mode,
            "axis_key": axis_key,
            "pairs_file": str(pairs_path),
            "base_seed": BASE_SEED,
            "random_seeds": seeds,
            "n_random_seeds": N_RANDOM_SEEDS,
            "train_fraction": TRAIN_FRACTION,
            "n_total_pairs": n_total,
            "hidden_size": int(hidden_size),
            "metric_mode": metric_meta["mode"],
            "metric_reg": metric_meta["reg"],
            "vocab_sample_size": VOCAB_SAMPLE_SIZE,
            "random_baseline_pairs_per_seed": RANDOM_BASELINE_PAIRS,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "ci_level": CI_LEVEL,
            "aggregated_across_seeds": aggregated,
            "pooled_scores": pooled,
            "notes": [
                f"Each seed defines a different random {int(TRAIN_FRACTION*100)}/{int((1-TRAIN_FRACTION)*100)} train/validation split.",
                "The concept direction is learned only on the train split of that seed.",
                "Validation metrics are fully out-of-sample relative to that seed's concept.",
                "Bootstrap confidence intervals are computed over the per-seed metric distribution.",
                "Because seeds reuse the same finite pool of pairs under different splits, the CIs reflect split variability rather than a population-level guarantee.",
            ],
        }

        output_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        output_per_seed.write_text(json.dumps(per_seed_results, ensure_ascii=False, indent=2), encoding="utf-8")
        output_per_pair_oos.write_text(json.dumps(per_pair_oos_summary, ensure_ascii=False, indent=2), encoding="utf-8")

        if SAVE_CONCEPTS_PT:
            torch.save(
                {
                    "model_key": model_key,
                    "model_id": model_id,
                    "representation_key": representation_key,
                    "rep_family": rep_family,
                    "rep_mode": rep_mode,
                    "axis_key": axis_key,
                    "pairs_file": str(pairs_path),
                    "base_seed": BASE_SEED,
                    "random_seeds": seeds,
                    "metric_mode": metric_meta["mode"],
                    "metric_reg": metric_meta["reg"],
                    "hidden_size": int(hidden_size),
                    "concepts_per_seed": concepts_to_save,
                },
                output_concepts_pt,
            )

        return {
            "ok": True,
            "model_key": model_key,
            "representation_key": representation_key,
            "axis_key": axis_key,
            "out_dir": str(out_dir),
            "n_pairs": n_total,
        }

    finally:
        unload_model(model)


# ============================================================
# MAIN
# ============================================================

def main():
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    manifest = {
        "models": MODELS,
        "representations": REPRESENTATIONS,
        "axes": {k: str(v) for k, v in AXES.items()},
        "train_fraction": TRAIN_FRACTION,
        "n_random_seeds": N_RANDOM_SEEDS,
        "metric_mode": METRIC_MODE,
        "results_root": str(RESULTS_ROOT),
    }
    (RESULTS_ROOT / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    all_jobs = [
        (mk, mid, rk, rv["rep_family"], rv["rep_mode"], ak, ap)
        for mk, mid in MODELS.items()
        for rk, rv in REPRESENTATIONS.items()
        for ak, ap in AXES.items()
    ]

    results = []
    errors = []

    outer_bar = tqdm(all_jobs, desc="All jobs", position=0)

    for model_key, model_id, representation_key, rep_family, rep_mode, axis_key, axis_path in outer_bar:
        outer_bar.set_postfix({"model": model_key, "rep": representation_key, "axis": axis_key})

        if not axis_path.exists():
            err = {
                "model_key": model_key,
                "model_id": model_id,
                "representation_key": representation_key,
                "rep_family": rep_family,
                "rep_mode": rep_mode,
                "axis_key": axis_key,
                "pairs_file": str(axis_path),
                "error": "pairs file does not exist",
            }
            errors.append(err)
            continue

        try:
            result = run_one_job(
                model_key=model_key,
                model_id=model_id,
                representation_key=representation_key,
                rep_family=rep_family,
                rep_mode=rep_mode,
                axis_key=axis_key,
                pairs_path=axis_path,
            )
            results.append(result)
        except Exception as e:
            err = {
                "model_key": model_key,
                "model_id": model_id,
                "representation_key": representation_key,
                "rep_family": rep_family,
                "rep_mode": rep_mode,
                "axis_key": axis_key,
                "pairs_file": str(axis_path),
                "error": repr(e),
            }
            errors.append(err)

            err_dir = RESULTS_ROOT / model_key / representation_key / axis_key
            err_dir.mkdir(parents=True, exist_ok=True)
            (err_dir / "error.json").write_text(
                json.dumps(err, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    summary = {
        "n_jobs_total": len(all_jobs),
        "n_jobs_ok": len(results),
        "n_jobs_failed": len(errors),
        "results": results,
        "errors": errors,
    }

    (RESULTS_ROOT / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nDone.")
    print(f"Results root: {RESULTS_ROOT}")
    print(f"Successful jobs: {len(results)}")
    print(f"Failed jobs: {len(errors)}")


if __name__ == "__main__":
    main()
