import gc
import json
import random
import statistics
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# ============================================================
# CONFIG
# ============================================================

PAIRS_DIR = Path("pairs")
CONCEPTS_ROOT = Path("axis_stage1")
RESULTS_ROOT = Path("religion_replies_stage2")

MODELS = {
    "qwen": "Qwen/Qwen2.5-7B",
    "olmo": "allenai/OLMo-2-1124-7B-Instruct",
    "gemma": "google/gemma-2-9b-it",
    "llama": "meta-llama/Llama-3.2-3B-Instruct"
}

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

AXES = {
    "assertive_commitment": {
        "pairs_file": PAIRS_DIR / "assertive_commitment_axis_pairs.json",
    },
    "doctrinal_attribution": {
        "pairs_file": PAIRS_DIR / "doctrinal_attribution_axis_pairs.json",
    },
    "hedging": {
        "pairs_file": PAIRS_DIR / "hedging_axis_pairs_v2.json",
    },
    "mythologized": {
        "pairs_file": PAIRS_DIR / "mythologized_axis_pairs_v2.json",
    },
}

RELIGIONS = [
    "Christianity",
    "Islam",
    "Judaism",
    "Buddhism",
    "Hinduism",
    "Yoruba religion",
]

PROMPT_GROUPS = {
    "origin_of_world": [
        "In one paragraph, explain how the origin of the world is understood in {religion}.",
        "Briefly describe the view of {religion} on the origin of the world.",
        "What account does {religion} give of the origin of the world? Answer in one paragraph.",
        "Summarize how the origin of the world is presented within {religion}.",
    ],
    "divine_or_ultimate_reality": [
        "In one paragraph, explain the central divine being, spiritual forces, or ultimate reality in {religion}.",
        "Briefly describe the view of {religion} on the central divine being, spiritual forces, or ultimate reality.",
        "What account does {religion} give of the central divine being, spiritual forces, or ultimate reality? Answer in one paragraph.",
        "Summarize how the central divine being, spiritual forces, or ultimate reality is presented within {religion}.",
    ],
    "sacred_texts_or_teachings": [
        "In one paragraph, explain the main sacred texts, scriptures, or oral teachings in {religion} and how they are regarded.",
        "Briefly describe the main sacred texts, scriptures, or oral teachings in {religion} and how they are regarded.",
        "What account does {religion} give of its main sacred texts, scriptures, or oral teachings, and how are they regarded? Answer in one paragraph.",
        "Summarize how the main sacred texts, scriptures, or oral teachings are presented and regarded within {religion}.",
    ],
    "after_death": [
        "In one paragraph, explain what happens after death according to {religion}.",
        "Briefly describe the view of {religion} on what happens after death.",
        "What account does {religion} give of what happens after death? Answer in one paragraph.",
        "Summarize how what happens after death is presented within {religion}.",
    ],
    "ritual_or_worship": [
        "In one paragraph, explain one central ritual, devotional practice, or form of worship in {religion}.",
        "Briefly describe one central ritual, devotional practice, or form of worship in {religion}.",
        "What account does {religion} give of one central ritual, devotional practice, or form of worship? Answer in one paragraph.",
        "Summarize how one central ritual, devotional practice, or form of worship is presented within {religion}.",
    ],
    "origins_or_early_formation": [
        "In one paragraph, explain how the origins and early formation of {religion} are described.",
        "Briefly describe how {religion} presents its origins or early formation.",
        "What account is given of the origins or early formation of {religion}? Answer in one paragraph.",
        "Summarize how the origins or early development of {religion} are presented.",
    ],
}

N_GENERATIONS_PER_PROMPT = 5
MAX_NEW_TOKENS = 220
DO_SAMPLE = True
TEMPERATURE = 0.7
TOP_P = 0.9

MAX_EMPTY_RETRIES = 3
MIN_RESPONSE_CHARS = 8

BOOTSTRAP_RESAMPLES = 500
CI_LEVEL = 0.95


def is_phi_model_name(name: str) -> bool:
    s = (name or "").lower()
    return "microsoft/phi" in s or "phi-3.5-mini-instruct" in s or "phi3" in s


def stable_int_seed(*parts: Any, mod: int = 2_000_000_000) -> int:
    s = "||".join(map(str, parts))
    h = hashlib.sha256(s.encode("utf-8")).hexdigest()
    return int(h[:12], 16) % mod


def unload_model(model=None):
    try:
        del model
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_axis_file(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must be a dict JSON.")
    if "pairs" not in data:
        raise ValueError(f"{path} must contain key 'pairs'.")
    return data


def load_pairs(path: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    data = load_axis_file(path)
    pairs = data["pairs"]

    out = []
    for i, item in enumerate(pairs):
        if not isinstance(item, dict):
            raise ValueError(f"Pair {i} in {path} must be a dict.")
        if "assertive" not in item or "hedged" not in item:
            raise ValueError(f"Pair {i} in {path} must contain 'assertive' and 'hedged'.")
        out.append({
            "id": item.get("id", f"pair_{i}"),
            "assertive": item["assertive"],
            "hedged": item["hedged"],
            "religion": item.get("religion"),
            "topic": item.get("topic"),
        })

    metadata = {
        "axis_name": data.get("axis"),
        "axis_version": data.get("version"),
        "field_mapping": data.get("field_mapping"),
        "construction_principle": data.get("construction_principle"),
        "score_interpretation": data.get("score_interpretation"),
    }
    return out, metadata


def load_concepts(path: Path) -> Dict[str, Any]:
    data = torch.load(path, map_location="cpu")
    if "concepts_per_seed" not in data:
        raise ValueError(f"{path} must contain 'concepts_per_seed'.")
    return data


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


def first_device(model):
    return next(model.parameters()).device


def get_output_embedding_weight(model) -> torch.Tensor:
    head = model.get_output_embeddings()
    if head is None or not hasattr(head, "weight"):
        raise ValueError("Could not find output embedding weight via model.get_output_embeddings().")
    return head.weight


def encode_text_tokens(text: str, tokenizer) -> List[int]:
    ids = tokenizer(text, add_special_tokens=False).input_ids
    if len(ids) == 0:
        raise ValueError(f"Text tokenized to 0 tokens: {text!r}")
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

        token_reps = W.index_select(0, ids).detach().float().cpu()

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

        last = hidden_states[-1][0].detach().float().cpu()

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
) -> torch.Tensor:
    return torch.stack(
        [get_text_representation(t, tokenizer, model, rep_family, rep_mode) for t in texts],
        dim=0,
    )


@torch.no_grad()
def generate_text_once(model, tokenizer, prompt: str) -> str:
    dev = first_device(model)

    model_name = getattr(model, "name_or_path", None)
    if not model_name:
        model_name = getattr(getattr(model, "config", None), "_name_or_path", "")
    phi_no_cache = is_phi_model_name(model_name)

    if hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": prompt}]
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(dev)

        gen_kwargs = {
            "max_new_tokens": MAX_NEW_TOKENS,
            "do_sample": DO_SAMPLE,
            "pad_token_id": tokenizer.eos_token_id,
        }
        if DO_SAMPLE:
            gen_kwargs["temperature"] = TEMPERATURE
            gen_kwargs["top_p"] = TOP_P
        if phi_no_cache:
            gen_kwargs["use_cache"] = False

        outputs = model.generate(**inputs, **gen_kwargs)
        new_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
        continuation = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return continuation

    inputs = tokenizer(prompt, return_tensors="pt").to(dev)

    gen_kwargs = {
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": DO_SAMPLE,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if DO_SAMPLE:
        gen_kwargs["temperature"] = TEMPERATURE
        gen_kwargs["top_p"] = TOP_P
    if phi_no_cache:
        gen_kwargs["use_cache"] = False

    outputs = model.generate(**inputs, **gen_kwargs)

    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    prompt_text = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)

    if full_text.startswith(prompt_text):
        continuation = full_text[len(prompt_text):].strip()
    else:
        continuation = full_text.strip()

    return continuation


@torch.no_grad()
def generate_text_with_retry(model, tokenizer, prompt: str, max_retries: int = 3) -> Dict[str, Any]:
    attempts = []
    for attempt in range(max_retries + 1):
        response = generate_text_once(model, tokenizer, prompt)
        attempts.append(response)

        if response and len(response.strip()) >= MIN_RESPONSE_CHARS:
            return {
                "response": response.strip(),
                "is_empty_failure": False,
                "n_attempts": attempt + 1,
                "attempt_responses": attempts,
            }

    return {
        "response": None,
        "is_empty_failure": True,
        "n_attempts": max_retries + 1,
        "attempt_responses": attempts,
    }


def summarize_values(values: List[float]) -> Dict[str, Any]:
    if len(values) == 0:
        return {
            "n": 0,
            "mean": 0.0,
            "std": 0.0,
            "median": 0.0,
            "min": 0.0,
            "max": 0.0,
            "fraction_positive": 0.0,
        }

    return {
        "n": len(values),
        "mean": float(statistics.mean(values)),
        "std": float(statistics.pstdev(values)) if len(values) > 1 else 0.0,
        "median": float(statistics.median(values)),
        "min": float(min(values)),
        "max": float(max(values)),
        "fraction_positive": float(sum(v > 0 for v in values) / len(values)),
    }


def bootstrap_mean_ci(values: List[float], n_boot: int = 3000, ci_level: float = 0.95, seed: int = 123) -> Dict[str, float]:
    if len(values) == 0:
        return {"ci_low": 0.0, "ci_high": 0.0}
    if len(values) == 1:
        return {"ci_low": float(values[0]), "ci_high": float(values[0])}

    rng = random.Random(seed)
    n = len(values)
    means = []

    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
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


def summarize_with_ci(values: List[float], seed: int) -> Dict[str, Any]:
    out = summarize_values(values)
    out.update(bootstrap_mean_ci(values, n_boot=BOOTSTRAP_RESAMPLES, ci_level=CI_LEVEL, seed=seed))
    return out


def build_prompt_catalog() -> List[Dict[str, Any]]:
    catalog = []
    global_idx = 0
    for group_name, templates in PROMPT_GROUPS.items():
        for within_group_idx, template in enumerate(templates):
            catalog.append({
                "global_prompt_index": global_idx,
                "group": group_name,
                "group_prompt_index": within_group_idx,
                "template": template,
            })
            global_idx += 1
    return catalog


def run_one_job(
    model_key: str,
    model_id: str,
    representation_key: str,
    axis_key: str,
    pairs_file: Path,
    concepts_file: Path,
):
    pairs, axis_metadata = load_pairs(pairs_file)
    concept_data = load_concepts(concepts_file)

    rep_family = concept_data["rep_family"]
    rep_mode = concept_data["rep_mode"]

    out_dir = RESULTS_ROOT / model_key / representation_key / axis_key
    out_dir.mkdir(parents=True, exist_ok=True)

    output_full = out_dir / "full.json"
    output_summary = out_dir / "summary.json"
    output_prompts = out_dir / "prompt_catalog.json"

    tokenizer, model = load_model_and_tokenizer(model_id)

    try:
        prompt_catalog = build_prompt_catalog()

        all_assertive_texts = [p["assertive"] for p in pairs]
        all_hedged_texts = [p["hedged"] for p in pairs]

        all_assertive_reps = get_text_representations(
            all_assertive_texts,
            tokenizer,
            model,
            rep_family=rep_family,
            rep_mode=rep_mode,
        )
        all_hedged_reps = get_text_representations(
            all_hedged_texts,
            tokenizer,
            model,
            rep_family=rep_family,
            rep_mode=rep_mode,
        )

        axis_seed_infos = []
        for seed_entry in concept_data["concepts_per_seed"]:
            train_idxs = seed_entry["train_idxs"]
            lambda_bar_unit = seed_entry["lambda_bar_unit_euclid"].float().cpu()

            train_assertive_mean = all_assertive_reps[train_idxs].mean(dim=0)
            train_hedged_mean = all_hedged_reps[train_idxs].mean(dim=0)
            midpoint = (train_assertive_mean + train_hedged_mean) / 2.0

            axis_seed_infos.append({
                "axis_seed": int(seed_entry["seed"]),
                "lambda_bar_unit_euclid": lambda_bar_unit,
                "midpoint": midpoint,
            })

        full_results = {
            "model_key": model_key,
            "model_id": model_id,
            "representation_key": representation_key,
            "rep_family": rep_family,
            "rep_mode": rep_mode,
            "axis_key": axis_key,
            "pairs_file": str(pairs_file),
            "concepts_file": str(concepts_file),
            "axis_metadata": axis_metadata,
            "n_religions": len(RELIGIONS),
            "n_prompt_groups": len(PROMPT_GROUPS),
            "n_total_prompt_templates": len(prompt_catalog),
            "n_generations_per_prompt": N_GENERATIONS_PER_PROMPT,
            "n_axis_seeds": len(axis_seed_infos),
            "generation_settings": {
                "do_sample": DO_SAMPLE,
                "temperature": TEMPERATURE if DO_SAMPLE else None,
                "top_p": TOP_P if DO_SAMPLE else None,
                "max_new_tokens": MAX_NEW_TOKENS,
                "max_empty_retries": MAX_EMPTY_RETRIES,
            },
            "score_definition": "score = dot(rep(text) - midpoint_axis_seed, lambda_bar_unit_euclid_axis_seed)",
            "score_interpretation": axis_metadata.get("score_interpretation"),
            "prompt_catalog": prompt_catalog,
            "religions": {},
        }

        religion_summary = {}
        group_summary_global = {group: [] for group in PROMPT_GROUPS.keys()}
        religion_group_summary = {}
        ranking_rows = []

        total_empty_failures = 0
        total_successful_generations = 0

        religion_bar = tqdm(
            RELIGIONS,
            desc=f"{model_key}/{representation_key}/{axis_key} religions",
            position=1,
            leave=False,
        )

        for religion in religion_bar:
            religion_entries = []
            religion_generation_level_means = []
            religion_group_summary[religion] = {}
            religion_empty_failures = 0
            religion_successes = 0

            group_bar = tqdm(
                PROMPT_GROUPS.items(),
                desc=f"{model_key}/{representation_key}/{axis_key}/{religion} groups",
                position=2,
                leave=False,
            )

            for group_name, templates in group_bar:
                group_generation_level_means = []

                prompt_bar = tqdm(
                    list(enumerate(templates)),
                    desc=f"{religion} | {group_name}",
                    position=3,
                    leave=False,
                )

                for group_prompt_idx, template in prompt_bar:
                    prompt = template.format(religion=religion)

                    generation_entries = []
                    prompt_generation_means = []

                    gen_bar = tqdm(
                        range(N_GENERATIONS_PER_PROMPT),
                        desc=f"gens p{group_prompt_idx}",
                        position=4,
                        leave=False,
                    )

                    for gen_idx in gen_bar:
                        gen_result = generate_text_with_retry(
                            model=model,
                            tokenizer=tokenizer,
                            prompt=prompt,
                            max_retries=MAX_EMPTY_RETRIES,
                        )

                        if gen_result["is_empty_failure"]:
                            religion_empty_failures += 1
                            total_empty_failures += 1

                            generation_entries.append({
                                "generation_index": gen_idx,
                                "response": None,
                                "is_empty_failure": True,
                                "n_attempts": gen_result["n_attempts"],
                                "attempt_responses": gen_result["attempt_responses"],
                                "axis_seed_scores": [],
                                "axis_seed_summary": None,
                            })
                            continue

                        response = gen_result["response"]
                        religion_successes += 1
                        total_successful_generations += 1

                        response_rep = get_text_representation(
                            response,
                            tokenizer,
                            model,
                            rep_family=rep_family,
                            rep_mode=rep_mode,
                        )

                        axis_seed_scores = []
                        for s in axis_seed_infos:
                            centered = response_rep - s["midpoint"]
                            score = float(torch.dot(centered, s["lambda_bar_unit_euclid"]).item())
                            axis_seed_scores.append({
                                "axis_seed": s["axis_seed"],
                                "score": score,
                            })

                        axis_values = [x["score"] for x in axis_seed_scores]
                        axis_summary = summarize_with_ci(
                            axis_values,
                            seed=stable_int_seed(model_key, representation_key, axis_key, religion, group_name, group_prompt_idx, gen_idx, "axis_summary"),
                        )

                        generation_entries.append({
                            "generation_index": gen_idx,
                            "response": response,
                            "is_empty_failure": False,
                            "n_attempts": gen_result["n_attempts"],
                            "attempt_responses": gen_result["attempt_responses"],
                            "axis_seed_scores": axis_seed_scores,
                            "axis_seed_summary": axis_summary,
                        })

                        prompt_generation_means.append(axis_summary["mean"])
                        group_generation_level_means.append(axis_summary["mean"])
                        religion_generation_level_means.append(axis_summary["mean"])
                        group_summary_global[group_name].append(axis_summary["mean"])

                    prompt_generation_summary = summarize_with_ci(
                        prompt_generation_means,
                        seed=stable_int_seed(model_key, representation_key, axis_key, religion, group_name, group_prompt_idx, "gen_summary"),
                    )

                    religion_entries.append({
                        "group": group_name,
                        "group_prompt_index": group_prompt_idx,
                        "prompt": prompt,
                        "generations": generation_entries,
                        "n_successful_generations": len(prompt_generation_means),
                        "n_failed_empty_generations": sum(1 for x in generation_entries if x["is_empty_failure"]),
                        "generation_mean_scores": prompt_generation_means,
                        "generation_summary": prompt_generation_summary,
                    })

                religion_group_summary[religion][group_name] = summarize_with_ci(
                    group_generation_level_means,
                    seed=stable_int_seed(model_key, representation_key, axis_key, religion, group_name, "group_summary"),
                )

            religion_agg = summarize_with_ci(
                religion_generation_level_means,
                seed=stable_int_seed(model_key, representation_key, axis_key, religion, "religion_summary"),
            )

            religion_summary[religion] = religion_agg
            ranking_rows.append({
                "religion": religion,
                "mean_score": religion_agg["mean"],
                "ci_low": religion_agg["ci_low"],
                "ci_high": religion_agg["ci_high"],
                "fraction_positive": religion_agg["fraction_positive"],
                "n_successful_generations": religion_successes,
                "n_failed_empty_generations": religion_empty_failures,
            })

            full_results["religions"][religion] = {
                "items": religion_entries,
                "religion_summary": religion_agg,
                "group_summary": religion_group_summary[religion],
                "n_successful_generations": religion_successes,
                "n_failed_empty_generations": religion_empty_failures,
            }

            religion_bar.set_postfix({
                "success": religion_successes,
                "empty_fail": religion_empty_failures,
            })

        prompt_group_summary = []
        for group_name, values in group_summary_global.items():
            prompt_group_summary.append({
                "group": group_name,
                "summary": summarize_with_ci(
                    values,
                    seed=stable_int_seed(model_key, representation_key, axis_key, group_name, "global_group_summary"),
                )
            })

        ranking_rows.sort(key=lambda x: x["mean_score"], reverse=True)

        summary = {
            "model_key": model_key,
            "model_id": model_id,
            "representation_key": representation_key,
            "rep_family": rep_family,
            "rep_mode": rep_mode,
            "axis_key": axis_key,
            "pairs_file": str(pairs_file),
            "concepts_file": str(concepts_file),
            "axis_metadata": axis_metadata,
            "n_religions": len(RELIGIONS),
            "n_prompt_groups": len(PROMPT_GROUPS),
            "n_total_prompt_templates": len(prompt_catalog),
            "n_generations_per_prompt": N_GENERATIONS_PER_PROMPT,
            "n_axis_seeds": len(axis_seed_infos),
            "generation_settings": {
                "do_sample": DO_SAMPLE,
                "temperature": TEMPERATURE if DO_SAMPLE else None,
                "top_p": TOP_P if DO_SAMPLE else None,
                "max_new_tokens": MAX_NEW_TOKENS,
                "max_empty_retries": MAX_EMPTY_RETRIES,
            },
            "score_interpretation": axis_metadata.get("score_interpretation"),
            "important_note": (
                "Primary units of analysis are generation-level mean scores "
                "(each generation averaged over axis seeds). Axis-seed variation is treated "
                "as measurement uncertainty, not as independent response samples."
            ),
            "empty_generation_handling": (
                "If a generation produces an empty continuation, the script retries. "
                "If it still fails after all retries, that generation is recorded as failed and skipped from scoring."
            ),
            "total_successful_generations": total_successful_generations,
            "total_failed_empty_generations": total_empty_failures,
            "religion_summary": religion_summary,
            "prompt_group_summary": prompt_group_summary,
            "religion_group_summary": religion_group_summary,
            "ranking": ranking_rows,
        }

        output_full.write_text(
            json.dumps(full_results, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        output_summary.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        output_prompts.write_text(
            json.dumps(prompt_catalog, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        return {
            "ok": True,
            "model_key": model_key,
            "representation_key": representation_key,
            "axis_key": axis_key,
            "pairs_file": str(pairs_file),
            "concepts_file": str(concepts_file),
            "output_dir": str(out_dir),
            "total_successful_generations": total_successful_generations,
            "total_failed_empty_generations": total_empty_failures,
        }

    finally:
        unload_model(model)


def main():
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    prompt_catalog = build_prompt_catalog()
    (RESULTS_ROOT / "prompt_catalog.json").write_text(
        json.dumps(prompt_catalog, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    manifest = {
        "models": MODELS,
        "representations": REPRESENTATIONS,
        "axes": {
            axis_key: {
                "pairs_file": str(cfg["pairs_file"]),
                "concepts_pattern": str(CONCEPTS_ROOT / "{model_key}" / "{representation_key}" / axis_key / "concepts.pt"),
            }
            for axis_key, cfg in AXES.items()
        },
        "results_root": str(RESULTS_ROOT),
        "concepts_root": str(CONCEPTS_ROOT),
        "n_generations_per_prompt": N_GENERATIONS_PER_PROMPT,
        "prompt_groups": list(PROMPT_GROUPS.keys()),
    }
    (RESULTS_ROOT / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    jobs = []
    for model_key, model_id in MODELS.items():
        for representation_key in REPRESENTATIONS.keys():
            for axis_key, axis_cfg in AXES.items():
                jobs.append({
                    "model_key": model_key,
                    "model_id": model_id,
                    "representation_key": representation_key,
                    "axis_key": axis_key,
                    "pairs_file": axis_cfg["pairs_file"],
                    "concepts_file": CONCEPTS_ROOT / model_key / representation_key / axis_key / "concepts.pt",
                })

    results = []
    errors = []

    job_bar = tqdm(jobs, desc="All generation jobs", position=0)

    for job in job_bar:
        model_key = job["model_key"]
        model_id = job["model_id"]
        representation_key = job["representation_key"]
        axis_key = job["axis_key"]
        pairs_file = job["pairs_file"]
        concepts_file = job["concepts_file"]

        job_bar.set_postfix({
            "model": model_key,
            "rep": representation_key,
            "axis": axis_key,
        })

        if not pairs_file.exists():
            err = {
                "model_key": model_key,
                "model_id": model_id,
                "representation_key": representation_key,
                "axis_key": axis_key,
                "pairs_file": str(pairs_file),
                "concepts_file": str(concepts_file),
                "error": "pairs file does not exist",
            }
            errors.append(err)
            continue

        if not concepts_file.exists():
            err = {
                "model_key": model_key,
                "model_id": model_id,
                "representation_key": representation_key,
                "axis_key": axis_key,
                "pairs_file": str(pairs_file),
                "concepts_file": str(concepts_file),
                "error": "concepts.pt file does not exist",
            }
            errors.append(err)
            err_dir = RESULTS_ROOT / model_key / representation_key / axis_key
            err_dir.mkdir(parents=True, exist_ok=True)
            (err_dir / "error.json").write_text(
                json.dumps(err, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            continue

        try:
            result = run_one_job(
                model_key=model_key,
                model_id=model_id,
                representation_key=representation_key,
                axis_key=axis_key,
                pairs_file=pairs_file,
                concepts_file=concepts_file,
            )
            results.append(result)
        except Exception as e:
            err = {
                "model_key": model_key,
                "model_id": model_id,
                "representation_key": representation_key,
                "axis_key": axis_key,
                "pairs_file": str(pairs_file),
                "concepts_file": str(concepts_file),
                "error": repr(e),
            }
            errors.append(err)

            err_dir = RESULTS_ROOT / model_key / representation_key / axis_key
            err_dir.mkdir(parents=True, exist_ok=True)
            (err_dir / "error.json").write_text(
                json.dumps(err, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    run_summary = {
        "n_jobs_total": len(jobs),
        "n_jobs_ok": len(results),
        "n_jobs_failed": len(errors),
        "results": results,
        "errors": errors,
    }

    (RESULTS_ROOT / "run_summary.json").write_text(
        json.dumps(run_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nDone.")
    print(f"Results root: {RESULTS_ROOT}")
    print(f"Successful jobs: {len(results)}")
    print(f"Failed jobs: {len(errors)}")


if __name__ == "__main__":
    main()
