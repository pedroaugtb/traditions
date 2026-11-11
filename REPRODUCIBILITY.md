# Reproducibility and artifact guide

This document maps the paper workflow to the files in the repository. All paths and commands assume the repository root as the working directory.

## Experimental design

The audit has two main stages:

1. **Axis learning and validation.** For each model, representation, and framing axis, 30 seeded 70/30 train-validation splits are created from 72 minimal pairs. Directions are learned on the training split and evaluated out of sample.
2. **Response generation and scoring.** Each model responds to 24 prompt templates (six prompt groups with four variants) for six religious traditions. Five responses are generated per prompt. Each response score is averaged across the 30 learned axis directions.

This produces 32 Stage 1 jobs and 32 Stage 2 jobs:

```text
4 models × 2 representations × 4 axes = 32 jobs per stage
```

Each Stage 2 job contains 720 successful responses:

```text
6 religions × 6 prompt groups × 4 templates × 5 generations = 720
```

## Models and representations

| Short key | Hugging Face model ID |
| --- | --- |
| `qwen` | `Qwen/Qwen2.5-7B` |
| `olmo` | `allenai/OLMo-2-1124-7B-Instruct` |
| `gemma` | `google/gemma-2-9b-it` |
| `llama` | `meta-llama/Llama-3.2-3B-Instruct` |

| Representation key | Description |
| --- | --- |
| `contextual_last_mean` | Mean of contextual hidden states from the final layer |
| `output_projection_mean` | Mean of token vectors from the model's output projection matrix |

The contextual representation is the paper's main analysis; the output-projection representation is included as a comparison.

## Artifact schemas

### Minimal pairs

Each file in `pairs/` is a JSON object with axis metadata and a `pairs` list. Every pair has an ID, religion, topic, and the generic fields `assertive` and `hedged`. Those two field names are historical: their axis-specific meanings are given by each file's `field_mapping` and `score_interpretation` objects.

### Stage 1: axis learning

Path pattern:

```text
axis_stage1/{model}/{representation}/{axis}/
```

| File | Contents |
| --- | --- |
| `summary.json` | Configuration, metrics aggregated across splits, and pooled score summaries |
| `per_seed.json` | Train, validation, baseline, and effect-size results for each split |
| `per_pair_oos.json` | Out-of-sample results at the pair level |
| `concepts.pt` | Learned directions and associated PyTorch tensors for all splits |

### Stage 2: generated responses

Path pattern:

```text
religion_replies_stage2/{model}/{representation}/{axis}/
```

| File | Contents |
| --- | --- |
| `full.json` | Prompts, generated text, generation settings, and response-level scores |
| `summary.json` | Aggregates by religion, prompt group, and religion–prompt combination |
| `prompt_catalog.json` | The prompt templates used for that job |

The root `religion_replies_stage2/prompt_catalog.json` is a convenient copy of the complete prompt catalog.

### Human annotation

`annotators/` contains ratings for all 288 minimal pairs (four axes × 72 pairs) from three annotators. `results_annotator/` contains pair-level aggregates and Gwet AC2, exact-agreement, and adjacent-agreement summaries.

### Lexical ablation

`lexical_initial_ablation/` contains response-level and aggregate stability analyses after removing high-salience lexical cues. `lexical_ablation_report.json` combines the principal tables in one object; the CSV files expose the same results for downstream analysis.

## Script reference

| Script | Purpose | Main inputs | Main outputs |
| --- | --- | --- | --- |
| `stage1_axes_both_reps.py` | Learn and validate axes | `pairs/` | `axis_stage1/` |
| `stage2_religions_both_reps.py` | Generate and score responses | `pairs/`, `axis_stage1/` | `religion_replies_stage2/` |
| `annotation_aggregate_gwet_ac2.py` | Aggregate minimal-pair ratings | `annotators/` | `results_annotator/` |
| `ablation.py` | Remove lexical cues and test stability | Stage 1 and Stage 2 artifacts | `lexical_initial_ablation/` |
| `robustness_framing_diagnostics.py` | Run extended ablation, lexical baselines, residualization, and optional placebo axes | Stage 1 and Stage 2 artifacts | `robustness_diagnostics/` by default |
| `print_robustness_paper_results.py` | Format robustness outputs as tables and prose | `robustness_diagnostics/` | `robustness_diagnostics/paper_results/` |
| `select_human_eval_examples.py` | Select a diverse generated-response evaluation pack | `religion_replies_stage2/` | `human_eval_pack/` |
| `analyze_generated_output_annotations.py` | Analyze ratings collected for a generated-response pack | A generated pack and its ratings | `human_eval_analysis/` |

The final two utilities concern a separate generated-output evaluation workflow. Its annotation pack is not part of the minimal-pair annotation data in `annotators/`.

## Recompute annotation agreement

This is the lightest-weight reproduction step and does not require a GPU:

```bash
python scripts/annotation_aggregate_gwet_ac2.py \
  --inputs annotators/annotator3.json annotators/annotator1.json annotators/annotator2.json \
  --output-dir results_annotator
```

## Rerun the model pipeline

The Stage 1 and Stage 2 scripts are configured with constants near the top of each file. By default, they run all models, representations, and axes.

```bash
python scripts/stage1_axes_both_reps.py
python scripts/stage2_religions_both_reps.py
```

Before running, ensure that:

- the machine has sufficient GPU memory for the selected model;
- Hugging Face access has been granted for gated models;
- enough disk space is available for model weights and generated artifacts; and
- model revisions are recorded if exact future reproduction is required.

## Run robustness analyses

The included initial ablation can be recomputed with:

```bash
python scripts/ablation.py --outdir lexical_initial_ablation
```

The extended diagnostic suite is optional:

```bash
python scripts/robustness_framing_diagnostics.py \
  --pairs-dir pairs \
  --concepts-root axis_stage1 \
  --results-root religion_replies_stage2 \
  --outdir robustness_diagnostics

python scripts/print_robustness_paper_results.py \
  --outdir robustness_diagnostics
```

Add `--run-placebo` to the first command to run the more expensive placebo-axis analysis.

## Reproducibility limitations

- The exact original Python package versions and hardware/software environment were not recorded in the archive. `requirements.txt` therefore lists dependencies without claiming exact version reproduction.
- Model repositories can change. Record Hugging Face commit revisions before a new run if bit-level provenance matters.
- Sampling is seeded in the scripts, but GPU kernels, library changes, and model revisions can still affect generated text or floating-point results.
- Confidence intervals across Stage 1 seeds describe sensitivity to repeated train-validation splits of the same finite pair pool; they are not population-level guarantees.
- Generated artifact metadata retains some original directory names. The data themselves have not been rewritten during repository cleanup.
