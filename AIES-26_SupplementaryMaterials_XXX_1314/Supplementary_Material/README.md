# Supplementary Material

Supplementary material for **“Epistemic Framing Tilt in LLMs: Auditing Religious Discourse Across Traditions.”**

This archive contains the data, model outputs, learned axis artifacts, validation summaries, and code used for the paper.

## Folder overview

```text
/
├── pairs/                    # Minimal-pair datasets for the four framing axes
├── annotators/               # Raw annotation files from three annotators
├── results_annotator/        # Aggregated Gwet AC2 annotation results
├── axis_stage1/              # Learned axis directions and validation outputs
├── religion_replies_stage2/  # Generated audit responses and score summaries
├── codes/                    # Scripts for running the experiments and diagnostics               
```

## Main data files

The `pairs/` folder contains four framing-axis datasets:

- `assertive_commitment_axis_pairs.json`
- `doctrinal_attribution_axis_pairs.json`
- `hedging_axis_pairs_v2.json`
- `mythologized_axis_pairs_v2.json`

Each axis contains 72 minimal pairs. Higher scores correspond to the first pole of the axis: more assertive commitment, stronger doctrinal/scriptural grounding, less hedging, and less mythologized framing.

## Model outputs

Generated responses and audit scores are in:

```text
religion_replies_stage2/{model}/{representation}/{axis}/
```

Each run usually contains:

- `full.json` — generated responses and generation-level scores
- `summary.json` — aggregate score summaries
- `prompt_catalog.json` — prompt templates used for the audit

The models are `qwen`, `gemma`, `llama`, and `olmo`. The main representation used in the paper is `contextual_last_mean`; `output_projection_mean` is also included.

## Validation outputs

Axis-learning and held-out validation files are in:

```text
axis_stage1/{model}/{representation}/{axis}/
```

Important files include:

- `summary.json` — aggregate validation summary
- `per_seed.json` — validation metrics by random seed
- `per_pair_oos.json` — held-out pair scores
- `concepts.pt` — saved learned axis/concept artifacts

## Code

The main scripts are in `codes/`:

- `stage1_axes_both_reps.py` — learn and validate framing axes
- `stage2_religions_both_reps.py` — generate and score religious-audit responses
- `annotation_aggregate_gwet_ac2.py` — compute annotation agreement
- `ablation.py` — run lexical cue ablations


## Basic setup

A typical Python environment should include:

```bash
python >= 3.10

The generation scripts require enough GPU memory to load the evaluated models.

## Quick start

```bash
unzip Supplementary_Material.zip

# Recompute annotation agreement
python codes/annotation_aggregate_gwet_ac2.py

# Relearn axes
python codes/stage1_axes_both_reps.py

# Regenerate and score audit responses
python codes/stage2_religions_both_reps.py
```

