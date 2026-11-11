# Epistemic Framing Tilt in LLMs

Research artifacts for **“Epistemic Framing Tilt in LLMs: Auditing Religious Discourse Across Traditions.”**

This repository contains the minimal-pair datasets, human annotations, learned framing-axis artifacts, generated model responses, aggregate results, and analysis code used in the paper. The included outputs make it possible to inspect the reported experiments without rerunning large language models.

> [!NOTE]
> This is a research artifact. Model-generated descriptions of religions may be incomplete, inaccurate, or culturally insensitive; they should not be treated as authoritative accounts of any tradition.

## At a glance

- **4 framing axes**, each built from 72 minimal pairs
- **4 language models**: Qwen 2.5 7B, OLMo 2 7B Instruct, Gemma 2 9B IT, and Llama 3.2 3B Instruct
- **2 representation methods**: contextual last-layer mean and output-projection mean
- **6 religious traditions** and **6 prompt groups**
- **23,040 successful generations** across the complete audit
- **3 annotators** evaluating 288 minimal pairs

## Framing axes

Scores are oriented toward the positive pole shown below. The folder name `mythologized` is retained for compatibility, but a higher score on that axis means *less* mythologized framing.

| Axis key | Positive pole (higher score) | Negative pole (lower score) |
| --- | --- | --- |
| `assertive_commitment` | Direct assertive commitment | Belief attributed to adherents or communities |
| `doctrinal_attribution` | Doctrinal, scriptural, or internal-tradition grounding | Generic believer or practitioner attribution |
| `hedging` | Unqualified direct assertion | Hedged, modal, or interpretive qualification |
| `mythologized` | Sacred, doctrinal, or practice framing | Mythic, symbolic, legendary, or folkloric framing |

The full construction principles and field mappings are stored with each dataset in [`pairs/`](pairs/).

## Repository structure

```text
.
├── pairs/                       # Four 72-item minimal-pair datasets
├── annotators/                  # Raw minimal-pair ratings from three annotators
├── results_annotator/           # Aggregated agreement and Gwet AC2 results
├── axis_stage1/                 # Learned axes and held-out validation artifacts
├── religion_replies_stage2/     # Generated audit responses and score summaries
├── lexical_initial_ablation/    # Lexical-cue ablation outputs
├── scripts/                     # Experiment and analysis scripts
├── REPRODUCIBILITY.md           # Detailed workflow and artifact guide
└── requirements.txt             # Python dependencies
```

The two main result trees use the same hierarchy:

```text
{stage}/{model}/{representation}/{axis}/
```

For a fast review, start with:

- [`pairs/`](pairs/) for the framing constructs and minimal pairs;
- [`results_annotator/paper_summary_gwet_ac2.json`](results_annotator/paper_summary_gwet_ac2.json) for annotation agreement;
- any `axis_stage1/.../summary.json` for held-out axis validation;
- any `religion_replies_stage2/.../summary.json` for aggregate audit results;
- [`lexical_initial_ablation/axis_response_summary.csv`](lexical_initial_ablation/axis_response_summary.csv) and [`lexical_initial_ablation/axis_aggregate_summary.csv`](lexical_initial_ablation/axis_aggregate_summary.csv) for ablation summaries.

## Setup

Python 3.10 or newer is recommended.

```bash
git clone https://github.com/pedroaugtb/traditions.git
cd traditions

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Stage 1, Stage 2, and the ablation analyses require enough GPU memory to load the evaluated models. Gemma and Llama may also require accepting their model terms and authenticating with Hugging Face.

## Reproducing the workflow

Run commands from the repository root. The core pipeline is:

```bash
# 1. Recompute human-annotation agreement
python scripts/annotation_aggregate_gwet_ac2.py \
  --inputs annotators/annotator3.json annotators/annotator1.json annotators/annotator2.json \
  --output-dir results_annotator

# 2. Relearn and validate the framing axes
python scripts/stage1_axes_both_reps.py

# 3. Regenerate and score the religious-discourse audit
python scripts/stage2_religions_both_reps.py

# 4. Recompute the contextual lexical-cue ablation
python scripts/ablation.py --outdir lexical_initial_ablation
```

These commands overwrite result files in their respective output directories. The model-running steps are computationally expensive; use the checked-in summaries when rerunning is unnecessary. See [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) for every script, artifact schema, and reproducibility note.

## Data notes

- `full.json` files contain generation-level responses and scores; `summary.json` files contain aggregates intended for quick inspection.
- `concepts.pt` files are PyTorch artifacts containing learned directions for each random split. Load them only from a trusted source.
- Some generated JSON metadata retains Portuguese directory names from the original experiment environment (for example, `pares` and `resultados_eixos_reps`). These strings are provenance only; the corresponding repository directories are `pairs/` and `axis_stage1/`.
- The included environment was reconstructed from imports because the exact original package versions were not recorded. Small numerical differences may occur with newer library or model revisions.
- A few agreement outputs contain `NaN`, which Python's `json` module accepts but strict JSON parsers may reject.

## Citation

The paper citation will be added here when publication metadata is available. Until then, please cite the paper title and this repository URL.

## License

No license has been specified yet. Unless and until a license is added, the repository remains under standard copyright terms.
