# Comprehensive RPS Pipeline

A complete implementation of the Robust Preference Selection (RPS) pipeline. The code and README reflect the current repository state.

## Pipeline Overview

```
1. Baseline Generation    2. RPS Generation         3. Scoring & Selection    4. A/B Comparison
   ├─ SFT baseline       ├─ Angle perturbation    ├─ Reward model scoring  ├─ Parallel GPT calls
   └─ Multiple models    └─ Multi-response gen     └─ Best selection        └─ Summary statistics
```

## Directory Structure

```
rps_code/
├── baseline_generation/          # Baseline response generation
│   └── sft_baseline.py
├── rps/                          # RPS perturbation generation  
│   └── rps_generation.py
├── scoring/                      # Reward model scoring and best selection
│   ├── reward_scoring.py
│   └── best_selection.py
├── comparison/                   # A/B comparison
│   └── parallel_compare.py
└── utils/                        # Utilities
    ├── env_detection.py
    ├── model_loaders.py
    ├── data_utils.py
    └── directions.py
```

## Quick Start

Note on directions: the repository uses v1–v8 to denote preference directions.

### 2. Individual Steps

#### 1) Generate Baseline (SFT)
```bash
python baseline_generation/sft_baseline.py \
  --dataset nvidia/HelpSteer \
  --model mistralai/Mistral-7B-Instruct-v0.2 \
  --output_dir ./outputs/baseline \
  --batch_size 16 \
  --num_responses 3
```

Outputs are written as `./outputs/baseline/baseline_responses_v{X}.csv`.

#### 2) Generate RPS Responses (angle perturbations)
```bash
python rps/rps_generation.py \
  --dataset nvidia/HelpSteer \
  --model mistralai/Mistral-7B-Instruct-v0.2 \
  --output_dir ./outputs \
  --angle_range -40,40 \
  --step 5 \
  --theta_max 30 \
  --top_k 5
```

Outputs are written per direction under `./outputs/rps/v{X}/rps_angle{angle}.csv`.

#### 3) Score and Select Best
```bash
## Example: score RPS outputs for direction v1, then select best
python scoring/reward_scoring.py \
  --input_dir ./outputs/rps/v1 \
  --output_dir ./outputs/rps_scored/v1

python scoring/best_selection.py \
  --input_dir ./outputs/rps_scored/v1 \
  --output_path ./outputs/rps_best/v1_best.csv
```

Repeat scoring and selection for v1–v8 to produce `./outputs/rps_best/v{X}_best.csv`.

#### 4) Compare Baseline vs RPS
```bash
python comparison/parallel_compare.py \
  --baseline_dir ./outputs/baseline_best \
  --rps_dir ./outputs/rps_best \
  --output_dir ./outputs/comparison \
  --model gpt-4o-mini \
  --max_workers 8
```

## Features

### Environment Detection
- **Colab**: Automatic Google Drive mounting, memory optimization
- **Server**: vLLM acceleration, GPU memory management
- **Local**: CPU fallback, local storage

### Model Support
- **Generation**: Mistral-7B-Instruct-v0.2 (others can be plugged in)
- **Scoring**: RewardModel-Mistral-7B-for-DPA-v1 
- **Judging**: GPT-4o-mini (via OpenAI/OpenRouter)

### Robustness Features
- **Resume capability**: Automatic detection of existing outputs
- **Error recovery**: Graceful handling of model loading failures
- **Memory management**: Dynamic batch size adjustment
- **Parallel processing**: Multi-worker GPT judging with rate limiting

### Data Compatibility
- **HelpSteer**: nvidia/HelpSteer (validation/train splits)
- **Custom datasets**: With a `prompt` column

## Configuration

Set environment variables:
```bash
export HF_TOKEN="your_huggingface_token"
export OPENROUTER_API_KEY="your_openrouter_key" 
export OPENAI_API_KEY="your_openai_key"
```

For Chinese users, optionally set:
```bash
export HF_ENDPOINT="https://hf-mirror.com"
```

## Outputs

All outputs are CSV/JSONL text files:
- Baseline: `baseline_responses_v{X}.csv`
- RPS raw: `rps/v{X}/rps_angle{angle}.csv`
- Best after scoring: `rps_best/v{X}_best.csv` (or `baseline_best/v{X}_best.csv` if you score baseline)
- A/B judgments: `{v{X}}_judged.jsonl`
- Summary: `comparison_summary.csv`

No model weights or caches are included in outputs.