# Comprehensive RPS Pipeline

A complete implementation of the Robust Preference Selection (RPS) pipeline with full functionality preserved from the original experiments.

## Pipeline Overview

```
1. Baseline Generation    2. RPS Generation         3. Scoring & Selection    4. A/B Comparison
   ├─ SFT baseline       ├─ Angle perturbation    ├─ Reward model scoring  ├─ Pairwise judging
   ├─ DPO baseline       ├─ Multi-response gen    ├─ Best selection        ├─ Parallel GPT calls
   └─ Multiple models    └─ vLLM acceleration     └─ Batch processing      └─ Win rate analysis
```

## Directory Structure

```
rps_code/
├── baseline_generation/           # Baseline response generation
│   ├── sft_baseline.py
│   ├── dpo_baseline.py
│   └── simpo_baseline.py
├── rps/                # RPS perturbation generation  
│   ├── angle_perturbation.py
│   ├── rps_generation.py
│   └── rps_vllm.py
├── scoring/            # Reward model scoring
│   ├── reward_scoring.py
│   └── best_selection.py
├── comparison/         # A/B comparison
│   ├── pairwise_judge.py
│   └── parallel_compare.py
├── utils/              # Utilities
│   ├── env_detection.py
│   ├── model_loaders.py
│   ├── data_utils.py
│   └── directions.py
├── configs/            # Configuration files
│   ├── models.yaml
│   └── datasets.yaml
└── pipelines/          # Complete pipeline scripts
    ├── run_helpsteer_pipeline.py
    ├── run_ultra_pipeline.py
    └── run_full_comparison.py
```

## Quick Start

### 1. Complete HelpSteer Pipeline
```bash
python pipelines/run_helpsteer_pipeline.py \
  --dataset nvidia/HelpSteer \
  --model mistralai/Mistral-7B-Instruct-v0.2 \
  --output_dir ./outputs \
  --num_prompts 2000
```

### 2. Individual Steps

#### Generate Baseline
```bash
python baseline_generation/sft_baseline.py \
  --dataset nvidia/HelpSteer \
  --model mistralai/Mistral-7B-Instruct-v0.2 \
  --output_dir ./outputs/baseline \
  --batch_size 16 \
  --num_responses 3
```

#### Generate RPS Responses
```bash
python rps/rps_generation.py \
  --dataset nvidia/HelpSteer \
  --model mistralai/Mistral-7B-Instruct-v0.2 \
  --output_dir ./outputs/rps \
  --angle_range -40,40 \
  --step 5 \
  --theta_max 30 \
  --top_k 5
```

#### Score and Select Best
```bash
python scoring/reward_scoring.py \
  --input_dir ./outputs/baseline \
  --output_dir ./outputs/baseline_scored

python scoring/best_selection.py \
  --input_dir ./outputs/baseline_scored \
  --output_dir ./outputs/baseline_best
```

#### Compare Baseline vs RPS
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
- **Generation**: Mistral-7B-Instruct-v0.2, Zephyr-7B-Beta, Gemma-2-9B-IT-SimPO
- **Scoring**: RewardModel-Mistral-7B-for-DPA-v1 
- **Judging**: GPT-4o-mini, GPT-4, OpenRouter models

### Robustness Features
- **Resume capability**: Automatic detection of existing outputs
- **Error recovery**: Graceful handling of model loading failures
- **Memory management**: Dynamic batch size adjustment
- **Parallel processing**: Multi-worker GPT judging with rate limiting

### Data Compatibility
- **HelpSteer**: nvidia/HelpSteer (validation/train splits)
- **UltraFeedback**: HuggingFaceH4/ultrafeedback_binarized
- **Custom datasets**: With prompt/response columns

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
- `responses_v{X}.csv`: Generated responses per direction
- `v{X}_best.csv`: Best responses after scoring
- `v{X}_judged.jsonl`: A/B comparison judgments  
- `comparison_summary.csv`: Win rate statistics

No model weights or caches are included in outputs.