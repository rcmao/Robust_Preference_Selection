"""
Reward Model Scoring
Scores responses with RewardModel-Mistral-7B-for-DPA-v1 and writes *_scored.csv files.
Also supports resume and periodic temp saves.
"""

import os
import sys
import glob
import argparse
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import torch

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.env_detection import setup_environment
from utils.model_loaders import load_reward_model
from utils.data_utils import read_csv_safe


def score_response(prompt: str, response: str, model, tokenizer, device) -> tuple:
    template = (
        "[INST] You must read the following conversation carefully and rate the assistant's "
        "response from score 0-100 in these aspects: helpfulness, correctness, coherence, "
        "honesty, complexity, verbosity\n\nUser: {prompt}\n\nAssistant: {response} [/INST]"
    )
    try:
        inputs = tokenizer(
            template.format(prompt=prompt, response=response),
            return_tensors="pt", truncation=True, max_length=2048,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits.squeeze().detach().cpu().numpy()
        helpfulness = float(logits[9])
        verbosity = float(logits[4])
        return helpfulness, verbosity
    except Exception:
        return 0.0, 0.0


def run_reward_scoring(input_dir: str, scored_dir: str, model, tokenizer, device):
    os.makedirs(scored_dir, exist_ok=True)
    files = sorted([p for p in glob.glob(os.path.join(input_dir, "*.csv"))])
    print(f"📁 Found {len(files)} CSV files to process in {input_dir}")
    
    for file_path in files:
        file = os.path.basename(file_path)
        print(f"\n📄 Processing: {file}")
        df = read_csv_safe(file_path)
        if df is None:
            continue
        if "prompt" not in df.columns or "response" not in df.columns:
            print("   ⚠️ Missing prompt/response columns; skipping file")
            continue

        # Ensure helpfulness/verbosity columns
        if "helpfulness" not in df.columns:
            df["helpfulness"] = np.nan
        if "verbosity" not in df.columns:
            df["verbosity"] = np.nan

        need = df[df["helpfulness"].isna() | df["verbosity"].isna()].shape[0]
        print(f"   🎯 Need to score {need} rows")
        if need == 0:
            print("   ✅ Already scored, skipping")
        else:
            for i, row in tqdm(df.iterrows(), total=len(df), desc=f"Scoring {file}"):
                if pd.notnull(row["helpfulness"]) and pd.notnull(row["verbosity"]):
                    continue
                try:
                    h, v = score_response(str(row["prompt"]), str(row["response"]), model, tokenizer, device)
                    df.loc[i, "helpfulness"] = h
                    df.loc[i, "verbosity"] = v
                    if (i + 1) % 200 == 0:
                        tmp = os.path.join(scored_dir, file.replace(".csv", "_temp_scored.csv"))
                        df.to_csv(tmp, index=False)
                except Exception as e:
                    print(f"   ❌ Row {i} error: {e}")
                    continue

        save_path = os.path.join(scored_dir, file.replace(".csv", "_scored.csv"))
        df.to_csv(save_path, index=False)
        print(f"   💾 Saved: {save_path}")
        tmp = os.path.join(scored_dir, file.replace(".csv", "_temp_scored.csv"))
        if os.path.exists(tmp):
            os.remove(tmp)


def main():
    ap = argparse.ArgumentParser(description="Reward Model Scoring")
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--environment", type=str, default=None, choices=['colab','server','local'])
    args = ap.parse_args()

    config = setup_environment(args.environment)
    device = config['device']
    model, tokenizer = load_reward_model(config)
    print(f"Loaded reward model on {device}")

    run_reward_scoring(args.input_dir, args.output_dir, model, tokenizer, device)


if __name__ == "__main__":
    main()


