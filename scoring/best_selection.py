"""
Best Response Selection
Selects best response per prompt based on weighted DPA score using (v1, v2).
"""

import os
import sys
import glob
import argparse
import pandas as pd
import numpy as np

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_utils import read_csv_safe
from utils.directions import parse_direction_vector


def pick_best(df: pd.DataFrame) -> pd.DataFrame:
    # Detect columns
    prompt_col = None
    for cand in ["prompt", "question", "input", "user", "query"]:
        if cand in df.columns:
            prompt_col = cand
            break
    response_col = None
    for cand in ["response", "output", "answer", "completion"]:
        if cand in df.columns:
            response_col = cand
            break
    if prompt_col is None or response_col is None:
        return pd.DataFrame()

    # main_v1/main_v2 may be present; otherwise parse from direction_vector
    have_main = ("main_v1" in df.columns) and ("main_v2" in df.columns)

    group_key = "prompt_id" if "prompt_id" in df.columns else prompt_col
    results = []

    for gid, g in df.groupby(group_key):
        scored = []
        for idx, row in g.iterrows():
            prompt = str(row[prompt_col])
            response = str(row[response_col])
            h = float(row.get("helpfulness", np.nan))
            v = float(row.get("verbosity", np.nan))
            if np.isnan(h) or np.isnan(v):
                # require pre-scored df (from reward_scoring)
                continue
            if have_main:
                v1 = float(row["main_v1"])
                v2 = float(row["main_v2"])
            else:
                v1, v2 = parse_direction_vector(row.get("direction_vector"))
            dpa = v1 * h + v2 * v
            scored.append({
                "idx": idx,
                "prompt": prompt,
                "response": response,
                "helpfulness": h,
                "verbosity": v,
                "dpa_score": float(dpa),
                "response_id": row.get("response_id"),
            })
        if not scored:
            continue
        best = max(scored, key=lambda x: x["dpa_score"])
        src = df.loc[best["idx"]]
        # build output row
        if have_main:
            v1 = float(src["main_v1"]) ; v2 = float(src["main_v2"])
        else:
            v1, v2 = parse_direction_vector(src.get("direction_vector"))
        results.append({
            "prompt_id": src.get("prompt_id", gid),
            "prompt": best["prompt"],
            "best_response": best["response"],
            "helpfulness": best["helpfulness"],
            "verbosity": best["verbosity"],
            "v1": v1, "v2": v2,
            "dpa_score": best["dpa_score"],
        })
    return pd.DataFrame(results)


def main():
    ap = argparse.ArgumentParser(description="Select best response per prompt from scored CSVs")
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--output_path", required=True)
    args = ap.parse_args()

    scored_files = sorted([p for p in glob.glob(os.path.join(args.input_dir, "*_scored.csv"))])
    if not scored_files:
        print(f"❌ No scored files in {args.input_dir}")
        return
    dfs = []
    for p in scored_files:
        df = read_csv_safe(p)
        if df is not None:
            dfs.append(df)
    if not dfs:
        print("❌ No valid scored data")
        return
    df_all = pd.concat(dfs, ignore_index=True)
    best = pick_best(df_all)
    if len(best) == 0:
        print("❌ No best rows to save")
        return
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    best.to_csv(args.output_path, index=False)
    print(f"🏆 Best saved: {args.output_path} ({len(best)} rows)")


if __name__ == "__main__":
    main()


