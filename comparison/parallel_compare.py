"""
Parallel A/B Comparison between Baseline and RPS best responses.
Uses GPT (OpenAI or OpenRouter) to judge pairwise examples with (v1, v2) weights.
"""

import os
import re
import json
import argparse
import random
from pathlib import Path
import pandas as pd
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


def build_prompt(prompt: str, a: str, b: str, v1: float, v2: float) -> str:
    return (
        "You are a careful evaluation judge. Decide which response better aligns with the user "
        "preference that balances two criteria with given weights:\n"
        f"- Helpfulness (weight = {v1})\n- Verbosity (weight = {v2})\n"
        "(Do not reward length itself; penalize fluff and inaccuracies.)\n\n"
        "Output format (exactly):\n"
        "Comparison: <one-sentence comparison and reason>\n"
        "More aligned: A | B | Tie\n\n"
        f"Query: {prompt}\n\nResponse A: {a}\n\nResponse B: {b}"
    )


def judge_one(item: dict, model_name: str) -> str:
    try:
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError("pip install openai") from e
    api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENROUTER_API_KEY or OPENAI_API_KEY in environment")
    base_url = "https://openrouter.ai/api/v1" if os.environ.get("OPENROUTER_API_KEY") else None
    client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)

    prompt = build_prompt(item['prompt'], item['A'], item['B'], item['v1'], item['v2'])
    resp = client.chat.completions.create(
        model=model_name, messages=[{"role": "user", "content": prompt}], temperature=0.1, max_tokens=256
    )
    content = resp.choices[0].message.content.strip()
    m = re.search(r"More aligned:\s*(A|B|Tie)", content, re.IGNORECASE)
    winner = (m.group(1) if m else "Tie").upper()
    if winner not in {"A", "B", "TIE"}:
        winner = "TIE"
    return winner


def load_best(csv_dir: Path) -> dict:
    all_files = sorted([p for p in csv_dir.glob('**/*.csv')])
    by_dir = {}
    for p in all_files:
        m = re.search(r"v[1-8]", p.stem)
        if not m:
            continue
        d = m.group(0)
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        if not {'prompt_id','prompt'}.issubset(df.columns):
            continue
        if 'best_response' not in df.columns and 'response' in df.columns:
            df = df.rename(columns={'response':'best_response'})
        if 'best_response' not in df.columns:
            continue
        by_dir.setdefault(d, []).append(df)
    for k, v in by_dir.items():
        by_dir[k] = pd.concat(v, ignore_index=True).drop_duplicates(subset=['prompt_id'])
    return by_dir


def run_parallel_judging(pairs: list, model: str, batch_size: int = 50, max_workers: int = 8):
    results = []
    results_lock = threading.Lock()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(judge_one, item, model): i for i, item in enumerate(pairs)}
        batch = []
        batch_num = 0
        with tqdm(total=len(pairs), desc=f"Judging {model} (parallel)") as pbar:
            for future in as_completed(futures):
                try:
                    winner = future.result()
                    item_idx = futures[future]
                    item = pairs[item_idx]
                    out = dict(item)
                    out['judgment'] = winner
                    with results_lock:
                        results.append(out)
                        batch.append(out)
                    if len(batch) >= batch_size:
                        batch = []
                        batch_num += 1
                except Exception as e:
                    item_idx = futures[future]
                    item = pairs[item_idx]
                    fallback = dict(item)
                    fallback.update({'judgment':'TIE','error':str(e)})
                    with results_lock:
                        results.append(fallback)
                pbar.update(1)
    return results


def main():
    ap = argparse.ArgumentParser(description="Parallel A/B comparison between baseline and RPS")
    ap.add_argument('--baseline_dir', required=True)
    ap.add_argument('--rps_dir', required=True)
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--model', default='gpt-4o-mini')
    ap.add_argument('--max_workers', type=int, default=8)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base = load_best(Path(args.baseline_dir))
    rps = load_best(Path(args.rps_dir))

    summary_rows = []
    for d in [f"v{i}" for i in range(1,9)]:
        if d not in base or d not in rps:
            continue
        left = base[d][['prompt_id','prompt','best_response']].rename(columns={'best_response':'base'})
        right = rps[d][['prompt_id','prompt','best_response','v1','v2']].rename(columns={'best_response':'rps'})
        merged = pd.merge(left, right, on=['prompt_id','prompt'], how='inner')
        if merged.empty:
            continue

        pairs = []
        for _, row in merged.iterrows():
            a_is_base = random.random() < 0.5
            A = row['base'] if a_is_base else row['rps']
            B = row['rps'] if a_is_base else row['base']
            pairs.append({
                'prompt': str(row['prompt']),
                'A': str(A),
                'B': str(B),
                'a_is_baseline': a_is_base,
                'v1': float(row.get('v1', 0.7071)),
                'v2': float(row.get('v2', 0.7071)),
            })

        results = run_parallel_judging(pairs, args.model, max_workers=args.max_workers)
        out_jsonl = out_dir / f"{d}_judged.jsonl"
        with out_jsonl.open('w', encoding='utf-8') as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')

        base_wins = 0 ; rps_wins = 0 ; ties = 0
        for r in results:
            j = r['judgment']
            if j == 'A':
                base_wins += 1 if r.get('a_is_baseline') else 0
                rps_wins += 0 if r.get('a_is_baseline') else 1
            elif j == 'B':
                base_wins += 0 if r.get('a_is_baseline') else 1
                rps_wins += 1 if r.get('a_is_baseline') else 0
            else:
                ties += 1
        win_rate = rps_wins / (rps_wins + base_wins) if (rps_wins + base_wins) else 0.5
        summary_rows.append({'direction': d, 'total': len(results), 'baseline_wins': base_wins, 'rps_wins': rps_wins, 'ties': ties, 'rps_win_rate': win_rate})

    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(out_dir / 'comparison_summary.csv', index=False)
        print(f"Saved: {out_dir / 'comparison_summary.csv'}")


if __name__ == '__main__':
    main()


