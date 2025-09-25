"""
RPS Perturbation Generation
Generates responses along perturbed directions around each main direction (v1-v8).
Based on experiments' angle perturbation and generation logic, with vLLM optional.
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import torch

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.env_detection import setup_environment
from utils.model_loaders import load_generation_model, create_vllm_sampling_params, cleanup_model
from utils.directions import PREFERENCE_DIRECTIONS, get_angle_perturbations, build_dpa_input
from utils.data_utils import append_csv, setup_output_dir


def generate_batch_transformers(prompts_batch, prompt_ids_batch, v1, v2, model, tokenizer, device, max_new_tokens):
    messages_list = [build_dpa_input(p, v1, v2) for p in prompts_batch]
    input_ids_list = [
        tokenizer.apply_chat_template(m, add_generation_prompt=True, return_tensors="pt")[0]
        for m in messages_list
    ]
    input_ids_padded = torch.nn.utils.rnn.pad_sequence(
        input_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id
    ).to(device)
    attention_mask = (input_ids_padded != tokenizer.pad_token_id).to(device)
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids_padded,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
            top_p=0.9,
            repetition_penalty=1.1,
            use_cache=True,
        )
    responses = []
    for i, input_ids in enumerate(input_ids_list):
        generated_tokens = outputs[i][input_ids.shape[0]:]
        decoded = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        responses.append(decoded)
    return responses


def generate_batch_vllm(prompts_batch, v1, v2, vllm_model, tokenizer, sampling_params):
    inputs = []
    for p in prompts_batch:
        msgs = build_dpa_input(p, v1, v2)
        text = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        inputs.append(text)
    outputs = vllm_model.generate(inputs, sampling_params)
    return [out.outputs[0].text for out in outputs]


def main():
    ap = argparse.ArgumentParser(description="RPS Perturbation Generation")
    ap.add_argument("--dataset", type=str, default="nvidia/HelpSteer")
    ap.add_argument("--split", type=str, default="validation")
    ap.add_argument("--num_prompts", type=int, default=2000)
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--output_dir", type=str, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--use_vllm", action="store_true")
    ap.add_argument("--angle_range", type=str, default="-40,40")
    ap.add_argument("--step", type=int, default=5)
    ap.add_argument("--theta_max", type=int, default=30)
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--environment", type=str, default=None, choices=['colab','server','local'])
    args = ap.parse_args()

    # Env
    config = setup_environment(args.environment)
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.output_dir is None:
        args.output_dir = config.get('default_output_dir', './outputs')
    out_root = setup_output_dir(args.output_dir, 'rps')

    # Dataset
    from datasets import load_dataset
    print("📦 Loading dataset...")
    ds = load_dataset(args.dataset, split=args.split)
    if "prompt" not in ds.column_names:
        raise ValueError("Dataset must contain 'prompt' column")
    prompts = ds["prompt"][:args.num_prompts]
    prompt_ids = list(range(len(prompts)))
    print(f"✅ Loaded {len(prompts)} prompts")

    # Model
    model, tokenizer = load_generation_model(args.model, config, args.use_vllm)
    sampling_params = create_vllm_sampling_params(config) if args.use_vllm else None

    # Angle settings
    angle_lo, angle_hi = [int(x) for x in args.angle_range.split(',')]

    try:
        for name, info in PREFERENCE_DIRECTIONS.items():
            v_main = np.array(info['vector'])
            valid_vs, valid_angles = get_angle_perturbations(
                v_main=v_main,
                angle_range=(angle_lo, angle_hi),
                step=args.step,
                theta_max=args.theta_max,
                top_k=args.top_k,
            )
            dir_out = os.path.join(out_root, name)
            os.makedirs(dir_out, exist_ok=True)

            print(f"\n🎯 Direction {name}: {info['vector']} ({info['angle']}°) → {len(valid_vs)} perturbations")
            for j, (v_vec, angle_deg) in enumerate(zip(valid_vs, valid_angles)):
                v1, v2 = float(v_vec[0]), float(v_vec[1])
                out_csv = os.path.join(dir_out, f"rps_angle{int(angle_deg)}.csv")

                done_ids = set()
                if os.path.exists(out_csv):
                    try:
                        ex = pd.read_csv(out_csv)
                        done_ids = set(ex.get("prompt_id", pd.Series(dtype=int)).unique())
                        print(f"🔁 Resume {name}@{angle_deg}°: {len(done_ids)} done")
                    except Exception:
                        pass

                bs = config['batch_size']
                for start in tqdm(range(0, len(prompts), bs), desc=f"{name}@{angle_deg}°"):
                    end = min(start + bs, len(prompts))
                    batch_prompts_all = prompts[start:end]
                    batch_ids_all = prompt_ids[start:end]
                    idxs = [i for i, pid in enumerate(batch_ids_all) if pid not in done_ids]
                    if not idxs:
                        continue
                    batch_prompts = [batch_prompts_all[i] for i in idxs]
                    batch_ids = [batch_ids_all[i] for i in idxs]

                    try:
                        if args.use_vllm:
                            batch_texts = generate_batch_vllm(batch_prompts, v1, v2, model, tokenizer, sampling_params)
                        else:
                            batch_texts = generate_batch_transformers(batch_prompts, batch_ids, v1, v2, model, tokenizer, config['device'], config['max_new_tokens'])
                        rows = []
                        for pid, prompt, resp in zip(batch_ids, batch_prompts, batch_texts):
                            rows.append({
                                "prompt_id": pid,
                                "prompt": prompt,
                                "response": resp,
                                "v1": v1, "v2": v2,
                                "direction_name": name,
                                "direction_vector": f"({v1:.4f}, {v2:.4f})",
                                "angle_degrees": float(angle_deg),
                                "main_v1": float(v_main[0]),
                                "main_v2": float(v_main[1]),
                            })
                        if rows:
                            append_csv(pd.DataFrame(rows), out_csv)
                    except Exception as e:
                        print(f"⚠️ Batch {start}-{end} failed: {e}")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

        print(f"\n✅ RPS generation completed. Outputs in {out_root}")
    finally:
        cleanup_model(model, tokenizer)


if __name__ == "__main__":
    main()


