"""
SFT Baseline Generation
Generates baseline responses using standard fine-tuned models.
Based on experiments/helpsteer/sft/sft_baseline_helpsteer.py
"""

import os
import sys
import argparse
import time
import pandas as pd
import torch
from datasets import load_dataset
from tqdm.auto import tqdm

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.env_detection import setup_environment
from utils.model_loaders import load_generation_model, cleanup_model
from utils.directions import PREFERENCE_DIRECTIONS, build_dpa_input
from utils.data_utils import append_csv, setup_output_dir


def generate_responses_for_direction(
    prompt: str,
    prompt_id: int,
    direction_name: str,
    direction_info: dict,
    model,
    tokenizer,
    device: torch.device,
    config: dict,
    num_responses: int = 3,
):
    """
    Generate multiple responses for a single prompt in a specific direction.
    
    Args:
        prompt: Input prompt
        prompt_id: Unique prompt identifier
        direction_name: Direction name (v1-v8)
        direction_info: Direction metadata (vector, angle)
        model: Generation model
        tokenizer: Model tokenizer
        device: Torch device
        config: Environment configuration
        num_responses: Number of responses to generate
        
    Returns:
        List of response dictionaries
    """
    try:
        v1, v2 = direction_info["vector"]
        angle = direction_info["angle"]
        
        # Build input using DPA format
        messages = build_dpa_input(prompt, v1, v2)
        input_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(device)
        
        # Generation parameters
        max_new_tokens = config['max_new_tokens']
        
        with torch.no_grad():
            attention_mask = torch.ones_like(input_ids)
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                num_return_sequences=num_responses,
                pad_token_id=tokenizer.eos_token_id,
                top_p=0.9,
                repetition_penalty=1.1,
                use_cache=config.get('use_cache', True),
            )
        
        responses = []
        for i in range(num_responses):
            generated_tokens = outputs[i][input_ids.shape[1]:]
            decoded = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            responses.append({
                "prompt_id": prompt_id,
                "prompt": prompt,
                "direction_name": direction_name,
                "direction_vector": f"({v1:.4f}, {v2:.4f})",
                "angle_degrees": angle,
                "response_id": i + 1,
                "response": decoded,
                "v1": v1,
                "v2": v2,
                "main_v1": v1,  # For baseline, main == current
                "main_v2": v2,
            })
        
        return responses
        
    except Exception as e:
        print(f"⚠️ Error generating responses for prompt {prompt_id} direction {direction_name}: {e}")
        return []


def generate_all_directions(
    prompts: list,
    prompt_ids: list,
    model,
    tokenizer,
    device: torch.device,
    config: dict,
    output_dir: str,
    num_responses: int = 3,
):
    """
    Generate responses for all directions with resume capability.
    
    Args:
        prompts: List of input prompts
        prompt_ids: List of prompt IDs
        model: Generation model
        tokenizer: Model tokenizer
        device: Torch device
        config: Environment configuration
        output_dir: Output directory
        num_responses: Number of responses per prompt
    """
    start_time = time.time()
    total_responses = 0
    
    for direction_name, direction_info in PREFERENCE_DIRECTIONS.items():
        print(f"\n🎯 Processing direction {direction_name}: {direction_info['vector']} ({direction_info['angle']}°)")
        
        output_file = os.path.join(output_dir, f"baseline_responses_{direction_name}.csv")
        
        # Resume capability: check existing results
        done_prompt_ids = set()
        if os.path.exists(output_file):
            try:
                existing_df = pd.read_csv(output_file)
                done_prompt_ids = set(existing_df["prompt_id"].unique())
                print(f"🔁 Found existing results for {len(done_prompt_ids)} prompts in {direction_name}")
            except Exception as e:
                print(f"⚠️ Error loading existing file for {direction_name}: {e}")
        
        remaining_prompt_ids = [pid for pid in prompt_ids if pid not in done_prompt_ids]
        print(f"📊 {direction_name}: Completed {len(done_prompt_ids)}, Remaining {len(remaining_prompt_ids)}")
        
        if not remaining_prompt_ids:
            print(f"✅ Direction {direction_name} already completed")
            continue
        
        # Batch processing
        batch_size = config['batch_size']
        direction_results = []
        
        for start in tqdm(range(0, len(prompts), batch_size), desc=f"Generating {direction_name}"):
            end = min(start + batch_size, len(prompts))
            batch_prompts = prompts[start:end]
            batch_ids = prompt_ids[start:end]
            
            # Skip already processed prompts
            unprocessed_indices = [i for i, pid in enumerate(batch_ids) if pid not in done_prompt_ids]
            if not unprocessed_indices:
                continue
            
            batch_results = []
            for i in unprocessed_indices:
                prompt = batch_prompts[i]
                prompt_id = batch_ids[i]
                
                responses = generate_responses_for_direction(
                    prompt, prompt_id, direction_name, direction_info,
                    model, tokenizer, device, config, num_responses
                )
                
                if responses:
                    batch_results.extend(responses)
            
            # Save batch results
            if batch_results:
                append_csv(pd.DataFrame(batch_results), output_file)
                direction_results.extend(batch_results)
                
                # Memory management
                if config['environment'] == 'colab':
                    torch.cuda.empty_cache()
        
        total_responses += len(direction_results)
        print(f"✅ Completed direction {direction_name}: {len(direction_results)} responses")
    
    elapsed = time.time() - start_time
    print(f"\n🏁 All directions completed in {elapsed:.1f} seconds")
    print(f"📊 Total responses generated: {total_responses}")


def main():
    """Main function for SFT baseline generation."""
    parser = argparse.ArgumentParser(description="SFT Baseline Generation")
    parser.add_argument("--dataset", type=str, default="nvidia/HelpSteer", 
                       help="Dataset name")
    parser.add_argument("--split", type=str, default="validation",
                       help="Dataset split")
    parser.add_argument("--num_prompts", type=int, default=2000,
                       help="Number of prompts to process")
    parser.add_argument("--model", type=str, required=True,
                       help="Model name or path")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory")
    parser.add_argument("--num_responses", type=int, default=3,
                       help="Number of responses per prompt")
    parser.add_argument("--batch_size", type=int, default=None,
                       help="Batch size (auto-detect if not set)")
    parser.add_argument("--environment", type=str, default=None,
                       choices=['colab', 'server', 'local'],
                       help="Force specific environment")
    parser.add_argument("--use_vllm", action="store_true",
                       help="Use vLLM for acceleration")
    
    args = parser.parse_args()
    
    # Setup environment
    config = setup_environment(args.environment)
    if args.batch_size:
        config['batch_size'] = args.batch_size
    
    # Setup output directory
    if args.output_dir is None:
        args.output_dir = config.get('default_output_dir', './outputs')
    output_dir = setup_output_dir(args.output_dir, 'baseline')
    
    print(f"🚀 SFT Baseline Generation")
    print(f"📊 Dataset: {args.dataset} ({args.split})")
    print(f"🤖 Model: {args.model}")
    print(f"📁 Output: {output_dir}")
    print(f"🔧 Environment: {config['environment']}")
    
    # Load dataset
    print("📦 Loading dataset...")
    ds = load_dataset(args.dataset, split=args.split)
    if "prompt" not in ds.column_names:
        raise ValueError("Dataset must contain 'prompt' column")
    
    prompts = ds["prompt"][:args.num_prompts]
    prompt_ids = list(range(len(prompts)))
    print(f"✅ Loaded {len(prompts)} prompts")
    
    # Load model
    model, tokenizer = load_generation_model(args.model, config, args.use_vllm)
    
    try:
        # Generate responses
        print(f"\n🎯 Processing {len(PREFERENCE_DIRECTIONS)} directions:")
        for name, info in PREFERENCE_DIRECTIONS.items():
            print(f"  {name}: {info['vector']} ({info['angle']}°)")
        
        generate_all_directions(
            prompts, prompt_ids, model, tokenizer, config['device'], config,
            output_dir, args.num_responses
        )
        
        print(f"\n✅ Generation completed! Results saved to: {output_dir}")
        
        # Show final statistics
        print(f"\n📈 Final statistics:")
        for direction_name in PREFERENCE_DIRECTIONS.keys():
            output_file = os.path.join(output_dir, f"baseline_responses_{direction_name}.csv")
            if os.path.exists(output_file):
                df = pd.read_csv(output_file)
                print(f"  {direction_name}: {len(df)} responses")
        
    finally:
        # Cleanup
        cleanup_model(model, tokenizer)


if __name__ == "__main__":
    main()
