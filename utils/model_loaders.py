"""
Model loading utilities with error handling and optimization.
Supports multiple model types and environments.
"""

import os
import torch
from typing import Tuple, Optional, Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from .env_detection import setup_hf_access, optimize_memory


def ensure_pad_token(tokenizer):
    """Ensure tokenizer has a pad token."""
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        # For some models, set a different pad token to avoid confusion
        if tokenizer.pad_token_id == tokenizer.eos_token_id:
            tokenizer.pad_token = "<pad>"
            try:
                tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("<pad>")
            except:
                pass  # Keep eos as pad if conversion fails
    return tokenizer


def setup_chat_template(tokenizer, model_name: str):
    """Setup chat template for models that don't have one."""
    if tokenizer.chat_template is None:
        if "MetaAligner" in model_name or "metaaligner" in model_name.lower():
            # Simple template for MetaAligner
            tokenizer.chat_template = "{{ messages[0]['content'] }}"
            print("🔧 Set simple chat template for MetaAligner model")
        else:
            # Generic template for other models
            tokenizer.chat_template = (
                "{% for message in messages %}"
                "{% if message['role'] == 'user' %}{{ message['content'] }}{% endif %}"
                "{% endfor %}"
            )
            print("🔧 Set generic chat template")
    return tokenizer


def load_generation_model(
    model_id: str, 
    config: Dict[str, Any],
    use_vllm: bool = False
) -> Tuple[Any, AutoTokenizer]:
    """
    Load generation model with environment-appropriate settings.
    
    Args:
        model_id: Model identifier
        config: Environment configuration
        use_vllm: Whether to use vLLM for acceleration
        
    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"🤖 Loading generation model: {model_id}")
    
    if use_vllm:
        return _load_vllm_model(model_id, config)
    else:
        return _load_transformers_model(model_id, config)


def _load_transformers_model(model_id: str, config: Dict[str, Any]) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load model using transformers library."""
    hf_kwargs = setup_hf_access(model_id)
    
    common_kwargs = {
        'torch_dtype': config['torch_dtype'],
        'trust_remote_code': True,
        'resume_download': True,
        'low_cpu_mem_usage': config['low_cpu_mem_usage'],
        **hf_kwargs
    }
    
    try:
        # Try with device_map first
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=config['device_map'],
            **common_kwargs
        )
        print(f"✅ Model loaded with device_map: {config['device_map']}")
        
    except Exception as e:
        print(f"⚠️ Device map loading failed: {e}")
        print("🔄 Retrying with explicit device placement...")
        
        # Fallback to explicit device placement
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=None,
            **common_kwargs
        ).to(config['device'])
        print(f"✅ Model loaded on device: {config['device']}")
    
    # Enable optimizations
    if hasattr(model, 'gradient_checkpointing_enable') and config.get('use_cache', False):
        model.gradient_checkpointing_enable()
        print("🔧 Enabled gradient checkpointing")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        **hf_kwargs
    )
    tokenizer.padding_side = "left"  # Important for decoder-only models
    tokenizer = ensure_pad_token(tokenizer)
    tokenizer = setup_chat_template(tokenizer, model_id)
    
    print("✅ Generation model and tokenizer loaded successfully!")
    return model, tokenizer


def _load_vllm_model(model_id: str, config: Dict[str, Any]) -> Tuple[Any, AutoTokenizer]:
    """Load model using vLLM for acceleration."""
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        raise ImportError("vLLM not installed. Run: pip install vllm")
    
    print("🚀 Loading model with vLLM acceleration...")
    
    vllm_kwargs = {
        'model': model_id,
        'tensor_parallel_size': 1,
        'gpu_memory_utilization': config.get('gpu_memory_utilization', 0.6),
        'max_model_len': 4096,
        'trust_remote_code': True,
    }
    
    # Add authentication if needed
    hf_kwargs = setup_hf_access(model_id)
    if 'token' in hf_kwargs:
        # vLLM may need token through environment
        os.environ['HF_TOKEN'] = hf_kwargs['token']
    
    try:
        model = LLM(**vllm_kwargs)
        print("✅ vLLM model loaded successfully!")
        
        # Load tokenizer separately
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
            **hf_kwargs
        )
        tokenizer.padding_side = "left"
        tokenizer = ensure_pad_token(tokenizer)
        tokenizer = setup_chat_template(tokenizer, model_id)
        
        return model, tokenizer
        
    except Exception as e:
        print(f"❌ vLLM loading failed: {e}")
        print("🔄 Falling back to transformers...")
        return _load_transformers_model(model_id, config)


def load_reward_model(config: Dict[str, Any]) -> Tuple[AutoModelForSequenceClassification, AutoTokenizer]:
    """
    Load reward model for scoring.
    
    Args:
        config: Environment configuration
        
    Returns:
        Tuple of (reward_model, tokenizer)
    """
    model_id = "Haoxiang-Wang/RewardModel-Mistral-7B-for-DPA-v1"
    print(f"🎯 Loading reward model: {model_id}")
    
    hf_kwargs = setup_hf_access(model_id)
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        trust_remote_code=True,
        resume_download=True,
        torch_dtype=config['torch_dtype'],
        low_cpu_mem_usage=config['low_cpu_mem_usage'],
        device_map=config['device_map'],
        **hf_kwargs
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        **hf_kwargs
    )
    tokenizer = ensure_pad_token(tokenizer)
    
    print("✅ Reward model loaded successfully!")
    return model, tokenizer


def create_vllm_sampling_params(config: Dict[str, Any], num_responses: int = 1) -> Any:
    """Create vLLM sampling parameters."""
    try:
        from vllm import SamplingParams
        return SamplingParams(
            n=num_responses,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            max_tokens=config['max_new_tokens'],
            min_tokens=50 if num_responses > 1 else 1,  # Avoid empty responses
        )
    except ImportError:
        raise ImportError("vLLM not installed")


def cleanup_model(model, tokenizer=None):
    """Clean up model and free memory."""
    if hasattr(model, 'cpu'):
        model.cpu()
    del model
    if tokenizer:
        del tokenizer
    optimize_memory()
    print("🗑️ Model cleaned up and memory freed")
