"""
Environment detection and optimization utilities.
Handles Colab, Server, and Local environments with appropriate optimizations.
"""

import os
import torch
import psutil
from typing import Dict, Any, Optional
from pathlib import Path


def detect_environment() -> str:
    """
    Detect the current runtime environment.
    
    Returns:
        Environment type: 'colab', 'server', or 'local'
    """
    # Check for Colab
    try:
        import google.colab
        return 'colab'
    except ImportError:
        pass
    
    # Check for server environment (has GPU and high memory)
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        if gpu_memory >= 20:  # Server-grade GPU
            return 'server'
    
    return 'local'


def setup_environment(environment: Optional[str] = None, seed: int = 42) -> Dict[str, Any]:
    """
    Setup environment with appropriate configurations.
    
    Args:
        environment: Force specific environment type
        seed: Random seed
        
    Returns:
        Configuration dictionary
    """
    if environment is None:
        environment = detect_environment()
    
    # Set random seeds
    import numpy as np
    import random
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    config = {
        'environment': environment,
        'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        'seed': seed,
    }
    
    if environment == 'colab':
        config.update(_setup_colab())
    elif environment == 'server':
        config.update(_setup_server())
    else:
        config.update(_setup_local())
    
    print(f"🔧 Environment: {environment}")
    print(f"🔧 Device: {config['device']}")
    print(f"🔧 Batch size: {config['batch_size']}")
    print(f"🔧 Max tokens: {config['max_new_tokens']}")
    
    return config


def _setup_colab() -> Dict[str, Any]:
    """Setup Colab-specific configurations."""
    # Mount Google Drive if available
    try:
        from google.colab import drive
        drive.mount('/content/drive', force_remount=False)
        default_output = "/content/drive/MyDrive/rps_outputs"
        print("📎 Google Drive mounted successfully")
    except Exception:
        default_output = "/content/rps_outputs"
        print("⚠️ Google Drive mounting failed, using local storage")
    
    # Set HF mirror for Chinese users
    if os.environ.get('HF_ENDPOINT') != 'https://huggingface.co':
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
        print("🌏 Using HF mirror for better connectivity")
    
    # Memory optimization for Colab
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    return {
        'batch_size': 8,
        'max_new_tokens': 256,
        'torch_dtype': torch.float16,
        'device_map': 'auto',
        'low_cpu_mem_usage': True,
        'default_output_dir': default_output,
        'use_cache': True,
    }


def _setup_server() -> Dict[str, Any]:
    """Setup server-specific configurations."""
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    # Optimize based on GPU memory
    if gpu_memory >= 40:  # A100, L40S
        batch_size = 16
        max_tokens = 512
        dtype = torch.bfloat16
    elif gpu_memory >= 24:  # V100, RTX 3090
        batch_size = 12
        max_tokens = 512
        dtype = torch.float16
    else:  # T4, K80
        batch_size = 8
        max_tokens = 256
        dtype = torch.float16
    
    print(f"🔧 GPU Memory: {gpu_memory:.1f} GB - Using optimized settings")
    
    # Enable TF32 for performance
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    return {
        'batch_size': batch_size,
        'max_new_tokens': max_tokens,
        'torch_dtype': dtype,
        'device_map': 'auto',
        'low_cpu_mem_usage': True,
        'default_output_dir': './outputs',
        'use_cache': True,
        'gpu_memory_utilization': 0.6,  # For vLLM
    }


def _setup_local() -> Dict[str, Any]:
    """Setup local environment configurations."""
    # Check available memory
    memory_gb = psutil.virtual_memory().total / 1e9
    
    # Conservative settings for local environments
    batch_size = 4 if torch.cuda.is_available() else 2
    max_tokens = 256 if torch.cuda.is_available() else 128
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    print(f"🔧 System Memory: {memory_gb:.1f} GB - Using conservative settings")
    
    return {
        'batch_size': batch_size,
        'max_new_tokens': max_tokens,
        'torch_dtype': dtype,
        'device_map': 'auto' if torch.cuda.is_available() else None,
        'low_cpu_mem_usage': True,
        'default_output_dir': './outputs',
        'use_cache': False,  # Save disk space locally
    }


def get_hf_token() -> Optional[str]:
    """Get Hugging Face token from environment variables."""
    return (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
        or os.environ.get("TRANSFORMERS_TOKEN")
    )


def setup_hf_access(model_id: str) -> Dict[str, Any]:
    """
    Setup Hugging Face access for gated models.
    
    Args:
        model_id: Model identifier
        
    Returns:
        Dictionary of authentication kwargs
    """
    kwargs = {}
    
    # Check if model needs authentication
    if any(org in model_id for org in ["mistralai/", "meta-llama/", "microsoft/"]):
        # For gated models, prefer official endpoint
        if os.environ.get("HF_ENDPOINT") != "https://huggingface.co":
            print("🔐 Switching to official HF endpoint for gated model access")
            os.environ["HF_ENDPOINT"] = "https://huggingface.co"
        
        hf_token = get_hf_token()
        if hf_token:
            kwargs["token"] = hf_token
            print("✅ Using HF token for model access")
        else:
            print("⚠️ No HF token found. Some models may be inaccessible.")
            print("   Get token at: https://huggingface.co/settings/tokens")
    
    return kwargs


def optimize_memory():
    """Optimize memory usage for current environment."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # Enable memory-efficient attention if available
        try:
            torch.backends.cuda.enable_flash_sdp(True)
        except AttributeError:
            pass
