"""
Direction vectors and angle calculations for RPS perturbation.
Based on the paper's preference direction definitions.
"""

import numpy as np
from typing import List, Tuple, Dict, Any

# Updated to v1-v8 as per paper
PREFERENCE_DIRECTIONS = {
    "v1": {"vector": (0.9848, 0.1736), "angle": 10},
    "v2": {"vector": (0.9659, 0.2588), "angle": 15},
    "v3": {"vector": (0.9397, 0.3420), "angle": 20},
    "v4": {"vector": (0.9063, 0.4226), "angle": 25},
    "v5": {"vector": (0.8660, 0.5000), "angle": 30},
    "v6": {"vector": (0.8192, 0.5736), "angle": 35},
    "v7": {"vector": (0.7660, 0.6428), "angle": 40},
    "v8": {"vector": (0.7071, 0.7071), "angle": 45},
}


def get_angle_perturbations(
    v_main: np.ndarray, 
    angle_range: Tuple[int, int] = (-40, 40), 
    step: int = 5, 
    theta_max: float = 30, 
    top_k: int = 5
) -> Tuple[List[np.ndarray], List[float]]:
    """
    Generate angle perturbations for RPS generation.
    
    Args:
        v_main: Main direction vector (v1, v2)
        angle_range: Range of angles to test in degrees
        step: Step size for angle generation
        theta_max: Maximum allowed angle difference from main direction
        top_k: Number of best perturbations to return
        
    Returns:
        Tuple of (valid_vectors, valid_angles)
    """
    def angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
        """Calculate angle between two vectors in degrees."""
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

    # Generate candidate angles
    angle_offsets = np.arange(angle_range[0], angle_range[1] + 1, step)
    perturbed_vs = []
    perturbed_angles = []
    angle_diffs = []

    for offset in angle_offsets:
        angle_rad = np.radians(offset)
        v = np.array([np.cos(angle_rad), np.sin(angle_rad)])
        angle_diff = angle_between(v, v_main)
        
        if angle_diff <= theta_max:
            perturbed_vs.append(v)
            perturbed_angles.append(offset)
            angle_diffs.append(angle_diff)

    # Select top_k closest to main direction
    sorted_indices = np.argsort(angle_diffs)
    top_indices = sorted_indices[:top_k]
    
    valid_vs = [perturbed_vs[i] for i in top_indices]
    valid_angles = [perturbed_angles[i] for i in top_indices]
    
    print(f"✅ Generated {len(valid_vs)} valid perturbation directions")
    for i, (v, a) in enumerate(zip(valid_vs, valid_angles)):
        print(f"  Perturbation {i+1}: angle={a:.1f}°, v=({v[0]:.4f}, {v[1]:.4f})")
    
    return valid_vs, valid_angles


def build_dpa_input(prompt: str, v1: float, v2: float) -> List[Dict[str, str]]:
    """
    Build DPA input with weighted instruction.
    
    Args:
        prompt: User prompt
        v1: Helpfulness weight (0-1)
        v2: Verbosity weight (0-1) 
        
    Returns:
        Chat format messages list
    """
    h = int(round(v1 * 100))
    v = int(round(v2 * 100))
    sys_instruction = (
        f"You are a helpful assistant. Your response should maximize weighted rating = "
        f"helpfulness*{h} + verbosity*{v}."
    )
    return [{"role": "user", "content": f"{sys_instruction}\n\n{prompt}"}]


def build_simpo_input(prompt: str, v1: float, v2: float) -> List[Dict[str, str]]:
    """
    Build SimPO/MetaAligner input format.
    
    Args:
        prompt: User prompt
        v1: Helpfulness weight (0-1)
        v2: Verbosity weight (0-1)
        
    Returns:
        Chat format messages list
    """
    h = int(round(v1 * 100))
    v = int(round(v2 * 100))
    
    query_prompt = (
        f'You are an assistant to human. You will be provided with a context and you need to provide a helpful response. '
        f'Your response should maximize weighted rating = helpfulness*{h} + verbosity*{v}. '
        f'Context: {prompt} | Response: '
    )
    
    return [{"role": "user", "content": query_prompt}]


def parse_direction_vector(direction_vector) -> Tuple[float, float]:
    """
    Parse direction vector from various formats.
    
    Args:
        direction_vector: String like "(0.9848, 0.1736)" or list/tuple
        
    Returns:
        Tuple of (v1, v2) floats
    """
    if isinstance(direction_vector, str):
        s = direction_vector.strip("[]()").replace(" ", "")
        v1s, v2s = s.split(",")
        return float(v1s), float(v2s)
    if isinstance(direction_vector, (list, tuple)) and len(direction_vector) == 2:
        return float(direction_vector[0]), float(direction_vector[1])
    raise ValueError(f"Invalid direction_vector format: {direction_vector}")
