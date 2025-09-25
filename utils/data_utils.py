"""
Data processing utilities for RPS pipeline.
Handles CSV operations, data validation, and file management.
"""

import os
import pandas as pd
from typing import List, Dict, Any, Optional
from pathlib import Path


def append_csv(df: pd.DataFrame, path: str, **kwargs) -> None:
    """
    Append DataFrame to CSV with header management.
    
    Args:
        df: DataFrame to append
        path: Output CSV path
        **kwargs: Additional arguments for to_csv
    """
    header = not os.path.exists(path)
    df.to_csv(path, mode='a', header=header, index=False, **kwargs)


def read_csv_safe(path: str, **kwargs) -> Optional[pd.DataFrame]:
    """
    Safely read CSV file with error handling.
    
    Args:
        path: CSV file path
        **kwargs: Additional arguments for read_csv
        
    Returns:
        DataFrame or None if reading fails
    """
    try:
        return pd.read_csv(path, **kwargs)
    except Exception as e:
        print(f"⚠️ Error reading {path}: {e}")
        return None


def validate_response_df(df: pd.DataFrame, required_columns: List[str] = None) -> bool:
    """
    Validate response DataFrame structure.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        
    Returns:
        True if valid, False otherwise
    """
    if required_columns is None:
        required_columns = ["prompt_id", "prompt", "response"]
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"❌ Missing required columns: {missing_columns}")
        return False
    
    # Check for empty responses
    if "response" in df.columns:
        empty_responses = df["response"].isna().sum()
        if empty_responses > 0:
            print(f"⚠️ Found {empty_responses} empty responses")
    
    return True


def setup_output_dir(base_dir: str, subdir: str = None) -> str:
    """
    Setup output directory with optional subdirectory.
    
    Args:
        base_dir: Base output directory
        subdir: Optional subdirectory name
        
    Returns:
        Full output directory path
    """
    if subdir:
        output_dir = os.path.join(base_dir, subdir)
    else:
        output_dir = base_dir
    
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def merge_csv_files(file_paths: List[str], output_path: str, **kwargs) -> pd.DataFrame:
    """
    Merge multiple CSV files into one.
    
    Args:
        file_paths: List of CSV file paths
        output_path: Output merged CSV path
        **kwargs: Additional arguments for read_csv
        
    Returns:
        Merged DataFrame
    """
    dfs = []
    for path in file_paths:
        df = read_csv_safe(path, **kwargs)
        if df is not None:
            dfs.append(df)
    
    if not dfs:
        raise ValueError("No valid CSV files found")
    
    merged_df = pd.concat(dfs, ignore_index=True)
    merged_df.to_csv(output_path, index=False)
    print(f"✅ Merged {len(dfs)} files into {output_path}: {len(merged_df)} rows")
    
    return merged_df


def clean_response_text(text: str) -> str:
    """
    Clean response text by removing unwanted characters.
    
    Args:
        text: Raw response text
        
    Returns:
        Cleaned text
    """
    if not isinstance(text, str):
        return str(text) if text is not None else ""
    
    # Remove common artifacts
    text = text.strip()
    
    # Remove repeated whitespace
    import re
    text = re.sub(r'\s+', ' ', text)
    
    return text


def filter_valid_responses(df: pd.DataFrame, min_length: int = 10) -> pd.DataFrame:
    """
    Filter DataFrame to keep only valid responses.
    
    Args:
        df: Input DataFrame
        min_length: Minimum response length
        
    Returns:
        Filtered DataFrame
    """
    initial_count = len(df)
    
    # Remove empty or very short responses
    if "response" in df.columns:
        df = df[df["response"].notna()]
        df = df[df["response"].astype(str).str.len() >= min_length]
    
    final_count = len(df)
    print(f"📊 Filtered responses: {initial_count} → {final_count} (-{initial_count - final_count})")
    
    return df


def get_file_statistics(file_path: str) -> Dict[str, Any]:
    """
    Get statistics about a CSV file.
    
    Args:
        file_path: Path to CSV file
        
    Returns:
        Dictionary with file statistics
    """
    if not os.path.exists(file_path):
        return {"exists": False}
    
    try:
        df = pd.read_csv(file_path)
        stats = {
            "exists": True,
            "rows": len(df),
            "columns": list(df.columns),
            "size_mb": os.path.getsize(file_path) / (1024 * 1024),
        }
        
        # Additional statistics for response data
        if "response" in df.columns:
            response_lengths = df["response"].astype(str).str.len()
            stats.update({
                "avg_response_length": response_lengths.mean(),
                "empty_responses": df["response"].isna().sum(),
            })
        
        if "prompt_id" in df.columns:
            stats["unique_prompts"] = df["prompt_id"].nunique()
        
        return stats
        
    except Exception as e:
        return {"exists": True, "error": str(e)}


def summarize_output_directory(output_dir: str) -> None:
    """
    Print summary of output directory contents.
    
    Args:
        output_dir: Directory to summarize
    """
    print(f"\n📁 Output Directory Summary: {output_dir}")
    print("-" * 60)
    
    if not os.path.exists(output_dir):
        print("❌ Directory does not exist")
        return
    
    csv_files = list(Path(output_dir).glob("**/*.csv"))
    
    if not csv_files:
        print("📄 No CSV files found")
        return
    
    total_size = 0
    total_rows = 0
    
    for file_path in sorted(csv_files):
        stats = get_file_statistics(str(file_path))
        if stats.get("exists") and "error" not in stats:
            print(f"📄 {file_path.name}: {stats['rows']} rows, {stats['size_mb']:.1f} MB")
            total_size += stats['size_mb']
            total_rows += stats['rows']
        else:
            print(f"⚠️ {file_path.name}: Error or empty")
    
    print("-" * 60)
    print(f"📊 Total: {len(csv_files)} files, {total_rows} rows, {total_size:.1f} MB")


def cleanup_temp_files(directory: str, pattern: str = "*_temp_*") -> None:
    """
    Clean up temporary files in directory.
    
    Args:
        directory: Directory to clean
        pattern: Glob pattern for temp files
    """
    temp_files = list(Path(directory).glob(pattern))
    for temp_file in temp_files:
        try:
            temp_file.unlink()
            print(f"🗑️ Removed temp file: {temp_file.name}")
        except Exception as e:
            print(f"⚠️ Failed to remove {temp_file.name}: {e}")
