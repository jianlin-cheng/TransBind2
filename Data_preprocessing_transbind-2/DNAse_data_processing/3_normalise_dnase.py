"""
normalize_dnase.py
Apply log1p + global z-score normalization to all DNase signals.

Usage:
    python normalize_dnase.py
"""

import numpy as np
import os
from glob import glob
import json
import time

# ============================================================
# CONFIG
# ============================================================
HOME_DIR = "/path/to/HOME_DIR"

input_dir = os.path.join(
    HOME_DIR,
    "data/dnase_processed"
)

output_dir = os.path.join(
    HOME_DIR,
    "data/dnase_normalized"
)

# ============================================================


def normalize_dnase_signals(input_dir, output_dir):
    """
    Two-pass normalization:
    1. Collect global statistics from ALL files
    2. Apply log1p + z-score normalization
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all .npy files (excluding metadata)
    npy_files = sorted(glob(os.path.join(input_dir, "*.npy")))
    npy_files = [f for f in npy_files if 'cell_idx' not in os.path.basename(f)]
    
    print(f"Found {len(npy_files)} .npy files\n")
    print("="*70)
    
    # ========== PASS 1: Collect global statistics ==========
    print("\nPASS 1: Collecting global statistics from all files...")
    print("="*70)
    
    all_nonzero_log = []
    total_nonzero = 0
    
    t_start = time.time()
    
    for i, npy_file in enumerate(npy_files, 1):
        filename = os.path.basename(npy_file)
        print(f"  [{i:3d}/{len(npy_files)}] Loading {filename:40s}", end=' ', flush=True)
        
        signals = np.load(npy_file)
        nonzero_mask = signals > 0
        nonzero_vals = signals[nonzero_mask]
        
        # Apply log1p and collect
        log_vals = np.log1p(nonzero_vals)
        all_nonzero_log.append(log_vals)
        
        total_nonzero += len(nonzero_vals)
        print(f"({len(nonzero_vals):>12,} non-zero)", flush=True)
        
        # Free memory
        del signals, nonzero_vals, log_vals
    
    print(f"\n  Total time: {time.time() - t_start:.1f}s")
    
    # Concatenate and compute global statistics
    print(f"\n  Concatenating {total_nonzero:,} non-zero values...", flush=True)
    all_log_values = np.concatenate(all_nonzero_log)
    
    print(f"  Computing statistics...", flush=True)
    global_mean = float(all_log_values.mean())
    global_std = float(all_log_values.std())
    global_min = float(all_log_values.min())
    global_max = float(all_log_values.max())
    
    print(f"\n{'='*70}")
    print(f"GLOBAL STATISTICS (after log1p, non-zero values only):")
    print(f"{'='*70}")
    print(f"  Total samples:  {len(all_log_values):>15,}")
    print(f"  Mean:           {global_mean:>15.4f}")
    print(f"  Std:            {global_std:>15.4f}")
    print(f"  Min:            {global_min:>15.4f}")
    print(f"  Max:            {global_max:>15.4f}")
    print(f"{'='*70}\n")
    
    # Free memory
    del all_nonzero_log, all_log_values
    
    # ========== PASS 2: Normalize and save ==========
    print("\nPASS 2: Normalizing and saving files...")
    print("="*70 + "\n")
    
    t_start = time.time()
    
    for i, npy_file in enumerate(npy_files, 1):
        filename = os.path.basename(npy_file)
        print(f"  [{i:3d}/{len(npy_files)}] Processing {filename:40s}", end=' ', flush=True)
        
        t_file = time.time()
        
        # Load
        signals = np.load(npy_file)
        
        # Step 1: Log transform
        signals_log = np.log1p(signals)
        
        # Step 2: Global z-score normalization
        signals_norm = (signals_log - global_mean) / global_std
        
        # Step 3: Clip extreme outliers
        signals_norm = np.clip(signals_norm, -5, 5)
        
        # Save as float32 to save space
        out_path = os.path.join(output_dir, filename)
        np.save(out_path, signals_norm.astype(np.float32))
        
        # Report statistics
        nonzero_mask = signals > 0
        nonzero_pct = 100 * nonzero_mask.sum() / signals.size
        
        print(f"({time.time() - t_file:4.1f}s)", flush=True)
        print(f"       Raw: [{signals.min():>7.1f}, {signals.max():>7.1f}] "
              f"| Non-zero: {nonzero_pct:>4.1f}%")
        print(f"       Norm: [{signals_norm.min():>6.3f}, {signals_norm.max():>6.3f}] "
              f"| Mean: {signals_norm.mean():>6.3f} | Std: {signals_norm.std():>5.3f}\n")
        
        del signals, signals_log, signals_norm
    
    print(f"  Total time: {time.time() - t_start:.1f}s")
    
    # ========== Save normalization parameters ==========
    norm_params = {
        'method': 'log1p + global_zscore + clip',
        'global_mean': global_mean,
        'global_std': global_std,
        'clip_min': -5.0,
        'clip_max': 5.0,
        'total_nonzero_samples': int(total_nonzero),
        'num_files': len(npy_files),
        'notes': 'Statistics computed from non-zero values across all train/val/test splits and all cell types'
    }
    
    param_path = os.path.join(output_dir, 'normalization_params.json')
    with open(param_path, 'w') as f:
        json.dump(norm_params, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"✓ NORMALIZATION COMPLETE!")
    print(f"{'='*70}")
    print(f"  Processed files:  {len(npy_files)}")
    print(f"  Output directory: {output_dir}")
    print(f"  Parameters saved: {param_path}")
    print(f"{'='*70}\n")
    
    # Copy cell_idx mapping if it exists
    mapping_file = os.path.join(input_dir, 'cell_idx_to_name.json')
    if os.path.exists(mapping_file):
        import shutil
        out_mapping = os.path.join(output_dir, 'cell_idx_to_name.json')
        shutil.copy(mapping_file, out_mapping)
        print(f"✓ Copied cell_idx_to_name.json\n")
    
    # Print example transformations
    print("\nEXAMPLE TRANSFORMATIONS:")
    print("="*70)
    example_values = [0, 10, 20, 50, 100, 200, 500, 1000]
    print(f"{'Raw Value':>12} {'log1p':>12} {'z-score':>12} {'clipped':>12}")
    print("-"*70)
    for val in example_values:
        log_val = np.log1p(val)
        z_val = (log_val - global_mean) / global_std
        clipped_val = np.clip(z_val, -5, 5)
        print(f"{val:>12.0f} {log_val:>12.4f} {z_val:>12.4f} {clipped_val:>12.4f}")
    print("="*70 + "\n")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("DNase Signal Normalization Pipeline")
    print("="*70)
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print("="*70 + "\n")
    
    normalize_dnase_signals(input_dir, output_dir)
    
    print("Done! 🎉\n")