import numpy as np
import pandas as pd
import pyBigWig
import os
from tqdm import tqdm

# ============================================================
# Extract uniqueness scores for all splits using coordinates
# Same approach as DNase extraction - uses SAME coordinate files
# ============================================================

print("=" * 70)
print("UNIQUENESS SCORE EXTRACTION")
print("=" * 70)

HOME_DIR = "/path/to/HOME_DIR"

coords_dir = os.path.join(
    HOME_DIR,
    "data/Binary_dataV1"
)

output_dir = os.path.join(
    HOME_DIR,
    "data/uniqueness_processed"
)

uniqueness_bw_path = os.path.join(
    HOME_DIR,
    "data/downloads/uniqueness/ENCFF000KTT.bigWig"
)

os.makedirs(output_dir, exist_ok=True)

# ============================================================
# Check if BigWig exists
# ============================================================

if not os.path.exists(uniqueness_bw_path):
    print(f"\n⚠️  BigWig file not found: {uniqueness_bw_path}")
    print("\nTo download the Duke uniqueness file:")
    print("  For hg19: Search UCSC Genome Browser for 'Duke uniqueness'")
    print("\nExiting without extracting.")
    exit(1)

print(f"✓ Found BigWig: {uniqueness_bw_path}")

# ============================================================
# Extract uniqueness for each split
# ============================================================

splits = ['train', 'val', 'test']

for split_name in splits:
    coords_path = os.path.join(coords_dir, f"{split_name}_coords.csv")
    
    if not os.path.exists(coords_path):
        print(f"⚠️  Coordinates file not found: {coords_path}")
        continue
    
    print(f"\n{'='*70}")
    print(f"Processing {split_name}")
    print(f"{'='*70}")
    
    # Load coordinates
    coords_df = pd.read_csv(coords_path)
    print(f"Loaded {len(coords_df):,} rows")
    
    # Open BigWig
    bw = pyBigWig.open(uniqueness_bw_path)
    
    # Get unique genomic positions (first half = originals, second half = RC of same positions)
    N = len(coords_df) // 2
    unique_coords = coords_df.iloc[:N]  # First N rows = unique positions
    
    print(f"Unique positions: {N:,}")
    print(f"With RC (in output): {len(coords_df):,}")
    
    # Extract uniqueness scores for unique positions
    uniqueness_scores = np.zeros(N, dtype=np.float32)
    missing_chroms = set()
    failed = 0
    
    print(f"\nExtracting uniqueness scores...")
    for idx in tqdm(range(N), desc=f"  {split_name}"):
        row = unique_coords.iloc[idx]
        chrom = row['chrom']
        win_start = int(row['window_start'])
        win_end = int(row['window_end'])
        
        try:
            # Get mean uniqueness value across the window
            # Duke uniqueness is per-base, so we average across the window
            values = bw.stats(chrom, win_start, win_end, type="mean")
            
            if values is None or values[0] is None:
                uniqueness_scores[idx] = 0.0
            else:
                uniqueness_scores[idx] = values[0]
        except RuntimeError:
            # Chromosome not in BigWig
            missing_chroms.add(chrom)
            uniqueness_scores[idx] = 0.0
            failed += 1
    
    bw.close()
    
    if missing_chroms:
        print(f"⚠️  Chromosomes not found in BigWig: {missing_chroms}")
    if failed > 0:
        print(f"⚠️  {failed:,} positions had no data")
    
    # Replicate for RC (same uniqueness for both strands)
    uniqueness_full = np.concatenate([uniqueness_scores, uniqueness_scores])
    
    # Save
    output_path = os.path.join(output_dir, f"{split_name}_uniqueness.npy")
    np.save(output_path, uniqueness_full)
    print(f"✓ Saved: {output_path}")
    print(f"  Shape: {uniqueness_full.shape}")
    
    # Statistics
    print(f"\nStatistics (unique positions only):")
    print(f"  Mean:   {np.mean(uniqueness_scores):.4f}")
    print(f"  Median: {np.median(uniqueness_scores):.4f}")
    print(f"  Std:    {np.std(uniqueness_scores):.4f}")
    print(f"  Min:    {np.min(uniqueness_scores):.4f}")
    print(f"  Max:    {np.max(uniqueness_scores):.4f}")
    print(f"  Q1:     {np.percentile(uniqueness_scores, 25):.4f}")
    print(f"  Q3:     {np.percentile(uniqueness_scores, 75):.4f}")
    
    # Distribution
    print(f"\n  Distribution:")
    bins_hist = np.linspace(0, 1, 11)
    hist, _ = np.histogram(uniqueness_scores, bins=bins_hist)
    for i in range(len(bins_hist)-1):
        count = hist[i]
        pct = 100 * count / len(uniqueness_scores)
        bar = "█" * int(pct / 2)
        print(f"    [{bins_hist[i]:.1f}-{bins_hist[i+1]:.1f}): {count:7d} ({pct:5.1f}%) {bar}")

print("\n" + "=" * 70)
print("DONE!")
print("=" * 70)

print(f"""
Output files created:
  - {output_dir}/train_uniqueness.npy    ({N*2:,} rows)
  - {output_dir}/val_uniqueness.npy      
  - {output_dir}/test_uniqueness.npy     

Each file has shape (num_sequences, ) with float32 values in [0, 1].

ALIGNMENT GUARANTEE:
  ✓ train_uniqueness[i] = uniqueness at coordinates train_coords[i]
  ✓ val_uniqueness[i]   = uniqueness at coordinates val_coords[i]
  ✓ test_uniqueness[i]  = uniqueness at coordinates test_coords[i]

Same as DNA and DNase!
""")
