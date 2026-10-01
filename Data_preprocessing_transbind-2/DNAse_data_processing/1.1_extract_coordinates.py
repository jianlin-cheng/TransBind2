"""
build_coords_csv.py
Rebuild genomic coordinates for every sequence in train/val/test .mat files
and save train_coords.csv, val_coords.csv, test_coords.csv.

unique_dna_idx in each CSV matches the index into *_unique_dna.npz.

Usage:
    python build_coords_csv.py
"""

import os
import numpy as np
import pandas as pd
from collections import defaultdict
import h5py
from tqdm import tqdm

# ============================================================
# CONFIG
# ============================================================
HOME_DIR = "/path/to/HOME_DIR"

# Created in 5_build_bedFile.py
bed_dir = os.path.join(
    HOME_DIR,
    "data/bed_files/train_test_val_bed_files"
)

train_mat_path = os.path.join(
    HOME_DIR,
    "data/train.mat"
)

out_dir = os.path.join(
    HOME_DIR,
    "data/Binary_dataV1"
)

bin_size = 200
test_chroms = {'chr8', 'chr9'}
val_chroms  = {'chr7'}
# ============================================================


# ============================================================
# Step 1: Rebuild 200bp bins from BED files
# ============================================================

def get_bins_for_region(start, end, bin_size=200):
    bins = set()
    idx = start // bin_size
    while True:
        bin_start = idx * bin_size
        bin_end = bin_start + bin_size
        overlap = max(0, min(bin_end, end) - max(bin_start, start))
        if overlap > bin_size // 2:
            bins.add(bin_start)
        if bin_end >= end:
            break
        idx += 1
    return bins

bins_per_chrom = defaultdict(set)
bed_files = sorted([f for f in os.listdir(bed_dir) if f.endswith('.bed')])

for bed_file in bed_files:
    with open(os.path.join(bed_dir, bed_file)) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 3:
                continue
            chrom, start, end = parts[0], int(parts[1]), int(parts[2])
            bins_per_chrom[chrom] |= get_bins_for_region(start, end, bin_size)

print("Bins rebuilt.")


# ============================================================
# Step 2: Build coordinate arrays in exact .mat order
#   .mat layout: [0..N-1] originals, [N..2N-1] reverse complements
#   (chromosomes sorted lexicographically, positions sorted)
# ============================================================

chroms_sorted = sorted(bins_per_chrom.keys())  # lexicographic

train_coords = []
val_coords   = []
test_coords  = []

for chrom in chroms_sorted:
    for bin_start in sorted(bins_per_chrom[chrom]):
        # 1000bp window centered on 200bp bin
        window_start = bin_start - 400
        window_end   = bin_start + 600
        coord = (chrom, bin_start, window_start, window_end)

        if chrom in test_chroms:
            test_coords.append(coord)
        elif chrom in val_chroms:
            val_coords.append(coord)
        else:
            train_coords.append(coord)

print(f"Train coords: {len(train_coords):,} (× 2 with RC = {len(train_coords)*2:,})")
print(f"Val coords:   {len(val_coords):,} (× 2 with RC = {len(val_coords)*2:,})")
print(f"Test coords:  {len(test_coords):,} (× 2 with RC = {len(test_coords)*2:,})")


# ============================================================
# Step 3: mat_idx -> coord mapping
# ============================================================

def mat_idx_to_coord(mat_idx, coords):
    """Given a .mat index, return (chrom, bin_start, window_start, window_end, is_rc)"""
    N = len(coords)
    if mat_idx < N:
        chrom, bin_start, ws, we = coords[mat_idx]
        return chrom, bin_start, ws, we, False
    else:
        chrom, bin_start, ws, we = coords[mat_idx - N]
        return chrom, bin_start, ws, we, True

print("\nSample mat_idx -> coord mapping (train):")
N_train = len(train_coords)
for idx in [0, 1, 2, N_train-1, N_train, N_train+1, 2*N_train-1]:
    print(f"  mat_idx {idx:>10,} -> {mat_idx_to_coord(idx, train_coords)}")


# ============================================================
# Step 4: Rebuild train dedup mapping (mat_idx -> unique_dna_idx)
#   Val and test are 1:1 (no duplicates)
# ============================================================

print("\nRebuilding train dedup mapping...")
f = h5py.File(train_mat_path, 'r')
xdata = f['trainxdata']

# Safety check: BED-derived bins must match train.mat exactly,
# otherwise every coordinate is silently shifted.
assert xdata.shape[0] == 2 * len(train_coords), (
    f"train.mat has {xdata.shape[0]:,} rows but BED bins give "
    f"{2 * len(train_coords):,} — BED files don't match train.mat"
)

seen_hashes = {}       # hash -> unique_idx (first occurrence)
mat_to_unique = np.zeros(xdata.shape[0], dtype=np.int32)
duplicates = []
unique_idx = 0

for i in tqdm(range(xdata.shape[0]), desc="Hashing train sequences"):
    seq_hash = hash(xdata[i].tobytes())
    if seq_hash not in seen_hashes:
        seen_hashes[seq_hash] = unique_idx
        mat_to_unique[i] = unique_idx
        unique_idx += 1
    else:
        mat_to_unique[i] = seen_hashes[seq_hash]
        duplicates.append((i, seen_hashes[seq_hash]))

f.close()

print(f"Unique: {unique_idx:,}, Duplicates: {len(duplicates)}")
for dup_mat_idx, dup_unique_idx in duplicates:
    coord = mat_idx_to_coord(dup_mat_idx, train_coords)
    print(f"  mat_idx {dup_mat_idx:,} (unique_idx {dup_unique_idx}) -> {coord}")


# ============================================================
# Step 5: Build unique_dna_idx -> coord tables and save
# ============================================================

# --- TRAIN ---
# Invert: unique_idx -> mat_idx (first occurrence)
unique_to_mat_train = np.zeros(unique_idx, dtype=np.int32)
first_seen = {}
for mat_idx in range(len(mat_to_unique)):
    u = mat_to_unique[mat_idx]
    if u not in first_seen:
        first_seen[u] = mat_idx
        unique_to_mat_train[u] = mat_idx

train_rows = []
for u_idx in range(unique_idx):
    mat_idx = unique_to_mat_train[u_idx]
    chrom, bin_start, ws, we, is_rc = mat_idx_to_coord(mat_idx, train_coords)
    train_rows.append({
        'unique_dna_idx': u_idx,
        'chrom': chrom,
        'bin_start': bin_start,
        'window_start': ws,
        'window_end': we,
        'is_reverse_complement': is_rc
    })
train_df = pd.DataFrame(train_rows)

# --- VAL (1:1, no dedup) ---
val_rows = []
N_val = len(val_coords)
for mat_idx in range(N_val * 2):
    chrom, bin_start, ws, we, is_rc = mat_idx_to_coord(mat_idx, val_coords)
    val_rows.append({
        'unique_dna_idx': mat_idx,
        'chrom': chrom,
        'bin_start': bin_start,
        'window_start': ws,
        'window_end': we,
        'is_reverse_complement': is_rc
    })
val_df = pd.DataFrame(val_rows)

# --- TEST (1:1, no dedup) ---
test_rows = []
N_test = len(test_coords)
for mat_idx in range(N_test * 2):
    chrom, bin_start, ws, we, is_rc = mat_idx_to_coord(mat_idx, test_coords)
    test_rows.append({
        'unique_dna_idx': mat_idx,
        'chrom': chrom,
        'bin_start': bin_start,
        'window_start': ws,
        'window_end': we,
        'is_reverse_complement': is_rc
    })
test_df = pd.DataFrame(test_rows)

# --- Save ---
os.makedirs(out_dir, exist_ok=True)

for name, df in [('train', train_df), ('val', val_df), ('test', test_df)]:
    path = os.path.join(out_dir, f"{name}_coords.csv")
    df.to_csv(path, index=False)
    print(f"\nSaved {name}_coords.csv: {len(df):,} rows")
    print(df.head(10).to_string())

print("\n\nDone! unique_dna_idx in each CSV matches the index into *_unique_dna.npz")