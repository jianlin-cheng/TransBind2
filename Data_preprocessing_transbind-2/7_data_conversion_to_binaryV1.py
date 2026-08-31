"""Create a standalone binary TF–DNA dataset."""

import numpy as np
import h5py
import pandas as pd
from tqdm import tqdm
import os

# Configuration

HOME_DIR = "/path/to/HOME_DIR"

DATA_DIR = os.path.join(HOME_DIR, "data")
MAPPING_PATH = os.path.join(DATA_DIR, "label_file_final_merged.csv")
OUTPUT_DIR = DATA_DIR

# Step 1: Create TF and Cell Mappings

def create_mappings(mapping_path, output_dir):
    """Create TF and Cell mappings from the full label file"""
    print("Creating TF and cell mappings...")
    
    mapping = pd.read_csv(mapping_path)
    
    unique_tfs = mapping['TF'].unique()
    tf_mapping = pd.DataFrame({
        'tf_idx': range(len(unique_tfs)),
        'TF': unique_tfs
    })
    
    if 'UniProt_ID' in mapping.columns:
        tf_info = mapping.drop_duplicates('TF')[['TF', 'UniProt_ID', 'TF_graph_file']]
        tf_mapping = tf_mapping.merge(tf_info, on='TF', how='left')
    
    print(f"  Unique TFs: {len(unique_tfs)}")
    
    unique_cells = mapping['Cell_Type'].unique()
    cell_mapping = pd.DataFrame({
        'cell_idx': range(len(unique_cells)),
        'Cell_Type': unique_cells
    })
    
    print(f"  Unique Cell types: {len(unique_cells)}")
    
    tf_mapping_path = os.path.join(output_dir, 'tf_mapping.csv')
    cell_mapping_path = os.path.join(output_dir, 'cell_mapping.csv')
    
    tf_mapping.to_csv(tf_mapping_path, index=False)
    cell_mapping.to_csv(cell_mapping_path, index=False)
    
    print(f"\n  Saved TF mapping to: {tf_mapping_path}")
    print(f"  Saved Cell mapping to: {cell_mapping_path}")
    
    tf_to_idx = {tf: idx for idx, tf in enumerate(unique_tfs)}
    cell_to_idx = {cell: idx for idx, cell in enumerate(unique_cells)}
    
    exp_to_tf_cell = {}
    for exp_idx, row in mapping.iterrows():
        tf_idx = tf_to_idx[row['TF']]
        cell_idx = cell_to_idx[row['Cell_Type']]
        exp_to_tf_cell[exp_idx] = (tf_idx, cell_idx)
    
    return tf_mapping, cell_mapping, exp_to_tf_cell

# Step 2: Process Each Split Independently to Extract DNA Sequences

def extract_sequences_by_split(mat_files, output_dir):
    """
    Extract DNA sequences from each split separately to avoid data leakage
    
    Args:
        mat_files: List of (split_name, mat_file_path) tuples
        output_dir: Directory to save the sequences
    """
    print("\nExtracting unique DNA sequences...")
    
    results = {}
    
    for split_name, mat_path in mat_files:
        print(f"\n  Processing {split_name} split...")
        
        if not os.path.exists(mat_path):
            print(f"    ⚠️  WARNING: {mat_path} not found, skipping")
            continue
        
        mat_file = h5py.File(mat_path, 'r')
        
        if split_name == 'train':
            dna_key = 'trainxdata'
        elif split_name == 'val':
            dna_key = 'validxdata'
        else:  # test
            dna_key = 'testxdata'
        
        dna_data = mat_file[dna_key]
        n_sequences = dna_data.shape[0]
        sequence_shape = dna_data.shape[1:]
        
        print(f"    Found {n_sequences:,} DNA sequences of shape {sequence_shape}")
        
        unique_sequences = {}
        dna_idx_mapping = {}
        next_idx = 0
        
        for i in tqdm(range(n_sequences), desc=f"    Finding unique sequences in {split_name}"):
            seq_hash = hash(dna_data[i].tobytes())
            
            if seq_hash not in unique_sequences:
                unique_sequences[seq_hash] = {
                    'idx': next_idx,
                    'sequence': dna_data[i]
                }
                next_idx += 1
            
            dna_idx_mapping[i] = unique_sequences[seq_hash]['idx']
        
        n_unique = len(unique_sequences)
        print(f"    Found {n_unique:,} unique DNA sequences in {split_name} split")
        
        unique_dna = np.zeros((n_unique, *sequence_shape), dtype=np.uint8)
        for seq_info in unique_sequences.values():
            unique_dna[seq_info['idx']] = seq_info['sequence']
        
        sequences_path = os.path.join(output_dir, f'{split_name}_unique_dna.npz')
        np.savez_compressed(sequences_path, dna_sequences=unique_dna)
        
        file_size_gb = unique_dna.nbytes / 1e9
        print(f"    Saved {split_name} sequences to {sequences_path} (~{file_size_gb:.2f} GB)")
        
        results[split_name] = {
            'path': sequences_path,
            'mapping': dna_idx_mapping,
            'size_gb': file_size_gb
        }
        
        mat_file.close()
    
    return results

# Step 3: Create Binary Indices for Each Split

def create_binary_indices_for_split(split_name, mat_file_path, output_path, 
                                    tf_mapping, cell_mapping, exp_to_tf_cell,
                                    dna_mapping):
    """
    Create binary indices that reference the unique DNA sequences
    
    Args:
        split_name: 'train', 'val', or 'test'
        mat_file_path: Path to the .mat file for this split
        output_path: Path to save the binary index
        tf_mapping: DataFrame with TF mappings
        cell_mapping: DataFrame with Cell mappings
        exp_to_tf_cell: Dict mapping experiment -> (tf_idx, cell_idx)
        dna_mapping: Dict mapping original DNA indices to new indices
    """
    print(f"\nCreating binary indices for {split_name}...")
    
    if not os.path.exists(mat_file_path):
        print(f"⚠️  WARNING: {mat_file_path} not found, skipping")
        return 0
    
    print(f"\nLoading {mat_file_path}...")
    mat_file = h5py.File(mat_file_path, 'r')
    
    if split_name == 'train':
        ydata_key = 'traindata'
    elif split_name == 'val':
        ydata_key = 'validdata'
    else:  # test
        ydata_key = 'testdata'
    
    testdata = mat_file[ydata_key]
    
    n_sequences = testdata.shape[0]
    n_experiments = testdata.shape[1]
    total_examples = n_sequences * n_experiments
    
    print(f"DNA sequences: {n_sequences:,}")
    print(f"Experiments: {n_experiments}")
    print(f"Total examples: {total_examples:,}")
    
    print(f"\nAllocating arrays...")
    new_dna_indices = np.zeros(total_examples, dtype=np.int32)
    tf_indices = np.zeros(total_examples, dtype=np.uint8)
    cell_indices = np.zeros(total_examples, dtype=np.uint8)
    labels = np.zeros(total_examples, dtype=np.uint8)
    
    print(f"\nGenerating indices...")
    idx = 0
    for dna_idx in tqdm(range(n_sequences), desc=f"  {split_name}"):
        for exp_idx in range(n_experiments):
            tf_idx, cell_idx = exp_to_tf_cell[exp_idx]
            
            new_dna_idx = dna_mapping[dna_idx]
            
            new_dna_indices[idx] = new_dna_idx
            tf_indices[idx] = tf_idx
            cell_indices[idx] = cell_idx
            labels[idx] = testdata[dna_idx, exp_idx]
            idx += 1
    
    print(f"Generated {idx:,} indexed examples")
    
    print(f"\nSaving to {output_path}...")
    np.savez_compressed(
        output_path,
        dna_idx=new_dna_indices,
        tf_idx=tf_indices,
        cell_idx=cell_indices,
        labels=labels
    )
    
    file_size_gb = (new_dna_indices.nbytes + tf_indices.nbytes + 
                    cell_indices.nbytes + labels.nbytes) / 1e9
    print(f"Saved! File size: ~{file_size_gb:.2f} GB")
    
    print(f"\nComputing {split_name} statistics...")
    compute_statistics(output_path, tf_mapping, cell_mapping, split_name)
    
    mat_file.close()
    
    return file_size_gb

# Step 4: Statistics

def compute_statistics(indices_path, tf_mapping, cell_mapping, split_name):
    """Compute and print statistics for one split"""
    
    indices_data = np.load(indices_path)
    tf_idx = indices_data['tf_idx']
    cell_idx = indices_data['cell_idx']
    labels = indices_data['labels']
    
    n_positive = np.sum(labels == 1)
    n_total = len(labels)
    pos_ratio = n_positive / n_total
    
    print(f"\n  {split_name.upper()} STATISTICS:")
    print(f"    Total examples: {n_total:,}")
    print(f"    Positive: {n_positive:,} ({pos_ratio:.2%})")
    print(f"    Negative: {n_total - n_positive:,} ({1-pos_ratio:.2%})")
    
    print(f"\n    Top 5 TFs by positive count:")
    tf_stats = []
    for tf_id in range(len(tf_mapping)):
        mask = tf_idx == tf_id
        if mask.sum() > 0:
            n_pos = np.sum(labels[mask] == 1)
            tf_stats.append((tf_mapping.loc[tf_id, 'TF'], n_pos))
    
    tf_stats.sort(key=lambda x: x[1], reverse=True)
    for tf_name, count in tf_stats[:5]:
        print(f"      {tf_name}: {count:,}")

# Main Function

def main():
    print("Creating standalone binary dataset...")
    print(f"\nData directory: {DATA_DIR}")
    print(f"Mapping file: {MAPPING_PATH}")
    
    tf_mapping, cell_mapping, exp_to_tf_cell = create_mappings(
        MAPPING_PATH, OUTPUT_DIR
    )
    
    splits_config = [
        ('train', 'train.mat'),
        ('val', 'valid.mat'),
        ('test', 'test.mat')
    ]
    
    mat_files = [(name, os.path.join(DATA_DIR, filename)) 
                 for name, filename in splits_config]
    
    sequences_by_split = extract_sequences_by_split(mat_files, OUTPUT_DIR)
    
    total_size = 0
    results = []
    
    for split_name, mat_filename in splits_config:
        mat_path = os.path.join(DATA_DIR, mat_filename)
        output_path = os.path.join(OUTPUT_DIR, f'standalone_{split_name}_indices.npz')
        
        if split_name not in sequences_by_split:
            continue
        
        dna_mapping = sequences_by_split[split_name]['mapping']
        
        file_size = create_binary_indices_for_split(
            split_name, mat_path, output_path,
            tf_mapping, cell_mapping, exp_to_tf_cell,
            dna_mapping
        )
        
        total_size += file_size + sequences_by_split[split_name]['size_gb']
        results.append((
            split_name, 
            os.path.basename(output_path), 
            file_size,
            os.path.basename(sequences_by_split[split_name]['path'])
        ))
    
    print("\nAll splits complete.")
    
    print("\nGenerated files:")
    print("  Shared mappings:")
    print(f"    - tf_mapping.csv (TF mappings)")
    print(f"    - cell_mapping.csv (Cell type mappings)")
    
    print("\n  Split-specific files:")
    for split_name, indices_file, indices_size, sequences_file in results:
        seq_size = sequences_by_split[split_name]['size_gb']
        print(f"    {split_name}:")
        print(f"      - {sequences_file} (unique DNA sequences, ~{seq_size:.2f} GB)")
        print(f"      - {indices_file} (binary indices, ~{indices_size:.2f} GB)")
    
    print(f"\n  Total size: ~{total_size:.2f} GB")
    

if __name__ == "__main__":
    main()