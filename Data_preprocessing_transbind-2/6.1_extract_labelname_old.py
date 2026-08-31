#!/usr/bin/env python3
"""Extract feature names from DeepSEA metadata file."""

import pandas as pd
import sys

def extract_feature_names():
    metadata_file = "/bml/shreya/TF_binding_site/dataset_test/DeepSEA_dataset/encoded_metadata.tsv"
    
    print("Reading metadata file...")
    try:
        df = pd.read_csv(metadata_file, sep='\t', low_memory=False)
        print(f"Successfully read metadata file!")
        print(f"Shape: {df.shape}")
        print(f"Total rows: {len(df)}")
        
        # Show column names
        print("\nColumns in metadata file:")
        for i, col in enumerate(df.columns):
            print(f"  {i}: {col}")
        
        # Show first few rows to understand structure
        print("\nFirst 5 rows:")
        print(df.head())
        
        # Look for target and cell type columns
        potential_target_cols = [col for col in df.columns if 'target' in col.lower() or 'tf' in col.lower()]
        potential_cell_cols = [col for col in df.columns if 'cell' in col.lower() or 'biosample' in col.lower()]
        potential_accession_cols = [col for col in df.columns if 'accession' in col.lower()]
        
        print(f"\nPotential target columns: {potential_target_cols}")
        print(f"Potential cell columns: {potential_cell_cols}")
        print(f"Potential accession columns: {potential_accession_cols}")
        
        # Create feature names based on available columns
        feature_names = []
        
        # Method 1: Try to find target and biosample columns
        target_col = None
        cell_col = None
        
        for col in df.columns:
            if 'target' in col.lower() and 'label' in col.lower():
                target_col = col
            elif 'biosample' in col.lower() and 'term' in col.lower():
                cell_col = col
        
        if target_col and cell_col:
            print(f"\nUsing columns: {target_col} and {cell_col}")
            for idx, row in df.iterrows():
                target = str(row[target_col]).replace(' ', '').replace('-', '').replace('.', '')
                cell = str(row[cell_col]).replace(' ', '').replace('-', '').replace('.', '')
                feature_name = f"{cell}-{target}"
                feature_names.append(feature_name)
        
        # Method 2: If method 1 fails, try accession IDs
        elif 'File accession' in df.columns:
            print("\nUsing File accession column")
            feature_names = df['File accession'].astype(str).tolist()
        
        # Method 3: Use row indices
        else:
            print("\nUsing generic feature names")
            feature_names = [f"Feature_{i:03d}" for i in range(len(df))]
        
        print(f"\nExtracted {len(feature_names)} feature names")
        print("First 10 features:")
        for i, name in enumerate(feature_names[:10]):
            print(f"  {i}: {name}")
        
        if len(feature_names) > 10:
            print("...")
            print("Last 5 features:")
            for i, name in enumerate(feature_names[-5:], len(feature_names)-5):
                print(f"  {i}: {name}")
        
        # Save to file
        output_file = "label_name.txt"
        with open(output_file, 'w') as f:
            f.write("Feature_Index\tFeature_Name\n")
            for i, name in enumerate(feature_names):
                f.write(f"{i}\t{name}\n")
        
        print(f"\nFeature names saved to: {output_file}")
        
        # Verification
        if len(feature_names) == 690:
            print("✓ Perfect! Found exactly 690 features")
        elif len(feature_names) > 690:
            print(f"⚠ Found {len(feature_names)} features (more than 690)")
            print("  You may need to apply filters to match your dataset")
        else:
            print(f"⚠ Found only {len(feature_names)} features (less than 690)")
        
        return feature_names
        
    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    feature_names = extract_feature_names()