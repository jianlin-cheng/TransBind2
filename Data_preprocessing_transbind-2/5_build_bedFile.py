#!/usr/bin/env python3
import os
import gzip

#input_file = '/bml/shreya/TF_binding_site/dataset_test/DEEPSEA_dataextraction/data/New_TF_data/peaks_with_labels_clean_TF.tsv.gz'
HOME_DIR="/path/to/HOME_DIR"
input_file = os.path.join(HOME_DIR, "data/processed/peaks_with_labels.tsv.gz")
#output_dir = os.path.join(HOME_DIR, "DeepSEA_dataset/new_tf/train_test_val_bed_files")
output_dir = os.path.join(HOME_DIR, "data/bed_files/train_test_val_bed_files")

os.makedirs(output_dir, exist_ok=True)

with gzip.open(input_file, 'rt') as f:
    #next(f)  # Skip header line
    for line_count, line in enumerate(f, 1):
        parts = line.strip().split('\t')
        
        if len(parts) < 5:
            print(f"Skipping line {line_count}: insufficient columns")
            continue
            
        try:
            chrom, start, end = parts[0], int(parts[1]), int(parts[2])
            accessions = [acc.strip() for acc in parts[4].split(',') if acc.strip()]
        except ValueError as e:
            print(f"Skipping line {line_count}: {e}")
            continue
            
        for acc in accessions:
            bed_path = os.path.join(output_dir, f"{acc.replace('/', '_')}.bed")
            with open(bed_path, 'a') as bed_file:
                bed_file.write(f"{chrom}\t{start}\t{end}\t{acc}\t1000\n")
                
        if line_count % 10000 == 0:
            print(f"Processed {line_count} lines...")

print(f"Finished. Created BED files in {output_dir}")