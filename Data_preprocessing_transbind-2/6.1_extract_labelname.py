#!/usr/bin/env python3
"""Create label-index mapping from DeepSEA metadata."""

import os
import pandas as pd

HOME_DIR = "/path/to/HOME_DIR"

metadata_file = os.path.join(
    HOME_DIR,
    "data/encoded_metadata.tsv"
)

output_file = os.path.join(
    HOME_DIR,
    "data/label_name.txt"
)

df = pd.read_csv(metadata_file, sep="\t")

feature_names = df["File accession"].astype(str).tolist()

with open(output_file, "w") as f:
    f.write("Feature_Index\tFeature_Name\n")

    for i, name in enumerate(feature_names):
        f.write(f"{i}\t{name}\n")

print(f"Found {len(feature_names)} features")
print(f"Saved label mapping to: {output_file}")

if len(feature_names) == 690:
    print("Found exactly 690 features")
else:
    print(f"Warning: expected 690 features, found {len(feature_names)}")