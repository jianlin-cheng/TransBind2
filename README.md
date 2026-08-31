# TransBind-2

## Table of Contents
- [Overview](#overview)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
  - [Requirements](#requirements)
- [Data Preprocessing Pipeline](#data-preprocessing-pipeline)
  - [Protein Features](#protein-features)
- [Training and Testing](#training-and-testing)
  - [Training Data Preparation](#training-data-preparation)
  - [Training Dataset](#training-dataset)
  - [Training](#training)
  - [Testing](#testing)
  - [Prediction](#prediction)
- [Citation](#citation)

## Overview

TransBind-2 is a deep learning framework for predicting transcription factor (TF)–DNA binding by integrating DNA sequence with the biological context surrounding potential binding sites. Building on our previous TransBind model, TransBind-2 incorporates DNase-seq chromatin accessibility and genome mappability alongside DNA sequence information. To capture TF-specific features, the model uses ProstT5 to encode both the amino acid sequence and 3D structural information of each TF. These DNA and protein representations are connected through bidirectional cross-attention, allowing information from each modality to inform the other and providing a more complete representation of TF–DNA interactions.

TransBind-2 formulates binding prediction as a binary classification task for individual DNA bin–TF–cell type combinations. This design enables the model to learn relationships across TFs and cellular contexts and, importantly, to make predictions for TFs and cell types that were not encountered during training. We trained the model on 690 ChIP-seq experiments covering 161 TFs and 91 human cell types. TransBind-2 achieved a macro AUROC of 0.9648 and an AUPR of 0.4215, representing a 12.67% relative improvement in AUPR over the original TransBind model and other baseline approaches. The model also showed strong transferability beyond human data, successfully performing zero-shot TF–DNA binding prediction on mouse datasets.

Together, these results suggest that incorporating TF structural features and chromatin context can substantially improve both the accuracy and generalizability of TF–DNA binding prediction. By capturing information from the TF, DNA sequence, and local chromatin environment within a unified framework, TransBind-2 provides a useful approach for investigating gene regulation, interpreting the regulatory effects of disease-associated variants, and supporting applications in synthetic biology.


## Model Architecture
![Model Architecture](final_module.png)

*Figure 1: TransBind2 model architecture for transcription factor binding site prediction*

## Installation

### Requirements
- PyTorch + PyTorch Lightning
- NumPy, scikit-learn, h5py
- CUDA recommended

### Quick Install
```bash
git clone https://github.com/jianlin-cheng/TransBind2.git
cd TransBind2
conda env create -f environment.yml
conda activate transBind
```

# Data Preprocessing Pipeline

# Data Preprocessing Pipeline

| Step | Script | Description |
|---|---|---|
| 1 | `0_download_data.py` | Download the hg19 human genome assembly and ENCODE TF ChIP-seq peak files |
| 2 | `1_preprocess_narrowPeaks_and_humanGenome.sh` | Sort TF peak files and divide the human genome into 200-bp bins |
| 3 | `2_compute_overlapping_using_batch.sh`<br>`3_postprocess.sh`<br>`4_merge_peaks_with_same_labels.ipynb` | Compute overlaps between TF peaks and genome bins, merge results, and assign TF experiment labels to each genomic region |
| 4 | `5_build_bedFile.py`<br>`5.1_convert_metadata.py` | Generate per-experiment BED files and convert UCSC metadata into the format required for dataset construction |
| 5 | `6_build_dataset.py`<br>`6.1_extract_labelname.py`<br>`6.2_creating_label_name_merged.ipynb` | Build chromosome-based train/validation/test datasets (`.mat`), extract experiment names, and generate the final experiment-to-TF/cell-type mapping |
| 6 | `7_data_conversion_to_binaryV1.py` | Convert the multi-label `.mat` datasets into binary long-format datasets containing `dna_idx`, `tf_idx`, `cell_idx`, and `labels` |
| 7 | `8_mapping_between_filename_TF.ipynb` | Generate TF mappings and associate each TF with its corresponding protein feature information |

## Transcription Factor

| Step | Script | Description |
|---|---|---|
| 1 | `1_download_fasta_from_uniprot.py` | Download amino-acid FASTA sequences from UniProt for each TF → saves to `Protein_data/fasta/` |
| 2 | `2_get_pdb_from_AF.py` | Download predicted protein structures (PDB) from AlphaFold for each TF → saves to `Protein_data/AF_structure/` |
| 3 | `3_PDB_to_3Di.sh` | Convert PDB structures to 3Di structural tokens using Foldseek → saves to `Protein_data/3Di_tokens/` |
| 4 | `4_generate_prostt5_embedding.py` | Generate per-residue ProstT5 embeddings from amino-acid and 3Di sequences → saves to `Protein_data/prostt5_featuresV1/` |
| 5 | `5_mapping_between_filename_protein.ipynb` | Map each TF index to its corresponding UniProt ID and ProstT5 feature file → saves the final TF–protein feature mapping |

## DNase Data

| Step | Script | Description |
|---|---|---|
| 1 | `0_download_md5.sh` | Download ENCODE Uniform DNase-seq peak files (`wgEncodeAwgDnaseUniform`) from UCSC and verify file integrity using MD5 checksums |
| 2 | `1_convert_to_bw.sh` | Convert `.narrowPeak.gz` files to bigWig (`.bw`) signal tracks using `bedtools merge`, `bedClip`, and `bedGraphToBigWig` |
| 3 | `2_extract_dnase_parallel.py` | Extract 1000-bp DNase signal vectors from bigWig files at each `{split}_coords.csv` genomic window, in parallel across available cell types |
| 4 | `3_normalise_dnase.py` | Apply `log1p`, global z-score normalization, and clipping to `[-5, 5]` to the extracted DNase signals |
| 5 | `4_convert_dnase_to_hdf5.py` | Combine normalized per-cell `.npy` files into `{split}_dnase.h5`, quantize signals to `uint8`, and fill missing cell types with a fixed default value |

## Uniqueness Data

| Step | Script | Description |
|---|---|---|
| 1 | `1_extract_uniqueness_score` | Extract mean uniqueness score per genomic window from the Duke uniqueness bigWig track, using the same {split}_coords.csv coordinates as DNase extraction → saves {split}_uniqueness.npy |

### Training Data Preparation

Before training the model, complete the following steps:

| Step | Process | Description |
|------|---------|-------------|
| 1 | Data preprocessing | Complete the main preprocessing pipeline, including DNA preprocessing, coordinate generation, DNase preprocessing, uniqueness preprocessing, and protein feature generation |
| 2 | Dataset organization | Ensure `train_unique_dna.npz`, `standalone_train_indices.npz`, and the corresponding validation/test files are stored in `data/Binary_dataV1/` |
| 3 | DNase data | Ensure `train_dnase.h5` and `val_dnase.h5` exist in `data/Binary_dataV1/` |
| 4 | Uniqueness data | Ensure `train_uniqueness.npy` and `val_uniqueness.npy` exist in `data/uniqueness_processed/` |
| 5 | Feature mapping | Ensure `tf_mapping_with_features_with_graphsV1.csv` exists in `data/Binary_dataV1/Binary_data_for_TF_splitV3/EGNN/` and maps each `tf_idx` to its corresponding protein feature file |
| 6 | Protein features | Ensure `Protein_data/prostt5_featuresV1/` contains the `.fea` embedding files referenced by the TF mapping |

### Training Dataset

