# TransBind2

## Table of Contents

* [Overview](#overview)
* [Model Architecture](#model-architecture)
* [Installation](#installation)

  * [Requirements](#requirements)
  * [Quick Install](#quick-install)
* [Data Preprocessing Pipeline](#data-preprocessing-pipeline)

  * [Transcription Factor](#transcription-factor)
  * [DNase Data](#dnase-data)
  * [Uniqueness Data](#uniqueness-data)
* [Training and Testing](#training-and-testing)

  * [Training Data Preparation](#training-data-preparation)
  * [Training](#training)
  * [Testing](#testing)
* [Predicting Binding for a New TF](#predicting-binding-for-a-new-tf)
* [Citation](#citation)

## Overview

TransBind2 is a deep learning framework for predicting transcription factor (TF)–DNA binding by integrating DNA sequence with the biological context surrounding potential binding sites. Building on our previous TransBind model, TransBind2 incorporates DNase-seq chromatin accessibility and genome mappability alongside DNA sequence information. To capture TF-specific features, the model uses ProstT5 to encode both the amino acid sequence and 3D structural information of each TF. These DNA and protein representations are connected through bidirectional cross-attention, allowing information from each modality to inform the other and providing a more complete representation of TF–DNA interactions.

TransBind2 formulates binding prediction as a binary classification task for individual DNA bin–TF–cell type combinations. The model was trained on 690 ChIP-seq experiments covering 161 TFs and 91 human cell types. By incorporating protein representations directly into the prediction framework, TransBind2 can also be applied to TFs not encountered during training when the corresponding protein features are provided. The model further demonstrated cross-species transferability through zero-shot evaluation on mouse datasets.

Together, these results suggest that incorporating TF structural features and chromatin context can improve both the accuracy and generalizability of TF–DNA binding prediction. By capturing information from the TF, DNA sequence, and local chromatin environment within a unified framework, TransBind2 provides an approach for investigating gene regulation and TF–DNA interactions.

## Model Architecture

![Model Architecture](final_module.png)

*Figure 1: TransBind2 model architecture for transcription factor binding site prediction.*

## Installation

### Requirements

* PyTorch
* PyTorch Lightning
* NumPy
* pandas
* scikit-learn
* h5py
* CUDA recommended for training and large-scale evaluation

### Quick Install

```bash
git clone https://github.com/jianlin-cheng/TransBind2.git

cd TransBind2

conda env create -f environment.yml

conda activate transBind
```

# Data Preprocessing Pipeline

| Step | Script                                                                                                 | Description                                                                                                                                         |
| ---- | ------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1    | `0_download_data.py`                                                                                   | Download the hg19 human genome assembly and ENCODE TF ChIP-seq peak files                                                                           |
| 2    | `1_preprocess_narrowPeaks_and_humanGenome.sh`                                                          | Sort TF peak files and divide the human genome into 200-bp bins                                                                                     |
| 3    | `2_compute_overlapping_using_batch.sh`<br>`3_postprocess.sh`<br>`4_merge_peaks_with_same_labels.ipynb` | Compute overlaps between TF peaks and genome bins, merge results, and assign TF experiment labels to each genomic region                            |
| 4    | `5_build_bedFile.py`<br>`5.1_convert_metadata.py`                                                      | Generate per-experiment BED files and convert UCSC metadata into the format required for dataset construction                                       |
| 5    | `6_build_dataset.py`<br>`6.1_extract_labelname.py`<br>`6.2_creating_label_name_merged.ipynb`           | Build chromosome-based train/validation/test datasets (`.mat`), extract experiment names, and generate the final experiment-to-TF/cell-type mapping |
| 6    | `7_data_conversion_to_binaryV1.py`                                                                     | Convert the multi-label `.mat` datasets into binary long-format datasets containing `dna_idx`, `tf_idx`, `cell_idx`, and `labels`                   |
| 7    | `8_mapping_between_filename_TF.ipynb`                                                                  | Generate TF mappings and associate each TF with its corresponding protein feature information                                                       |

## Transcription Factor

| Step | Script                                     | Description                                                                                                             |
| ---- | ------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------- |
| 1    | `1_download_fasta_from_uniprot.py`         | Download amino-acid FASTA sequences from UniProt for each TF → saves to `Protein_data/fasta/`                           |
| 2    | `2_get_pdb_from_AF.py`                     | Download predicted protein structures (PDB) from AlphaFold for each TF → saves to `Protein_data/AF_structure/`          |
| 3    | `3_PDB_to_3Di.sh`                          | Convert PDB structures to 3Di structural tokens using Foldseek → saves to `Protein_data/3Di_tokens/`                    |
| 4    | `4_generate_prostt5_embedding.py`          | Generate per-residue ProstT5 embeddings from amino-acid and 3Di sequences → saves to `Protein_data/prostt5_featuresV1/` |
| 5    | `5_mapping_between_filename_protein.ipynb` | Map each TF index to its corresponding UniProt ID and ProstT5 feature file → saves the final TF–protein feature mapping |

## DNase Data

| Step | Script                        | Description                                                                                                                                           |
| ---- | ----------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1    | `0_download_md5.sh`           | Download ENCODE Uniform DNase-seq peak files (`wgEncodeAwgDnaseUniform`) from UCSC and verify file integrity using MD5 checksums                      |
| 2    | `1_convert_to_bw.sh`          | Convert `.narrowPeak.gz` files to bigWig (`.bw`) signal tracks using `bedtools merge`, `bedClip`, and `bedGraphToBigWig`                              |
| 3    | `2_extract_dnase_parallel.py` | Extract 1000-bp DNase signal vectors from bigWig files at each `{split}_coords.csv` genomic window, in parallel across available cell types           |
| 4    | `3_normalise_dnase.py`        | Apply `log1p`, global z-score normalization, and clipping to `[-5, 5]` to the extracted DNase signals                                                 |
| 5    | `4_convert_dnase_to_hdf5.py`  | Combine normalized per-cell `.npy` files into `{split}_dnase.h5`, quantize signals to `uint8`, and fill missing cell types with a fixed default value |

## Uniqueness Data

| Step | Script                       | Description                                                                                                                                                                                     |
| ---- | ---------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1    | `1_extract_uniqueness_score` | Extract the mean uniqueness score per genomic window from the Duke uniqueness bigWig track using the same `{split}_coords.csv` coordinates as DNase extraction → saves `{split}_uniqueness.npy` |

# Training and Testing

### Training Data Preparation

Before training the model, complete the following steps:

| Step | Process              | Description                                                                                                                                                                          |
| ---- | -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 1    | Data preprocessing   | Complete the main preprocessing pipeline, including DNA preprocessing, coordinate generation, DNase preprocessing, uniqueness preprocessing, and protein feature generation          |
| 2    | Dataset organization | Ensure `train_unique_dna.npz`, `standalone_train_indices.npz`, and the corresponding validation/test files are stored in `data/Binary_dataV1/`                                       |
| 3    | DNase data           | Ensure `train_dnase.h5` and `val_dnase.h5` exist in `data/Binary_dataV1/`                                                                                                            |
| 4    | Uniqueness data      | Ensure `train_uniqueness.npy` and `val_uniqueness.npy` exist in `data/uniqueness_processed/`                                                                                         |
| 5    | Feature mapping      | Ensure `tf_mapping_with_features_with_graphsV1.csv` exists in `data/Binary_dataV1/Binary_data_for_TF_splitV3/EGNN/` and maps each `tf_idx` to its corresponding protein feature file |
| 6    | Protein features     | Ensure `Protein_data/prostt5_featuresV1/` contains the `.fea` embedding files referenced by the TF mapping                                                                           |

### Training

Before training, ensure that all preprocessing steps have been completed and the required DNA, DNase, uniqueness, and protein feature files are available as described above.

Update the repository root in the training script:

```python
HOME_DIR = "/path/to/HOME_DIR"
```

where `HOME_DIR` is the path to the cloned TransBind repository.

Run the training script:

```bash
cd scripts

python train.py
```

Model checkpoints are saved to:

```text
model_binary_best_sweep_v2/
```

### Testing

Before testing, ensure that all preprocessing steps have been completed and the required DNA, DNase, uniqueness, and protein feature files are available as described above.

Update the repository root in the testing script:

```python
HOME_DIR = "/path/to/HOME_DIR"
```

where `HOME_DIR` is the path to the cloned TransBind repository.

Run the testing script:

```bash
cd scripts

python test.py
```

Evaluation results are saved to:

```text
evaluation_results/
```

# Prediction

TransBind2 can be used to predict TF–DNA binding for a new TF by providing its AA + 3Di ProstT5 protein representation.

```bash
# Step 1: Prepare a 1000-bp DNA sequence
# The sequence should contain the target genomic region and its flanking sequence.

DNA_SEQUENCE="ACGT...1000bp..."


# Step 2: Generate protein features for the new TF
# Generate AA + 3Di ProstT5 features using the protein preprocessing pipeline.
# This produces a 2048-dimensional per-residue .fea file.

# Example output:
# example/new_TF.fea


# Step 3: Move to the scripts directory

cd scripts


# Step 4: Run TF binding prediction

python predict.py \
  --dna "$DNA_SEQUENCE" \
  --protein_feature ../example/new_TF.fea \
  --uniqueness 0.82


# DNase is optional. If available, provide the normalized
# 1000-bp DNase signal:

python predict.py \
  --dna "$DNA_SEQUENCE" \
  --protein_feature ../example/new_TF.fea \
  --dnase ../example/dnase.npy \
  --uniqueness 0.82


# Step 5: View the prediction

# Example output:
# TF binding probability: 0.573421
```

The DNA sequence must be exactly 1000 bp, and the uniqueness/mappability score must be between 0 and 1.

If DNase data are not provided, the model uses the predefined no-coverage DNase value for the input region.

## Citation

If you use TransBind2 in your research, please cite:

```text

```
