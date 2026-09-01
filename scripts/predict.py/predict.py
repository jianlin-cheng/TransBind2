#!/usr/bin/env python3

import os
import argparse
import numpy as np
import torch

from train_new_tf import BinaryProteinAwareTransBind


# ============================================================
# Paths
# ============================================================

HOME_DIR = "/path/to/HOME_DIR"

CHECKPOINT_PATH = os.path.join(
    HOME_DIR,
    "model_binary_best_sweep_v2/"
    "Transbind2.ckpt"
)

HUMAN_TF_MAPPING = os.path.join(
    HOME_DIR,
    "data/Binary_dataV1/"
    "Binary_data_for_TF_splitV3/EGNN/"
    "tf_mapping_with_features_with_graphsV1.csv"
)

HUMAN_FEATURES = os.path.join(
    HOME_DIR,
    "Protein_data/prostt5_featuresV1"
)

MAX_PROTEIN_LEN = 512
DNASE_NO_COVERAGE_VALUE = -3.542


# ============================================================
# DNA
# ============================================================

def one_hot_encode_dna(sequence):
    """
    Convert a 1000-bp DNA sequence to shape (1000, 4).

    Channel order:
        A, C, G, T
    """

    sequence = sequence.upper()

    if len(sequence) != 1000:
        raise ValueError(
            f"DNA sequence must be exactly 1000 bp. "
            f"Received {len(sequence)} bp."
        )

    mapping = {
        "A": [1, 0, 0, 0],
        "C": [0, 1, 0, 0],
        "G": [0, 0, 1, 0],
        "T": [0, 0, 0, 1],
        "N": [0, 0, 0, 0],
    }

    encoded = np.zeros((1000, 4), dtype=np.float32)

    for i, base in enumerate(sequence):
        if base not in mapping:
            raise ValueError(
                f"Invalid DNA base '{base}' at position {i}"
            )

        encoded[i] = mapping[base]

    return encoded


# ============================================================
# New TF ProstT5 feature
# ============================================================

def load_protein_feature(feature_path):
    """
    Load AA + 3Di ProstT5 .fea file.

    Expected original shape:
        (protein_length, 2048)

    Returns:
        protein_tensor: (1, L, 2048)
        protein_mask:   (1, L)
    """

    with open(feature_path, "r") as f:
        numbers = [float(x) for x in f.read().split()]

    if len(numbers) % 2048 != 0:
        raise ValueError(
            f"Protein feature file contains {len(numbers)} values. "
            "Expected a multiple of 2048."
        )

    seq_len = len(numbers) // 2048

    feature = np.asarray(
        numbers,
        dtype=np.float32
    ).reshape(seq_len, 2048)

    print(f"Protein length: {seq_len} residues")

    # Same maximum used during training
    if seq_len > MAX_PROTEIN_LEN:
        print(
            f"Protein longer than {MAX_PROTEIN_LEN}. "
            f"Truncating to {MAX_PROTEIN_LEN} residues."
        )

        feature = feature[:MAX_PROTEIN_LEN]

    protein_tensor = torch.from_numpy(
        feature
    ).unsqueeze(0)

    # No padding because this is one protein
    protein_mask = torch.zeros(
        (1, feature.shape[0]),
        dtype=torch.bool
    )

    return protein_tensor, protein_mask


# ============================================================
# DNase
# ============================================================

def load_dnase(dnase_path):
    """
    Load normalized DNase signal.

    Expected:
        1000 values

    IMPORTANT:
        This should use the same normalized DNase representation
        used during training.
    """

    if dnase_path.endswith(".npy"):
        dnase = np.load(dnase_path)
    else:
        dnase = np.loadtxt(dnase_path)

    dnase = np.asarray(
        dnase,
        dtype=np.float32
    ).reshape(-1)

    if len(dnase) != 1000:
        raise ValueError(
            f"DNase must contain exactly 1000 values. "
            f"Received {len(dnase)}."
        )

    return dnase


# ============================================================
# Load trained model
# ============================================================

def load_model():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Original human TF mapping is still needed to initialize
    # the model/checkpoint correctly.
    import pandas as pd

    human_mapping = pd.read_csv(HUMAN_TF_MAPPING)
    num_human_tfs = len(human_mapping)

    print(f"Loading checkpoint on {device}...")

    model = BinaryProteinAwareTransBind.load_from_checkpoint(
        CHECKPOINT_PATH,
        mapping_path=HUMAN_TF_MAPPING,
        features_dir=HUMAN_FEATURES,
        num_tfs=num_human_tfs,
    )

    model = model.to(device)
    model.eval()

    return model, device


# ============================================================
# Prediction
# ============================================================

def predict(
    model,
    device,
    dna_sequence,
    dnase_path,
    uniqueness,
    protein_feature_path,
):

    # ---------------- DNA ----------------

    dna = one_hot_encode_dna(
        dna_sequence
    )

    dna_tensor = torch.from_numpy(
        dna
    ).unsqueeze(0).to(device)

    # shape:
    # (1, 1000, 4)

    # ---------------- DNase ----------------
    if dnase_path is not None:
        dnase = load_dnase(dnase_path)
        print("Using provided DNase signal.")
    else:
        dnase = np.full(
            1000,
            DNASE_NO_COVERAGE_VALUE,
            dtype=np.float32
        )
        print(
            f"No DNase provided. Using no-coverage value "
            f"{DNASE_NO_COVERAGE_VALUE}."
        )

    dnase_tensor = torch.from_numpy(
        dnase
    ).unsqueeze(0).to(device)

    # ---------------- uniqueness ----------------

    uniqueness_tensor = torch.tensor(
        [[uniqueness]],
        dtype=torch.float32,
        device=device,
    )

    # shape:
    # (1, 1)

    # ---------------- protein ----------------

    protein_tensor, protein_mask = load_protein_feature(
        protein_feature_path
    )

    protein_tensor = protein_tensor.to(device)
    protein_mask = protein_mask.to(device)

    print("\nInput shapes:")
    print(f"  DNA:        {dna_tensor.shape}")
    print(f"  DNase:      {dnase_tensor.shape}")
    print(f"  Uniqueness: {uniqueness_tensor.shape}")
    print(f"  Protein:    {protein_tensor.shape}")

    # ---------------- inference ----------------

    with torch.no_grad():

        logits = model.forward_with_protein(
            dna_seq=dna_tensor,
            dnase_signal=dnase_tensor,
            uniqueness_score=uniqueness_tensor,
            protein_raw=protein_tensor,
            protein_mask=protein_mask,
        )

        probability = torch.sigmoid(
            logits
        ).item()

    return probability


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Predict TF binding using a new TF protein embedding."
    )

    parser.add_argument(
        "--dna",
        required=True,
        help="1000-bp DNA sequence"
    )

    parser.add_argument(
        "--protein_feature",
        required=True,
        help="AA + 3Di ProstT5 .fea file for the new TF"
    )

    # parser.add_argument(
    #     "--dnase",
    #     required=True,
    #     help="Normalized DNase file (.npy or text), 1000 values"
    # )
    parser.add_argument(
        "--dnase",
        default=None,
        help="Optional normalized DNase file (.npy or text), 1000 values"
    )

    parser.add_argument(
        "--uniqueness",
        required=True,
        type=float,
        help="Uniqueness/mappability value"
    )

    args = parser.parse_args()

    if not 0.0 <= args.uniqueness <= 1.0:
        raise ValueError(
            "Uniqueness should be between 0 and 1."
        )

    model, device = load_model()

    probability = predict(
        model=model,
        device=device,
        dna_sequence=args.dna,
        dnase_path=args.dnase,
        uniqueness=args.uniqueness,
        protein_feature_path=args.protein_feature,
    )

    print("\n" + "=" * 50)
    print("Prediction")
    print("=" * 50)
    print(
        f"TF binding probability: {probability:.6f}"
    )


if __name__ == "__main__":
    main()