import os

import numpy as np
import torch
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm

from train import BinaryProteinAwareTransBind, BinaryTransBindDataset

# ===========================================================
# Paths
# ===========================================================
# ===========================================================
# Paths
# ===========================================================

HOME_DIR = "/path/to/HOME_DIR"

DATA_DIR = os.path.join(
    HOME_DIR,
    "data/Binary_dataV1"
)

TF_MAPPING_PATH = os.path.join(
    DATA_DIR,
    "Binary_data_for_TF_splitV3/EGNN/"
    "tf_mapping_with_features_with_graphsV1.csv"
)

FEATURES_DIR = os.path.join(
    HOME_DIR,
    "Protein_data/prostt5_featuresV1"
)

TEST_UNIQUENESS_PATH = os.path.join(
    HOME_DIR,
    "data/uniqueness_processed/test_uniqueness.npy"
)

CHECKPOINT_PATH = os.path.join(
    HOME_DIR,
    "model_binary_best_sweep_v2/"
    "binary-sweep-v2-epoch=59-val_aupr=0.4995.ckpt"
)

USE_DNASE = True
DNASE_NO_COVERAGE_VALUE = -3.6954

OUTPUT_DIR = os.path.join(
    HOME_DIR,
    "evaluation_results/sweep_v2_epoch59"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "aucs_binary_sweep_v2_epoch59.txt"
)

# ===========================================================
# Step 1: Load model
# ===========================================================
print("=" * 80)
print("BINARY PROTEIN-AWARE TRANSBIND  |  DNase + Uniqueness + Hybrid Cross-Attention")
print("SWEEP V2 CHECKPOINT  |  Epoch 59  |  val_aupr=0.4995")
print("EVALUATION ON TEST SET")
print("=" * 80)

print(f"\nLoading TF mapping from: {TF_MAPPING_PATH}")
tf_mapping = pd.read_csv(TF_MAPPING_PATH)
num_tfs = len(tf_mapping)
print(f"Number of TFs: {num_tfs}")

print(f"\nLoading checkpoint: {CHECKPOINT_PATH}")
model = BinaryProteinAwareTransBind.load_from_checkpoint(
    CHECKPOINT_PATH,
    mapping_path=TF_MAPPING_PATH,
    features_dir=FEATURES_DIR,
    num_tfs=num_tfs,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = model.to(device)
model.eval()

print(f"\nModel loaded on device: {device}")
print(f"Cross-attention enabled:   {model.use_cross_attention}")
print(f"d_model:                   {model.d_model}")
print(f"TF feature projection dim: {model.hparams.get('protein_target_len', 'N/A')}")

# ===========================================================
# Step 2: Build test dataset
# ===========================================================
print("\nLoading test data...")
print(f"Use DNase: {USE_DNASE}")

test_dna_path     = os.path.join(DATA_DIR, 'test_unique_dna.npz')
test_indices_path = os.path.join(DATA_DIR, 'standalone_test_indices.npz')

# DNase is optional here too. When USE_DNASE=False, or when test_dnase.h5
# simply isn't present, BinaryTransBindDataset falls back to a fixed
# no-coverage default signal for every test example (see its docstring).
if USE_DNASE:
    test_dnase_path = os.path.join(DATA_DIR, 'test_dnase.h5')
    if not os.path.exists(test_dnase_path):
        print(f"WARNING: USE_DNASE=True but {test_dnase_path} not found — "
              f"falling back to no-coverage default for all test examples")
        test_dnase_path = None
else:
    print("USE_DNASE=False — DNase channel will use the fixed "
          "no-coverage default for all test examples")
    test_dnase_path = None

test_dataset = BinaryTransBindDataset(
    dna_sequences_path=test_dna_path,
    indices_path=test_indices_path,
    uniqueness_path=TEST_UNIQUENESS_PATH,
    dnase_h5_path=test_dnase_path,
    split='test',
    no_coverage_value=DNASE_NO_COVERAGE_VALUE,
)

# Load raw index arrays for per-TF slicing
test_indices = np.load(test_indices_path)
test_labels  = test_indices['labels']
test_tf_idx  = test_indices['tf_idx']

print(f"\nTest set size:      {len(test_dataset):,} examples")
print(f"Positive samples:   {np.sum(test_labels):,} ({np.sum(test_labels)/len(test_labels):.2%})")
print(f"Negative samples:   {len(test_labels) - np.sum(test_labels):,}")
print(f"Unique TFs in test: {len(np.unique(test_tf_idx))}")

# ===========================================================
# Step 3: Inference
# ===========================================================
print("\nRunning inference...")

BATCH_SIZE = 3000

preds           = []
labels_list     = []
tf_indices_list = []

with torch.no_grad():
    for i in tqdm(range(0, len(test_dataset), BATCH_SIZE), desc="Batches"):
        batch_indices = range(i, min(i + BATCH_SIZE, len(test_dataset)))

        batch_dna        = []
        batch_dnase      = []
        batch_uniqueness = []
        batch_tf         = []
        batch_labels     = []

        for idx in batch_indices:
            dna_seq, dnase_signal, uniqueness_score, tf_index, label = test_dataset[idx]
            batch_dna.append(dna_seq)
            batch_dnase.append(dnase_signal)
            batch_uniqueness.append(uniqueness_score)
            batch_tf.append(tf_index)
            batch_labels.append(label)

        batch_dna        = torch.stack(batch_dna).to(device)
        batch_dnase      = torch.stack(batch_dnase).to(device)
        batch_uniqueness = torch.stack(batch_uniqueness).to(device)
        batch_tf         = torch.stack(batch_tf).to(device)
        batch_labels     = torch.stack(batch_labels)

        logits     = model(batch_dna, batch_dnase, batch_uniqueness, batch_tf)
        batch_pred = torch.sigmoid(logits).cpu().numpy()

        preds.append(batch_pred)
        labels_list.append(batch_labels.numpy())
        tf_indices_list.append(batch_tf.cpu().numpy())

pred_y     = np.concatenate(preds,           axis=0).flatten()
y_test     = np.concatenate(labels_list,     axis=0).flatten()
tf_indices = np.concatenate(tf_indices_list, axis=0).flatten()

print(f"\nPredictions shape: {pred_y.shape}")
print(f"Labels shape:      {y_test.shape}")
print(f"TF indices shape:  {tf_indices.shape}")

# ===========================================================
# Step 4: Overall metrics
# ===========================================================
print("\nCalculating overall metrics...")

try:
    overall_roc = roc_auc_score(y_test, pred_y)
    overall_pr  = average_precision_score(y_test, pred_y)
    print('\n' + '=' * 60)
    print('OVERALL PERFORMANCE (ALL SAMPLES)')
    print('=' * 60)
    print(f'Overall ROC AUC:  {overall_roc:.5f}')
    print(f'Overall PR  AUC:  {overall_pr:.5f}')
    print(f'Total samples:    {len(y_test):,}')
    print(f'Positive:         {int(np.sum(y_test)):,} ({np.sum(y_test)/len(y_test):.2%})')
    print(f'Negative:         {int(len(y_test) - np.sum(y_test)):,}')
    print('=' * 60)
except Exception as e:
    print(f"Error calculating overall metrics: {e}")
    overall_roc = np.nan
    overall_pr  = np.nan

# ===========================================================
# Step 5: Per-TF metrics
# ===========================================================
print("\nCalculating per-TF metrics...")

roc_scores       = []
pr_scores        = []
tf_sample_counts = []
tf_pos_counts    = []

with open(OUTPUT_FILE, 'w') as aucs_file:
    aucs_file.write('TF_Index\tN_Samples\tN_Positive\tPos_Ratio\tROC_AUC\tPR_AUC\n')

    for tf_idx in tqdm(range(num_tfs), desc="Per-TF metrics"):
        mask = tf_indices == tf_idx

        if not np.any(mask):
            roc_scores.append(np.nan)
            pr_scores.append(np.nan)
            tf_sample_counts.append(0)
            tf_pos_counts.append(0)
            aucs_file.write(f'{tf_idx}\t0\t0\t0.0000\tNaN\tNaN\n')
            continue

        tf_labels  = y_test[mask]
        tf_preds   = pred_y[mask]
        n_samples  = len(tf_labels)
        n_positive = int(np.sum(tf_labels))
        pos_ratio  = n_positive / n_samples if n_samples > 0 else 0.0

        tf_sample_counts.append(n_samples)
        tf_pos_counts.append(n_positive)

        try:
            if len(np.unique(tf_labels)) > 1 and n_samples >= 10:
                roc_auc = roc_auc_score(tf_labels, tf_preds)
                pr_auc  = average_precision_score(tf_labels, tf_preds)
                roc_scores.append(roc_auc)
                pr_scores.append(pr_auc)
                aucs_file.write(
                    f'{tf_idx}\t{n_samples}\t{n_positive}\t{pos_ratio:.4f}'
                    f'\t{roc_auc:.5f}\t{pr_auc:.5f}\n'
                )
            else:
                roc_scores.append(np.nan)
                pr_scores.append(np.nan)
                aucs_file.write(
                    f'{tf_idx}\t{n_samples}\t{n_positive}\t{pos_ratio:.4f}\tNaN\tNaN\n'
                )
        except ValueError as e:
            print(f"Error for TF {tf_idx}: {e}")
            roc_scores.append(np.nan)
            pr_scores.append(np.nan)
            aucs_file.write(
                f'{tf_idx}\t{n_samples}\t{n_positive}\t{pos_ratio:.4f}\tNaN\tNaN\n'
            )

    # Summary rows
    avg_roc    = np.nanmean(roc_scores)
    avg_pr     = np.nanmean(pr_scores)
    median_roc = np.nanmedian(roc_scores)
    median_pr  = np.nanmedian(pr_scores)

    valid_roc_count        = np.sum(~np.isnan(roc_scores))
    valid_pr_count         = np.sum(~np.isnan(pr_scores))
    total_tfs_with_samples = np.sum(np.array(tf_sample_counts) > 0)

    aucs_file.write(f'\nAVERAGE\t-\t-\t-\t{avg_roc:.5f}\t{avg_pr:.5f}\n')
    aucs_file.write(f'MEDIAN\t-\t-\t-\t{median_roc:.5f}\t{median_pr:.5f}\n')
    aucs_file.write(
        f'VALID_SCORES\t{total_tfs_with_samples}\t-\t-'
        f'\t{valid_roc_count}/{num_tfs}\t{valid_pr_count}/{num_tfs}\n'
    )
    try:
        aucs_file.write(
            f'\nOVERALL_ALL_SAMPLES\t{len(y_test)}\t{int(np.sum(y_test))}'
            f'\t{np.sum(y_test)/len(y_test):.4f}\t{overall_roc:.5f}\t{overall_pr:.5f}\n'
        )
    except Exception:
        pass

# ===========================================================
# Step 6: Print summary
# ===========================================================
print('\n' + '=' * 80)
print('BINARY PROTEIN-AWARE TRANSBIND  |  SWEEP V2  |  EPOCH 59  |  TEST SUMMARY')
print('=' * 80)
print(f'Checkpoint: {CHECKPOINT_PATH}')
print(f'\nModel configuration:')
print(f'  Cross-attention:     {model.use_cross_attention}')
print(f'  d_model:             {model.d_model}')
print(f'  Using DNase:         {test_dataset.has_dnase}')
print(f'  Using uniqueness:    True')
print(f'  Using protein feats: True')
print(f'\nOverall Results (All Samples):')
try:
    print(f'  Overall ROC AUC:  {overall_roc:.5f}')
    print(f'  Overall PR  AUC:  {overall_pr:.5f}')
    print(f'  Total samples:    {len(y_test):,}')
    print(f'  Positive samples: {int(np.sum(y_test)):,} ({np.sum(y_test)/len(y_test):.2%})')
except Exception:
    pass
print(f'\nPer-TF Results:')
print(f'  Average ROC AUC:  {avg_roc:.5f}')
print(f'  Average PR  AUC:  {avg_pr:.5f}')
print(f'  Median  ROC AUC:  {median_roc:.5f}')
print(f'  Median  PR  AUC:  {median_pr:.5f}')
print(f'  Valid ROC scores: {valid_roc_count}/{num_tfs}')
print(f'  Valid PR  scores: {valid_pr_count}/{num_tfs}')
print(f'  TFs with samples: {total_tfs_with_samples}/{num_tfs}')
print(f'\nResults saved to: {OUTPUT_FILE}')
print('=' * 80)

# Distribution summary
print('\nPer-TF Score Distribution:')
print(f'ROC AUC — Min: {np.nanmin(roc_scores):.3f}, '
      f'Max: {np.nanmax(roc_scores):.3f}, '
      f'Std: {np.nanstd(roc_scores):.3f}')
print(f'PR  AUC — Min: {np.nanmin(pr_scores):.3f}, '
      f'Max: {np.nanmax(pr_scores):.3f}, '
      f'Std: {np.nanstd(pr_scores):.3f}')

# Top 10 TFs by ROC
if valid_roc_count > 0:
    print('\nTop 10 TFs by ROC AUC:')
    top_indices = np.argsort(roc_scores)[-10:][::-1]
    for idx in top_indices:
        if not np.isnan(roc_scores[idx]):
            print(f'  TF {idx}: ROC={roc_scores[idx]:.3f}, PR={pr_scores[idx]:.3f}, '
                  f'N={tf_sample_counts[idx]}, Pos={tf_pos_counts[idx]} '
                  f'({tf_pos_counts[idx]/tf_sample_counts[idx]:.1%})')

# Bottom 10 TFs by ROC (≥100 samples)
if valid_roc_count > 0:
    valid_mask        = np.array([n >= 100 for n in tf_sample_counts])
    roc_scores_masked = np.where(valid_mask, roc_scores, np.nan)
    bottom_indices    = np.argsort(roc_scores_masked)[:10]
    print('\nBottom 10 TFs by ROC AUC (≥100 samples):')
    for idx in bottom_indices:
        if not np.isnan(roc_scores_masked[idx]) and tf_sample_counts[idx] >= 100:
            print(f'  TF {idx}: ROC={roc_scores[idx]:.3f}, PR={pr_scores[idx]:.3f}, '
                  f'N={tf_sample_counts[idx]}, Pos={tf_pos_counts[idx]} '
                  f'({tf_pos_counts[idx]/tf_sample_counts[idx]:.1%})')

# Top 10 TFs by sample count
if tf_sample_counts:
    print('\nTop 10 TFs by sample count:')
    most_samples_indices = np.argsort(tf_sample_counts)[-10:][::-1]
    for idx in most_samples_indices:
        if tf_sample_counts[idx] > 0:
            roc = roc_scores[idx] if not np.isnan(roc_scores[idx]) else -1
            pr  = pr_scores[idx]  if not np.isnan(pr_scores[idx])  else -1
            print(f'  TF {idx}: N={tf_sample_counts[idx]}, '
                  f'Pos={tf_pos_counts[idx]} ({tf_pos_counts[idx]/tf_sample_counts[idx]:.1%}), '
                  f'ROC={roc:.3f}, PR={pr:.3f}')

# Top 10 most imbalanced TFs
if tf_pos_counts:
    imbalance_ratios = []
    for i in range(len(tf_sample_counts)):
        if tf_sample_counts[i] > 0:
            pos_ratio = tf_pos_counts[i] / tf_sample_counts[i]
            imbalance = abs(0.5 - pos_ratio)
            imbalance_ratios.append((i, imbalance, pos_ratio))
        else:
            imbalance_ratios.append((i, 0, 0))

    imbalance_ratios_sorted = sorted(imbalance_ratios, key=lambda x: x[1], reverse=True)[:10]
    print('\nTop 10 most imbalanced TFs:')
    for idx, imbalance, pos_ratio in imbalance_ratios_sorted:
        if tf_sample_counts[idx] > 0:
            roc = roc_scores[idx] if not np.isnan(roc_scores[idx]) else -1
            pr  = pr_scores[idx]  if not np.isnan(pr_scores[idx])  else -1
            print(f'  TF {idx}: Pos={tf_pos_counts[idx]}/{tf_sample_counts[idx]} '
                  f'({pos_ratio:.1%}), ROC={roc:.3f}, PR={pr:.3f}')

print('\nTesting complete!')
print(f'Detailed results saved to: {OUTPUT_FILE}')