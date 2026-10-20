import os
import json

import numpy as np
import torch
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve,
)
from tqdm import tqdm

from train_binaryV1_resume_dnase_uniqueness_attentionV1_subsampling_validation_V1 import BinaryProteinAwareTransBind, BinaryTransBindDataset

# ===========================================================
# Paths
# ===========================================================
BASE_DIR = "/cheng-data/shreya/TransBind/"
DATA_DIR = os.path.join(BASE_DIR, "data/Binary_dataV1")

TF_MAPPING_PATH = os.path.join(
    DATA_DIR,
    'Binary_data_for_TF_splitV3/EGNN/tf_mapping_with_features_with_graphsV1.csv',
)
FEATURES_DIR = os.path.join(BASE_DIR, "Protein_data/prostt5_featuresV1")

TEST_UNIQUENESS_PATH = os.path.join(
    BASE_DIR, "uniqueness_processed/test_uniqueness.npy"
)

CHECKPOINT_PATH = os.path.join(
    BASE_DIR,
    "model_binary_best_sweep_v2/binary-sweep-v2-epoch=59-val_aupr=0.4995.ckpt"
)

TEST_INDICES_WITH_EXPIDX_PATH = os.path.join(
    DATA_DIR, 'standalone_test_indices_with_expidx.npz'
)

OUTPUT_DIR = os.path.join(BASE_DIR, "evaluation_results/sweep_v2_epoch59_per_experiment_690way")
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "aucs_binary_sweep_v2_epoch59_per_experiment_690way.txt"
)

CURVE_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "transbind2_curve_data.npz")

METHOD_NAME = "TransBind-2"
PLOT_COLOR  = "#1f77b4"

N_GRID_POINTS = 100
MEAN_FPR    = np.linspace(0.0, 1.0, N_GRID_POINTS)
MEAN_RECALL = np.linspace(0.0, 1.0, N_GRID_POINTS)

MIN_SAMPLES_FOR_CURVE = 10

CACHE_DIR = os.path.join(OUTPUT_DIR, "inference_cache")
os.makedirs(CACHE_DIR, exist_ok=True)
PRED_MEMMAP_PATH  = os.path.join(CACHE_DIR, "pred_y.memmap")
LABEL_MEMMAP_PATH = os.path.join(CACHE_DIR, "y_test.memmap")
PROGRESS_PATH      = os.path.join(CACHE_DIR, "progress.json")

BATCH_SIZE = 8000
CHECKPOINT_EVERY_N_BATCHES = 200

# ===========================================================
# Step 1: Load model
# ===========================================================
print("=" * 80)
print("BINARY PROTEIN-AWARE TRANSBIND  |  PER-EXPERIMENT (690-way) EVAL")
print("SWEEP V2 CHECKPOINT  |  Epoch 59  |  val_aupr=0.4995")
print("RESUMABLE VERSION -- predictions cached incrementally to disk")
print("NOW ALSO SAVES: macro-averaged curve npz + per-TF curve npz")
print("=" * 80)

print(f"\nLoading TF mapping from: {TF_MAPPING_PATH}")
tf_mapping = pd.read_csv(TF_MAPPING_PATH)
num_tfs = len(tf_mapping)
tf_names = dict(zip(tf_mapping['tf_idx'], tf_mapping['TF']))
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

# ===========================================================
# Step 2: Build test dataset
# ===========================================================
print("\nLoading test data...")

test_dna_path   = os.path.join(DATA_DIR, 'test_unique_dna.npz')
test_dnase_path = os.path.join(DATA_DIR, 'test_dnase.h5')

test_dataset = BinaryTransBindDataset(
    dna_sequences_path=test_dna_path,
    indices_path=TEST_INDICES_WITH_EXPIDX_PATH,
    dnase_h5_path=test_dnase_path,
    uniqueness_path=TEST_UNIQUENESS_PATH,
    split='test',
)

test_indices  = np.load(TEST_INDICES_WITH_EXPIDX_PATH)
test_labels   = test_indices['labels']
test_tf_idx   = test_indices['tf_idx']
test_cell_idx = test_indices['cell_idx']
test_exp_idx  = test_indices['experiment_idx']

n_total = len(test_dataset)
print(f"\nTest set size:      {n_total:,} examples")
print(f"Positive samples:   {np.sum(test_labels):,} ({np.sum(test_labels)/n_total:.2%})")
print(f"Unique TFs:         {len(np.unique(test_tf_idx))}")
print(f"Unique cell types:  {len(np.unique(test_cell_idx))}")

experiment_keys = np.unique(test_exp_idx)
num_experiments = len(experiment_keys)
print(f"Unique experiment_idx values in test set: {num_experiments} (expect up to 690)")

# ===========================================================
# Step 3: Resumable inference
# ===========================================================
print("\nSetting up resumable inference cache...")

if os.path.exists(PRED_MEMMAP_PATH) and os.path.exists(PROGRESS_PATH):
    print("Found existing cache -- resuming from last checkpoint.")
    pred_y_mm = np.memmap(PRED_MEMMAP_PATH, dtype=np.float32, mode='r+', shape=(n_total,))
    y_test_mm = np.memmap(LABEL_MEMMAP_PATH, dtype=np.uint8,   mode='r+', shape=(n_total,))
    with open(PROGRESS_PATH, 'r') as f:
        progress = json.load(f)
    start_batch = progress['last_completed_batch'] + 1
    print(f"Resuming from batch {start_batch} (already completed {start_batch} of "
          f"{(n_total + BATCH_SIZE - 1) // BATCH_SIZE} batches)")
else:
    print("No existing cache found -- starting fresh.")
    pred_y_mm = np.memmap(PRED_MEMMAP_PATH, dtype=np.float32, mode='w+', shape=(n_total,))
    y_test_mm = np.memmap(LABEL_MEMMAP_PATH, dtype=np.uint8,   mode='w+', shape=(n_total,))
    start_batch = 0
    progress = {'last_completed_batch': -1}

n_batches = (n_total + BATCH_SIZE - 1) // BATCH_SIZE
batch_starts = list(range(0, n_total, BATCH_SIZE))

print(f"\nRunning inference (batches {start_batch} to {n_batches - 1})...")

with torch.no_grad():
    for batch_num in tqdm(range(start_batch, n_batches), initial=start_batch,
                           total=n_batches, desc="Batches"):
        i = batch_starts[batch_num]
        batch_indices = range(i, min(i + BATCH_SIZE, n_total))

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
        batch_pred = torch.sigmoid(logits).cpu().numpy().flatten()
        batch_lab  = batch_labels.numpy().flatten()

        start_i = i
        end_i   = min(i + BATCH_SIZE, n_total)
        pred_y_mm[start_i:end_i] = batch_pred.astype(np.float32)
        y_test_mm[start_i:end_i] = batch_lab.astype(np.uint8)

        if batch_num % CHECKPOINT_EVERY_N_BATCHES == 0:
            pred_y_mm.flush()
            y_test_mm.flush()
            with open(PROGRESS_PATH, 'w') as f:
                json.dump({'last_completed_batch': batch_num}, f)

    pred_y_mm.flush()
    y_test_mm.flush()
    with open(PROGRESS_PATH, 'w') as f:
        json.dump({'last_completed_batch': n_batches - 1}, f)

print("\nInference complete. Loading final arrays into memory for metric computation...")
pred_y = np.array(pred_y_mm)
y_test = np.array(y_test_mm)

print(f"Predictions shape: {pred_y.shape}")
print(f"Labels shape:      {y_test.shape}")

assert len(pred_y) == len(test_exp_idx), "Row count mismatch -- investigate before trusting results."

# ===========================================================
# Step 4: Overall (pooled/micro) metrics -- for reference only
# ===========================================================
print("\nCalculating overall (pooled/micro) metrics for reference...")
try:
    overall_roc = roc_auc_score(y_test, pred_y)
    overall_pr  = average_precision_score(y_test, pred_y)
    print(f'Overall (pooled) ROC AUC:  {overall_roc:.5f}')
    print(f'Overall (pooled) PR  AUC:  {overall_pr:.5f}')
    print('(This is NOT comparable to the macro-averaged numbers below.)')
except Exception as e:
    print(f"Error calculating overall metrics: {e}")

# ===========================================================
# Step 5: Per-EXPERIMENT metrics -- TRUE 690-way macro-average
#         + per-experiment curves, interpolated onto a common grid
#         + per-TF bookkeeping
# ===========================================================
print("\nCalculating true per-experiment (690-way) metrics...")

roc_scores = []
pr_scores  = []

curve_tpr_list       = []
curve_precision_list = []
curve_roc_scores     = []
curve_pr_scores      = []
curve_tf_idx_list    = []

with open(OUTPUT_FILE, 'w') as aucs_file:
    aucs_file.write('Experiment_Idx\tTF_Index\tCell_Index\tTF_Name\tN_Samples\tN_Positive\tPos_Ratio\tROC_AUC\tPR_AUC\n')

    for exp_idx_val in tqdm(experiment_keys, desc="Per-experiment metrics"):
        mask = (test_exp_idx == exp_idx_val)

        n_samples = int(np.sum(mask))
        if n_samples == 0:
            roc_scores.append(np.nan)
            pr_scores.append(np.nan)
            aucs_file.write(f'{exp_idx_val}\t-\t-\t-\t0\t0\t0.0000\tNaN\tNaN\n')
            continue

        tf_idx_val   = int(test_tf_idx[mask][0])
        cell_idx_val = int(test_cell_idx[mask][0])
        tf_name = tf_names.get(tf_idx_val, f"TF_{tf_idx_val}")

        exp_labels = y_test[mask]
        exp_preds  = pred_y[mask]
        n_positive = int(np.sum(exp_labels))
        pos_ratio  = n_positive / n_samples if n_samples > 0 else 0.0

        try:
            if len(np.unique(exp_labels)) > 1 and n_samples >= MIN_SAMPLES_FOR_CURVE:
                roc_auc = roc_auc_score(exp_labels, exp_preds)
                pr_auc  = average_precision_score(exp_labels, exp_preds)
                roc_scores.append(roc_auc)
                pr_scores.append(pr_auc)
                aucs_file.write(
                    f'{exp_idx_val}\t{tf_idx_val}\t{cell_idx_val}\t{tf_name}\t{n_samples}\t{n_positive}'
                    f'\t{pos_ratio:.4f}\t{roc_auc:.5f}\t{pr_auc:.5f}\n'
                )

                exp_fpr, exp_tpr, _ = roc_curve(exp_labels, exp_preds)
                interp_tpr = np.interp(MEAN_FPR, exp_fpr, exp_tpr)
                interp_tpr[0] = 0.0
                curve_tpr_list.append(interp_tpr)

                exp_precision, exp_recall, _ = precision_recall_curve(exp_labels, exp_preds)
                order = np.argsort(exp_recall)
                interp_precision = np.interp(
                    MEAN_RECALL, exp_recall[order], exp_precision[order]
                )
                curve_precision_list.append(interp_precision)

                curve_roc_scores.append(roc_auc)
                curve_pr_scores.append(pr_auc)
                curve_tf_idx_list.append(tf_idx_val)
            else:
                roc_scores.append(np.nan)
                pr_scores.append(np.nan)
                aucs_file.write(
                    f'{exp_idx_val}\t{tf_idx_val}\t{cell_idx_val}\t{tf_name}\t{n_samples}\t{n_positive}'
                    f'\t{pos_ratio:.4f}\tNaN\tNaN\n'
                )
        except ValueError as e:
            print(f"Error for experiment_idx {exp_idx_val}: {e}")
            roc_scores.append(np.nan)
            pr_scores.append(np.nan)
            aucs_file.write(
                f'{exp_idx_val}\t{tf_idx_val}\t{cell_idx_val}\t{tf_name}\t{n_samples}\t{n_positive}'
                f'\t{pos_ratio:.4f}\tNaN\tNaN\n'
            )

    avg_roc    = np.nanmean(roc_scores)
    avg_pr     = np.nanmean(pr_scores)
    median_roc = np.nanmedian(roc_scores)
    median_pr  = np.nanmedian(pr_scores)

    valid_roc_count = int(np.sum(~np.isnan(roc_scores)))
    valid_pr_count  = int(np.sum(~np.isnan(pr_scores)))

    aucs_file.write(f'\nAVERAGE\t-\t-\t-\t-\t-\t-\t{avg_roc:.5f}\t{avg_pr:.5f}\n')
    aucs_file.write(f'MEDIAN\t-\t-\t-\t-\t-\t-\t{median_roc:.5f}\t{median_pr:.5f}\n')
    aucs_file.write(
        f'VALID_SCORES\t-\t-\t-\t{num_experiments}\t-\t-'
        f'\t{valid_roc_count}/{num_experiments}\t{valid_pr_count}/{num_experiments}\n'
    )

# ===========================================================
# Step 5b: Save macro-averaged curve npz -- DanQ-compatible schema
# ===========================================================
print("\nAveraging per-experiment ROC and PR curves...")

n_valid_curves = len(curve_roc_scores)
if n_valid_curves > 0:
    curve_tpr_arr       = np.stack(curve_tpr_list, axis=0)
    curve_precision_arr = np.stack(curve_precision_list, axis=0)

    mean_tpr = curve_tpr_arr.mean(axis=0)
    mean_tpr[-1] = 1.0
    mean_precision = curve_precision_arr.mean(axis=0)

    curve_roc_scores_arr = np.array(curve_roc_scores, dtype=np.float64)
    curve_pr_scores_arr  = np.array(curve_pr_scores,  dtype=np.float64)

    np.savez(
        CURVE_OUTPUT_PATH,
        method_name=METHOD_NAME,
        mean_fpr=MEAN_FPR,
        mean_tpr=mean_tpr,
        mean_recall=MEAN_RECALL,
        mean_precision=mean_precision,
        avg_roc_auc=float(curve_roc_scores_arr.mean()),
        avg_pr_auc=float(curve_pr_scores_arr.mean()),
        roc_scores=curve_roc_scores_arr,
        pr_scores=curve_pr_scores_arr,
        valid_features=n_valid_curves,
        total_features=num_experiments,
        color=PLOT_COLOR,
    )
    print(f"Saved curve data to: {CURVE_OUTPUT_PATH}")
else:
    print("No experiments had enough samples/class balance to compute curves -- "
          "skipping curve npz.")

# ===========================================================
# Step 5c: Aggregate per-experiment curves up to per-TF curves
#          (BOTH ROC and PR) and save them for multi-TF plotting
# ===========================================================
print("\nAggregating per-experiment curves into per-TF curves (ROC + PR)...")

PER_TF_CURVE_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "transbind2_per_tf_curve_data.npz")

if len(curve_tpr_list) > 0:
    curve_tpr_arr       = np.stack(curve_tpr_list, axis=0)        # (n_valid_curves, 100)
    curve_precision_arr = np.stack(curve_precision_list, axis=0)  # (n_valid_curves, 100)
    curve_tf_idx_arr    = np.array(curve_tf_idx_list)

    unique_tf_idxs = sorted(set(curve_tf_idx_arr.tolist()))
    per_tf_mean_tpr       = []
    per_tf_mean_precision = []
    per_tf_names          = []
    per_tf_auroc          = []
    per_tf_aupr           = []
    per_tf_n_exps         = []

    for tf_idx_val in unique_tf_idxs:
        mask = (curve_tf_idx_arr == tf_idx_val)

        mean_tpr_this_tf = curve_tpr_arr[mask].mean(axis=0)
        mean_tpr_this_tf[0]  = 0.0
        mean_tpr_this_tf[-1] = 1.0
        per_tf_mean_tpr.append(mean_tpr_this_tf)

        mean_precision_this_tf = curve_precision_arr[mask].mean(axis=0)
        per_tf_mean_precision.append(mean_precision_this_tf)

        per_tf_names.append(tf_names.get(tf_idx_val, f"TF_{tf_idx_val}"))
        per_tf_auroc.append(float(np.array(curve_roc_scores)[mask].mean()))
        per_tf_aupr.append(float(np.array(curve_pr_scores)[mask].mean()))
        per_tf_n_exps.append(int(mask.sum()))

    per_tf_mean_tpr       = np.stack(per_tf_mean_tpr, axis=0)        # (n_tfs, 100)
    per_tf_mean_precision = np.stack(per_tf_mean_precision, axis=0)  # (n_tfs, 100)

    np.savez(
        PER_TF_CURVE_OUTPUT_PATH,
        mean_fpr=MEAN_FPR,                                  # shared x-axis for ROC
        mean_recall=MEAN_RECALL,                             # shared x-axis for PR
        tf_names=np.array(per_tf_names),
        tf_idx=np.array(unique_tf_idxs),
        tpr_matrix=per_tf_mean_tpr,                           # (n_tfs, 100) -- ROC curves
        precision_matrix=per_tf_mean_precision,                # (n_tfs, 100) -- PR curves
        auroc_per_tf=np.array(per_tf_auroc),
        aupr_per_tf=np.array(per_tf_aupr),
        n_experiments_per_tf=np.array(per_tf_n_exps),
        method_name=METHOD_NAME,
    )
    print(f"Saved per-TF curve data to: {PER_TF_CURVE_OUTPUT_PATH}")
    print(f"  {len(unique_tf_idxs)} TFs, ROC + PR curves each averaged across their own experiments/cell types")
else:
    print("No valid curves to aggregate -- skipping per-TF curve npz.")

# ===========================================================
# Step 6: Print summary
# ===========================================================
print('\n' + '=' * 80)
print('TRUE PER-EXPERIMENT RESULTS -- 690-way macro-average')
print('=' * 80)
print(f'Total unique experiments: {num_experiments}')
print(f'Average ROC AUC:  {avg_roc:.5f}')
print(f'Average PR  AUC:  {avg_pr:.5f}')
print(f'Median  ROC AUC:  {median_roc:.5f}')
print(f'Median  PR  AUC:  {median_pr:.5f}')
print(f'Valid ROC scores: {valid_roc_count}/{num_experiments}')
print(f'Valid PR  scores: {valid_pr_count}/{num_experiments}')
print(f'\nResults saved to:   {OUTPUT_FILE}')
print(f'Curve data saved to: {CURVE_OUTPUT_PATH}')
print(f'Per-TF curve data saved to: {PER_TF_CURVE_OUTPUT_PATH}')
print('=' * 80)

print("\nCleaning up inference cache (comment this out if you want to keep it)...")
# import shutil
# shutil.rmtree(CACHE_DIR)
print(f"Cache retained at: {CACHE_DIR} (delete manually if no longer needed)")