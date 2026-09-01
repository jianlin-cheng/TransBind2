import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
import pandas as pd
import math
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score, f1_score
import h5py
from typing import Optional, Tuple

import torch.serialization
if hasattr(torch.serialization, "add_safe_globals"):
    torch.serialization.add_safe_globals([np.core.multiarray.scalar, np.dtype])

torch.set_float32_matmul_precision('medium')
from lightning_fabric.plugins.io.torch_io import TorchCheckpointIO

class SafeCheckpointIO(TorchCheckpointIO):
    """Custom checkpoint IO that allows loading with weights_only=False"""
    def load_checkpoint(self, path, map_location=None):
        return torch.load(path, map_location=map_location, weights_only=False)

class PositionWeightedPooling(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.W = nn.Linear(d_model, 1)

    def forward(self, x):
        scores  = self.W(x)                        # (B, M, 1)
        weights = torch.softmax(scores, dim=1)     # (B, M, 1)
        pooled  = (weights * x).sum(dim=1)         # (B, d_model)
        return pooled


# ===========================================================
# ProteinReduceVariable
# ===========================================================
class ProteinReduceVariable(nn.Module):
    def __init__(
        self,
        protein_in_dim: int = 1024,
        d_model: int = 128,
        target_len: int = 200,
        nhead: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.target_len = target_len
        self.d_model    = d_model

        self.input_proj = nn.Sequential(
            nn.Linear(protein_in_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.query = nn.Parameter(
            torch.randn(1, target_len, d_model) * (d_model ** -0.5)
        )

        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )

        self.norm = nn.LayerNorm(d_model)
        self.ff   = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(
        self,
        protein_emb: torch.Tensor,
        protein_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        protein_emb = self.input_proj(protein_emb)
        B = protein_emb.shape[0]

        q = self.query.expand(B, -1, -1)
        out, _ = self.attn(
            query=q,
            key=protein_emb,
            value=protein_emb,
            key_padding_mask=protein_mask,
            need_weights=False,
        )
        out = self.norm(out + self.ff(out))
        return out, None


# ===========================================================
# FFNBlock
# ===========================================================
class FFNBlock(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.2):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.ff(x))


# ===========================================================
# HybridCrossAttentionEncoder
# ===========================================================
class HybridCrossAttentionEncoder(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 3,
        num_bidir_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_bidir_layers = num_bidir_layers

        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "dna_to_prot_attn": nn.MultiheadAttention(
                    embed_dim=d_model, num_heads=nhead,
                    dropout=dropout, batch_first=True,
                ),
                "prot_to_dna_attn": nn.MultiheadAttention(
                    embed_dim=d_model, num_heads=nhead,
                    dropout=dropout, batch_first=True,
                ),
                "norm_dna":  nn.LayerNorm(d_model),
                "norm_prot": nn.LayerNorm(d_model),
                "ffn_dna":   FFNBlock(d_model, dropout),
                "ffn_prot":  FFNBlock(d_model, dropout),
            })
            for _ in range(num_layers)
        ])

    def forward(
        self,
        protein: torch.Tensor,
        dna: torch.Tensor,
        protein_mask: Optional[torch.Tensor] = None,
        dna_mask: Optional[torch.Tensor] = None,
        return_both: bool = True,
        return_attention: bool = False,
    ):
        prot_ctx = protein
        dna_ctx  = dna

        for layer_idx, layer in enumerate(self.layers):
            dna_out, _ = layer["dna_to_prot_attn"](
                query=dna_ctx,
                key=prot_ctx,
                value=prot_ctx,
                key_padding_mask=protein_mask,
                need_weights=False,
            )
            dna_ctx = layer["norm_dna"](dna_ctx + dna_out)
            dna_ctx = layer["ffn_dna"](dna_ctx)

            if layer_idx < self.num_bidir_layers:
                prot_out, _ = layer["prot_to_dna_attn"](
                    query=prot_ctx,
                    key=dna_ctx,
                    value=dna_ctx,
                    key_padding_mask=dna_mask,
                    need_weights=False,
                )
                prot_ctx = layer["norm_prot"](prot_ctx + prot_out)
                prot_ctx = layer["ffn_prot"](prot_ctx)

        if return_both:
            return dna_ctx, prot_ctx
        return dna_ctx


# ===========================================================
# BalancedBatchSampler — with TF coverage check
# ===========================================================
class BalancedBatchSampler(Sampler):
    """Creates batches with equal numbers of positive and negative examples."""

    def __init__(self, labels, batch_size, max_batches=3000, tf_idx=None, num_tfs=161):
        self.labels     = labels
        self.batch_size = batch_size
        self.tf_idx     = tf_idx
        self.num_tfs    = num_tfs

        self.positive_indices = np.where(labels == 1)[0]
        self.negative_indices = np.where(labels == 0)[0]
        self.half_batch = batch_size // 2
        self.num_batches = min(
            len(self.positive_indices) // self.half_batch,
            len(self.negative_indices) // self.half_batch,
            max_batches,
        )
        print(f"Created balanced sampler: {len(self.positive_indices)} pos, "
              f"{len(self.negative_indices)} neg → {self.num_batches} batches/epoch")

    def __iter__(self):
        pos_indices = self.positive_indices.copy()
        neg_indices = self.negative_indices.copy()
        np.random.shuffle(pos_indices)
        np.random.shuffle(neg_indices)

        selected_pos = pos_indices[:self.num_batches * self.half_batch]

        # TF coverage check — prints once per epoch
        if self.tf_idx is not None:
            tfs_covered = np.unique(self.tf_idx[selected_pos])
            missing     = sorted(set(range(self.num_tfs)) - set(tfs_covered.tolist()))
            print(f"TFs covered this epoch: {len(tfs_covered)}/{self.num_tfs} "
                  f"(missing indices: {missing if missing else 'none'})")

        pos_batches = selected_pos.reshape(self.num_batches, self.half_batch)
        neg_batches = neg_indices[:self.num_batches * self.half_batch].reshape(
            self.num_batches, self.half_batch
        )

        for i in range(self.num_batches):
            batch_indices = np.concatenate([pos_batches[i], neg_batches[i]])
            np.random.shuffle(batch_indices)
            yield batch_indices

    def __len__(self):
        return self.num_batches


# ===========================================================
# Dataset
# ===========================================================
class BinaryTransBindDataset(Dataset):
    """
    DNase is optional. If dnase_h5_path is None or doesn't exist, every
    example gets a flat "no coverage" default DNase signal instead of real
    per-cell-type data. Pass the default via no_coverage_value — it should
    equal what a truly-zero raw DNase signal maps to after log1p + global
    z-score normalization (see normalize_dnase.py / normalization_params.json
    for how this value is derived). This keeps the input in-distribution for
    a model trained with real DNase, rather than feeding it an arbitrary
    or zero-valued placeholder. If no_coverage_value isn't provided, a
    hardcoded fallback verified against this project's real data is used.
    """

    DNASE_MIN_DEFAULT = -10.0
    DNASE_MAX_DEFAULT = 5.0
    # Fallback if normalization_params.json isn't available. This is the
    # value real "zero raw DNase signal" positions take on throughout the
    # dataset (verified directly from normalized .npy files).
    DNASE_NO_COVERAGE_FALLBACK = -3.6954

    def __init__(self, dna_sequences_path, indices_path, uniqueness_path,
                 dnase_h5_path=None, split='train',
                 no_coverage_value=None):
        import time
        start_time = time.time()

        dna_data = np.load(dna_sequences_path)
        self.dna_sequences = dna_data['dna_sequences']

        indices_data  = np.load(indices_path)
        self.dna_idx  = indices_data['dna_idx']
        self.tf_idx   = indices_data['tf_idx']
        self.cell_idx = indices_data['cell_idx']
        self.labels   = indices_data['labels']

        print(f"[{split}] Loaded {len(self.labels):,} examples")
        print(f"[{split}] Unique DNA sequences: {len(self.dna_sequences):,}")

        self.has_dnase = dnase_h5_path is not None and os.path.exists(dnase_h5_path)

        if self.has_dnase:
            print(f"[{split}] Loading DNase from {dnase_h5_path} ...")
            with h5py.File(dnase_h5_path, 'r') as f:
                self.dnase_min  = float(f.attrs['dnase_min'])
                self.dnase_max  = float(f.attrs['dnase_max'])
                self.dnase_data = f['dnase'][:]
            print(f"[{split}] DNase shape: {self.dnase_data.shape}")
        else:
            print(f"[{split}] No DNase file provided — using fixed no-coverage "
                  f"default DNase signal for all examples")
            self.dnase_min = self.DNASE_MIN_DEFAULT
            self.dnase_max = self.DNASE_MAX_DEFAULT

            default_z = no_coverage_value if no_coverage_value is not None \
                else self.DNASE_NO_COVERAGE_FALLBACK
            print(f"[{split}] Using no-coverage default: {default_z:.4f}")

            default_z = float(np.clip(default_z, self.dnase_min, self.dnase_max))
            self._default_dnase_signal = np.full(1000, default_z, dtype=np.float32)

        self.uniqueness_data = np.load(uniqueness_path)
        print(f"[{split}] Uniqueness shape: {self.uniqueness_data.shape}")
        print(f"[{split}] Dataset ready in {time.time() - start_time:.1f}s")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        dna_tensor = torch.FloatTensor(self.dna_sequences[self.dna_idx[idx]])

        unique_dna_idx = self.dna_idx[idx]

        if self.has_dnase:
            cell_idx_val    = int(self.cell_idx[idx])
            dnase_quantized = self.dnase_data[unique_dna_idx, cell_idx_val, :]
            dnase_signal    = (dnase_quantized.astype(np.float32) / 255.0 *
                               (self.dnase_max - self.dnase_min) + self.dnase_min)
        else:
            dnase_signal = self._default_dnase_signal

        dnase_tensor = torch.FloatTensor(dnase_signal)

        uniqueness_tensor = torch.FloatTensor([self.uniqueness_data[unique_dna_idx]])
        tf_index          = torch.tensor(self.tf_idx[idx], dtype=torch.long)
        label             = torch.FloatTensor([self.labels[idx]])

        return dna_tensor, dnase_tensor, uniqueness_tensor, tf_index, label


# ===========================================================
# Protein feature loader
# ===========================================================
class CSVProteinFeatureLoader:
    def __init__(self, mapping_path, features_dir):
        self.mapping_path = mapping_path
        self.features_dir = features_dir
        self.load_mapping()
        self.load_protein_features()

    def load_mapping(self):
        print(f"\nLoading TF feature mapping from {self.mapping_path}")
        self.tf_mapping = pd.read_csv(self.mapping_path)
        print(f"Loaded mapping with {len(self.tf_mapping)} TFs")
        print(f"Available columns: {list(self.tf_mapping.columns)}")

        if 'prot5_feature_path' in self.tf_mapping.columns:
            self.path_column = 'prot5_feature_path'
        elif 'matched_filename' in self.tf_mapping.columns:
            self.path_column = 'matched_filename'
        else:
            path_columns = [c for c in self.tf_mapping.columns
                            if 'path' in c.lower() or 'file' in c.lower()]
            if path_columns:
                self.path_column = path_columns[0]
            else:
                raise ValueError("Could not find a column with file paths in the CSV file")

        print(f"Using column '{self.path_column}' for feature paths")
        mapped_tfs = self.tf_mapping[self.path_column].notna().sum()
        print(f"Feature paths found for {mapped_tfs}/{len(self.tf_mapping)} TFs")

    def load_single_feature_file(self, path_or_filename):
        feature_path = (path_or_filename if os.path.isabs(path_or_filename)
                        else os.path.join(self.features_dir, path_or_filename))

        if not os.path.exists(feature_path):
            print(f"Warning: Feature file not found: {feature_path}")
            return None

        try:
            with open(feature_path, 'r') as f:
                numbers = [float(x) for x in f.read().split()]
            data = np.array(numbers)

            if len(numbers) % 2048 == 0:
                seq_len = len(numbers) // 2048
                return data.reshape(seq_len, 2048)
            else:
                print(f"Warning: {len(numbers)} elements not divisible by 2048")
                return None
        except Exception as e:
            print(f"Error loading {feature_path}: {e}")
            return None

    def load_protein_features(self):
        print("\nLoading protein features (sequence mode) ...")
        MAX_PROTEIN_LEN = 512
        num_tfs   = len(self.tf_mapping)
        sequences = []
        lengths   = []

        loaded_count = 0
        for _, row in self.tf_mapping.iterrows():
            if pd.notna(row[self.path_column]):
                seq = self.load_single_feature_file(row[self.path_column])
                if seq is not None:
                    sequences.append(seq)
                    lengths.append(len(seq))
                    loaded_count += 1
                    continue
            sequences.append(np.zeros((1, 2048), dtype=np.float32))
            lengths.append(1)

        print(f"Successfully loaded {loaded_count}/{num_tfs} TF features")
        print(f"Protein length range: {min(lengths)}–{max(lengths)}, "
              f"mean: {np.mean(lengths):.1f}")

        max_len = min(max(lengths), MAX_PROTEIN_LEN)
        print(f"Pre-padding all sequences to length {max_len}")

        self.protein_features = np.zeros((num_tfs, max_len, 2048), dtype=np.float32)
        self.protein_masks    = np.ones((num_tfs, max_len), dtype=bool)

        for i, (seq, length) in enumerate(zip(sequences, lengths)):
            actual_len = min(length, MAX_PROTEIN_LEN)
            self.protein_features[i, :actual_len] = seq[:actual_len]
            self.protein_masks[i, :actual_len]    = False

        self.protein_features = torch.FloatTensor(self.protein_features)
        self.protein_masks    = torch.BoolTensor(self.protein_masks)

        print(f"Protein features tensor: {self.protein_features.shape}")
        print(f"Protein masks tensor:    {self.protein_masks.shape}")

        if loaded_count == 0:
            print("⚠️  WARNING: No protein features were loaded!")


# ===========================================================
# Main model
# ===========================================================
class BinaryProteinAwareTransBind(pl.LightningModule):
    def __init__(self, mapping_path, features_dir, num_tfs,
                 learning_rate=0.000328, dropout=0.088,
                 weight_decay=0.028, use_cross_attention=True,
                 d_model=320, nhead_cross=8, num_cross_layers=3,
                 num_bidir_layers=2, protein_target_len=200):
        super().__init__()
        self.save_hyperparameters()
        self.num_tfs = num_tfs
        self.d_model = d_model

        self.conv1d      = nn.Conv1d(in_channels=6, out_channels=320, kernel_size=26, padding=0)
        self.maxpool     = nn.MaxPool1d(kernel_size=13, stride=13)
        self.dropout_cnn = nn.Dropout(dropout)

        self.bilstm = nn.LSTM(
            input_size=320, hidden_size=160, num_layers=2,
            batch_first=True, bidirectional=True, dropout=dropout,
        )

        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=320, nhead=16, dim_feedforward=1024,
            dropout=dropout, activation='relu', batch_first=True,
        )

        print("Loading protein features...")
        self.protein_loader = CSVProteinFeatureLoader(mapping_path, features_dir)
        self.register_buffer('protein_features', self.protein_loader.protein_features)
        self.register_buffer('protein_masks',    self.protein_loader.protein_masks)

        self.use_cross_attention = use_cross_attention
        if self.use_cross_attention:
            self.protein_reducer = ProteinReduceVariable(
                protein_in_dim=2048,
                d_model=d_model,
                target_len=protein_target_len,
                nhead=nhead_cross,
                dropout=dropout,
            )
            self.cross_attention_encoder = HybridCrossAttentionEncoder(
                d_model=d_model,
                nhead=nhead_cross,
                num_layers=num_cross_layers,
                num_bidir_layers=num_bidir_layers,
                dropout=dropout,
            )

        self.position_weighted_pool = PositionWeightedPooling(320)
        self.fc1    = nn.Linear(320, 512)
        self.fc2    = nn.Linear(512, 256)
        self.fc_out = nn.Linear(256, 1)

        self.criterion = nn.BCEWithLogitsLoss()
        self.validation_step_outputs = []

    def get_protein_features(self, tf_indices):
        return self.protein_features[tf_indices], self.protein_masks[tf_indices]

    def forward(self, dna_seq, dnase_signal, uniqueness_score, tf_idx):
        dnase_exp      = dnase_signal.unsqueeze(2)
        uniq_broadcast = uniqueness_score.unsqueeze(2).expand(-1, 1000, -1)
        combined_input = torch.cat([dna_seq, dnase_exp, uniq_broadcast], dim=2)  # (B, 1000, 6)

        x = F.relu(self.conv1d(combined_input.transpose(1, 2)))
        x = self.maxpool(x)
        x = self.dropout_cnn(x)
        x = x.transpose(1, 2)

        lstm_out, _        = self.bilstm(x)
        dna_sequence_feats = self.transformer_layer(lstm_out)

        target_dna_len = 256
        if dna_sequence_feats.shape[1] != target_dna_len:
            dna_sequence_feats = F.interpolate(
                dna_sequence_feats.transpose(1, 2),
                size=target_dna_len, mode='linear', align_corners=False,
            ).transpose(1, 2)

        if self.use_cross_attention:
            protein_raw, protein_mask = self.get_protein_features(tf_idx)
            protein_reduced, _        = self.protein_reducer(protein_raw, protein_mask)

            dna_ctx, _ = self.cross_attention_encoder(
                protein=protein_reduced,
                dna=dna_sequence_feats,
                protein_mask=None,
                dna_mask=None,
                return_both=True,
            )
            dna_final = dna_sequence_feats + dna_ctx
        else:
            dna_final = dna_sequence_feats

        dna_global = self.position_weighted_pool(dna_final)

        x = F.relu(self.fc1(dna_global))
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=0.1, training=self.training)
        logits = self.fc_out(x)

        return logits
    def forward_with_protein(
        self,
        dna_seq,
        dnase_signal,
        uniqueness_score,
        protein_raw,
        protein_mask=None,
    ):
        """
        Forward pass using a protein embedding directly instead of tf_idx.

        protein_raw shape:
            (B, protein_length, 2048)

        protein_mask shape:
            (B, protein_length)
            False = real residue
            True  = padding
        """

        dnase_exp = dnase_signal.unsqueeze(2)

        uniq_broadcast = uniqueness_score.unsqueeze(2).expand(
            -1, 1000, -1
        )

        combined_input = torch.cat(
            [dna_seq, dnase_exp, uniq_broadcast],
            dim=2
        )

        # DNA encoder
        x = F.relu(
            self.conv1d(combined_input.transpose(1, 2))
        )

        x = self.maxpool(x)
        x = self.dropout_cnn(x)
        x = x.transpose(1, 2)

        lstm_out, _ = self.bilstm(x)

        dna_sequence_feats = self.transformer_layer(
            lstm_out
        )

        # Same DNA resampling as normal forward()
        target_dna_len = 256

        if dna_sequence_feats.shape[1] != target_dna_len:
            dna_sequence_feats = F.interpolate(
                dna_sequence_feats.transpose(1, 2),
                size=target_dna_len,
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)

        # Protein embedding is supplied directly
        if self.use_cross_attention:

            protein_reduced, _ = self.protein_reducer(
                protein_raw,
                protein_mask
            )

            dna_ctx, _ = self.cross_attention_encoder(
                protein=protein_reduced,
                dna=dna_sequence_feats,
                protein_mask=None,
                dna_mask=None,
                return_both=True,
            )

            dna_final = dna_sequence_feats + dna_ctx

        else:
            dna_final = dna_sequence_feats

        # Same classifier as normal forward()
        dna_global = self.position_weighted_pool(
            dna_final
        )

        x = F.relu(self.fc1(dna_global))
        x = F.dropout(
            x,
            p=0.2,
            training=self.training
        )

        x = F.relu(self.fc2(x))
        x = F.dropout(
            x,
            p=0.1,
            training=self.training
        )

        logits = self.fc_out(x)

        return logits

    def training_step(self, batch, batch_idx):
        dna_seq, dnase_signal, uniqueness_score, tf_idx, labels = batch
        logits = self(dna_seq, dnase_signal, uniqueness_score, tf_idx)
        loss   = self.criterion(logits, labels)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        dna_seq, dnase_signal, uniqueness_score, tf_idx, labels = batch
        logits = self(dna_seq, dnase_signal, uniqueness_score, tf_idx)
        loss   = self.criterion(logits, labels)
        y_hat  = torch.sigmoid(logits)
        self.validation_step_outputs.append({
            'y_hat': y_hat.detach().cpu(),
            'y':     labels.detach().cpu(),
            'loss':  loss.detach().cpu(),
        })
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        if not self.validation_step_outputs:
            return

        all_preds   = np.concatenate([x['y_hat'] for x in self.validation_step_outputs])
        all_targets = np.concatenate([x['y']     for x in self.validation_step_outputs])
        y_true, y_pred = all_targets, all_preds
        y_pred_binary  = (y_pred > 0.5).astype(np.int32)

        try:
            auroc     = roc_auc_score(y_true, y_pred)
            aupr      = average_precision_score(y_true, y_pred)
            accuracy  = accuracy_score(y_true, y_pred_binary)
            precision = precision_score(y_true, y_pred_binary, zero_division=0)
            recall    = recall_score(y_true, y_pred_binary, zero_division=0)
            f1        = f1_score(y_true, y_pred_binary, zero_division=0)

            self.log('val_auroc',     auroc,     on_epoch=True, prog_bar=True)
            self.log('val_aupr',      aupr,      on_epoch=True, prog_bar=True)
            self.log('val_accuracy',  accuracy,  on_epoch=True)
            self.log('val_precision', precision, on_epoch=True)
            self.log('val_recall',    recall,    on_epoch=True)
            self.log('val_f1',        f1,        on_epoch=True)

            pos_count = int(np.sum(y_true))
            neg_count = len(y_true) - pos_count
            print(f"\n{'='*50}")
            print(f"EPOCH {self.current_epoch} VALIDATION RESULTS")
            print(f"{'='*50}")
            print(f"ROC AUC:   {auroc:.5f}")
            print(f"PR  AUC:   {aupr:.5f}")
            print(f"Accuracy:  {accuracy:.5f}")
            print(f"Precision: {precision:.5f}")
            print(f"Recall:    {recall:.5f}")
            print(f"F1 Score:  {f1:.5f}")
            print(f"Class balance: {pos_count} pos, {neg_count} neg "
                  f"({pos_count/len(y_true):.2%} positive)")
            print(f"[NOTE] Validation on fixed 2M subset. Run full eval on best checkpoint for paper numbers.")
            print(f"{'='*50}\n")

        except Exception as e:
            print(f"Error calculating metrics: {e}")
            import traceback; traceback.print_exc()

        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=60, eta_min=1e-6,
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


# ===========================================================
# DataModule
# ===========================================================
class BalancedBinaryTransBindDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=2048, num_workers=0,
                 val_batch_size=None, max_batches=3000,
                 val_subsample_size=2_000_000,
                 use_dnase=True,
                 dnase_no_coverage_value=None,
                 train_uniqueness_path=None,
                 val_uniqueness_path=None):
        super().__init__()
        self.data_dir           = data_dir
        self.batch_size         = batch_size
        self.num_workers        = num_workers
        self.val_batch_size     = val_batch_size or batch_size * 2
        self.max_batches        = max_batches
        self.val_subsample_size = val_subsample_size
        self.use_dnase          = use_dnase

        print("Loading datasets ...")
        train_dna_path          = os.path.join(data_dir, 'train_unique_dna.npz')
        self.train_indices_path = os.path.join(data_dir, 'standalone_train_indices.npz')

        val_dna_path            = os.path.join(data_dir, 'val_unique_dna.npz')
        val_indices_path        = os.path.join(data_dir, 'standalone_val_indices.npz')

        # DNase is optional. When use_dnase=False, both paths resolve to
        # None and BinaryTransBindDataset falls back to a fixed
        # no-coverage default signal for every example.
        if self.use_dnase:
            train_dnase_path = os.path.join(data_dir, 'train_dnase.h5')
            val_dnase_path   = os.path.join(data_dir, 'val_dnase.h5')
        else:
            print("use_dnase=False — DNase channel will use the fixed "
                  "no-coverage default for all examples")
            train_dnase_path = None
            val_dnase_path   = None

        # Uniqueness paths must be supplied explicitly (no hardcoded
        # machine-specific defaults) — pass them in when constructing
        # this DataModule.
        if train_uniqueness_path is None or val_uniqueness_path is None:
            raise ValueError(
                "train_uniqueness_path and val_uniqueness_path must be provided"
            )

        tf_mapping_path  = os.path.join(data_dir, 'tf_mapping.csv')
        self.tf_mapping  = pd.read_csv(tf_mapping_path)
        self.num_tfs     = len(self.tf_mapping)

        self.train_dataset = BinaryTransBindDataset(
            train_dna_path, self.train_indices_path, train_uniqueness_path,
            dnase_h5_path=train_dnase_path, split='train',
            no_coverage_value=dnase_no_coverage_value,
        )
        self.val_dataset = BinaryTransBindDataset(
            val_dna_path, val_indices_path, val_uniqueness_path,
            dnase_h5_path=val_dnase_path, split='val',
            no_coverage_value=dnase_no_coverage_value,
        )

        # ── Build fixed val subset ONCE (same indices every epoch) ──────────
        if self.val_subsample_size and self.val_subsample_size < len(self.val_dataset):
            rng    = np.random.default_rng(42)           # fixed seed → reproducible
            chosen = rng.choice(
                len(self.val_dataset),
                size=self.val_subsample_size,
                replace=False,
            )
            self.val_subset = torch.utils.data.Subset(self.val_dataset, chosen)
            print(f"Val subset: {self.val_subsample_size:,} / {len(self.val_dataset):,} "
                  f"examples (seed=42, fixed — same every epoch)")
            # Print expected class balance in subset
            val_indices_data = np.load(val_indices_path)
            val_labels       = val_indices_data['labels']
            subset_labels    = val_labels[chosen]
            n_pos = int(np.sum(subset_labels))
            n_neg = len(subset_labels) - n_pos
            print(f"Val subset class balance: {n_pos:,} pos, {n_neg:,} neg "
                  f"({n_pos/len(subset_labels):.2%} positive)")
        else:
            self.val_subset = self.val_dataset
            print("Val subset: using full val dataset (subsample_size >= dataset size)")
        # ────────────────────────────────────────────────────────────────────

        train_indices      = np.load(self.train_indices_path)
        self.train_labels  = train_indices['labels']
        self.train_tf_idx  = train_indices['tf_idx']
        self.pos_count     = int(np.sum(self.train_labels))
        self.neg_count     = len(self.train_labels) - self.pos_count
        print(f"Training set: {self.pos_count} positive, {self.neg_count} negative")
        print("Datasets ready!")

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        sampler = BalancedBatchSampler(
            self.train_labels,
            self.batch_size,
            max_batches=self.max_batches,
            tf_idx=self.train_tf_idx,
            num_tfs=self.num_tfs,
        )
        return DataLoader(
            self.train_dataset,
            batch_sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        # Always uses the fixed subset — same 2M examples every epoch
        return DataLoader(
            self.val_subset,              # <-- fixed subset, not full val_dataset
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


def main():

    pl.seed_everything(42)

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

    CHECKPOINT_PATH = ""

    # DNase
    USE_DNASE = True
    DNASE_NO_COVERAGE_VALUE = -3.542

    # Uniqueness
    TRAIN_UNIQUENESS_PATH = os.path.join(
        HOME_DIR,
        "data/uniqueness_processed/train_uniqueness.npy"
    )

    VAL_UNIQUENESS_PATH = os.path.join(
        HOME_DIR,
        "data/uniqueness_processed/val_uniqueness.npy"
    )

    # ── Architecture (Trial 25 best config) ────────────────────────────────
    USE_CROSS_ATTENTION  = True
    BATCH_SIZE           = 3500
    VAL_BATCH_SIZE        = 5000
    MAX_BATCHES          = 5000
    VAL_SUBSAMPLE_SIZE   = 5_000_000
    D_MODEL              = 320
    NHEAD_CROSS          = 8
    NUM_CROSS_LAYERS     = 2      # was 3
    NUM_BIDIR_LAYERS     = 2      # unchanged
    PROTEIN_TARGET_LEN   = 160    # was 128

    # ── Optimized hyperparameters from Optuna sweep (Trial 25) ─────────────
    OPTIMIZED_PARAMS = {
        'learning_rate': 0.00032017,
        'dropout':       0.0574,
        'weight_decay':  0.03366,
    }

    print("=" * 80)
    print("BINARY PROTEIN-AWARE TRANSBIND  |  DNase + Uniqueness + Hybrid Cross-Attention")
    print("=" * 80)
    print(f"Data dir:      {DATA_DIR}")
    print(f"TF mapping:    {TF_MAPPING_PATH}")
    print(f"Features dir:  {FEATURES_DIR}")
    print(f"Use DNase:     {USE_DNASE}")
    print(f"\nCross-Attention Architecture:")
    print(f"  d_model:            {D_MODEL}")
    print(f"  num_heads:          {NHEAD_CROSS}")
    print(f"  num_layers:         {NUM_CROSS_LAYERS}")
    print(f"  num_bidir_layers:   {NUM_BIDIR_LAYERS}")
    print(f"  protein_target_len: {PROTEIN_TARGET_LEN}")
    print(f"\nTraining:")
    print(f"  batch_size:         {BATCH_SIZE}")
    print(f"  val_batch_size:     {VAL_BATCH_SIZE}")
    print(f"  max_batches/epoch:  {MAX_BATCHES}")
    print(f"  val_subsample_size: {VAL_SUBSAMPLE_SIZE:,} (fixed, seed=42)")
    print(f"\nHyperparameters (Optuna Trial 25):")
    for k, v in OPTIMIZED_PARAMS.items():
        print(f"  {k}: {v}")

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    devices     = [0]   if torch.cuda.is_available() else 1

    data_module = BalancedBinaryTransBindDataModule(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
        val_batch_size=VAL_BATCH_SIZE,
        num_workers=0,
        max_batches=MAX_BATCHES,
        val_subsample_size=VAL_SUBSAMPLE_SIZE,
        use_dnase=USE_DNASE,
        dnase_no_coverage_value=DNASE_NO_COVERAGE_VALUE,
        train_uniqueness_path=TRAIN_UNIQUENESS_PATH,
        val_uniqueness_path=VAL_UNIQUENESS_PATH,
    )
    data_module.setup()

    model = BinaryProteinAwareTransBind(
        mapping_path=TF_MAPPING_PATH,
        features_dir=FEATURES_DIR,
        num_tfs=data_module.num_tfs,
        use_cross_attention=USE_CROSS_ATTENTION,
        d_model=D_MODEL,
        nhead_cross=NHEAD_CROSS,
        num_cross_layers=NUM_CROSS_LAYERS,
        num_bidir_layers=NUM_BIDIR_LAYERS,
        protein_target_len=PROTEIN_TARGET_LEN,
        **OPTIMIZED_PARAMS,
    )

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(HOME_DIR, "model_binary_best_sweep_v2"),
        filename="binary-sweep-v2-{epoch:02d}-{val_aupr:.4f}",
        save_top_k=3,
        monitor="val_aupr",
        mode="max",
        save_last=True,
    )
    early_stopping = EarlyStopping(monitor="val_aupr", patience=15, mode="max")
    logger = TensorBoardLogger(
        os.path.join(HOME_DIR, "tb_logs"), name="binary_best_sweep_v2",
    )

    trainer = pl.Trainer(
        max_epochs=60,
        accelerator=accelerator,
        devices=devices,
        callbacks=[checkpoint_callback, early_stopping, TQDMProgressBar(refresh_rate=50)],
        logger=logger,
        log_every_n_steps=50,
        val_check_interval=1.0,
        enable_progress_bar=True,
        precision="16-mixed",
        plugins=[SafeCheckpointIO()],
    )

    print("\nStarting training...")
    resume = CHECKPOINT_PATH and os.path.exists(CHECKPOINT_PATH)
    if resume:
        print(f"Resuming from: {CHECKPOINT_PATH}")
        trainer.fit(model, data_module, ckpt_path=CHECKPOINT_PATH)
    else:
        trainer.fit(model, data_module)

    final_path = os.path.join(
        HOME_DIR, "model_binary_best_sweep_v2/binary_sweep_v2_final.ckpt"
    )
    trainer.save_checkpoint(final_path)
    print(f"\nTraining complete. Model saved to {final_path}")
    print(f"NOTE: Run full validation on best checkpoint for paper-reported metrics.")


if __name__ == "__main__":
    main()