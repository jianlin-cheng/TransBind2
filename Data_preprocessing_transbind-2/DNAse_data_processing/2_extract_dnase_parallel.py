"""
extract_dnase_parallel.py
Parallel DNase signal extraction — crash-safe version.

Usage:
    python extract_dnase_parallel.py
"""

import os
import json
import numpy as np
import pandas as pd
import pyBigWig
from multiprocessing import Pool, cpu_count
import time
import traceback

# ============================================================
# CONFIG
# ============================================================
HOME_DIR = "/path/to/HOME_DIR"

data_dir = os.path.join(
    HOME_DIR,
    "data/Binary_dataV1"
)

bw_dir = os.path.join(
    HOME_DIR,
    "data/downloads/wgEncodeAwgDnaseUniform"
)

out_dir = os.path.join(
    HOME_DIR,
    "data/dnase_processed"
)

# Use fewer workers — bigWig random access is I/O bound, 
# more workers just thrash the disk
NUM_WORKERS = 8

cell_idx_to_info = {
    1:  ("GM12878",    "wgEncodeAwgDnaseUwdukeGm12878UniPk.bw"),
    2:  ("H1-hESC",    "wgEncodeAwgDnaseUwdukeH1hescUniPk.bw"),
    3:  ("HeLa-S3",    "wgEncodeAwgDnaseDukeHelas3ifna4hUniPk.bw"),
    4:  ("HepG2",      "wgEncodeAwgDnaseUwdukeHepg2UniPk.bw"),
    5:  ("HMEC",       "wgEncodeAwgDnaseUwdukeHmecUniPk.bw"),
    6:  ("HSMM",       "wgEncodeAwgDnaseUwdukeHsmmUniPk.bw"),
    7:  ("HSMMtube",   "wgEncodeAwgDnaseUwdukeHsmmtubeUniPk.bw"),
    8:  ("HUVEC",      "wgEncodeAwgDnaseUwdukeHuvecUniPk.bw"),
    9:  ("K562",       "wgEncodeAwgDnaseUwdukeK562UniPk.bw"),
    10: ("NH-A",       "wgEncodeAwgDnaseUwNhaUniPk.bw"),
    11: ("NHDF-Ad",    "wgEncodeAwgDnaseUwNhdfadUniPk.bw"),
    12: ("NHEK",       "wgEncodeAwgDnaseUwdukeNhekUniPk.bw"),
    13: ("NHLF",       "wgEncodeAwgDnaseUwNhlfUniPk.bw"),
    14: ("Osteobl",    "wgEncodeAwgDnaseDukeOsteoblUniPk.bw"),
    15: ("A549",       "wgEncodeAwgDnaseUwdukeA549UniPk.bw"),
    17: ("GM12891",    "wgEncodeAwgDnaseDukeGm12891UniPk.bw"),
    18: ("GM12892",    "wgEncodeAwgDnaseDukeGm12892UniPk.bw"),
    19: ("HCT-116",    "wgEncodeAwgDnaseUwHct116UniPk.bw"),
    20: ("PANC-1",     "wgEncodeAwgDnaseUwPanc1UniPk.bw"),
    22: ("SK-N-MC",    "wgEncodeAwgDnaseUwSknmcUniPk.bw"),
    24: ("SK-N-SH_RA","wgEncodeAwgDnaseUwSknshraUniPk.bw"),
    25: ("T-47D",      "wgEncodeAwgDnaseDukeT47dUniPk.bw"),
    39: ("MCF-7",      "wgEncodeAwgDnaseDukeMcf7hypoxiaUniPk.bw"),
    40: ("NB4",        "wgEncodeAwgDnaseUwNb4UniPk.bw"),
    41: ("NT2-D1",     "wgEncodeAwgDnaseUwNt2d1UniPk.bw"),
    47: ("Fibrobl",    "wgEncodeAwgDnaseDukeFibroblUniPk.bw"),
    48: ("Gliobla",    "wgEncodeAwgDnaseDukeGlioblaUniPk.bw"),
    49: ("GM19238",    "wgEncodeAwgDnaseDukeGm19238UniPk.bw"),
    50: ("GM19239",    "wgEncodeAwgDnaseDukeGm19239UniPk.bw"),
    51: ("GM19240",    "wgEncodeAwgDnaseDukeGm19240UniPk.bw"),
    52: ("ProgFib",    "wgEncodeAwgDnaseDukeProgfibUniPk.bw"),
    53: ("AG04449",    "wgEncodeAwgDnaseUwAg04449UniPk.bw"),
    54: ("AG04450",    "wgEncodeAwgDnaseUwAg04450UniPk.bw"),
    55: ("AG09309",    "wgEncodeAwgDnaseUwAg09309UniPk.bw"),
    56: ("AG09319",    "wgEncodeAwgDnaseUwAg09319UniPk.bw"),
    57: ("AG10803",    "wgEncodeAwgDnaseUwAg10803UniPk.bw"),
    58: ("AoAF",       "wgEncodeAwgDnaseUwAoafUniPk.bw"),
    59: ("BE2_C",      "wgEncodeAwgDnaseUwBe2cUniPk.bw"),
    60: ("BJ",         "wgEncodeAwgDnaseUwBjUniPk.bw"),
    61: ("Caco-2",     "wgEncodeAwgDnaseUwCaco2UniPk.bw"),
    62: ("GM06990",    "wgEncodeAwgDnaseUwGm06990UniPk.bw"),
    64: ("GM12864",    "wgEncodeAwgDnaseUwGm12864UniPk.bw"),
    65: ("GM12865",    "wgEncodeAwgDnaseUwGm12865UniPk.bw"),
    70: ("HAc",        "wgEncodeAwgDnaseUwHacUniPk.bw"),
    71: ("HA-sp",      "wgEncodeAwgDnaseUwHaspUniPk.bw"),
    72: ("HBMEC",      "wgEncodeAwgDnaseUwHbmecUniPk.bw"),
    73: ("HCFaa",      "wgEncodeAwgDnaseUwHcfaaUniPk.bw"),
    74: ("HCM",        "wgEncodeAwgDnaseUwHcmUniPk.bw"),
    75: ("HCPEpiC",    "wgEncodeAwgDnaseUwHcpepicUniPk.bw"),
    76: ("HEEpiC",     "wgEncodeAwgDnaseUwHeepicUniPk.bw"),
    77: ("HFF",        "wgEncodeAwgDnaseUwHffUniPk.bw"),
    78: ("HFF-Myc",    "wgEncodeAwgDnaseUwHffmycUniPk.bw"),
    79: ("HL-60",      "wgEncodeAwgDnaseUwHl60UniPk.bw"),
    80: ("HMF",        "wgEncodeAwgDnaseUwHmfUniPk.bw"),
    81: ("HPAF",       "wgEncodeAwgDnaseUwHpafUniPk.bw"),
    82: ("HPF",        "wgEncodeAwgDnaseUwHpfUniPk.bw"),
    83: ("HRE",        "wgEncodeAwgDnaseUwHreUniPk.bw"),
    84: ("HRPEpiC",    "wgEncodeAwgDnaseUwHrpepicUniPk.bw"),
    85: ("HVMF",       "wgEncodeAwgDnaseUwHvmfUniPk.bw"),
    86: ("NHDF-neo",   "wgEncodeAwgDnaseUwNhdfneoUniPk.bw"),
    87: ("RPTEC",      "wgEncodeAwgDnaseUwRptecUniPk.bw"),
    88: ("SAEC",       "wgEncodeAwgDnaseUwSaecUniPk.bw"),
    89: ("WERI-Rb-1",  "wgEncodeAwgDnaseUwWerirb1UniPk.bw"),
    90: ("WI-38",      "wgEncodeAwgDnaseUwWi38UniPk.bw"),
}
# ============================================================


def extract_dnase_for_split(bw_path, coords_df):
    """
    Extract 1000x1 DNase signal for every row in coords_df.
    Returns (n_seqs, 1000) float32 array.
    """
    bw = pyBigWig.open(bw_path)
    n_seqs  = len(coords_df)
    signals = np.zeros((n_seqs, 1000), dtype=np.float32)

    # Pre-convert to numpy for speed — iloc is slow in a tight loop
    chroms  = coords_df["chrom"].values
    starts  = coords_df["window_start"].values.astype(int)
    ends    = coords_df["window_end"].values.astype(int)

    for i in range(n_seqs):
        chrom = chroms[i]
        start = starts[i]
        end   = ends[i]

        try:
            if start < 0:
                pad    = -start
                signal = bw.values(chrom, 0, end)
                signal = np.nan_to_num(signal, nan=0.0)
                signals[i, pad:] = signal
            else:
                signal = bw.values(chrom, start, end)
                signals[i] = np.nan_to_num(signal, nan=0.0)
        except Exception:
            pass  # stays zero

        if (i + 1) % 500_000 == 0:
            print(f"    [{os.path.basename(bw_path)}] {i+1:,}/{n_seqs:,}", flush=True)

    bw.close()
    return signals


def _worker(args):
    """
    Per-job worker. Writes to a .tmp file first, then renames
    to .npy atomically — so a crash mid-write won't leave a
    corrupt file that gets skipped next time.
    """
    split_name, cell_name, bw_path, coords_csv_path, out_path = args
    tag = f"{split_name}_{cell_name}"
    # np.save() silently appends .npy if the path doesn't already end in .npy.
    # "foo.npy" + ".tmp" = "foo.npy.tmp" → np.save writes "foo.npy.tmp.npy"
    # → os.rename("foo.npy.tmp", ...) fails because that file doesn't exist.
    # Fix: put .tmp BEFORE .npy so np.save sees the .npy and doesn't append.
    tmp_path = out_path.replace(".npy", ".tmp.npy")

    # --- skip if already successfully completed ---
    if os.path.exists(out_path):
        return {"tag": tag, "status": "skipped"}

    try:
        coords_df = pd.read_csv(coords_csv_path)
        n_seqs    = len(coords_df)

        print(f"  [START] {tag} — {n_seqs:,} seqs | pid {os.getpid()}", flush=True)
        t0 = time.time()

        signals = extract_dnase_for_split(bw_path, coords_df)

        # Write to .tmp first, then rename atomically
        np.save(tmp_path, signals)
        os.rename(tmp_path, out_path)

        nonzero = int(np.sum(signals.sum(axis=1) > 0))
        elapsed = time.time() - t0

        print(
            f"  [DONE]  {tag} — shape {signals.shape}, "
            f"non-zero {nonzero:,}/{n_seqs:,}, "
            f"time {elapsed:.1f}s",
            flush=True,
        )
        return {"tag": tag, "status": "done", "nonzero": nonzero}

    except Exception as e:
        # Clean up tmp file if it exists
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        print(f"  [FAIL]  {tag} — {e}", flush=True)
        traceback.print_exc()
        return {"tag": tag, "status": "failed", "error": str(e)}


def main():
    os.makedirs(out_dir, exist_ok=True)

    # --- validate coords CSVs exist ---
    split_csv = {}
    for split in ("train", "val", "test"):
        path = os.path.join(data_dir, f"{split}_coords.csv")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing coords file: {path}")
        split_csv[split] = path

    # --- validate all bigWig files exist upfront ---
    print("Checking bigWig files...")
    missing_bw = []
    for cidx, (cell_name, bw_name) in sorted(cell_idx_to_info.items()):
        bw_path = os.path.join(bw_dir, bw_name)
        if not os.path.isfile(bw_path):
            missing_bw.append(bw_name)
    if missing_bw:
        print(f"  ✗ {len(missing_bw)} missing bigWig files:")
        for name in missing_bw:
            print(f"    {name}")
        raise FileNotFoundError("Fix missing bigWig files before running.")
    print(f"  ✓ All {len(cell_idx_to_info)} bigWig files found\n")

    # --- clean up any leftover .tmp files from previous crashes ---
    stale_tmp = [f for f in os.listdir(out_dir) if '.tmp.' in f]
    if stale_tmp:
        print(f"Cleaning up {len(stale_tmp)} stale .tmp files from previous run...")
        for f in stale_tmp:
            os.remove(os.path.join(out_dir, f))

    # --- build job list ---
    jobs = []
    for cidx, (cell_name, bw_name) in sorted(cell_idx_to_info.items()):
        bw_path = os.path.join(bw_dir, bw_name)
        for split_name in ("train", "val", "test"):
            out_path = os.path.join(out_dir, f"{split_name}_{cell_name}.npy")
            jobs.append((split_name, cell_name, bw_path, split_csv[split_name], out_path))

    # Count already done
    already_done = sum(1 for j in jobs if os.path.exists(j[4]))
    remaining    = len(jobs) - already_done

    print(f"{'='*60}")
    print(f"  Total jobs: {len(jobs)} | Done: {already_done} | Remaining: {remaining}")
    print(f"  Workers: {NUM_WORKERS}")
    print(f"{'='*60}\n")

    if remaining == 0:
        print("All jobs already completed!")
        return

    t_start = time.time()

    with Pool(processes=NUM_WORKERS) as pool:
        results = pool.map(_worker, jobs)

    # --- summary ---
    done    = [r for r in results if r["status"] == "done"]
    skipped = [r for r in results if r["status"] == "skipped"]
    failed  = [r for r in results if r["status"] == "failed"]

    print(f"\n{'='*60}")
    print(f"  Finished in {time.time() - t_start:.1f}s")
    print(f"  Done: {len(done)} | Skipped: {len(skipped)} | Failed: {len(failed)}")
    if failed:
        print(f"\n  Failed jobs (re-run script to retry):")
        for r in failed:
            print(f"    {r['tag']}: {r['error']}")
    print(f"{'='*60}\n")

    # --- save cell_idx -> cell_name mapping ---
    mapping = {cidx: cell_name for cidx, (cell_name, _) in cell_idx_to_info.items()}
    mapping_path = os.path.join(out_dir, "cell_idx_to_name.json")
    with open(mapping_path, "w") as f:
        json.dump(mapping, f, indent=2)
    print(f"Saved cell_idx -> name mapping to {mapping_path}")


if __name__ == "__main__":
    main()