import h5py
import numpy as np
import pandas as pd
import os
import time
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import shutil
import sys


# ============================================================
# CONFIG
# ============================================================

HOME_DIR = "/path/to/HOME_DIR"

CELL_MAPPING_PATH = os.path.join(
    HOME_DIR,
    "data/Binary_dataV1/cell_mapping.csv"
)

DNASE_DIR = os.path.join(
    HOME_DIR,
    "data/dnase_normalized"
)

FINAL_OUTPUT = os.path.join(
    HOME_DIR,
    "data/Binary_dataV1"
)

# Temporary local storage used while creating large HDF5 files
LOCAL_OUTPUT = "/tmp/dnase_conversion"

# ============================================================


def load_and_quantize_cell(args):
    """
    Load a single DNase file and quantize it.
    Memory-efficient version.
    """

    (
        cell_idx,
        cell_name,
        dnase_path,
        num_dna,
        num_positions,
        dtype,
        dnase_min,
        dnase_max,
        default_val
    ) = args

    try:
        # Load DNase data with memory mapping
        dnase_data = np.load(dnase_path, mmap_mode="r")

        # Verify shape
        if dnase_data.shape != (num_dna, num_positions):
            return (
                cell_idx,
                None,
                f"Shape mismatch: {dnase_data.shape}"
            )

        # Convert to target dtype in chunks to save memory
        if dtype == "uint8":

            dnase_quantized = np.empty(
                (num_dna, num_positions),
                dtype=np.uint8
            )

            chunk_size = 100000

            for i in range(0, num_dna, chunk_size):

                end = min(i + chunk_size, num_dna)

                chunk = dnase_data[i:end].copy()

                chunk_clipped = np.clip(
                    chunk,
                    dnase_min,
                    dnase_max
                )

                dnase_quantized[i:end] = (
                    (
                        (chunk_clipped - dnase_min)
                        / (dnase_max - dnase_min)
                        * 255
                    )
                    .astype(np.uint8)
                )

            return (
                cell_idx,
                dnase_quantized,
                None
            )

        elif dtype == "float16":

            result = np.empty(
                (num_dna, num_positions),
                dtype=np.float16
            )

            chunk_size = 100000

            for i in range(0, num_dna, chunk_size):

                end = min(
                    i + chunk_size,
                    num_dna
                )

                result[i:end] = (
                    dnase_data[i:end]
                    .astype(np.float16)
                )

            return (
                cell_idx,
                result,
                None
            )

        else:

            return (
                cell_idx,
                dnase_data[:].copy(),
                None
            )

    except Exception as e:

        return (
            cell_idx,
            None,
            str(e)
        )


def convert_dnase_to_hdf5_parallel(
    dnase_dir,
    output_file,
    split="train",
    cell_mapping_path=CELL_MAPPING_PATH,
    dtype="uint8",
    compression_level=4,
    num_workers=4
):
    """
    Convert DNase .npy files to HDF5 format (PARALLELIZED).
    Memory-efficient version with fillvalue.
    """

    print(f"\n{'=' * 60}")
    print(
        f"Converting {split} DNase data "
        f"to HDF5 (Parallel)"
    )
    print(f"{'=' * 60}")
    print(
        f"Using {num_workers} parallel workers"
    )

    # Load cell mapping
    cell_mapping = pd.read_csv(
        cell_mapping_path
    )

    cell_idx_to_name = dict(
        zip(
            cell_mapping["cell_idx"],
            cell_mapping["Cell_Type"]
        )
    )

    num_cells = len(cell_mapping)

    print(
        f"Total cell types in mapping: "
        f"{num_cells}"
    )

    # --------------------------------------------------------
    # Check which DNase files exist
    # --------------------------------------------------------

    print(
        f"\nScanning for available DNase files "
        f"in {dnase_dir}..."
    )

    available_files = {}

    for cell_idx in range(num_cells):

        if cell_idx in cell_idx_to_name:

            cell_name = (
                cell_idx_to_name[cell_idx]
            )

            dnase_path = os.path.join(
                dnase_dir,
                f"{split}_{cell_name}.npy"
            )

            if os.path.exists(dnase_path):

                available_files[
                    cell_idx
                ] = dnase_path

    available_count = len(
        available_files
    )

    missing_count = (
        num_cells - available_count
    )

    print(
        f"Found {available_count}/"
        f"{num_cells} DNase files"
    )

    print(
        f"Missing: {missing_count}/"
        f"{num_cells} files "
        f"(will use default value)"
    )

    if available_count == 0:

        print(
            "ERROR: No DNase files found!"
        )

        return

    # --------------------------------------------------------
    # Get dimensions
    # --------------------------------------------------------

    first_cell_idx = list(
        available_files.keys()
    )[0]

    first_file = (
        available_files[
            first_cell_idx
        ]
    )

    print(
        f"\nReading dimensions from: "
        f"{os.path.basename(first_file)}"
    )

    first_data = np.load(
        first_file,
        mmap_mode="r"
    )

    num_dna, num_positions = (
        first_data.shape
    )

    print(
        f"DNA sequences: "
        f"{num_dna:,}"
    )

    print(
        f"Positions per sequence: "
        f"{num_positions}"
    )

    # --------------------------------------------------------
    # Calculate sizes
    # --------------------------------------------------------

    bytes_per_value = (
        1 if dtype == "uint8"
        else (
            2 if dtype == "float16"
            else 4
        )
    )

    print(
        f"Using {dtype}"
    )

    uncompressed_size = (
        num_dna
        * num_cells
        * num_positions
        * bytes_per_value
        / (1024 ** 3)
    )

    print(
        f"Uncompressed size: "
        f"{uncompressed_size:.1f} GB"
    )

    print(
        f"Expected compressed size: "
        f"{uncompressed_size * 0.2:.1f} - "
        f"{uncompressed_size * 0.4:.1f} GB"
    )

    # --------------------------------------------------------
    # Quantization parameters
    # --------------------------------------------------------

    if dtype == "uint8":

        dnase_min = -10.0
        dnase_max = 5.0

        default_quantized = int(
            (
                (-3.542 - dnase_min)
                / (dnase_max - dnase_min)
                * 255
            )
        )

        default_val = (
            default_quantized
        )

    else:

        dnase_min = None
        dnase_max = None

        default_val = -3.542
        # -3.6954

    print(
        f"Default value for missing cells: "
        f"{default_val}"
    )

    # --------------------------------------------------------
    # Memory estimate
    # --------------------------------------------------------

    mem_per_worker = (
        num_dna
        * num_positions
        * bytes_per_value
        / (1024 ** 3)
    )

    total_mem = (
        mem_per_worker
        * num_workers
    )

    print(
        f"Estimated RAM usage: "
        f"{mem_per_worker:.1f} GB "
        f"per worker, "
        f"{total_mem:.1f} GB total"
    )

    # --------------------------------------------------------
    # Create HDF5
    # --------------------------------------------------------

    print(
        f"\nCreating HDF5 file: "
        f"{output_file}"
    )

    with h5py.File(
        output_file,
        "w",
        libver="latest"
    ) as f:

        print(
            f"Creating dataset with "
            f"fillvalue={default_val} "
            f"(auto-fills missing cells)"
        )

        dset = f.create_dataset(
            "dnase",
            shape=(
                num_dna,
                num_cells,
                num_positions
            ),
            dtype=dtype,
            chunks=(
                min(num_dna, 10000),
                1,
                num_positions
            ),
            compression="gzip",
            compression_opts=(
                compression_level
            ),
            fillvalue=default_val,
            track_times=False
        )

        # ----------------------------------------------------
        # Metadata
        # ----------------------------------------------------

        dset.attrs["split"] = split
        dset.attrs["num_dna"] = (
            num_dna
        )
        dset.attrs["num_cells"] = (
            num_cells
        )
        dset.attrs["num_positions"] = (
            num_positions
        )
        dset.attrs["dtype"] = str(
            dtype
        )
        dset.attrs["creation_date"] = (
            time.strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        )
        dset.attrs["available_cells"] = (
            available_count
        )
        dset.attrs["total_cells"] = (
            num_cells
        )

        if dtype == "uint8":

            f.attrs["dnase_min"] = (
                dnase_min
            )

            f.attrs["dnase_max"] = (
                dnase_max
            )

        f.attrs["default_value"] = (
            default_val
        )

        # ----------------------------------------------------
        # Store cell names
        # ----------------------------------------------------

        cell_names_list = [
            cell_idx_to_name.get(
                i,
                "unknown"
            )
            for i in range(
                num_cells
            )
        ]

        f.create_dataset(
            "cell_names",
            data=np.array(
                cell_names_list,
                dtype="S50"
            )
        )

        # ----------------------------------------------------
        # Store which cell types have DNase
        # ----------------------------------------------------

        cell_available = np.zeros(
            num_cells,
            dtype=bool
        )

        for cell_idx in (
            available_files.keys()
        ):

            cell_available[
                cell_idx
            ] = True

        f.create_dataset(
            "cell_available",
            data=cell_available
        )

        # ----------------------------------------------------
        # Prepare parallel jobs
        # ----------------------------------------------------

        print(
            f"\nLoading {available_count} "
            f"DNase files in parallel "
            f"({num_workers} workers)..."
        )

        load_args = []

        for (
            cell_idx,
            dnase_path
        ) in available_files.items():

            cell_name = (
                cell_idx_to_name[
                    cell_idx
                ]
            )

            args = (
                cell_idx,
                cell_name,
                dnase_path,
                num_dna,
                num_positions,
                dtype,
                dnase_min,
                dnase_max,
                default_val
            )

            load_args.append(
                args
            )

        # ----------------------------------------------------
        # Load files in parallel
        # ----------------------------------------------------

        loaded_count = 0
        error_count = 0

        with ProcessPoolExecutor(
            max_workers=num_workers
        ) as executor:

            futures = {
                executor.submit(
                    load_and_quantize_cell,
                    args
                ): args[1]
                for args in load_args
            }

            with tqdm(
                total=len(load_args),
                desc="Loading cells"
            ) as pbar:

                for future in (
                    as_completed(
                        futures
                    )
                ):

                    cell_name = (
                        futures[future]
                    )

                    try:

                        (
                            cell_idx,
                            data,
                            error
                        ) = future.result()

                        if (
                            error is None
                            and data is not None
                        ):

                            # Write using cell_idx
                            dset[
                                :,
                                cell_idx,
                                :
                            ] = data

                            loaded_count += 1

                            del data

                        else:

                            if error:

                                tqdm.write(
                                    f"  ✗ "
                                    f"{cell_name}: "
                                    f"{error}"
                                )

                            error_count += 1

                    except Exception as e:

                        tqdm.write(
                            f"  ✗ "
                            f"{cell_name}: "
                            f"Exception - {e}"
                        )

                        error_count += 1

                    pbar.update(1)

                    if (
                        loaded_count
                        + error_count
                    ) % 10 == 0:

                        f.flush()

        # Final flush
        print(
            "\nFlushing data to disk..."
        )

        f.flush()

        print(
            f"Missing cells "
            f"({missing_count}) already "
            f"filled with default value "
            f"via fillvalue"
        )

    # --------------------------------------------------------
    # Verify file
    # --------------------------------------------------------

    print(
        "\nVerifying output file..."
    )

    file_size = (
        os.path.getsize(
            output_file
        )
        / (1024 ** 3)
    )

    compression_ratio = (
        uncompressed_size
        / file_size
        if file_size > 0
        else 0
    )

    print(
        f"\n{'=' * 60}"
    )

    print(
        "Conversion complete!"
    )

    print(
        f"{'=' * 60}"
    )

    print(
        f"Successfully loaded: "
        f"{loaded_count}/"
        f"{available_count} "
        f"available cell types"
    )

    print(
        f"Errors: "
        f"{error_count} cell types"
    )

    print(
        f"Missing (auto-filled): "
        f"{missing_count} cell types"
    )

    print(
        f"Output file: "
        f"{output_file}"
    )

    print(
        f"File size: "
        f"{file_size:.2f} GB"
    )

    print(
        f"Compression ratio: "
        f"{compression_ratio:.1f}x"
    )

    print(
        f"{'=' * 60}\n"
    )

    return output_file


def inspect_hdf5_file(
    h5_file_path
):
    """
    Inspect what's in an HDF5 file.
    """

    print(
        f"\n{'=' * 60}"
    )

    print(
        f"Inspecting: "
        f"{h5_file_path}"
    )

    print(
        f"{'=' * 60}"
    )

    with h5py.File(
        h5_file_path,
        "r"
    ) as f:

        print(
            "\nDatasets:"
        )

        for key in f.keys():

            dset = f[key]

            if isinstance(
                dset,
                h5py.Dataset
            ):

                print(
                    f"  {key}: "
                    f"shape={dset.shape}, "
                    f"dtype={dset.dtype}"
                )

        print(
            "\nMetadata "
            "(attributes):"
        )

        for (
            key,
            value
        ) in f.attrs.items():

            print(
                f"  {key}: "
                f"{value}"
            )

        # Show available cells
        if "cell_available" in f:

            cell_available = (
                f["cell_available"][:]
            )

            cell_names = (
                f["cell_names"][:]
                .astype(str)
            )

            available_indices = (
                np.where(
                    cell_available
                )[0]
            )

            missing_indices = (
                np.where(
                    ~cell_available
                )[0]
            )

            print(
                f"\nAvailable cells "
                f"({len(available_indices)}):"
            )

            for idx in (
                available_indices[:10]
            ):

                print(
                    f"  {idx}: "
                    f"{cell_names[idx]}"
                )

            if (
                len(
                    available_indices
                ) > 10
            ):

                print(
                    f"  ... and "
                    f"{len(available_indices) - 10} "
                    f"more"
                )

            print(
                f"\nMissing cells "
                f"({len(missing_indices)}):"
            )

            for idx in (
                missing_indices[:10]
            ):

                print(
                    f"  {idx}: "
                    f"{cell_names[idx]}"
                )

            if (
                len(
                    missing_indices
                ) > 10
            ):

                print(
                    f"  ... and "
                    f"{len(missing_indices) - 10} "
                    f"more"
                )

    print(
        f"{'=' * 60}\n"
    )


if __name__ == "__main__":

    # --------------------------------------------------------
    # Output configuration
    # --------------------------------------------------------

    USE_LOCAL_TEMP = True

    if USE_LOCAL_TEMP:

        os.makedirs(
            LOCAL_OUTPUT,
            exist_ok=True
        )

        os.makedirs(
            FINAL_OUTPUT,
            exist_ok=True
        )

        print(
            "=" * 60
        )

        print(
            "CONFIGURATION"
        )

        print(
            "=" * 60
        )

        print(
            f"Writing to LOCAL disk: "
            f"{LOCAL_OUTPUT}"
        )

        print(
            f"Will copy to final storage: "
            f"{FINAL_OUTPUT}"
        )

        # Check local disk space
        stat = shutil.disk_usage(
            LOCAL_OUTPUT
        )

        free_gb = (
            stat.free
            / (1024 ** 3)
        )

        print(
            f"Local disk available: "
            f"{free_gb:.1f} GB"
        )

        if free_gb < 100:

            print(
                f"ERROR: Need at least "
                f"100 GB free on local disk, "
                f"have {free_gb:.1f} GB"
            )

            sys.exit(1)

        OUTPUT_DIR = (
            LOCAL_OUTPUT
        )

    else:

        OUTPUT_DIR = (
            FINAL_OUTPUT
        )

        print(
            f"Writing directly to: "
            f"{OUTPUT_DIR}"
        )

    # --------------------------------------------------------
    # Workers
    # --------------------------------------------------------

    num_workers = 4

    print(
        f"System has "
        f"{mp.cpu_count()} CPU cores"
    )

    print(
        f"Using "
        f"{num_workers} workers"
    )

    print(
        "=" * 60
    )

    # ========================================================
    # TRAIN
    # ========================================================

    print(
        "\n" + "=" * 60
    )

    print(
        "CONVERTING TRAIN DATA"
    )

    print(
        "=" * 60
    )

    train_file_temp = os.path.join(
        OUTPUT_DIR,
        "train_dnase.h5"
    )

    train_file_final = os.path.join(
        FINAL_OUTPUT,
        "train_dnase.h5"
    )

    train_file = (
        convert_dnase_to_hdf5_parallel(
            dnase_dir=DNASE_DIR,
            output_file=train_file_temp,
            split="train",
            cell_mapping_path=(
                CELL_MAPPING_PATH
            ),
            dtype="uint8",
            compression_level=4,
            num_workers=num_workers
        )
    )

    if train_file:

        inspect_hdf5_file(
            train_file
        )

        if (
            USE_LOCAL_TEMP
            and train_file_temp
            != train_file_final
        ):

            print(
                f"\n{'=' * 60}"
            )

            print(
                "COPYING TO FINAL STORAGE"
            )

            print(
                f"{'=' * 60}"
            )

            print(
                f"Source: "
                f"{train_file_temp}"
            )

            print(
                f"Destination: "
                f"{train_file_final}"
            )

            try:

                shutil.copy2(
                    train_file_temp,
                    train_file_final
                )

                local_size = (
                    os.path.getsize(
                        train_file_temp
                    )
                    / (1024 ** 3)
                )

                final_size = (
                    os.path.getsize(
                        train_file_final
                    )
                    / (1024 ** 3)
                )

                print(
                    "✓ Copy complete"
                )

                print(
                    f"  Local: "
                    f"{local_size:.2f} GB"
                )

                print(
                    f"  Final: "
                    f"{final_size:.2f} GB"
                )

                if abs(
                    local_size
                    - final_size
                ) < 0.1:

                    print(
                        "  ✓ File sizes match"
                    )

                    os.remove(
                        train_file_temp
                    )

                    print(
                        "  Removed local temp file"
                    )

                else:

                    print(
                        "  ⚠ File sizes don't match "
                        "- keeping local copy"
                    )

            except Exception as e:

                print(
                    f"✗ Copy failed: "
                    f"{e}"
                )

                print(
                    f"Local file kept at: "
                    f"{train_file_temp}"
                )

    # ========================================================
    # VALIDATION
    # ========================================================

    print(
        "\n" + "=" * 60
    )

    print(
        "CONVERTING VAL DATA"
    )

    print(
        "=" * 60
    )

    val_file_temp = os.path.join(
        OUTPUT_DIR,
        "val_dnase.h5"
    )

    val_file_final = os.path.join(
        FINAL_OUTPUT,
        "val_dnase.h5"
    )

    val_file = (
        convert_dnase_to_hdf5_parallel(
            dnase_dir=DNASE_DIR,
            output_file=val_file_temp,
            split="val",
            cell_mapping_path=(
                CELL_MAPPING_PATH
            ),
            dtype="uint8",
            compression_level=4,
            num_workers=num_workers
        )
    )

    if val_file:

        inspect_hdf5_file(
            val_file
        )

        if (
            USE_LOCAL_TEMP
            and val_file_temp
            != val_file_final
        ):

            print(
                f"\n{'=' * 60}"
            )

            print(
                "COPYING TO FINAL STORAGE"
            )

            print(
                f"{'=' * 60}"
            )

            print(
                f"Source: "
                f"{val_file_temp}"
            )

            print(
                f"Destination: "
                f"{val_file_final}"
            )

            try:

                shutil.copy2(
                    val_file_temp,
                    val_file_final
                )

                local_size = (
                    os.path.getsize(
                        val_file_temp
                    )
                    / (1024 ** 3)
                )

                final_size = (
                    os.path.getsize(
                        val_file_final
                    )
                    / (1024 ** 3)
                )

                print(
                    "✓ Copy complete"
                )

                print(
                    f"  Local: "
                    f"{local_size:.2f} GB"
                )

                print(
                    f"  Final: "
                    f"{final_size:.2f} GB"
                )

                if abs(
                    local_size
                    - final_size
                ) < 0.1:

                    print(
                        "  ✓ File sizes match"
                    )

                    os.remove(
                        val_file_temp
                    )

                    print(
                        "  Removed local temp file"
                    )

                else:

                    print(
                        "  ⚠ File sizes don't match "
                        "- keeping local copy"
                    )

            except Exception as e:

                print(
                    f"✗ Copy failed: "
                    f"{e}"
                )

                print(
                    f"Local file kept at: "
                    f"{val_file_temp}"
                )

    # ========================================================
    # COMPLETE
    # ========================================================

    print(
        "\n" + "=" * 60
    )

    print(
        "ALL CONVERSIONS COMPLETE!"
    )

    print(
        "=" * 60
    )

    print(
        f"Train: "
        f"{train_file_final}"
    )

    print(
        f"Val: "
        f"{val_file_final}"
    )

    print(
        "=" * 60
    )