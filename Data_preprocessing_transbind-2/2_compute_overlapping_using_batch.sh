#!/bin/bash

# ==============================================
# FINAL OPTIMIZED GENOMIC PROCESSING SCRIPT
# Configured for System 1 (515GB RAM, high-load)
# ==============================================

# --------------------------
# CONFIGURATION
# --------------------------
TOTAL_PEAK_FILES=690
FILES_PER_BATCH=8          # Optimal for I/O-bound system
MEMORY_PER_JOB=3          # GB (conservative estimate)
MAX_PARALLEL_JOBS=25      # Capped below theoretical max
STABILIZATION_DELAY=20    # Longer pauses for I/O-heavy system
IONICE_PRIORITY="ionice -c2 -n7"  # Best-effort low-priority I/O

# Create a unique completion tracker file
COMPLETION_TRACKER="/tmp/batch_completion_$(date +%s).tracker"
touch "$COMPLETION_TRACKER"

# --------------------------
# PROCESSING SCRIPT
# --------------------------
cat > process_batch.sh << 'EOF'
#!/bin/bash

# Memory constraints
ulimit -v $((3 * 1024 * 1024))  # 3GB hard limit

# Parameters
BATCH_ID=$1
FILES_PER_BATCH=$2
TOTAL_FILES=$3

# Calculate file range
start_idx=$(( FILES_PER_BATCH * (BATCH_ID - 1) + 1 ))
end_idx=$(( FILES_PER_BATCH * BATCH_ID ))
(( end_idx > TOTAL_FILES )) && end_idx=$TOTAL_FILES

echo "[$(date +'%T')] Processing batch $BATCH_ID (files $start_idx-$end_idx)"

# Path configuration
HOME_DIR="/path/to/HOME_DIR"
data_root="$HOME_DIR/data/processed/"
sorted_peak_files_dir="${data_root}wgEncodeAwgTfbsUniform_sorted/"
windowed_sorted_human_genome_filepath="${data_root}hg19.genome.windowed.sorted.gz"
out_peaks_dir="${data_root}peaks/"

mkdir -p "$out_peaks_dir"

# Process files with error handling
processed=0
for (( counter=start_idx; counter<=end_idx; counter++ )); do
    filepath="${sorted_peak_files_dir}$(ls ${sorted_peak_files_dir} | sed -n ${counter}p)"
    output_file="${out_peaks_dir}${counter}.sorted.gz"
    
    # Skip if output already exists
    if [[ -f "$output_file" ]]; then
        echo "Output exists for file $counter - skipping"
        ((processed++))
        continue
    fi
    
    if [[ -f "$filepath" ]]; then
        echo "Processing file $counter: $(basename "$filepath")"
        
        # Run bedtools intersect and then use awk to insert the filename
        if $IONICE_PRIORITY bedtools intersect \
            -a "$windowed_sorted_human_genome_filepath" \
            -b "$filepath" \
            -wo \
            -f .5 \
            -sorted \
            > "${out_peaks_dir}${counter}.temp"; then
            
            # Extract just the filename without extension
            filename=$(basename "$filepath" .narrowPeak.sorted.gz)
            
            # Post-process to insert the filename as an additional column
            awk -v filename="$filename" '{print $1"\t"$2"\t"$3"\t"filename"\t"$4"\t"$5"\t"$6"\t"$7"\t"$8"\t"$9"\t"$10"\t"$11"\t"$12"\t"$13}' \
                "${out_peaks_dir}${counter}.temp" > "${out_peaks_dir}${counter}.sorted"
            
            # Clean up and compress
            rm "${out_peaks_dir}${counter}.temp"
            gzip "${out_peaks_dir}${counter}.sorted" -f
            ((processed++))
        else
            echo "[ERROR] Failed processing file $counter"
        fi
    fi
done

echo "Completed $processed/$FILES_PER_BATCH files in batch $BATCH_ID"
(( processed == FILES_PER_BATCH ))  # Return success only if all files processed
EOF
chmod +x process_batch.sh

# --------------------------
# RESOURCE-AWARE CONTROLLER
# --------------------------
calculate_safe_jobs() {
    # 1. CPU availability (adjusted for high-load systems)
    local load_capacity=$(echo "scale=2; $(nproc) / ($(cut -d' ' -f1 /proc/loadavg) + 1)" | bc | cut -d. -f1)
    local cpu_based=$(( load_capacity * 70 / 100 ))  # Use only 70% of estimated capacity
    
    # 2. Memory availability
    local available_mem=$(free -g | awk '/Mem:/ {print $7}')
    local mem_based=$(( available_mem / MEMORY_PER_JOB ))
    
    # Use the more restrictive limit
    local safe_jobs=$(( cpu_based < mem_based ? cpu_based : mem_based ))
    
    # Apply absolute limits
    (( safe_jobs = safe_jobs < 1 ? 1 : safe_jobs ))
    (( safe_jobs = safe_jobs > MAX_PARALLEL_JOBS ? MAX_PARALLEL_JOBS : safe_jobs ))
    
    echo $safe_jobs
}

# --------------------------
# EXECUTION WITH MONITORING
# --------------------------
N_BATCHES=$(( (TOTAL_PEAK_FILES + FILES_PER_BATCH - 1) / FILES_PER_BATCH ))
retry_queue=()
declare -A retry_counts  # Track how many times each batch has been retried
next_batch=1

echo "===== JOB STARTED $(date) ====="
echo "Configuration:"
echo "- Files: $TOTAL_PEAK_FILES | Batches: $N_BATCHES"
echo "- Files/Batch: $FILES_PER_BATCH | Max Parallel: $MAX_PARALLEL_JOBS"
echo "- Memory/Job: ${MEMORY_PER_JOB}GB | I/O Priority: $IONICE_PRIORITY"

while (( $(wc -l < "$COMPLETION_TRACKER") < N_BATCHES )) || (( ${#retry_queue[@]} > 0 )); do
    current_jobs=$(calculate_safe_jobs)
    completed=$(wc -l < "$COMPLETION_TRACKER")
    
    # System status
    echo "--------------------------------------------------"
    echo "[$(date +'%Y-%m-%d %T')] SYSTEM STATUS"
    echo "CPU: $(nproc) cores | Load: $(cat /proc/loadavg)"
    echo "Memory: $(free -h | awk '/Mem:/ {printf "Free: %s / Avail: %s", $4, $7}')"
    echo "Progress: $completed/$N_BATCHES | Retry Queue: ${#retry_queue[@]}"
    echo "Allocating $current_jobs parallel jobs"
    echo "--------------------------------------------------"
    
    # Process retries first
    temp_retry_queue=()
    for batch in "${retry_queue[@]}"; do
        if (( ${retry_counts[$batch]:-0} < 3 )); then
            temp_retry_queue+=("$batch")
        else
            echo "[WARNING] Batch $batch failed too many times - skipping"
        fi
    done
    retry_queue=("${temp_retry_queue[@]}")
    
    if (( ${#retry_queue[@]} > 0 )); then
        batches_to_run=("${retry_queue[@]:0:$current_jobs}")
        retry_queue=("${retry_queue[@]:$current_jobs}")
    else
        # Start new batches
        batches_to_run=()
        while (( ${#batches_to_run[@]} < current_jobs )) && (( next_batch <= N_BATCHES )); do
            if ! grep -q "^$next_batch$" "$COMPLETION_TRACKER"; then
                batches_to_run+=("$next_batch")
            fi
            ((next_batch++))
        done
    fi
    
    # Execute batches with proper locking
    for batch in "${batches_to_run[@]}"; do
        (
            flock -x 200
            
            if ./process_batch.sh "$batch" "$FILES_PER_BATCH" "$TOTAL_PEAK_FILES"; then
                echo "$batch" >> "$COMPLETION_TRACKER"
                echo "[SUCCESS] Completed batch $batch"
            else
                retry_queue+=("$batch")
                retry_counts[$batch]=$(( ${retry_counts[$batch]:-0} + 1 ))
                echo "[WARNING] Batch $batch failed (attempt ${retry_counts[$batch]})"
            fi
        ) 200>"$COMPLETION_TRACKER.lock" &
    done
    
    wait
    sleep $STABILIZATION_DELAY
done

rm -f "$COMPLETION_TRACKER" "$COMPLETION_TRACKER.lock"
echo "===== JOB COMPLETED SUCCESSFULLY $(date) ====="