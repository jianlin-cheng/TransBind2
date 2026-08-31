#!/bin/bash

HOME_DIR="/path/to/HOME_DIR"

DNASE_DIR="$HOME_DIR/data/downloads/wgEncodeAwgDnaseUniform"
CHROM_SIZES="$HOME_DIR/data/downloads/hg19_latest/hg19.chrom.sizes"

cd "$DNASE_DIR" || exit 1

for narrowpeak in *.narrowPeak.gz; do

    basename="${narrowpeak%.narrowPeak.gz}"

    echo "Converting $narrowpeak..."

    # Extract chromosome, start, end, and signalValue (column 7)
    zcat "$narrowpeak" |
        awk 'BEGIN{OFS="\t"}{print $1, $2, $3, $7}' |
        sort -k1,1 -k2,2n \
        > "${basename}.bedGraph"

    # Merge overlapping regions and retain the maximum signal
    bedtools merge \
        -i "${basename}.bedGraph" \
        -c 4 \
        -o max \
        > "${basename}.merged.bedGraph"

    # Clip intervals to hg19 chromosome boundaries
    bedClip \
        "${basename}.merged.bedGraph" \
        "$CHROM_SIZES" \
        "${basename}.clipped.bedGraph"

    # Convert bedGraph to bigWig
    bedGraphToBigWig \
        "${basename}.clipped.bedGraph" \
        "$CHROM_SIZES" \
        "${basename}.bw"

    # Remove temporary files
    rm -f \
        "${basename}.bedGraph" \
        "${basename}.merged.bedGraph" \
        "${basename}.clipped.bedGraph"

    echo "Created ${basename}.bw"
done

echo "All conversions complete!"