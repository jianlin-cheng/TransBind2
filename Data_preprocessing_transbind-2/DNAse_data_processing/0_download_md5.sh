#!/bin/bash

HOME_DIR="/path/to/HOME_DIR"

OUTPUT_DIR="$HOME_DIR/data/downloads/wgEncodeAwgDnaseUniform"
MD5_FILE="$HOME_DIR/data/downloads/wgEncodeAwgDnaseUniform/md5sum.txt"

mkdir -p "$OUTPUT_DIR"
cd "$OUTPUT_DIR" || exit 1

# Download files listed in md5sum.txt
cut -d' ' -f1 "$MD5_FILE" | while read -r filename; do
    wget -c \
        "https://hgdownload.soe.ucsc.edu/goldenPath/hg19/encodeDCC/wgEncodeAwgDnaseUniform/${filename}"
done

# Verify downloaded files
awk '{print $2, $1}' "$MD5_FILE" | md5sum -c -