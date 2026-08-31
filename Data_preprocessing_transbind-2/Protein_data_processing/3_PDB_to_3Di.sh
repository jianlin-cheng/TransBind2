#!/bin/bash

HOME_DIR="/path/to/HOME_DIR"

PDB_DIR="$HOME_DIR/Protein_data/AF_structure"
OUTPUT_DIR="$HOME_DIR/Protein_data/3Di_tokens"
TEMP_DB="/tmp/foldseek_temp_db"

mkdir -p "$OUTPUT_DIR"

echo "Creating structure database..."
foldseek createdb "$PDB_DIR" "$TEMP_DB"

echo "Linking header database..."
foldseek lndb "${TEMP_DB}_h" "${TEMP_DB}_ss_h"

echo "Converting 3Di to FASTA..."
foldseek convert2fasta "${TEMP_DB}_ss" "$OUTPUT_DIR/all_3di.fasta"

echo "Done! 3Di sequences saved to $OUTPUT_DIR/all_3di.fasta"

if [ ! -f "$OUTPUT_DIR/all_3di.fasta" ]; then
    echo "ERROR: FASTA file was not created!"
    exit 1
fi

echo "Splitting into individual 3Di files..."
cd "$OUTPUT_DIR" || exit 1

awk '
/^>/ {
    if (filename) close(filename)
    filename = substr($1, 2) ".3di"
    next
}
{
    print > filename
}
' all_3di.fasta

for file in *.3di; do
    if [ -f "$file" ]; then
        name="${file%.3di}"
        seq=$(cat "$file")

        echo ">$name" > "$file.tmp"
        echo "$seq" >> "$file.tmp"

        mv "$file.tmp" "$file"
    fi
done

num_files=$(find . -maxdepth 1 -name "*.3di" | wc -l)

echo "Created $num_files individual 3Di files"

rm -rf "${TEMP_DB}"*

echo "All done!"