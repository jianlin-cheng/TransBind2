import csv
import re

# Function to parse the metadata line
def parse_metadata_line(line):
    parts = line.strip().split('\t')
    if len(parts) < 2:
        return None
    
    file_name = parts[0]
    metadata_str = parts[1]
    
    # Parse the key-value pairs
    metadata = {}
    for pair in metadata_str.split('; '):
        if '=' in pair:
            key, value = pair.split('=', 1)
            metadata[key.strip()] = value.strip()
    
    return file_name, metadata

# Function to extract the accession from the file name
def extract_accession(file_name):
    # Remove file extension
    accession = file_name
    if '.narrowPeak.gz' in accession:
        accession = accession.replace('.narrowPeak.gz', '')
    elif '.bed.gz' in accession:
        accession = accession.replace('.bed.gz', '')
    elif '.bed' in accession:
        accession = accession.replace('.bed', '')
    
    return accession

# Read your metadata file and convert it
def convert_metadata(input_file, output_file):
    with open(output_file, 'w', newline='') as out_f:
        writer = csv.writer(out_f, delimiter='\t')
        
        # Write the header expected by the DeepSEA script
        writer.writerow([
            'File accession', 'Output type', 'Experiment accession', 'Biological replicate(s)', 
            'Technical replicate(s)', 'Assay', 'Biosample term name', 'Biosample type', 
            'Biosample treatments', 'Biosample genetic modifications', 'File format', 'Assembly'
        ])
        
        with open(input_file, 'r') as in_f:
            for line in in_f:
                parsed = parse_metadata_line(line)
                if not parsed:
                    continue
                
                file_name, metadata = parsed
                accession = extract_accession(file_name)
                
                # Map your metadata fields to ENCODE format
                output_type = 'peaks'
                experiment_accession = metadata.get('dccAccession', f'EXP-{accession}')
                bio_rep = '1'
                tech_rep = '1'
                assay = 'ChIP-seq'
                biosample = metadata.get('cell', 'unknown')
                biosample_type = 'cell line'
                treatments = metadata.get('treatment', '')
                if treatments == 'None':
                    treatments = ''
                genetic_mods = ''
                file_format = 'bed'
                assembly = 'hg19'  # Assuming hg19 for all
                
                writer.writerow([
                    accession, output_type, experiment_accession, bio_rep, tech_rep,
                    assay, biosample, biosample_type, treatments, genetic_mods, file_format, assembly
                ])

# Example usage
#convert_metadata('../data/downloads/wgEncodeAwgTfbsUniform/files.txt', '../data/encoded_metadata.tsv')