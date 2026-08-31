import os
import requests
import glob

# Function to extract UniProt ID from FASTA header
def extract_uniprot_id(fasta_header):
    try:
        parts = fasta_header.split('|')
        if len(parts) >= 2:
            return parts[1]  # The UniProt ID is the second part
        else:
            raise ValueError("The FASTA header format is not as expected.")
    except Exception as e:
        print(f"Error extracting UniProt ID: {e}")
        return None

# Function to process FASTA file and extract UniProt ID
def process_fasta_file(filepath):
    try:
        with open(filepath, 'r') as file:
            header = file.readline().strip()  # Read the header line
            if header.startswith('>'):
                return extract_uniprot_id(header)
            else:
                print(f"The file {filepath} does not appear to be in FASTA format.")
    except FileNotFoundError:
        print(f"The file {filepath} was not found.")
    except Exception as e:
        print(f"An error occurred while processing the file {filepath}: {e}")
    return None

# Function to download PDB file from AlphaFold
def download_pdb_from_alphafold(fasta_filename, uniprot_id, output_dir):
    #pdb_url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.pdb"
    pdb_url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v6.pdb"
    response = requests.get(pdb_url)
    if response.status_code == 200:
        # Use the FASTA filename (without extension) directly for the PDB
        base_name = os.path.splitext(os.path.basename(fasta_filename))[0]
        pdb_filename = os.path.join(output_dir, f"{base_name}.pdb")
        
        with open(pdb_filename, 'wb') as file:
            file.write(response.content)
        print(f"PDB file for {base_name} downloaded successfully.")
        return pdb_filename
    else:
        print(f"Error: Unable to fetch PDB file for UniProt ID {uniprot_id}. Response Code: {response.status_code}")
        return None

# Main function to process all FASTA files and download PDB files
def main():

    HOME_DIR = "/path/to/HOME_DIR"

    fasta_folder = os.path.join(
        HOME_DIR,
        "Protein_data/fasta"
    )

    output_dir = os.path.join(
        HOME_DIR,
        "Protein_data/AF_structure"
    )

    if not os.path.exists(fasta_folder):
        print(f"FASTA folder {fasta_folder} does not exist.")
        return

    os.makedirs(output_dir, exist_ok=True)

    fasta_files = glob.glob(os.path.join(fasta_folder, '*.fasta'))
    fasta_files += glob.glob(os.path.join(fasta_folder, '*.fa'))
    fasta_files += glob.glob(os.path.join(fasta_folder, '*.faa'))

    if not fasta_files:
        print(f"No FASTA files found in {fasta_folder}")
        return

    print(f"Found {len(fasta_files)} FASTA files to process.\n")

    for fasta_file in fasta_files:
        print(f"Processing {os.path.basename(fasta_file)}...")

        uniprot_id = process_fasta_file(fasta_file)

        if uniprot_id:
            download_pdb_from_alphafold(
                fasta_file,
                uniprot_id,
                output_dir
            )
        else:
            print(
                f"Skipping {fasta_file} - "
                "could not extract UniProt ID.\n"
            )

    print("\nProcessing completed.")

# Main script execution
if __name__ == "__main__":
    main()