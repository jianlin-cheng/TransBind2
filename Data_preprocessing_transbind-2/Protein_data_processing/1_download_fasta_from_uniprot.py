import os
import requests
import csv

HOME_DIR = "/path/to/HOME_DIR"


def download_fasta_from_uniprot(transcription_factor, uniprot_id, output_dir):
    fasta_url = f"https://www.uniprot.org/uniprot/{uniprot_id}.fasta"

    try:
        response = requests.get(fasta_url)

        if response.status_code == 200:
            safe_tf_name = (
                transcription_factor
                .replace('/', '_')
                .replace(' ', '_')
            )

            fasta_filename = os.path.join(
                output_dir,
                f"{safe_tf_name}_{uniprot_id}.fasta"
            )

            with open(fasta_filename, 'w') as file:
                file.write(response.text)

            print(
                f"FASTA file for {transcription_factor} "
                f"(UniProt ID: {uniprot_id}) downloaded successfully."
            )

            return fasta_filename

        else:
            print(
                f"Error: Unable to fetch FASTA file for "
                f"UniProt ID {uniprot_id}. "
                f"Response Code: {response.status_code}"
            )
            return None

    except Exception as e:
        print(
            f"Error downloading FASTA for UniProt ID "
            f"{uniprot_id}: {e}"
        )
        return None


def read_tf_data_from_csv(csv_file):
    tf_data = []

    try:
        with open(csv_file, 'r') as file:
            reader = csv.DictReader(file)

            print(f"CSV Headers: {reader.fieldnames}")

            for row in reader:
                tf_info = {
                    'transcription_factor':
                        row['Transcription Factor'].strip(),
                    'cell_type':
                        row['Cell Type'].strip(),
                    'filename':
                        row['Filename'].strip(),
                    'uniprot_id':
                        row['UniProt ID'].strip()
                }

                tf_data.append(tf_info)

    except Exception as e:
        print(f"Error reading CSV file {csv_file}: {e}")

    return tf_data


def main():

    csv_file = os.path.join(
        HOME_DIR,
        "data/tf_celltype_list_uniprotID_161_standardized.csv"
    )

    output_dir = os.path.join(
        HOME_DIR,
        "Protein_data/fasta"
    )

    if not os.path.exists(csv_file):
        print(f"CSV file {csv_file} does not exist.")
        return

    os.makedirs(output_dir, exist_ok=True)

    tf_data = read_tf_data_from_csv(csv_file)

    if not tf_data:
        print("No transcription factor data found in the CSV.")
        return

    downloaded_combinations = set()

    for tf_info in tf_data:

        transcription_factor = tf_info['transcription_factor']
        cell_type = tf_info['cell_type']
        uniprot_id = tf_info['uniprot_id']

        unique_key = f"{transcription_factor}_{uniprot_id}"

        if unique_key not in downloaded_combinations:

            print(
                f"Processing {transcription_factor} "
                f"(Cell Type: {cell_type}) "
                f"with UniProt ID {uniprot_id}..."
            )

            success = download_fasta_from_uniprot(
                transcription_factor,
                uniprot_id,
                output_dir
            )

            if success:
                downloaded_combinations.add(unique_key)

        else:
            print(
                f"FASTA for {transcription_factor} "
                f"(UniProt ID: {uniprot_id}) "
                f"already downloaded, skipping..."
            )

    print(
        "FASTA download completed. "
        f"Total unique sequences downloaded: "
        f"{len(downloaded_combinations)}"
    )


if __name__ == "__main__":
    main()